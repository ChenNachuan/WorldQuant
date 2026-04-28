import time
import sys
import random
import re
import json
import difflib
from typing import List, Dict, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_db import get_alpha_db, AlphaDB
from core.data_fetcher import OperatorFetcher, DataFieldFetcher, SmartSearch
from core.ast_validator import ASTValidator
from core.expression_compiler import ExpressionCompiler
from core.simulation_slot_manager import SimulationSlotManager
from core.region_config import get_region_config
from core.self_optimizer import SelfOptimizer
from core.quality_monitor import QualityMonitor, AlphaMetrics
from core.log_manager import setup_logger
from evolution.genetic_engine import GeneticEngine
from evolution.bandits import OperatorFieldBandit
from evolution.similarity import TemplateSimilarity
from core.hypothesis_manager import HypothesisManager

logger = setup_logger(__name__, "pipeline")


class AlphaMiningPipeline:
    def __init__(
        self,
        region_key: str = "USA",
        model_name: str = "deepseek-coder-v2:16b",
        target_submittable: int = 2,
        max_concurrent: int = 2,
        mode: str = "auto",
    ):
        self.region_key = region_key
        self.target = target_submittable
        self.max_concurrent = max_concurrent
        self.mode = mode
        self.region_config = get_region_config(region_key)

        logger.info(f"Initializing pipeline for {region_key} ({self.region_config.universe}) | Mode: {mode}")

        self.generator = AlphaGenerator(
            max_concurrent=max_concurrent,
            region_key=region_key,
        )
        self.generator.model_name = model_name
        self.generator.initial_model = model_name

        self.operator_fetcher = OperatorFetcher(session=self.generator.sess)
        self.data_field_fetcher = DataFieldFetcher(session=self.generator.sess)
        self.smart_search = SmartSearch()
        self.ast_validator = ASTValidator()
        self.slot_manager = SimulationSlotManager(max_concurrent=max_concurrent)
        self.self_optimizer = SelfOptimizer()
        self.quality_monitor = QualityMonitor()
        self.template_similarity = TemplateSimilarity()

        # SQLite storage (improvement #1)
        self.db: AlphaDB = get_alpha_db()

        self.operators: List[Dict] = []
        self.operator_names: List[str] = []
        self.field_ids: List[str] = []
        self.fields: List[Dict] = []

        self.bandit = OperatorFieldBandit()
        self.genetic_engine: Optional[GeneticEngine] = None
        self.expression_compiler: Optional[ExpressionCompiler] = None

        self.successful_alphas: List[Tuple[str, float]] = []
        self.failed_alphas: List[Tuple[str, str]] = []
        self.generation_count = 0
        self.alphas_to_heal: Dict[str, Dict] = {}  # {base_expr: {"iteration": 1, "feedback": ""}}
        self.alphas_to_polish: Dict[str, Dict] = {} # {expr: {"sharpe": 1.5, "universe": "TOP3000"}}
        self.hypothesis_manager = HypothesisManager(generator=self.generator)


    def initialize(self):
        logger.info("Loading cached operators and data fields...")
        self.generator.initialize() # New semantic-aware init
        self.hypothesis_manager.sync_insights()
        
        self.operators = self.operator_fetcher.fetch_operators()
        self.operator_names = [op.get("name", "") for op in self.operators if op.get("name")]
        logger.info(f"  Operators: {len(self.operator_names)}")

        cache_key = f"{self.region_config.region}_{self.region_config.universe}_{self.region_config.delay}"
        self.fields = self.data_field_fetcher.fetch_data_fields(
            region=self.region_config.region,
            universe=self.region_config.universe,
            delay=self.region_config.delay,
        )
        self.smart_search.update_fields(self.data_field_fetcher.data_fields)
        numeric_fields = self.data_field_fetcher.get_numeric_fields(cache_key)
        self.field_ids = [f.get("id", "") for f in numeric_fields if f.get("id")]
        logger.info(f"  Data fields: {len(self.field_ids)} numeric")

        self.ast_validator = ASTValidator(
            known_operators=set(self.operator_names),
            known_fields=set(self.field_ids),
            strict=True,
        )

        self.bandit = OperatorFieldBandit(
            operators=self.operator_names,
            fields=self.field_ids[:100],
        )

        self.genetic_engine = GeneticEngine(
            operators=self.operator_names,
            fields=self.field_ids[:50],
            windows=self.self_optimizer.get_recommended_windows(),
            groups=self.self_optimizer.get_recommended_groups(),
        )

        self.expression_compiler = ExpressionCompiler(
            operators=self.operator_names,
            fields=self.field_ids[:50],
        )

        logger.info("Pipeline initialized!")

    def count_submittable(self) -> int:
        """Count alphas in DB that meet submittable criteria (Sharpe >= 1.25, Fitness >= 1.0, Turnover <= 0.7)."""
        successes = self.db.get_successful_alphas(min_fitness=1.0, min_sharpe=1.25, limit=1000)
        # Filter by turnover <= 0.7 if not already handled by get_successful_alphas
        # Our save_alpha already filters for success, but let's be explicit
        return len([a for a in successes if a.get("turnover", 1.0) <= 0.7])

    def run(self, max_batches: int = 50):
        self.initialize()

        logger.info(f"Starting closed-loop mining. Target: {self.target} submittable alphas")
        logger.info(f"Mode: AI generate → test → genetic evolve → bandit guide → repeat")

        batch_num = 0
        while batch_num < max_batches:
            submittable = self.count_submittable()
            logger.info(f"{'='*60}")
            logger.info(f"Batch #{batch_num + 1} | Submittable: {submittable}/{self.target} | "
                       f"Successful: {len(self.successful_alphas)} | Failed: {len(self.failed_alphas)}")

            if submittable >= self.target:
                logger.info(f"TARGET REACHED! {submittable} submittable alphas.")
                break

            batch_num += 1

            phase = self._select_phase()
            logger.info(f"Phase: {phase}")

            if phase == "healer_evolve":
                expressions = self._healer_evolve_phase()
            elif phase == "ai_generate":
                expressions = self._ai_generate_phase()
            elif phase == "genetic_evolve":
                expressions = self._genetic_evolve_phase()
            elif phase == "template_compile":
                expressions = self._template_compile_phase()
            else:
                expressions = self._ai_generate_phase()

            if not expressions:
                logger.warning("No expressions generated, sleeping...")
                time.sleep(10)
                continue

            validated, invalid = self.ast_validator.validate_batch(expressions)
            if invalid:
                logger.info(f"AST filtered: {len(invalid)} invalid, {len(validated)} kept")

            deduped = self.template_similarity.deduplicate(validated)
            if len(deduped) < len(validated):
                logger.info(f"Dedup: {len(validated)} -> {len(deduped)}")

            expressions = deduped[:10]

            logger.info(f"Testing {len(expressions)} expressions...")
            self._test_and_record(expressions)

            self._update_bandits()

            summary = self.self_optimizer.get_optimization_summary()
            quality = self.quality_monitor.get_summary()
            logger.info(f"Optimizer: success_rate={summary['success_rate']}, "
                       f"top_ops={summary['top_operators'][:3]}")
            logger.info(f"Quality: 24h_rate={quality['last_24h_success_rate']}, "
                       f"degraded={quality['degradation']['is_degraded']}")

            self.generation_count += 1

            sleep_time = 20 if phase == "ai_generate" else 5
            logger.info(f"Sleeping {sleep_time}s...")
            time.sleep(sleep_time)

        self._print_final_report()

    def _select_phase(self) -> str:
        if self.mode == "grid":
            return "template_compile"
            
        if self.alphas_to_heal:
            return "healer_evolve"
        if self.alphas_to_polish:
            return "polisher_phase"
        if len(self.successful_alphas) >= 3 and self.generation_count % 3 == 1:
            return "genetic_evolve"
        if self.generation_count % 4 == 2:
            return "template_compile"
        return "ai_generate"

    def _healer_evolve_phase(self) -> List[str]:
        if not self.alphas_to_heal:
            return []
            
        base_expr = list(self.alphas_to_heal.keys())[0]
        state = self.alphas_to_heal[base_expr]
        
        if state["iteration"] > 3:
            logger.info(f"Alpha {base_expr[:30]}... failed 3 healing iterations. Discarding.")
            del self.alphas_to_heal[base_expr]
            return self._ai_generate_phase()
            
        logger.info(f"Phase: Healer Evolve (Iteration {state['iteration']}/3) for {base_expr[:40]}...")
        mutations = self.generator.heal_high_turnover_alpha(base_expr, state["feedback"])
        
        if not mutations:
            logger.warning("Turnover healer returned no mutations.")
            del self.alphas_to_heal[base_expr]
            return []
            
        expressions = []
        for mut in mutations:
            expressions.append(mut)
            
        # We process this batch. The next time we evaluate the queue, we will increment iteration 
        # based on results. For simplicity, we increment iteration now, and rely on _test_and_record
        # to re-insert or update feedback if it fails. But wait, _test_and_record is stateless.
        # Let's pass the state tracking to where we evaluate results in _test_and_record.
        
        state["iteration"] += 1
        state["current_mutations"] = mutations  # track what we are testing
        return expressions

    def _polisher_phase(self) -> List[Tuple[str, Dict]]:
        if not self.alphas_to_polish:
            return []
            
        expr = list(self.alphas_to_polish.keys())[0]
        state = self.alphas_to_polish.pop(expr)
        
        logger.info(f"Phase: Alpha Polisher (Optimizing settings for {expr[:40]}...)")
        
        # Grid Search Settings
        variations = []
        neutralizations = ["INDUSTRY", "SUBINDUSTRY", "SECTOR"]
        decays = [0, 4, 8, 12]
        universes = ["TOP3000", "TOP1000", "TOP500"]
        
        # We pick a subset to keep the batch size reasonable (~12-15)
        for neut in neutralizations:
            for decay in decays:
                # Use a representative universe or the original one
                univ = state.get("universe", "TOP3000")
                variations.append((expr, {
                    "neutralization": neut,
                    "decay": decay,
                    "universe": univ
                }))
        
        # Also try different universes with best neut/decay if we wanted to be more thorough,
        # but for now, 3*4=12 is a solid batch.
        return variations

    def _ai_generate_phase(self) -> List[str]:
        logger.info("Phase: AI Generate (Ollama + Forum Insights)")
        data_fields = self.generator.get_data_fields()
        operators = self.generator.get_operators()

        if not data_fields or not operators:
            logger.error("No data fields or operators available")
            return []

        # Fetch random hypotheses for inspiration (AlphaSpire style)
        hypotheses = self.hypothesis_manager.get_random_theses(count=2)
        
        # Generate expressions with semantic tags and hypotheses
        ideas = self.generator.generate_alphas(data_fields, operators, hypothesis_context=hypotheses)
        logger.info(f"AI generated {len(ideas)} raw ideas")
        return ideas

    def _genetic_evolve_phase(self) -> List[str]:
        if not self.successful_alphas or not self.genetic_engine:
            logger.info("Not enough successful alphas for evolution, falling back to AI")
            return self._ai_generate_phase()

        logger.info(f"Phase: Genetic Evolution (from {len(self.successful_alphas)} successful alphas)")

        recommended_windows = self.self_optimizer.get_recommended_windows()
        recommended_groups = self.self_optimizer.get_recommended_groups()
        self.genetic_engine.windows = recommended_windows if recommended_windows else [5, 20, 60, 120, 252]
        self.genetic_engine.groups = recommended_groups if recommended_groups else ["sector", "industry"]

        offspring = self.genetic_engine.evolve(self.successful_alphas)
        logger.info(f"Genetic engine produced {len(offspring)} offspring")
        return offspring

    def _template_compile_phase(self) -> List[Union[str, Tuple[str, Dict]]]:
        if not self.expression_compiler:
            return self._ai_generate_phase()

        logger.info(f"Phase: Template Compilation (Mode: {self.mode})")

        recommended_windows = self.self_optimizer.get_recommended_windows()
        recommended_groups = self.self_optimizer.get_recommended_groups()

        k_fields = 30 if self.mode == "grid" else 10
        bandit_combo = self.bandit.select_combination(k_operators=10, k_fields=k_fields)
        bandit_fields = bandit_combo.get("fields", self.field_ids[:k_fields])
        bandit_ops = bandit_combo.get("operators", self.operator_names[:10])

        templates = ExpressionCompiler.get_default_templates()
        
        if self.mode == "grid":
            replacements = {
                "FIELD": bandit_fields[:30],
                "FIELD2": bandit_fields[:15],
                "WINDOW": [str(w) for w in [5, 10, 20, 40, 60, 120, 252]],
                "WINDOW2": [str(w) for w in [10, 20, 60, 120]],
                "GROUP": ["sector", "industry", "subindustry"],
                "RANK": ["5", "10", "20"],
                "OP": bandit_ops[:10],
            }
            max_expr = 150
        else:
            replacements = {
                "FIELD": bandit_fields[:15],
                "FIELD2": bandit_fields[:10],
                "WINDOW": [str(w) for w in (recommended_windows or [5, 20, 60, 120, 252])],
                "WINDOW2": [str(w) for w in [60, 120, 252]],
                "GROUP": recommended_groups or ["sector", "industry"],
                "RANK": ["5", "20"],
                "OP": bandit_ops[:5],
            }
            max_expr = 30

        expressions = self.expression_compiler.compile_templates(
            templates, replacements, max_expressions=max_expr
        )
        logger.info(f"Template compiler produced {len(expressions)} expressions")
        
        if self.mode == "grid":
            grid_items = []
            universes = ["TOP3000", "TOP1000", "TOP500", "TOP200"]
            for expr in expressions:
                # The expression compiler might have put {GROUP} inside the expression.
                # We extract the group name from the expression if possible to match neutralization.
                neut = "INDUSTRY"
                if "subindustry" in expr: neut = "SUBINDUSTRY"
                elif "sector" in expr: neut = "SECTOR"
                
                # Pick a random universe for each or iterate (random is better to avoid explosion)
                import random
                univ = random.choice(universes)
                
                grid_items.append((expr, {"neutralization": neut, "universe": univ}))
            return grid_items
            
        return expressions

    def _test_and_record(self, items: List[Union[str, Tuple[str, Dict]]]):
        """Test expressions using native API batch submission."""
        novel = []
        for item in items:
            expr = item if isinstance(item, str) else item[0]
            settings = None if isinstance(item, str) else item[1]
            univ = settings.get("universe") if settings else None
            neut = settings.get("neutralization") if settings else None
            
            if not self.db.expression_exists(expr, self.region_key, univ, neut):
                novel.append((expr, settings))

        if len(novel) < len(items):
            logger.info(f"SQLite dedup: {len(items)} -> {len(novel)} novel expressions")
        if not novel:
            # If we are in healer phase and all are deduped (meaning we tried them), 
            # we should mark them as failed to trigger next iteration.
            self._evaluate_healing_results([])
            return

        # Native Batch Submission
        batch_results = self.generator.test_alphas_batch(novel)
        
        results = []
        for res in batch_results:
            expr = res.get("expr")
            if not expr:
                continue
                
            if res.get("status") == "error":
                error_msg = res.get("message", "Unknown error")
                logger.error(f"Failed to submit: {expr[:120]} | Error: {error_msg}")
                self.failed_alphas.append((expr, error_msg))
                self.db.save_error(expr, "submit_error", error_msg)
                self._record_failure(expr)
                results.append({"expr": expr, "success": False})
                continue
                
            alpha_data = res.get("alpha_data")
            if alpha_data:
                is_data = alpha_data.get("is", {})
                fitness = is_data.get("fitness")
                sharpe = is_data.get("sharpe")
                turnover = is_data.get("turnover")
                returns_val = is_data.get("returns")

                success = (
                    fitness is not None and fitness >= 1.0
                    and sharpe is not None and sharpe >= 1.25
                    and turnover is not None and turnover <= 0.7
                )

                # Find the settings used for this expression to pass to db
                settings = next((s for e, s in novel if e == expr), None)
                univ = settings.get("universe") if settings else self.region_config.universe
                neut = settings.get("neutralization") if settings else self.region_config.neutralization
                
                # Check for Turnover Healing (high sharpe, high turnover)
                if sharpe is not None and sharpe >= 1.25 and turnover is not None and turnover > 0.7:
                    logger.info(f"🩺 Alpha Healer triggered! Sharpe={sharpe:.2f}, Tov={turnover:.2f}, Expr={expr[:40]}...")
                    self.alphas_to_heal[expr] = {
                        "iteration": 1,
                        "best_sharpe": sharpe,
                        "best_turnover": turnover,
                        "feedback": "Initial high-turnover formula.\n",
                        "current_mutations": []
                    }

                # Save to database
                self.db.save_alpha(
                    expression=expr,
                    alpha_data=alpha_data,
                    source="pipeline",
                    settings={
                        "universe": univ,
                        "neutralization": neut,
                        "region": self.region_key
                    }
                )

                # Monitor quality and operators
                metrics = AlphaMetrics(
                    expression=expr,
                    fitness=fitness,
                    sharpe=sharpe,
                    turnover=turnover,
                    returns=returns_val,
                    source="pipeline"
                )
                self.quality_monitor.record(metrics)

                # Update SelfOptimizer
                self.self_optimizer.record_result(
                    expression=expr,
                    fitness=fitness,
                    sharpe=sharpe,
                    turnover=turnover,
                    success=success
                )

                if success:
                    self.successful_alphas.append((expr, alpha_data))
                    logger.info(f"SUCCESS: {expr[:60]}... fit={fitness} shp={sharpe} tov={turnover}")
                    
                    # Alpha Polisher trigger: if potential is high, queue for settings optimization
                    if sharpe is not None and sharpe >= 1.2 and fitness is not None and fitness >= 0.8:
                        if expr not in self.alphas_to_polish:
                            logger.info(f"✨ Alpha Polisher triggered for potential gem: {expr[:40]}...")
                            self.alphas_to_polish[expr] = {
                                "sharpe": sharpe,
                                "fitness": fitness,
                                "universe": univ
                            }

                    # Rich Metadata Management
                    alpha_id = res.get("alpha")
                    if alpha_id:
                        tag_list = ["ai-gen"]
                        if turnover < 0.1: tag_list.append("low-turnover")
                        if sharpe > 2.0: tag_list.append("elite")
                        
                        source_tag = "[Grid]" if self.mode == "grid" else "[AI-Gen]"
                        name = f"{source_tag} Alpha {alpha_id[:6]} [S:{sharpe:.2f}]"
                        desc = f"Auto-generated alpha. Fitness: {fitness:.4f}, Sharpe: {sharpe:.4f}, Turnover: {turnover:.4f}"
                        self._update_alpha_metadata(alpha_id, name, tag_list, desc)

                    # Generate DPO Pair (Chosen)
                    if hasattr(self.generator, "current_prompt_context") and self.generator.current_prompt_context:
                        self.db.save_dpo_pair(
                            prompt=self.generator.current_prompt_context,
                            chosen=expr,
                            rejected=""
                        )
                else:
                    self.failed_alphas.append((expr, "Sub-par metrics"))
                    self._record_failure(expr)
                    
                    # Generate DPO Pair (Rejected)
                    if hasattr(self.generator, "current_prompt_context") and self.generator.current_prompt_context:
                        # Only reject if it had a valid simulation but was weak, to teach the model
                        if fitness is not None:
                            self.db.save_dpo_pair(
                                prompt=self.generator.current_prompt_context,
                                chosen="",
                                rejected=expr
                            )
                    
                    fmt_fit = round(fitness, 2) if fitness else fitness
                    fmt_shp = round(sharpe, 2) if sharpe else sharpe
                    fmt_tov = round(turnover, 4) if turnover else turnover
                    logger.info(f"Rejected: {expr[:60]}... fit={fmt_fit} shp={fmt_shp} tov={fmt_tov}")

                    # Alpha Flipper: recycle extremely negative-Sharpe alphas
                    if not success and sharpe is not None and sharpe <= -1.0:
                        logger.info(f"💡 Alpha Flipper triggered! Sharpe={sharpe:.2f}, flipping sign...")
                        if expr.startswith("-rank("):
                            flipped_expr = expr.replace("-rank(", "rank(", 1)
                        elif expr.startswith("rank("):
                            flipped_expr = "-1 * " + expr
                        else:
                            flipped_expr = f"-1 * ({expr})"
                        existing_exprs = (
                            {e for e, _ in self.failed_alphas}
                            | {e for e, _ in self.successful_alphas}
                        )
                        if flipped_expr not in existing_exprs and not self.db.expression_exists(flipped_expr, self.region_key):
                            self.generator.retry_queue.add(flipped_expr)
                            logger.info(f"  ↪ Queued flipped: {flipped_expr[:60]}...")
                
                results.append({"expr": expr, "success": success, "sharpe": sharpe, "turnover": turnover})
            else:
                self.failed_alphas.append((expr, "no alpha data"))
                results.append({"expr": expr, "success": False})

        self._evaluate_healing_results(results)
    def _evaluate_healing_results(self, results: List[Dict]):
        # Check if any base_expr in healing queue was just tested
        for base_expr, state in list(self.alphas_to_heal.items()):
            mutations = state.get("current_mutations", [])
            if not mutations:
                continue
                
            # Find results for these mutations
            mut_results = [r for r in results if r["expr"] in mutations]
            if not mut_results:
                # None of the mutations were successfully simulated or all were duplicates
                state["feedback"] += f"Iteration {state['iteration']-1}: All mutations failed simulation or were duplicates.\n"
                continue
                
            # Check if any succeeded
            success = any(r["success"] for r in mut_results)
            if success:
                logger.info(f"Successfully healed alpha! Removing {base_expr[:40]} from queue.")
                del self.alphas_to_heal[base_expr]
                continue
                
            # If failed, find the one with best Sharpe to generate feedback
            best_mut = max(mut_results, key=lambda x: x.get("sharpe", -99) if x.get("sharpe") is not None else -99)
            sharpe = best_mut.get("sharpe", 0)
            tov = best_mut.get("turnover", 0)
            expr = best_mut["expr"]
            
            if sharpe is not None and sharpe < 1.0:
                state["feedback"] += f"Attempt {state['iteration']-1}: Expression `{expr}` resulted in Sharpe={sharpe:.4f} and Turnover={tov:.4f}. This is a massive DROP in Sharpe ratio. DO NOT use that structural approach again.\n"
            elif tov is not None and tov > 0.7:
                state["feedback"] += f"Attempt {state['iteration']-1}: Expression `{expr}` resulted in Sharpe={sharpe:.4f} but Turnover={tov:.4f}. Turnover is STILL TOO HIGH. You need to use a STRONGER reduction method like Discretization or Boolean Masking.\n"
            else:
                state["feedback"] += f"Attempt {state['iteration']-1}: Failed for other reasons. Try again.\n"

    def _record_failure(self, expr: str):
        """Record a failed expression to optimizer and quality monitor."""
        self.self_optimizer.record_result(
            expression=expr, fitness=None, sharpe=None,
            turnover=None, success=False,
        )
        self.quality_monitor.record(AlphaMetrics(
            expression=expr, source="pipeline", fitness=None,
        ))

    def _try_self_correct(self, expr: str, error_msg: str) -> Optional[str]:
        """Attempt to fix a broken expression based on the API error message.

        Improvement #2: Template Self-Correction.
        """
        if not error_msg:
            return None

        import re as _re

        # Fix 1: Unknown variable — find closest matching field
        unknown_var_match = _re.search(r'unknown variable "([^"]+)"', error_msg, _re.IGNORECASE)
        if unknown_var_match:
            bad_field = unknown_var_match.group(1)
            closest = difflib.get_close_matches(bad_field, self.field_ids, n=1, cutoff=0.5)
            if closest:
                fixed = expr.replace(bad_field, closest[0])
                logger.info(f"  🔧 Fix unknown var: '{bad_field}' → '{closest[0]}'")
                return fixed
            return None

        # Fix 2: Unknown operator — find closest matching operator
        unknown_op_match = _re.search(r'unknown (?:function|operator) "([^"]+)"', error_msg, _re.IGNORECASE)
        if unknown_op_match:
            bad_op = unknown_op_match.group(1)
            closest = difflib.get_close_matches(bad_op, self.operator_names, n=1, cutoff=0.5)
            if closest:
                fixed = expr.replace(bad_op, closest[0])
                logger.info(f"  🔧 Fix unknown op: '{bad_op}' → '{closest[0]}'")
                return fixed
            return None

        # Fix 3: Missing lookback / wrong arg count — try adding/removing a window param
        if "lookback" in error_msg.lower() or "input count" in error_msg.lower():
            # Try adding a default window of 20
            func_match = _re.search(r'(\w+)\(([^)]+)\)', expr)
            if func_match:
                func_name = func_match.group(1)
                func_args = func_match.group(2)
                if "," not in func_args:
                    fixed = expr.replace(
                        f"{func_name}({func_args})",
                        f"{func_name}({func_args}, 20)"
                    )
                    logger.info(f"  🔧 Fix missing lookback: added window=20")
                    return fixed

        return None

    def _update_bandits(self):
        for expr, fitness in self.successful_alphas[-20:]:
            ops = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr)
            fields = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", expr, re.IGNORECASE)
            non_op_fields = [f for f in fields if not any(f.startswith(p) for p in ["ts_", "group_", "vec_"])]
            reward = min(fitness / 2.0, 1.0) if fitness > 0 else 0.0
            self.bandit.record_result(operators=ops, fields=non_op_fields, fitness=fitness)

        for expr, reason in self.failed_alphas[-20:]:
            ops = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr)
            fields = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", expr, re.IGNORECASE)
            non_op_fields = [f for f in fields if not any(f.startswith(p) for p in ["ts_", "group_", "vec_"])]
            self.bandit.record_result(operators=ops, fields=non_op_fields, fitness=0.0)

    def _update_alpha_metadata(self, alpha_id: str, name: str, tags: List[str], description: str = ""):
        """Rich metadata management (borrowed from forum framework).
        Automatically categorizes, tags, and names alphas on the BRAIN platform.
        """
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
            params = {
                "color": "green",
                "name": name,
                "tags": tags,
                "combo": {"description": description},
            }
            resp = self.generator.sess.patch(url, json=params, timeout=15)
            if resp.status_code in (200, 204):
                logger.info(f"Updated metadata for alpha {alpha_id} on BRAIN platform.")
            else:
                logger.warning(f"Failed to update metadata for {alpha_id}: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Error updating metadata for {alpha_id}: {e}")

    def _print_final_report(self):
        submittable = self.count_submittable()
        optimizer_summary = self.self_optimizer.get_optimization_summary()
        quality_summary = self.quality_monitor.get_summary()
        bandit_stats = self.bandit.get_stats()

        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Submittable alphas: {submittable}")
        logger.info(f"Successful (fitness>=1): {len(self.successful_alphas)}")
        logger.info(f"Failed: {len(self.failed_alphas)}")
        logger.info(f"Total tested: {optimizer_summary['total_tested']}")
        logger.info(f"Overall success rate: {optimizer_summary['success_rate']}")
        logger.info(f"Top operators: {optimizer_summary['top_operators']}")
        logger.info(f"Top windows: {optimizer_summary['top_windows']}")
        logger.info(f"Quality degradation: {quality_summary['degradation']['is_degraded']}")
        logger.info(f"Bandit top operators: {bandit_stats['operators']['top_arms'][:3]}")
        logger.info(f"Bandit top fields: {bandit_stats['fields']['top_arms'][:3]}")

        if self.successful_alphas:
            logger.info(f"\nTop 5 successful alphas:")
            sorted_alphas = sorted(self.successful_alphas, key=lambda x: x[1], reverse=True)
            for i, (expr, fitness) in enumerate(sorted_alphas[:5], 1):
                logger.info(f"  {i}. fitness={fitness:.4f} | {expr[:80]}")

        if self.failed_alphas:
            error_stats: Dict[str, int] = {}
            for _, reason in self.failed_alphas:
                key = reason[:50]
                error_stats[key] = error_stats.get(key, 0) + 1
            top_errors = sorted(error_stats.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"\nTop failure reasons:")
            for reason, count in top_errors:
                logger.info(f"  [{count}x] {reason}")

        # Retrospect analysis (improvement #5)
        self._print_retrospect_report()

        # Export DPO training dataset
        self.export_dpo_dataset()

    def _print_retrospect_report(self):
        """Print SQLite-backed retrospect analysis. (Improvement #5)"""
        try:
            report = self.db.get_retrospect_report(days=7)

            logger.info(f"\n{'='*60}")
            logger.info(f"RETROSPECT ANALYSIS (last 7 days)")
            logger.info(f"{'='*60}")
            logger.info(f"Total alphas in DB: {report['total_alphas']}")
            logger.info(f"Recent (7d): {report['recent_alphas']}")
            logger.info(f"Successful: {report['successful_alphas']}")

            if report['daily_summary']:
                logger.info(f"\n📊 Daily Summary:")
                for day in report['daily_summary'][:7]:
                    logger.info(
                        f"  {day['day']}: tested={day['total_tested']}, "
                        f"success={day['successes']}, "
                        f"avg_fit={day['avg_fitness']}, max_fit={day['max_fitness']}, "
                        f"max_sharpe={day['max_sharpe']}"
                    )

            if report['top_operators']:
                logger.info(f"\n🏆 Top Operators (by avg fitness):")
                for op in report['top_operators'][:5]:
                    logger.info(
                        f"  {op['operator']}: avg={op['avg_fitness']}, "
                        f"max={op['max_fitness']}, count={op['count']}"
                    )

            if report['top_fields']:
                logger.info(f"\n📋 Top Fields (by avg fitness):")
                for field in report['top_fields'][:5]:
                    logger.info(
                        f"  {field['field']}: avg={field['avg_fitness']}, "
                        f"count={field['count']}"
                    )

            if report['error_stats']:
                logger.info(f"\n🔧 Error Self-Correction Stats:")
                for err in report['error_stats'][:5]:
                    logger.info(
                        f"  {err['error_type']}: {err['count']}x, "
                        f"fixed={err['fixed_count']} ({err['fix_rate']}%)"
                    )
        except Exception as e:
            logger.warning(f"Retrospect report error: {e}")

    def export_dpo_dataset(self, filename: str = "dpo_training_data.jsonl"):
        """Export mining results as DPO fine-tuning pairs (chosen=success, rejected=fail)."""
        if not self.successful_alphas or not self.failed_alphas:
            logger.info("Not enough data for DPO export (need both successes and failures)")
            return

        import os
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)

        pairs_written = 0
        with open(filepath, "a", encoding="utf-8") as f:
            for i in range(min(len(self.successful_alphas), len(self.failed_alphas))):
                chosen_expr = self.successful_alphas[i][0]
                rejected_expr = self.failed_alphas[i][0]
                dpo_pair = {
                    "prompt": (
                        "You are an elite Quantitative Researcher participating in the "
                        "WorldQuant IQC. Generate a HIGH-SHARPE FASTEXPR alpha expression."
                    ),
                    "chosen": chosen_expr,
                    "rejected": rejected_expr,
                }
                f.write(json.dumps(dpo_pair) + "\n")
                pairs_written += 1

        logger.info(f"✅ DPO dataset updated: {filepath} (+{pairs_written} pairs)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Closed-loop Alpha Mining Pipeline")
    parser.add_argument("--region", type=str, default="USA", help="Region key")
    parser.add_argument("--model", type=str, default="deepseek-coder-v2:16b", help="Ollama model")
    parser.add_argument("--target", type=int, default=2, help="Target submittable alphas")
    parser.add_argument("--max-batches", type=int, default=50, help="Max batches")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent simulations")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "grid"], help="Run mode: auto (AI+Genetic) or grid (Template Grid Search)")
    args = parser.parse_args()

    pipeline = AlphaMiningPipeline(
        region_key=args.region,
        model_name=args.model,
        target_submittable=args.target,
        max_concurrent=args.concurrent,
        mode=args.mode,
    )
    pipeline.run(max_batches=args.max_batches)


if __name__ == "__main__":
    main()

