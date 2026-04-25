import time
import sys
import random
import re
import json
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, ".")

from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_store import load_all_alphas, save_alpha
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

logger = setup_logger(__name__, "pipeline")


class AlphaMiningPipeline:
    def __init__(
        self,
        region_key: str = "USA",
        model_name: str = "deepseek-coder-v2:16b",
        target_submittable: int = 2,
        max_concurrent: int = 2,
    ):
        self.region_key = region_key
        self.target = target_submittable
        self.region_config = get_region_config(region_key)

        logger.info(f"Initializing pipeline for {region_key} ({self.region_config.universe})")

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

    def initialize(self):
        logger.info("Loading cached operators and data fields...")
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
        alphas = load_all_alphas()
        count = 0
        for a in alphas:
            checks = a.get("backtest", {}).get("checks", [])
            if not any(c.get("result") == "FAIL" for c in checks):
                count += 1
        return count

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

            if phase == "ai_generate":
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
        if len(self.successful_alphas) >= 3 and self.generation_count % 3 == 1:
            return "genetic_evolve"
        if self.generation_count % 4 == 2:
            return "template_compile"
        return "ai_generate"

    def _ai_generate_phase(self) -> List[str]:
        logger.info("Phase: AI Generate (Ollama)")
        data_fields = self.generator.get_data_fields()
        operators = self.generator.get_operators()

        if not data_fields or not operators:
            logger.error("No data fields or operators available")
            return []

        ideas = self.generator.generate_alpha_ideas_with_ollama(data_fields, operators)
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

    def _template_compile_phase(self) -> List[str]:
        if not self.expression_compiler:
            return self._ai_generate_phase()

        logger.info("Phase: Template Compilation")

        recommended_ops = self.self_optimizer.get_recommended_operators(10)
        recommended_windows = self.self_optimizer.get_recommended_windows()
        recommended_groups = self.self_optimizer.get_recommended_groups()

        bandit_combo = self.bandit.select_combination(k_operators=5, k_fields=10)
        bandit_fields = bandit_combo.get("fields", self.field_ids[:10])
        bandit_ops = bandit_combo.get("operators", self.operator_names[:5])

        templates = ExpressionCompiler.get_default_templates()
        replacements = {
            "FIELD": bandit_fields[:15],
            "FIELD2": bandit_fields[:10],
            "WINDOW": [str(w) for w in (recommended_windows or [5, 20, 60, 120, 252])],
            "WINDOW2": [str(w) for w in [60, 120, 252]],
            "GROUP": recommended_groups or ["sector", "industry"],
            "RANK": ["5", "20"],
            "OP": bandit_ops[:5],
        }

        expressions = self.expression_compiler.compile_templates(
            templates, replacements, max_expressions=30
        )
        logger.info(f"Template compiler produced {len(expressions)} expressions")
        return expressions

    def _test_and_record(self, expressions: List[str]):
        for expr in expressions:
            try:
                result = self.generator._test_alpha_impl(expr)

                if result.get("status") == "error":
                    error_msg = result.get("message", "")
                    self.failed_alphas.append((expr, error_msg))
                    self.self_optimizer.record_result(
                        expression=expr, fitness=None, sharpe=None,
                        turnover=None, success=False,
                    )
                    self.quality_monitor.record(AlphaMetrics(
                        expression=expr, source="pipeline", fitness=None,
                    ))
                    continue

                progress_url = result.get("result", {}).get("progress_url")
                if not progress_url:
                    self.failed_alphas.append((expr, "no progress url"))
                    continue

                alpha_data = self._monitor_simulation(progress_url)
                if alpha_data:
                    is_data = alpha_data.get("is", {})
                    fitness = is_data.get("fitness")
                    sharpe = is_data.get("sharpe")
                    turnover = is_data.get("turnover")
                    returns_val = is_data.get("returns")

                    success = fitness is not None and fitness >= 1.0

                    self.self_optimizer.record_result(
                        expression=expr, fitness=fitness, sharpe=sharpe,
                        turnover=turnover, success=success,
                    )
                    self.quality_monitor.record(AlphaMetrics(
                        expression=expr, fitness=fitness, sharpe=sharpe,
                        turnover=turnover, returns=returns_val, source="pipeline",
                    ))

                    if success:
                        self.successful_alphas.append((expr, fitness or 0))
                        save_alpha(expr, alpha_data, source="pipeline")
                        alpha_id = alpha_data.get("id")
                        if alpha_id:
                            self._mark_green(alpha_id)
                        logger.info(f"SUCCESS: {expr[:60]}... fitness={fitness:.4f} sharpe={sharpe}")
                    else:
                        self.failed_alphas.append((expr, f"fitness={fitness}"))
                        logger.info(f"Failed: {expr[:60]}... fitness={fitness}")
                else:
                    self.failed_alphas.append((expr, "simulation failed"))

            except Exception as e:
                logger.error(f"Error testing {expr[:40]}: {e}")
                self.failed_alphas.append((expr, str(e)))

            time.sleep(2)

    def _monitor_simulation(self, progress_url: str, timeout: int = 600) -> Optional[Dict]:
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.generator.sess.get(progress_url, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    return None

                data = resp.json()
                status = data.get("status")

                if status == "COMPLETE":
                    alpha_id = data.get("alpha")
                    if alpha_id:
                        alpha_resp = self.generator.sess.get(
                            f"https://api.worldquantbrain.com/alphas/{alpha_id}",
                            timeout=30,
                        )
                        if alpha_resp.status_code == 200:
                            return alpha_resp.json()
                    return None

                if status == "ERROR":
                    return None

                retry_after = resp.headers.get("Retry-After")
                wait = int(float(retry_after)) if retry_after else 5
                time.sleep(wait)

            except Exception as e:
                logger.warning(f"Monitor error: {e}")
                time.sleep(5)

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

    def _mark_green(self, alpha_id: str):
        try:
            url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
            resp = self.generator.sess.patch(url, json={"color": "green"}, timeout=15)
            if resp.status_code in (200, 204):
                logger.info(f"Marked alpha {alpha_id} as GREEN")
            else:
                logger.warning(f"Failed to mark {alpha_id} green: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Error marking {alpha_id} green: {e}")

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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Closed-loop Alpha Mining Pipeline")
    parser.add_argument("--region", type=str, default="USA", help="Region key")
    parser.add_argument("--model", type=str, default="deepseek-coder-v2:16b", help="Ollama model")
    parser.add_argument("--target", type=int, default=2, help="Target submittable alphas")
    parser.add_argument("--max-batches", type=int, default=50, help="Max batches")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent simulations")
    args = parser.parse_args()

    pipeline = AlphaMiningPipeline(
        region_key=args.region,
        model_name=args.model,
        target_submittable=args.target,
        max_concurrent=args.concurrent,
    )
    pipeline.run(max_batches=args.max_batches)


if __name__ == "__main__":
    main()
