import argparse
import requests
import json
import os
from time import sleep
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Optional
import time
import re
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from .log_manager import setup_logger
from .data_fetcher import OperatorFetcher, DataFieldFetcher, SmartSearch
from .ast_validator import ASTValidator
from .expression_compiler import ExpressionCompiler
from .simulation_slot_manager import SimulationSlotManager
from .region_config import get_region_config, RegionConfig
from .self_optimizer import SelfOptimizer
from .quality_monitor import QualityMonitor, AlphaMetrics
from evolution.similarity import TemplateSimilarity

logger = setup_logger(__name__, "generator")

class RetryQueue:
    def __init__(self, generator, max_retries=3, retry_delay=60):
        self.queue = Queue()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.generator = generator  # Store reference to generator
        self.worker = Thread(target=self._process_queue, daemon=True)
        self.worker.start()
    
    def add(self, alpha: str, retry_count: int = 0):
        self.queue.put((alpha, retry_count))
    
    def _process_queue(self):
        while True:
            if not self.queue.empty():
                alpha, retry_count = self.queue.get()
                if retry_count >= self.max_retries:
                    logger.error(f"Max retries exceeded for alpha: {alpha}")
                    continue
                    
                try:
                    result = self.generator._test_alpha_impl(alpha)  # Use _test_alpha_impl to avoid recursion
                    if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                        logger.info(f"Simulation limit exceeded, requeueing alpha: {alpha}")
                        time.sleep(self.retry_delay)
                        self.add(alpha, retry_count + 1)
                    else:
                        self.generator.results.append({
                            "alpha": alpha,
                            "result": result
                        })
                except Exception as e:
                    logger.error(f"Error processing alpha: {str(e)}")
                    
            time.sleep(1)  # Prevent busy waiting

class AlphaGenerator:
    def __init__(self, ollama_url: str = None, max_concurrent: int = 2, region_key: str = "USA"):
        from .config import load_credentials, get_ollama_url
        self.sess = requests.Session()
        self._fix_proxy()
        self.setup_auth()
        self.ollama_url = ollama_url or get_ollama_url()
        self.results = []
        self.pending_results = {}
        self.retry_queue = RetryQueue(self)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.vram_cleanup_interval = 10
        self.operation_count = 0
        
        self.initial_model = getattr(self, 'model_name', 'qwen3.5:35b')
        self.error_count = 0
        self.max_errors_before_downgrade = 3
        self.model_fleet = [
            'qwen3.5:35b',
            'qwen3-coder:latest',
            'deepseek-coder-v2:16b'
        ]
        self.current_model_index = 0

        self.region_config = get_region_config(region_key)
        self.operator_fetcher = OperatorFetcher(session=self.sess)
        self.data_field_fetcher = DataFieldFetcher(session=self.sess)
        self.smart_search = SmartSearch()
        self.ast_validator = ASTValidator()
        self.slot_manager = SimulationSlotManager(
            max_concurrent=self.region_config.max_concurrent
        )
        self.self_optimizer = SelfOptimizer()
        self.quality_monitor = QualityMonitor()
        self.template_similarity = TemplateSimilarity()
        self.expression_compiler = ExpressionCompiler()
        
        self._cached_operators: List[Dict] = []
        self._cached_fields: List[Dict] = []
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        operators = self.get_operators()
        fields = self.get_data_fields()
        if operators or fields:
            op_names = set(op.get("name", "") for op in operators if op.get("name"))
            field_ids = set(f.get("id", "") for f in fields if f.get("id"))
            self.ast_validator = ASTValidator(
                known_operators=op_names,
                known_fields=field_ids,
            )
            self.expression_compiler = ExpressionCompiler(
                operators=list(op_names),
                fields=list(field_ids)[:50],
            )
            self._initialized = True
            logger.info(f"Lazy-init AST validator with {len(op_names)} operators, {len(field_ids)} fields")
    
    def _fix_proxy(self) -> None:
        from .config import fix_session_proxy
        fix_session_proxy(self.sess)
    
    def setup_auth(self) -> None:
        from .config import load_credentials
        logger.info("Loading credentials from .env")
        username, password = load_credentials()
        self.sess.auth = HTTPBasicAuth(username, password)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info("Authenticating with WorldQuant Brain...")
                response = self.sess.post('https://api.worldquantbrain.com/authentication', timeout=(15, 30))
                logger.info(f"Authentication response status: {response.status_code}")
                
                if response.status_code != 201:
                    raise Exception(f"Authentication failed: {response.text}")
                return
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Auth SSL/Connection error attempt {attempt+1}/{max_retries}: {str(e)[:100]}")
                    self._fix_proxy()
                    sleep(5)
                else:
                    raise
    
    def cleanup_vram(self):
        """Perform VRAM cleanup by forcing garbage collection and waiting."""
        try:
            import gc
            gc.collect()
            logger.info("Performed VRAM cleanup")
            # Add a small delay to allow GPU memory to be freed
            time.sleep(2)
        except Exception as e:
            logger.warning(f"VRAM cleanup failed: {e}")
        
    def get_data_fields(self) -> List[Dict]:
        if self._cached_fields:
            return self._cached_fields

        fields = self.data_field_fetcher.fetch_data_fields(
            region=self.region_config.region,
            universe=self.region_config.universe,
            delay=self.region_config.delay,
        )

        self.smart_search.update_fields(self.data_field_fetcher.data_fields)

        numeric_fields = self.data_field_fetcher.get_numeric_fields(
            f"{self.region_config.region}_{self.region_config.universe}_{self.region_config.delay}"
        )

        self._cached_fields = numeric_fields
        logger.info(f"Using {len(numeric_fields)} numeric data fields (cached)")
        return numeric_fields

    def get_operators(self) -> List[Dict]:
        if self._cached_operators:
            return self._cached_operators

        operators = self.operator_fetcher.fetch_operators()
        self._cached_operators = operators
        logger.info(f"Using {len(operators)} operators (cached)")
        return operators

    def clean_alpha_ideas(self, ideas: List[str]) -> List[str]:
        self._ensure_initialized()
        cleaned_ideas = []
        for idea in ideas:
            if re.match(r'^\d+\.?$|^[a-zA-Z]+$', idea):
                continue
            
            common_words = ['it', 'the', 'is', 'are', 'captures', 'provides', 'measures']
            if any(word in idea.lower() for word in common_words):
                continue
            
            if ('=' in idea) or (';' in idea) or idea.startswith('Comment:'):
                continue
            
            is_valid, errors = self.ast_validator.validate(idea)
            if not is_valid:
                logger.debug(f"AST rejected: {idea[:60]} | errors: {errors}")
                continue
            
            cleaned_ideas.append(idea)
        
        if len(cleaned_ideas) > 1:
            cleaned_ideas = self.template_similarity.deduplicate(cleaned_ideas)
        
        return cleaned_ideas

    def generate_alpha_ideas_with_ollama(self, data_fields: List[Dict], operators: List[Dict]) -> List[str]:
        """Generate alpha ideas using Ollama with FinGPT model."""
        print("Organizing operators by category...")
        operator_by_category = {}
        for op in operators:
            category = op['category']
            if category not in operator_by_category:
                operator_by_category[category] = []
            operator_by_category[category].append({
                'name': op['name'],
                'type': op.get('type', 'SCALAR'),
                'definition': op['definition'],
                'description': op['description']
            })

        try:
            # Clear tested expressions if we hit token limit in previous attempt
            if hasattr(self, '_hit_token_limit'):
                logger.info("Clearing tested expressions due to previous token limit")
                self.results = []
                delattr(self, '_hit_token_limit')

            # Randomly sample ~35% of operators from each category (tighter, higher-precision set)
            sampled_operators = {}
            for category, ops in operator_by_category.items():
                sample_size = max(1, int(len(ops) * 0.35))  # At least 1 operator per category
                sampled_operators[category] = random.sample(ops, sample_size)

            print("Preparing prompt for FinGPT...")
            # Format operators with their types, definitions, and descriptions
            def format_operators(ops):
                formatted = []
                for op in ops:
                    formatted.append(f"{op['name']} ({op['type']})\n"
                                   f"  Definition: {op['definition']}\n"
                                   f"  Description: {op['description']}")
                return formatted

            sampled_fields = random.sample(data_fields, min(30, len(data_fields)))
            field_ids_for_prompt = [field['id'] for field in sampled_fields]

            prompt = f"""Generate 5 DIVERSE and CREATIVE FASTEXPR alpha expressions. Each expression should use a DIFFERENT structure/pattern. Return ONLY the expressions, one per line, with no comments or explanations.

Available Data Fields:
{field_ids_for_prompt}

Available Operators by Category:
Time Series:
{chr(10).join(format_operators(sampled_operators.get('Time Series', [])))}

Cross Sectional:
{chr(10).join(format_operators(sampled_operators.get('Cross Sectional', [])))}

Arithmetic:
{chr(10).join(format_operators(sampled_operators.get('Arithmetic', [])))}

Logical:
{chr(10).join(format_operators(sampled_operators.get('Logical', [])))}

Vector:
{chr(10).join(format_operators(sampled_operators.get('Vector', [])))}

Transformational:
{chr(10).join(format_operators(sampled_operators.get('Transformational', [])))}

Group:
{chr(10).join(format_operators(sampled_operators.get('Group', [])))}

Rules:
1. Output ONLY single-line FASTEXPR expressions. No variables, no comments, no semicolons.
2. Use ONLY the provided data fields and operators above.
3. Time windows: only use {{5, 20, 60, 120, 180, 252}}.
4. Max nesting depth: 4 levels.
5. group_neutralize second arg must be one of: sector, industry, subindustry, market (no quotes).
6. Do NOT use categorical fields (sector, industry, etc.) as numeric inputs to ts_ operators.
7. All data fields must be numeric. Do NOT invent variable names.

DIVERSITY REQUIREMENTS - use a DIFFERENT pattern for each expression:
- Pattern A: group_neutralize(zscore(ts_xxx(field, window)), sector)
- Pattern B: rank(ts_corr(field1, field2, window))
- Pattern C: group_mean(divide(ts_delta(field, window), ts_std_dev(field, window)), industry)
- Pattern D: ts_rank(subtract(field, ts_mean(field, window)), window)
- Pattern E: group_zscore(multiply(rank(field1), rank(field2)), subindustry)
- Pattern F: zscore(ts_av_diff(field, window))
- Pattern G: group_neutralize(divide(ts_mean(field1, w1), ts_mean(field2, w2)), market)
- Pattern H: rank(ts_covariance(field1, field2, window))

Pick 5 DIFFERENT patterns from above. Do NOT repeat the same pattern.

Example diverse output:
group_neutralize(zscore(ts_mean(returns, 120)), sector)
rank(ts_corr(volume, close, 60))
group_mean(divide(ts_delta(revenue, 20), ts_std_dev(revenue, 60)), industry)
ts_rank(subtract(returns, ts_mean(returns, 20)), 120)
group_zscore(multiply(rank(volume), rank(returns)), subindustry)
"""

            # Prepare Ollama API request
            model_name = getattr(self, 'model_name', self.model_fleet[self.current_model_index])
            ollama_data = {
                'model': model_name,
                'prompt': prompt,
                'stream': False,
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 1000
            }

            print("Sending request to Ollama API...")
            try:
                response = requests.post(
                    f'{self.ollama_url}/api/generate',
                    json=ollama_data,
                    timeout=600  # 10 minutes timeout for large models
                )

                print(f"Ollama API response status: {response.status_code}")
                print(f"Ollama API response: {response.text[:500]}...")  # Print first 500 chars

                if response.status_code == 500:
                    logger.error(f"Ollama API returned 500 error: {response.text}")
                    # Trigger model downgrade for 500 errors
                    self._handle_ollama_error("500_error")
                    return []
                elif response.status_code != 200:
                    raise Exception(f"Ollama API request failed: {response.text}")
                    
            except requests.exceptions.Timeout:
                logger.error("Ollama API request timed out (600s)")
                # Trigger model downgrade for timeouts
                self._handle_ollama_error("timeout")
                return []
            except requests.exceptions.ConnectionError as e:
                if "Read timed out" in str(e):
                    logger.error("Ollama API read timeout")
                    # Trigger model downgrade for read timeouts
                    self._handle_ollama_error("read_timeout")
                    return []
                else:
                    raise e

            response_data = response.json()
            print(f"Ollama API response JSON keys: {list(response_data.keys())}")

            if 'response' not in response_data:
                raise Exception(f"Unexpected Ollama API response format: {response_data}")

            print("Processing Ollama API response...")
            content = response_data['response']
            
            # Extract pure alpha expressions by:
            # 1. Remove markdown backticks
            # 2. Remove numbering (e.g., "1. ", "2. ")
            # 3. Skip comments
            alpha_ideas = []
            for line in content.split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('*'):
                    continue
                # Remove numbering and backticks
                line = line.replace('`', '')
                if '. ' in line:
                    line = line.split('. ', 1)[1]
                if line and not line.startswith('Comment:'):
                    alpha_ideas.append(line)
            
            print(f"Generated {len(alpha_ideas)} alpha ideas")
            for i, alpha in enumerate(alpha_ideas, 1):
                print(f"Alpha {i}: {alpha}")
            
            # Clean and validate ideas
            cleaned_ideas = self.clean_alpha_ideas(alpha_ideas)
            logger.info(f"Found {len(cleaned_ideas)} valid alpha expressions")
            
            return cleaned_ideas

        except Exception as e:
            if "token limit" in str(e).lower():
                self._hit_token_limit = True  # Mark that we hit token limit
            logger.error(f"Error generating alpha ideas: {str(e)}")
            return []
    
    def _handle_ollama_error(self, error_type: str):
        """Handle Ollama errors by downgrading model if needed."""
        self.error_count += 1
        logger.warning(f"Ollama error ({error_type}) - Count: {self.error_count}/{self.max_errors_before_downgrade}")
        
        if self.error_count >= self.max_errors_before_downgrade:
            self._downgrade_model()
            self.error_count = 0  # Reset error count after downgrade
    
    def _downgrade_model(self):
        """Downgrade to the next smaller model in the fleet."""
        if self.current_model_index >= len(self.model_fleet) - 1:
            logger.error("Already using the smallest model in the fleet!")
            # Reset to initial model if we've exhausted all options
            self.current_model_index = 0
            self.model_name = self.initial_model
            logger.info(f"Reset to initial model: {self.initial_model}")
            return
        
        old_model = self.model_fleet[self.current_model_index]
        self.current_model_index += 1
        new_model = self.model_fleet[self.current_model_index]
        
        logger.warning(f"Downgrading model: {old_model} -> {new_model}")
        self.model_name = new_model
        
        # Update the model in the orchestrator if it exists
        try:
            # Try to update the orchestrator's model fleet manager
            if hasattr(self, 'orchestrator') and hasattr(self.orchestrator, 'model_fleet_manager'):
                self.orchestrator.model_fleet_manager.current_model_index = self.current_model_index
                self.orchestrator.model_fleet_manager.save_state()
                logger.info(f"Updated orchestrator model fleet to use: {new_model}")
        except Exception as e:
            logger.warning(f"Could not update orchestrator model fleet: {e}")
        
        logger.info(f"Successfully downgraded to {new_model}")

    def test_alpha_batch(self, alphas: List[str]) -> None:
        """Submit a batch of alphas for testing with monitoring, respecting concurrent limits."""
        logger.info(f"Starting batch test of {len(alphas)} alphas")
        for alpha in alphas:
            logger.info(f"Alpha expression: {alpha}")
        
        # Submit alphas in smaller chunks to respect concurrent limits
        max_concurrent = self.executor._max_workers
        submitted = 0
        queued = 0
        
        for i in range(0, len(alphas), max_concurrent):
            chunk = alphas[i:i + max_concurrent]
            logger.info(f"Submitting chunk {i//max_concurrent + 1}/{(len(alphas)-1)//max_concurrent + 1} ({len(chunk)} alphas)")
            
            # Submit chunk
            futures = []
            for j, alpha in enumerate(chunk, 1):
                logger.info(f"Submitting alpha {i+j}/{len(alphas)}")
                future = self.executor.submit(self._test_alpha_impl, alpha)
                futures.append((alpha, future))
            
            # Process results for this chunk
            for alpha, future in futures:
                try:
                    result = future.result()
                    if result.get("status") == "error":
                        if "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
                            self.retry_queue.add(alpha)
                            queued += 1
                            logger.info(f"Queued for retry: {alpha}")
                        else:
                            logger.error(f"Simulation error for {alpha}: {result.get('message')}")
                        continue
                        
                    sim_id = result.get("result", {}).get("id")
                    progress_url = result.get("result", {}).get("progress_url")
                    if sim_id and progress_url:
                        self.pending_results[sim_id] = {
                            "alpha": alpha,
                            "progress_url": progress_url,
                            "status": "pending",
                            "attempts": 0
                        }
                        submitted += 1
                        logger.info(f"Successfully submitted {alpha} (ID: {sim_id})")
                        
                except Exception as e:
                    logger.error(f"Error submitting alpha {alpha}: {str(e)}")
            
            # Wait between chunks to avoid overwhelming the API
            if i + max_concurrent < len(alphas):
                logger.info(f"Waiting 20 seconds before next chunk...")
                sleep(20)
        
        logger.info(f"Batch submission complete: {submitted} submitted, {queued} queued for retry")
        
        # Monitor progress until all complete or need retry
        total_successful = 0
        max_monitoring_time = 21600  # 6 hours maximum monitoring time
        start_time = time.time()
        
        while self.pending_results:
            # Check for timeout
            if time.time() - start_time > max_monitoring_time:
                logger.warning(f"Monitoring timeout reached ({max_monitoring_time}s), stopping monitoring")
                logger.warning(f"Remaining pending simulations: {list(self.pending_results.keys())}")
                break
                
            logger.info(f"Monitoring {len(self.pending_results)} pending simulations...")
            completed = self.check_pending_results()
            total_successful += completed
            sleep(5)  # Wait between checks
        
        logger.info(f"Batch complete: {total_successful} successful simulations")
        return total_successful

    def check_pending_results(self) -> int:
        """Check status of all pending simulations with proper retry handling."""
        completed = []
        retry_queue = []
        successful = 0
        
        for sim_id, info in self.pending_results.items():
            if info["status"] == "pending":
                # Check if simulation has been pending too long (30 minutes)
                if "start_time" not in info:
                    info["start_time"] = time.time()
                elif time.time() - info["start_time"] > 1800:  # 30 minutes
                    logger.warning(f"Simulation {sim_id} has been pending for too long, marking as failed")
                    completed.append(sim_id)
                    continue
                try:
                    sim_progress_resp = self.sess.get(info["progress_url"])
                    logger.info(f"Checking simulation {sim_id} for alpha: {info['alpha'][:50]}...")
                    
                    # Handle rate limits
                    if sim_progress_resp.status_code == 429:
                        logger.info("Rate limit hit, will retry later")
                        continue
                        
                    # Handle simulation limits
                    if "SIMULATION_LIMIT_EXCEEDED" in sim_progress_resp.text:
                        logger.info(f"Simulation limit exceeded for alpha: {info['alpha']}")
                        retry_queue.append((info['alpha'], sim_id))
                        continue
                        
                    # Handle retry-after
                    retry_after = sim_progress_resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = int(float(retry_after))  # Handle decimal values like "2.5"
                            logger.info(f"Need to wait {wait_time}s before next check")
                            time.sleep(wait_time)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid Retry-After header: {retry_after}, using default 5s")
                            time.sleep(5)
                        continue
                    
                    sim_result = sim_progress_resp.json()
                    status = sim_result.get("status")
                    logger.info(f"Simulation {sim_id} status: {status}")
                    
                    # Log additional details for debugging
                    if status == "PENDING":
                        logger.debug(f"Simulation {sim_id} still pending...")
                    elif status == "RUNNING":
                        logger.debug(f"Simulation {sim_id} is running...")
                    elif status not in ["COMPLETE", "ERROR"]:
                        logger.warning(f"Simulation {sim_id} has unknown status: {status}")
                    
                    if status == "COMPLETE":
                        alpha_id = sim_result.get("alpha")
                        if alpha_id:
                            alpha_resp = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}')
                            if alpha_resp.status_code == 200:
                                alpha_data = alpha_resp.json()
                                fitness = alpha_data.get("is", {}).get("fitness")
                                logger.info(f"Alpha {alpha_id} completed with fitness: {fitness}")
                                
                                self.results.append({
                                    "alpha": info["alpha"],
                                    "result": sim_result,
                                    "alpha_data": alpha_data
                                })
                                
                                self.log_hopeful_alpha(info["alpha"], alpha_data)
                                sharpe = alpha_data.get("is", {}).get("sharpe")
                                if fitness is not None and fitness > 1:
                                    logger.info(f"Found promising alpha! Fitness: {fitness}{', Sharpe: ' + str(sharpe) if sharpe is not None else ''}")
                                    successful += 1
                                elif fitness is None:
                                    logger.warning(f"Alpha {alpha_id} has no fitness data")
                    elif status == "ERROR":
                        error_msg = sim_result.get("error", sim_result.get("message", ""))
                        logger.error(f"Simulation failed for alpha: {info['alpha']} | Error: {error_msg}")
                    completed.append(sim_id)
                    
                except Exception as e:
                    logger.error(f"Error checking result for {sim_id}: {str(e)}")
        
        # Remove completed simulations
        for sim_id in completed:
            del self.pending_results[sim_id]
        
        # Requeue failed simulations
        for alpha, sim_id in retry_queue:
            del self.pending_results[sim_id]
            self.retry_queue.add(alpha)
        
        return successful

    def test_alpha(self, alpha: str) -> Dict:
        result = self._test_alpha_impl(alpha)
        if result.get("status") == "error" and "SIMULATION_LIMIT_EXCEEDED" in result.get("message", ""):
            self.retry_queue.add(alpha)
            return {"status": "queued", "message": "Added to retry queue"}
        return result

    def _test_alpha_impl(self, alpha_expression: str) -> Dict:
        def submit_simulation():
            simulation_data = {
                'type': 'REGULAR',
                'settings': self.region_config.to_simulation_settings(),
                'regular': alpha_expression
            }
            return self.sess.post('https://api.worldquantbrain.com/simulations', json=simulation_data, timeout=(30, 120))

        max_ssl_retries = 2
        for attempt in range(max_ssl_retries + 1):
            try:
                sim_resp = submit_simulation()
                break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_ssl_retries:
                    logger.warning(f"SSL/Connection error on attempt {attempt+1}/{max_ssl_retries+1}, re-authenticating: {str(e)[:100]}")
                    self._fix_proxy()
                    self.setup_auth()
                    sleep(5)
                else:
                    logger.error(f"SSL/Connection error after {max_ssl_retries+1} attempts: {str(e)[:200]}")
                    return {"status": "error", "message": str(e)}
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout testing alpha {alpha_expression}: {str(e)[:100]}")
                return {"status": "error", "message": str(e)}

        try:
            if sim_resp.status_code == 401 or (
                sim_resp.status_code == 400 and 
                "authentication credentials" in sim_resp.text.lower()
            ):
                logger.warning("Authentication expired, refreshing session...")
                self.setup_auth()
                sim_resp = submit_simulation()
            
            if sim_resp.status_code != 201:
                return {"status": "error", "message": sim_resp.text}

            sim_progress_url = sim_resp.headers.get('location')
            if not sim_progress_url:
                return {"status": "error", "message": "No progress URL received"}

            return {
                "status": "success", 
                "result": {
                    "id": f"{time.time()}_{random.random()}",
                    "progress_url": sim_progress_url
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing alpha {alpha_expression}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def log_hopeful_alpha(self, expression: str, alpha_data: Dict) -> None:
        from .alpha_store import save_alpha
        save_alpha(expression, alpha_data, source="generator")

        is_data = alpha_data.get("is", {})
        fitness = is_data.get("fitness")
        sharpe = is_data.get("sharpe")
        turnover = is_data.get("turnover")
        returns_val = is_data.get("returns")
        success = fitness is not None and fitness >= 1.0

        self.self_optimizer.record_result(
            expression=expression,
            fitness=fitness,
            sharpe=sharpe,
            turnover=turnover,
            success=success,
        )

        self.quality_monitor.record(AlphaMetrics(
            expression=expression,
            fitness=fitness,
            sharpe=sharpe,
            turnover=turnover,
            returns=returns_val,
            source="generator",
        ))

    def get_results(self) -> List[Dict]:
        """Get all processed results including retried alphas."""
        return self.results

    def fetch_submitted_alphas(self):
        """Fetch submitted alphas from the WorldQuant API with retry logic"""
        url = "https://api.worldquantbrain.com/users/self/alphas"
        params = {
            "limit": 100,
            "offset": 0,
            "status!=": "UNSUBMITTED%1FIS-FAIL",
            "order": "-dateCreated",
            "hidden": "false"
        }
        
        max_retries = 3
        retry_delay = 60  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.sess.get(url, params=params)
                if response.status_code == 429:  # Too Many Requests
                    wait_time = int(response.headers.get('Retry-After', retry_delay))
                    logger.info(f"Rate limited. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()["results"]
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to fetch submitted alphas after {max_retries} attempts: {e}")
                    return []
        
        return []

def extract_expressions(alphas):
    """Extract expressions from submitted alphas"""
    expressions = []
    for alpha in alphas:
        if alpha.get("regular") and alpha["regular"].get("code"):
            expressions.append({
                "expression": alpha["regular"]["code"],
                "performance": {
                    "sharpe": alpha["is"].get("sharpe", 0),
                    "fitness": alpha["is"].get("fitness", 0)
                }
            })
    return expressions

def is_similar_to_existing(new_expression, existing_expressions, similarity_threshold=0.7):
    """Check if new expression is too similar to existing ones"""
    for existing in existing_expressions:
        # Basic similarity checks
        if new_expression == existing["expression"]:
            return True
            
        # Check for structural similarity
        if structural_similarity(new_expression, existing["expression"]) > similarity_threshold:
            return True
    
    return False

def calculate_similarity(expr1: str, expr2: str) -> float:
    """Calculate similarity between two expressions using token-based comparison."""
    # Normalize expressions
    expr1_tokens = set(tokenize_expression(normalize_expression(expr1)))
    expr2_tokens = set(tokenize_expression(normalize_expression(expr2)))
    
    if not expr1_tokens or not expr2_tokens:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(expr1_tokens.intersection(expr2_tokens))
    union = len(expr1_tokens.union(expr2_tokens))
    
    return intersection / union

def structural_similarity(expr1, expr2):
    """Calculate structural similarity between two expressions"""
    return calculate_similarity(expr1, expr2)  # Use our new similarity function

def normalize_expression(expr):
    """Normalize expression for comparison"""
    # Remove whitespace and convert to lowercase
    expr = re.sub(r'\s+', '', expr.lower())
    return expr

def tokenize_expression(expr):
    """Split expression into meaningful tokens"""
    # Split on operators and parentheses while keeping them
    tokens = re.findall(r'[\w._]+|[(),*/+-]', expr)
    return tokens

def generate_alpha():
    """Generate new alpha expression"""
    generator = AlphaGenerator()
    data_fields = generator.get_data_fields()
    operators = generator.get_operators()
    
    # Fetch existing alphas first
    submitted_alphas = generator.fetch_submitted_alphas()
    existing_expressions = extract_expressions(submitted_alphas)
    
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts:
        alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
        for idea in alpha_ideas:
            if not is_similar_to_existing(idea, existing_expressions):
                logger.info(f"Generated unique expression: {idea}")
                return idea
                
        attempts += 1
        logger.debug(f"Attempt {attempts}: All expressions were too similar")
    
    logger.warning("Failed to generate unique expression after maximum attempts")
    return None

def main():
    parser = argparse.ArgumentParser(description='Generate and test alpha factors using WorldQuant Brain API with Ollama/FinGPT')
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Directory to save results (default: ./results)')
    parser.add_argument('--batch-size', type=int, default=3,
                      help='Number of alpha factors to generate per batch (default: 3)')
    parser.add_argument('--sleep-time', type=int, default=10,
                      help='Sleep time between batches in seconds (default: 10)')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level (default: INFO)')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                      help='Ollama API URL (default: http://localhost:11434)')
    parser.add_argument('--ollama-model', type=str, default='qwen3.5:35b',
                                             help='Ollama model to use (default: qwen3.5:35b)')
    parser.add_argument('--max-concurrent', type=int, default=2,
                      help='Maximum concurrent simulations (default: 2)')
    
    args = parser.parse_args()
    
    import logging
    from .log_manager import setup_logger
    setup_logger(__name__, "generator", level=getattr(logging, args.log_level))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize alpha generator with Ollama
        generator = AlphaGenerator(ollama_url=args.ollama_url, max_concurrent=args.max_concurrent)
        generator.model_name = args.ollama_model  # Set the model name
        generator.initial_model = args.ollama_model  # Set the initial model for reset
        
        # Get data fields and operators once
        print("Fetching data fields and operators...")
        data_fields = generator.get_data_fields()
        operators = generator.get_operators()
        
        batch_number = 1
        total_successful = 0
        
        print(f"Starting continuous alpha mining with batch size {args.batch_size}")
        print(f"Results will be saved to {args.output_dir}")
        print(f"Using Ollama at {args.ollama_url}")
        
        while True:
            try:
                logger.info(f"\nProcessing batch #{batch_number}")
                logger.info("-" * 50)
                
                # Generate and submit batch using Ollama
                alpha_ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
                batch_successful = generator.test_alpha_batch(alpha_ideas)
                total_successful += batch_successful
                
                # Perform VRAM cleanup every few batches
                generator.operation_count += 1
                if generator.operation_count % generator.vram_cleanup_interval == 0:
                    generator.cleanup_vram()
                
                # Save batch results
                results = generator.get_results()
                timestamp = int(time.time())
                output_file = os.path.join(args.output_dir, f'batch_{batch_number}_{timestamp}.json')
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Batch {batch_number} results saved to {output_file}")
                logger.info(f"Batch successful: {batch_successful}")
                logger.info(f"Total successful alphas: {total_successful}")
                
                batch_number += 1
                
                # Sleep between batches
                print(f"Sleeping for {args.sleep_time} seconds...")
                sleep(args.sleep_time)
                
            except Exception as e:
                logger.error(f"Error in batch {batch_number}: {str(e)}")
                logger.info("Sleeping for 5 minutes before retrying...")
                sleep(300)
                continue
        
    except KeyboardInterrupt:
        logger.info("\nStopping alpha mining...")
        logger.info(f"Total batches processed: {batch_number - 1}")
        logger.info(f"Total successful alphas: {total_successful}")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
