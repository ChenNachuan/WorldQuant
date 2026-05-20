"""
WorldQuant Brain Alpha Miner

Simplified alpha mining pipeline using direct API calls.
Supports both Ollama and DeepSeek for LLM-based alpha generation.
"""

import os
import sys
import time
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.config import load_credentials
from core.log_manager import setup_logger
from core.alpha_db import get_alpha_db
from core.submission_quota import get_submission_quota
from core.llm_client import get_llm_client, DEFAULT_SYSTEM_PROMPT

logger = setup_logger(__name__, "alpha_miner")

# Constants
BASE_URL = "https://api.worldquantbrain.com"
MIN_SHARPE = 1.25
MIN_FITNESS = 1.0


class AlphaMiner:
    """Main alpha mining engine with direct API approach."""

    def __init__(self, llm_provider: str = "auto"):
        self.llm_client = get_llm_client(llm_provider)
        self.alpha_db = get_alpha_db()
        self.quota = get_submission_quota()

        # Session state
        self.session = None
        self.tested_expressions = set()

        # Queues
        self.llm_task_queue = queue.Queue()
        self.test_queue = queue.Queue()

        # Stats
        self.stats = {
            "tested": 0,
            "passed": 0,
            "failed": 0,
            "rescued": 0,
            "best_sharpe": -99.0
        }

    def authenticate(self) -> bool:
        """Authenticate with WorldQuant Brain API."""
        import requests
        from requests.auth import HTTPBasicAuth

        username, password = load_credentials()

        logger.info("Authenticating with WorldQuant Brain...")
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        })
        self.session.auth = HTTPBasicAuth(username, password)

        try:
            resp = self.session.post(
                f"{BASE_URL}/authentication",
                verify=False,
                timeout=15
            )
            if resp.status_code == 201:
                logger.info("Authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def generate_alphas(self, fields_data: Dict = None) -> List[Dict]:
        """Generate alpha expressions using LLM."""
        if not fields_data:
            fields_data = self._get_default_fields()

        prompt = f"""请利用以下提供的数据字段，进行模块内部或模块之间的交叉组合，
生成 5 个具有爆发力的全新因子。必须使用真实的字段名。

可用字段: {json.dumps(fields_data, ensure_ascii=False)}"""

        return self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

    def _get_default_fields(self) -> Dict:
        """Get default field data for alpha generation."""
        return {
            "PRICE&VOLUME": [
                {"Field": "close", "Description": "收盘价"},
                {"Field": "open", "Description": "开盘价"},
                {"Field": "high", "Description": "最高价"},
                {"Field": "low", "Description": "最低价"},
                {"Field": "volume", "Description": "成交量"},
                {"Field": "returns", "Description": "收益率"},
                {"Field": "vwap", "Description": "成交量加权平均价"}
            ],
            "FUNDAMENTAL": [
                {"Field": "market_cap", "Description": "市值"},
                {"Field": "pe_ratio", "Description": "市盈率"},
                {"Field": "pb_ratio", "Description": "市净率"},
                {"Field": "roe", "Description": "净资产收益率"}
            ]
        }

    def simulate_factor(self, factor: Dict) -> Dict:
        """Submit factor for backtesting and poll for results."""
        if not self.session:
            return {"error": "Not authenticated"}

        expression = factor.get("expression", "")
        if not expression:
            return {"error": "Empty expression"}

        # Build simulation settings
        settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "NONE",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False
        }

        # Apply custom settings if provided
        if isinstance(factor.get("settings"), dict):
            for k, v in factor["settings"].items():
                if k in settings:
                    settings[k] = v

        payload = {
            "type": "REGULAR",
            "settings": settings,
            "regular": expression
        }

        try:
            # Submit simulation
            resp = self.session.post(
                f"{BASE_URL}/simulations",
                json=payload,
                verify=False,
                timeout=30
            )

            # Handle auth failures
            if resp.status_code in [401, 403] or "Incorrect authentication" in resp.text:
                return {"error": "AUTH_FAILED"}

            if resp.status_code != 201:
                return {"error": f"Simulation failed: {resp.text[:200]}"}

            # Get simulation ID from Location header
            sim_id = resp.headers.get("Location", "").split("/")[-1]
            if not sim_id:
                return {"error": "No simulation ID returned"}

            # Poll for results
            return self._poll_simulation(sim_id)

        except Exception as e:
            return {"error": f"Exception: {str(e)}"}

    def _poll_simulation(self, sim_id: str, timeout: int = 600) -> Dict:
        """Poll simulation until complete or timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                resp = self.session.get(
                    f"{BASE_URL}/simulations/{sim_id}",
                    verify=False,
                    timeout=30
                )

                if resp.status_code != 200:
                    time.sleep(3)
                    continue

                data = resp.json()
                status = data.get("status")

                if status in ["FINISHED", "WARNING", "COMPLETE", "COMPLETED"]:
                    alpha_id = data.get("alpha")
                    if alpha_id:
                        # Get alpha performance
                        alpha_resp = self.session.get(
                            f"{BASE_URL}/alphas/{alpha_id}",
                            verify=False,
                            timeout=30
                        )
                        if alpha_resp.status_code == 200:
                            perf = alpha_resp.json().get("is", {})
                            return {
                                "alpha_id": alpha_id,
                                "sharpe": perf.get("sharpe", 0),
                                "fitness": perf.get("fitness", 0),
                                "turnover": perf.get("turnover", 0),
                                "margin": perf.get("margin", 0),
                                "message": data.get("message", "Success")
                            }
                    return {"alpha_id": alpha_id, "status": "completed"}

                elif status in ["ERROR", "FAILED"]:
                    return {"error": data.get("message", "Unknown error")}

                time.sleep(3)

            except Exception as e:
                logger.warning(f"Poll error: {e}")
                time.sleep(3)

        return {"error": "Simulation timeout"}

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit alpha to WorldQuant Brain."""
        try:
            resp = self.session.post(
                f"{BASE_URL}/alphas/{alpha_id}/submit",
                json={"type": "REGULAR"},
                verify=False,
                timeout=15
            )
            return resp.status_code == 201
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return False

    def process_result(self, factor: Dict, result: Dict):
        """Process simulation result and take action."""
        expression = factor.get("expression", "")

        if "error" in result:
            logger.error(f"Failed: {expression[:60]}... -> {result['error']}")
            self.stats["failed"] += 1

            # Handle auth failure
            if result["error"] == "AUTH_FAILED":
                logger.warning("Authentication expired, attempting re-login...")
                if self.authenticate():
                    self.test_queue.put(factor)  # Re-queue factor
                else:
                    logger.error("Re-login failed, stopping")
                    return False
            return True

        # Extract metrics
        sharpe = result.get("sharpe", 0)
        fitness = result.get("fitness", 0)
        alpha_id = result.get("alpha_id")

        logger.info(f"Result: S={sharpe:.2f} F={fitness:.2f} | {expression[:60]}...")

        # Update best sharpe
        if sharpe > self.stats["best_sharpe"]:
            self.stats["best_sharpe"] = sharpe

        # Check if meets submission criteria
        if sharpe >= MIN_SHARPE and fitness >= MIN_FITNESS:
            logger.info(f"Found alpha! S={sharpe:.2f} F={fitness:.2f}")

            # Check quota
            if self.quota.can_submit():
                if self.submit_alpha(alpha_id):
                    self.stats["passed"] += 1
                    self.quota.record_submission(alpha_id)
                    logger.info(f"Submitted alpha {alpha_id}")

                    # Save to database
                    self.alpha_db.add_alpha(
                        expression=expression,
                        sharpe=sharpe,
                        fitness=fitness,
                        alpha_id=alpha_id
                    )
            else:
                logger.info("Daily submission quota reached")

        # Rescue mechanism for borderline alphas
        elif (abs(sharpe) + abs(fitness)) > 1.7:
            logger.info(f"Rescuing borderline alpha: S={sharpe:.2f} F={fitness:.2f}")
            self.stats["rescued"] += 1

            # Add rescue task to LLM queue
            self.llm_task_queue.put({
                "type": "RESCUE",
                "expression": expression,
                "sharpe": sharpe,
                "fitness": fitness,
                "turnover": result.get("turnover", 0),
                "margin": result.get("margin", 0)
            })

        self.stats["tested"] += 1
        return True

    def llm_producer_worker(self):
        """Background thread for LLM alpha generation."""
        logger.info("LLM producer thread started")

        while True:
            try:
                # Control queue size
                if self.test_queue.qsize() > 15:
                    time.sleep(3)
                    continue

                # Process rescue tasks first
                if not self.llm_task_queue.empty():
                    task = self.llm_task_queue.get()
                    if task.get("type") == "RESCUE":
                        self._process_rescue_task(task)
                        continue

                # Generate new alphas
                alphas = self.generate_alphas()
                for alpha in alphas:
                    if alpha.get("expression"):
                        self.test_queue.put(alpha)

                # Small delay to prevent overwhelming
                time.sleep(1)

            except Exception as e:
                logger.error(f"LLM producer error: {e}")
                time.sleep(5)

    def _process_rescue_task(self, task: Dict):
        """Process a rescue task - generate variants of borderline alpha."""
        expression = task.get("expression", "")
        sharpe = task.get("sharpe", 0)
        fitness = task.get("fitness", 0)

        prompt = f"""这个因子表现一般，请给出 3 种不同视角的变种，必须保持 ts_decay_linear(group_neutralize(zscore(...))) 外壳不变。

原代码: {expression}
Sharpe: {sharpe:.2f}, Fitness: {fitness:.2f}
Turnover: {task.get('turnover', 0):.2f}, Margin: {task.get('margin', 0):.2f}

请生成 3 个变种因子。"""

        variants = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)
        for variant in variants:
            if variant.get("expression"):
                self.test_queue.put(variant)

    def run(self, max_workers: int = 2):
        """Main execution loop."""
        if not self.authenticate():
            logger.error("Failed to authenticate, exiting")
            return

        logger.info("Starting alpha miner...")

        # Start LLM producer thread
        producer = threading.Thread(
            target=self.llm_producer_worker,
            daemon=True
        )
        producer.start()

        # Main consumer loop
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            running_tasks = {}

            while True:
                try:
                    # Fill executor slots
                    while len(running_tasks) < max_workers and not self.test_queue.empty():
                        factor = self.test_queue.get()
                        expression = factor.get("expression", "")

                        # Skip if already tested
                        if not expression or expression in self.tested_expressions:
                            continue
                        self.tested_expressions.add(expression)

                        logger.info(f"Testing: {expression[:60]}...")
                        future = executor.submit(self.simulate_factor, factor)
                        running_tasks[future] = factor

                    if not running_tasks:
                        time.sleep(1)
                        continue

                    # Wait for any task to complete
                    done, _ = as_completed(
                        running_tasks.keys(),
                        timeout=1.0
                    ), None

                    for future in list(running_tasks.keys()):
                        if future.done():
                            factor = running_tasks.pop(future)
                            try:
                                result = future.result()
                                if not self.process_result(factor, result):
                                    return  # Fatal error
                            except Exception as e:
                                logger.error(f"Task exception: {e}")
                                self.stats["failed"] += 1

                    # Print stats periodically
                    if self.stats["tested"] % 10 == 0 and self.stats["tested"] > 0:
                        logger.info(
                            f"Stats: tested={self.stats['tested']} "
                            f"passed={self.stats['passed']} "
                            f"failed={self.stats['failed']} "
                            f"rescued={self.stats['rescued']} "
                            f"best_sharpe={self.stats['best_sharpe']:.2f}"
                        )

                except KeyboardInterrupt:
                    logger.info("Received interrupt, shutting down...")
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    time.sleep(1)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="WorldQuant Brain Alpha Miner")
    parser.add_argument(
        "--llm",
        choices=["auto", "ollama", "deepseek"],
        default="auto",
        help="LLM provider to use"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent simulation workers"
    )
    args = parser.parse_args()

    miner = AlphaMiner(llm_provider=args.llm)
    miner.run(max_workers=args.workers)


if __name__ == "__main__":
    main()
