"""
WorldQuant Brain Alpha Miner

Simplified alpha mining pipeline using direct API calls.
Supports both Ollama and DeepSeek for LLM-based alpha generation.
"""

import os
import sys
import time
import json
import glob
import queue
import random
import warnings
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

# Suppress SSL verification warnings for WorldQuant Brain API
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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
RESCUE_THRESHOLD = 1.7

# Parameter sweep settings for check failures
SETTINGS_SWEEP = [
    {"neutralization": "INDUSTRY", "truncation": 0.1, "decay": 5, "delay": 1},
    {"neutralization": "SUBINDUSTRY", "truncation": 0.15, "decay": 10, "delay": 1},
    {"neutralization": "SECTOR", "truncation": 0.08, "decay": 20, "delay": 0},
    {"neutralization": "MARKET", "truncation": 0.2, "decay": 40, "delay": 1},
]

# Check failure strategies
CHECK_STRATEGIES = {
    "TURNOVER": {
        "description": "换手率过高",
        "suggestions": [
            "加大时序平滑窗口: ts_mean(x, 10) → ts_mean(x, 20)",
            "增加衰减: ts_decay_linear(x, 5) → ts_decay_linear(x, 10)",
            "使用 ts_rank 替换 zscore 降低换手"
        ]
    },
    "SELF_CORRELATION": {
        "description": "自相关性过高",
        "suggestions": [
            "在settings中改变neutralization: INDUSTRY → SUBINDUSTRY/SECTOR/MARKET",
            "增加截断: truncation=0.08 → truncation=0.15",
            "使用 ts_corr(x, adv20, 20) 引入成交量因子"
        ]
    },
    "DRAWDOWN": {
        "description": "回撤过大",
        "suggestions": [
            "使用 ts_max(x, 60) 限制最大回撤",
            "增加衰减平滑: ts_decay_linear(x, 20)",
            "使用 -1 * x 翻转信号方向"
        ]
    }
}

# Rescue decision logic - check types
RESCUABLE_CHECKS = ["TURNOVER", "DRAWDOWN", "TURNOVER_RATE"]
NON_RESCUABLE_CHECKS = ["SELF_CORRELATION", "LOW_SUBMISSION_CORRELATION"]

# Data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIELDS_DIR = os.path.join(DATA_DIR, "fields")
OPERATORS_DIR = os.path.join(DATA_DIR, "operators")
SHARED_POOL_DIR = os.path.join(DATA_DIR, "shared_pool")

# Target field files to load (matching IQC approach)
# Note: options.csv removed - contains invalid field names, options fields are in price&volume.csv
TARGET_FIELD_FILES = [
    "price&volume.csv",
    "fundamental.csv",
    "analyst.csv",
    "sentiment.csv",
    "model.csv"
]


class AlphaMiner:
    """Main alpha mining engine with direct API approach."""

    def __init__(self, llm_provider: str = "auto", member_id: str = "default"):
        self.llm_client = get_llm_client(llm_provider)
        self.alpha_db = get_alpha_db()
        self.quota = get_submission_quota()
        self.member_id = member_id

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
            "flipped": 0,
            "rescue_pool": 0,
            "best_sharpe": -99.0
        }

        # Dynamic module weights (reinforcement learning style)
        # Note: OPTIONS removed - fields are in price&volume.csv
        self.module_stats = {
            "PRICE&VOLUME": {"tried": 0, "success": 0},
            "FUNDAMENTAL": {"tried": 0, "success": 0},
            "ANALYST": {"tried": 0, "success": 0},
            "SENTIMENT": {"tried": 0, "success": 0},
            "MODEL": {"tried": 0, "success": 0}
        }

        # Operator knowledge base
        self.operator_arity = self._load_operator_knowledge()

        # Create directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(FIELDS_DIR, exist_ok=True)
        os.makedirs(SHARED_POOL_DIR, exist_ok=True)

    def _load_operator_knowledge(self) -> Dict[str, int]:
        """Load operator arity from operators.csv."""
        import pandas as pd
        operators_file = os.path.join(OPERATORS_DIR, "operators.csv")

        if not os.path.exists(operators_file):
            logger.warning(f"Operators file not found: {operators_file}")
            return {}

        try:
            df = pd.read_csv(operators_file, encoding='utf-8-sig')
            # Extract arity from Definition column (count parameters)
            arity_dict = {}
            for _, row in df.iterrows():
                name = str(row.get('Name', '')).strip()
                definition = str(row.get('Definition', ''))
                # Count parameters: look for pattern like func(x, y, z)
                if '(' in definition:
                    params = definition.split('(')[1].split(')')[0]
                    arity = len([p.strip() for p in params.split(',') if p.strip()])
                    arity_dict[name] = arity
            logger.info(f"Loaded {len(arity_dict)} operators")
            return arity_dict
        except Exception as e:
            logger.warning(f"Failed to load operators: {e}")
            return {}

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

    # ==========================================
    # Field Loading (from CSV files)
    # ==========================================

    def load_fields_from_csvs(self) -> Dict[str, List[Dict]]:
        """Load field data from specified CSV files (matching IQC approach)."""
        import pandas as pd

        selected_fields = {}

        if not os.path.exists(FIELDS_DIR):
            logger.info("Fields directory not found, using default fields")
            return self._get_default_fields()

        # Only load target files (matching IQC approach)
        for file in TARGET_FIELD_FILES:
            filepath = os.path.join(FIELDS_DIR, file)
            if not os.path.exists(filepath):
                logger.warning(f"Target file not found: {file}")
                continue

            try:
                df = pd.read_csv(filepath)
                # Get top 10 fields with Field and Description columns
                if 'Field' in df.columns and 'Description' in df.columns:
                    top_10 = df.head(10)[['Field', 'Description']].to_dict(orient='records')
                    category = file.replace(".csv", "").upper()
                    selected_fields[category] = top_10
                    logger.info(f"Loaded {len(top_10)} fields from {file}")
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")

        # If no CSV files found, use defaults
        if not selected_fields:
            logger.info("No target CSV files found, using default fields")
            selected_fields = self._get_default_fields()

        return selected_fields

    def _get_default_fields(self) -> Dict[str, List[Dict]]:
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

    # ==========================================
    # Dynamic Module Weights
    # ==========================================

    def get_dynamic_modules(self, fields_data: Dict) -> tuple:
        """
        Select 1-2 modules based on historical success rates.
        Returns (selected_fields, modules_used)
        """
        # Check if we have enough data to use weighted selection
        ready = all(stat['success'] >= 2 for stat in self.module_stats.values())
        modules = list(self.module_stats.keys())

        if not ready:
            # Equal weights until we have enough data
            weights = [1.0] * len(modules)
        else:
            # Calculate win rates, minimum 0.1 base weight
            weights = [
                max(0.1, stat['success'] / max(1, stat['tried']))
                for stat in self.module_stats.values()
            ]

        # Select 1 or 2 modules
        num_to_select = random.choice([1, 2])
        selected = random.choices(modules, weights=weights, k=num_to_select)
        selected = list(set(selected))  # Remove duplicates

        # Get fields for selected modules
        selected_fields = {mod: fields_data.get(mod, []) for mod in selected}

        return selected_fields, selected

    def record_module_stat(self, modules_used: List[str], success: bool):
        """Record success/failure for module weight updates."""
        for mod in modules_used:
            if mod in self.module_stats:
                self.module_stats[mod]['tried'] += 1
                if success:
                    self.module_stats[mod]['success'] += 1

    # ==========================================
    # Shared Pool Management
    # ==========================================

    def load_shared_pool(self) -> List[Dict]:
        """Load and merge all shared pool files."""
        combined_pool = []
        search_pattern = os.path.join(SHARED_POOL_DIR, "shared_pool_*.json")

        for file_path in glob.glob(search_pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_pool.extend(json.load(f))
            except Exception:
                continue

        # Sort by Sharpe and keep top 500
        combined_pool.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
        return combined_pool[:500]

    def add_to_shared_pool(self, expression: str, sharpe: float, fitness: float, logic: str = ""):
        """Add factor to member's shared pool file."""
        my_pool_path = os.path.join(SHARED_POOL_DIR, f"shared_pool_{self.member_id}.json")

        my_pool = []
        if os.path.exists(my_pool_path):
            try:
                with open(my_pool_path, "r", encoding="utf-8") as f:
                    my_pool = json.load(f)
            except Exception:
                pass

        my_pool.append({
            "expression": expression,
            "sharpe": sharpe,
            "fitness": fitness,
            "logic": logic
        })

        # Sort and keep top 500
        my_pool.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
        my_pool = my_pool[:500]

        with open(my_pool_path, "w", encoding="utf-8") as f:
            json.dump(my_pool, f, ensure_ascii=False, indent=2)

    # ==========================================
    # Alpha Generation
    # ==========================================

    def generate_alphas(self, fields_data: Dict = None) -> List[Dict]:
        """Generate alpha expressions using LLM with dynamic module selection."""
        if not fields_data:
            fields_data = self.load_fields_from_csvs()

        # Use dynamic module selection
        target_fields, modules_used = self.get_dynamic_modules(fields_data)
        mod_names = "+".join(modules_used)

        logger.info(f"Generating alphas for modules: {mod_names}")

        prompt = f"""请利用以下提供的数据字段，进行模块内部或模块之间的交叉组合，
生成 5 个具有爆发力的全新因子。必须使用真实的字段名。

可用字段: {json.dumps(target_fields, ensure_ascii=False)}"""

        results = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

        # Tag results with modules used
        for res in results:
            res['modules_used'] = modules_used

        return results

    def generate_crossover_alphas(self) -> List[Dict]:
        """Generate alphas by crossing over elite factors from shared pool using non-linear operations."""
        pool = self.load_shared_pool()

        if len(pool) < 2:
            logger.info("Not enough factors in shared pool for crossover")
            return []

        # Select 2 random elite parents
        parents = random.sample(pool, 2)

        logger.info("Generating crossover from elite parents (non-linear)...")

        prompt = f"""两个已提交因子，禁止线性组合(无法通过相关性检测)。

【操作步骤】
1. 提取父因子的核心逻辑(去掉 ts_decay_linear(zscore(...)) 外壳)
2. 使用以下非线性方式杂交：
   - ts_corr(核心A, 核心B, d)：计算时序相关性
   - ts_cov(核心A, 核心B, d)：计算时序协方差
   - rank(核心A) / rank(核心B)：排名比值
   - sign(核心A) * abs(核心B)：符号组合
   - 交叉数据字段：保持父A结构，替换为父B的字段
   - 更换算子：ts_mean↔ts_std_dev, ts_rank↔ts_zscore
3. 重新套上外壳

【父因子】
父A: {parents[0]['expression']}
父B: {parents[1]['expression']}

【输出要求】
输出3个变种，外壳必须为: ts_decay_linear(zscore(...), 5)
中性化由settings控制，表达式中不要包含group_neutralize"""

        results = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

        # Tag as crossover
        for res in results:
            res['modules_used'] = []  # Crossover doesn't count for module stats

        return results

    # ==========================================
    # Simulation & Polling
    # ==========================================

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
            "neutralization": "INDUSTRY",
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
                            alpha_data = alpha_resp.json()
                            perf = alpha_data.get("is", {})
                            settings = alpha_data.get("settings", {})
                            return {
                                "alpha_id": alpha_id,
                                "sharpe": perf.get("sharpe", 0),
                                "fitness": perf.get("fitness", 0),
                                "turnover": perf.get("turnover", 0),
                                "margin": perf.get("margin", 0),
                                "returns": perf.get("returns", 0),
                                "long_count": perf.get("longCount", 0),
                                "short_count": perf.get("shortCount", 0),
                                "drawdown": perf.get("drawdown", 0),
                                "grade": alpha_data.get("grade", ""),
                                "checks": perf.get("checks", []),
                                "region": settings.get("region", "USA"),
                                "universe": settings.get("universe", "TOP3000"),
                                "delay": settings.get("delay", 1),
                                "decay": settings.get("decay", 0),
                                "neutralization": settings.get("neutralization", "NONE"),
                                "truncation": settings.get("truncation", 0.08),
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

    def submit_alpha(self, alpha_id: str) -> Dict:
        """Submit alpha to WorldQuant Brain. Returns dict with success status and details."""
        try:
            resp = self.session.post(
                f"{BASE_URL}/alphas/{alpha_id}/submit",
                json={"type": "REGULAR"},
                verify=False,
                timeout=15
            )

            if resp.status_code == 201:
                return {"success": True}
            else:
                # Capture error message from response
                error_msg = resp.text[:200] if resp.text else "Unknown error"
                return {"success": False, "error": error_msg}
        except Exception as e:
            logger.error(f"Submit error: {e}")
            return {"success": False, "error": str(e)}

    # ==========================================
    # Result Processing
    # ==========================================

    def _has_failed_checks(self, checks: list) -> bool:
        """Check if any checks have FAILED status."""
        if not checks:
            return False
        return any(check.get("result") == "FAIL" for check in checks)

    def _should_rescue_after_sweep(self, failed_checks: list) -> bool:
        """判断参数调优失败后是否应该进入 rescue_pool"""
        # 如果失败的检查包含不可 rescue 的类型，则丢弃
        for check in failed_checks:
            check_upper = check.upper()
            if any(nr in check_upper for nr in NON_RESCUABLE_CHECKS):
                return False
        # 如果有可 rescue 的检查类型，则进入 rescue_pool
        for check in failed_checks:
            check_upper = check.upper()
            if any(r in check_upper for r in RESCUABLE_CHECKS):
                return True
        # 默认不 rescue
        return False

    def process_result(self, factor: Dict, result: Dict) -> bool:
        """
        Process simulation result and take action.
        Returns False if fatal error occurred.
        """
        expression = factor.get("expression", "")
        mod_used = factor.get("modules_used", [])

        if "error" in result:
            logger.error(f"Failed: {expression[:60]}... -> {result['error']}")
            self.stats["failed"] += 1
            self.record_module_stat(mod_used, False)

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
        turnover = result.get("turnover", 0)
        margin = result.get("margin", 0)

        logger.info(
            f"Result: S={sharpe:.2f} F={fitness:.2f} T={turnover:.2f} M={margin:.2f} | "
            f"{expression[:60]}..."
        )

        # Update best sharpe
        if sharpe > self.stats["best_sharpe"]:
            self.stats["best_sharpe"] = sharpe

        # Check if meets submission criteria (Holy Grail)
        is_holy_grail = (sharpe >= MIN_SHARPE and fitness >= MIN_FITNESS)
        is_pool_worthy = is_holy_grail or (sharpe > 1.0 and fitness > 0.8)

        if is_pool_worthy:
            self.record_module_stat(mod_used, True)
            self.add_to_shared_pool(expression, sharpe, fitness, factor.get("logic", ""))
        else:
            self.record_module_stat(mod_used, False)

        # Only save alphas that meet submission criteria and pass all checks (no auto-submit)
        checks = result.get("checks", [])
        if is_holy_grail:
            # Check if any checks failed
            if self._has_failed_checks(checks):
                failed_checks = [c["name"] for c in checks if c.get("result") == "FAIL"]
                logger.warning(f"Alpha {alpha_id} failed checks: {failed_checks}")
                logger.warning(f"  Expression: {expression[:80]}...")
                logger.warning(f"  S={sharpe:.2f} F={fitness:.2f} T={turnover:.2f}")

                # Try parameter sweep first
                if self._try_parameter_sweep(alpha_id, expression, failed_checks):
                    logger.info(f"Alpha {alpha_id} saved via parameter sweep")
                else:
                    # Parameter sweep failed, check if should rescue
                    if self._should_rescue_after_sweep(failed_checks):
                        logger.info(f"Adding alpha {alpha_id} to rescue pool (rescuable checks: {failed_checks})")
                        self.alpha_db.add_to_rescue_pool(
                            alpha_id=alpha_id,
                            expression=expression,
                            sharpe=sharpe,
                            fitness=fitness,
                            turnover=turnover,
                            failed_checks=failed_checks,
                            modules_used=mod_used
                        )
                    else:
                        logger.info(f"Discarding alpha {alpha_id} (non-rescuable checks: {failed_checks})")
            else:
                logger.info(f"Found alpha! S={sharpe:.2f} F={fitness:.2f} (unsubmitted)")

                # Save to database as unsubmitted
                self.alpha_db.add_alpha(
                    expression=expression,
                    alpha_id=alpha_id,
                    sharpe=sharpe,
                    fitness=fitness,
                    turnover=turnover,
                    margin=margin,
                    returns=result.get("returns", 0),
                    long_count=result.get("long_count", 0),
                    short_count=result.get("short_count", 0),
                    drawdown=result.get("drawdown", 0),
                    grade=result.get("grade", ""),
                    checks=checks,
                    source="pipeline",
                    region=result.get("region", "USA"),
                    universe=result.get("universe", "TOP3000"),
                    delay=result.get("delay", 1),
                    decay=result.get("decay", 0),
                    neutralization=result.get("neutralization", "NONE"),
                    truncation=result.get("truncation", 0.08),
                    status="unsubmitted",
                )

        # Reverse factor detection (Sharpe < -0.8)
        elif sharpe < -0.8:
            logger.info(f"Found reverse factor (S={sharpe:.2f}), flipping sign...")
            self.stats["flipped"] += 1

            # Create flipped version
            flipped_factor = factor.copy()
            flipped_factor['expression'] = f"-1 * ({expression})"
            self.test_queue.put(flipped_factor)

        # Rescue mechanism for borderline alphas
        elif (abs(sharpe) + abs(fitness)) > RESCUE_THRESHOLD:
            logger.info(f"Rescuing borderline alpha: S={sharpe:.2f} F={fitness:.2f}")
            self.stats["rescued"] += 1

            # Add to rescue pool (will be picked up by rescue worker)
            if alpha_id:
                self.alpha_db.add_to_rescue_pool(
                    alpha_id=alpha_id,
                    expression=expression,
                    sharpe=sharpe,
                    fitness=fitness,
                    turnover=turnover,
                    failed_checks=[],
                    modules_used=mod_used
                )

        self.stats["tested"] += 1
        return True

    # ==========================================
    # LLM Producer Worker
    # ==========================================

    def llm_producer_worker(self, fields_data: Dict):
        """Background thread for LLM alpha generation."""
        logger.info("LLM producer thread started")

        while True:
            try:
                # Control queue size
                if self.test_queue.qsize() > 15:
                    time.sleep(3)
                    continue

                # Clean up rescue pool periodically
                self.alpha_db.cleanup_rescue_pool()

                # Decide between new generation, crossover, and rescue
                # 60% new generation, 20% crossover, 20% rescue
                rand = random.random()

                if rand < 0.6:
                    # 60% chance: Generate new alphas
                    alphas = self.generate_alphas(fields_data)
                elif rand < 0.8:
                    # 20% chance: Crossover from shared pool
                    pool = self.load_shared_pool()
                    if len(pool) >= 2:
                        alphas = self.generate_crossover_alphas()
                    else:
                        alphas = self.generate_alphas(fields_data)
                else:
                    # 20% chance: Rescue from rescue pool
                    rescue_count = self.alpha_db.count_rescue_pool()
                    if rescue_count > 0:
                        alphas = self.generate_rescue_alphas()
                    else:
                        alphas = self.generate_alphas(fields_data)

                for alpha in alphas:
                    if alpha.get("expression"):
                        self.test_queue.put(alpha)

                # Small delay to prevent overwhelming
                time.sleep(1)

            except Exception as e:
                logger.error(f"LLM producer error: {e}")
                time.sleep(5)

    def _try_parameter_sweep(self, alpha_id: str, expression: str, failed_checks: list) -> bool:
        """
        Try different parameter combinations to fix check failures.
        Returns True if any combination passes all checks.
        """
        logger.info(f"Trying parameter sweep for alpha {alpha_id}...")

        for i, settings in enumerate(SETTINGS_SWEEP):
            logger.info(f"  Sweep {i+1}/{len(SETTINGS_SWEEP)}: {settings}")
            result = self.simulate_factor({
                "expression": expression,
                "settings": settings
            })

            if "error" not in result:
                checks = result.get("checks", [])
                if not self._has_failed_checks(checks):
                    # Success! Save to database
                    logger.info(f"  Parameter sweep succeeded with: {settings}")
                    self.alpha_db.add_alpha(
                        expression=expression,
                        alpha_id=result.get("alpha_id"),
                        sharpe=result.get("sharpe"),
                        fitness=result.get("fitness"),
                        turnover=result.get("turnover"),
                        margin=result.get("margin"),
                        returns=result.get("returns", 0),
                        long_count=result.get("long_count", 0),
                        short_count=result.get("short_count", 0),
                        drawdown=result.get("drawdown", 0),
                        grade=result.get("grade", ""),
                        checks=checks,
                        source="pipeline",
                        region=result.get("region", "USA"),
                        universe=result.get("universe", "TOP3000"),
                        delay=settings.get("delay", 1),
                        decay=settings.get("decay", 0),
                        neutralization=settings.get("neutralization", "NONE"),
                        truncation=settings.get("truncation", 0.08),
                        status="unsubmitted",
                    )
                    return True

        logger.info(f"All parameter sweep combinations failed for alpha {alpha_id}")
        return False

    def _get_check_suggestions(self, failed_checks: list) -> str:
        """Get targeted suggestions based on failed checks."""
        suggestions = []
        for check_name in failed_checks:
            check_upper = check_name.upper()
            for key, strategy in CHECK_STRATEGIES.items():
                if key in check_upper:
                    suggestions.append(f"【{strategy['description']}】")
                    for s in strategy["suggestions"]:
                        suggestions.append(f"  - {s}")
                    break
        return "\n".join(suggestions) if suggestions else "无特定建议，请尝试通用优化"

    def generate_rescue_alphas(self) -> List[Dict]:
        """Generate rescue alphas from rescue pool."""
        candidate = self.alpha_db.get_rescue_candidate()
        if not candidate:
            return []

        alpha_id = candidate["alpha_id"]
        expression = candidate["expression"]
        sharpe = candidate["sharpe"]
        fitness = candidate["fitness"]
        turnover = candidate["turnover"]
        failed_checks = candidate.get("failed_checks", [])
        modules_used = candidate.get("modules_used", [])

        # Increment attempt count
        self.alpha_db.increment_rescue_attempt(alpha_id)

        # Determine rescue type based on failed checks
        has_check_failures = len(failed_checks) > 0

        if has_check_failures:
            # Case B: Check failures - use targeted suggestions
            check_suggestions = self._get_check_suggestions(failed_checks)
            prompt = f"""因子检查失败，需要针对性修复。

【当前状态】
原代码: {expression}
Sharpe={sharpe:.2f} Fitness={fitness:.2f} Turnover={turnover:.2f}
失败检查: {', '.join(failed_checks)}

【针对性建议】
{check_suggestions}

【重要原则】
- 只修改时序平滑参数，不要改变核心逻辑
- 失败检查为 TURNOVER → 加大窗口 (10→20, 20→40)
- 失败检查为 SELF_CORRELATION → 在settings中改变neutralization
- 失败检查为 DRAWDOWN → 增加衰减平滑

【输出要求】
输出3个变种，外壳不变: ts_decay_linear(zscore(...), 5)
中性化由settings控制，表达式中不要包含group_neutralize"""
        else:
            # Case A: Poor performance - general optimization
            prompt = f"""因子表现接近达标，需要提升性能。

【当前状态】
原代码: {expression}
Sharpe={sharpe:.2f} Fitness={fitness:.2f} Turnover={turnover:.2f}

【优化建议】
1. 引入新的数据字段（如基本面、分析师、情绪数据）
2. 更换核心算子：ts_mean↔ts_std_dev, ts_rank↔ts_zscore
3. 调整时序窗口：5→10, 10→20
4. 使用非线性变换：abs, log, sign, rank

【输出要求】
输出3个变种，外壳不变: ts_decay_linear(zscore(...), 5)
中性化由settings控制，表达式中不要包含group_neutralize"""

        results = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

        # Tag as rescue
        for res in results:
            res['modules_used'] = modules_used

        logger.info(f"Generated {len(results)} rescue variants for alpha {alpha_id} (attempt {candidate['attempt_count'] + 1})")
        return results

    def _process_rescue_task(self, task: Dict):
        """Process a rescue task - generate variants of borderline alpha."""
        expression = task.get("expression", "")
        sharpe = task.get("sharpe", 0)
        fitness = task.get("fitness", 0)
        turnover = task.get("turnover", 0)
        failed_checks = task.get("failed_checks", [])

        # Determine rescue type
        has_check_failures = len(failed_checks) > 0

        if has_check_failures:
            # Case B: Check failures - use targeted suggestions
            check_suggestions = self._get_check_suggestions(failed_checks)
            prompt = f"""因子检查失败，需要针对性修复。

【当前状态】
原代码: {expression}
Sharpe={sharpe:.2f} Fitness={fitness:.2f} Turnover={turnover:.2f}
失败检查: {', '.join(failed_checks)}

【针对性建议】
{check_suggestions}

【重要原则】
- 只修改时序平滑参数，不要改变核心逻辑
- 失败检查为 TURNOVER → 加大窗口 (10→20, 20→40)
- 失败检查为 SELF_CORRELATION → 在settings中改变neutralization
- 失败检查为 DRAWDOWN → 增加衰减平滑

【输出要求】
输出3个变种，外壳不变: ts_decay_linear(zscore(...), 5)
中性化由settings控制，表达式中不要包含group_neutralize"""
        else:
            # Case A: Poor performance - general optimization
            prompt = f"""因子表现接近达标，需要提升性能。

【当前状态】
原代码: {expression}
Sharpe={sharpe:.2f} Fitness={fitness:.2f} Turnover={turnover:.2f}

【优化建议】
1. 引入新的数据字段（如基本面、分析师、情绪数据）
2. 更换核心算子：ts_mean↔ts_std_dev, ts_rank↔ts_zscore
3. 调整时序窗口：5→10, 10→20
4. 使用非线性变换：abs, log, sign, rank

【输出要求】
输出3个变种，外壳不变: ts_decay_linear(zscore(...), 5)
中性化由settings控制，表达式中不要包含group_neutralize"""

        variants = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

        for variant in variants:
            if variant.get("expression"):
                variant['modules_used'] = task.get("modules_used", [])
                self.test_queue.put(variant)

    # ==========================================
    # Main Execution Loop
    # ==========================================

    def run(self, max_workers: int = 2):
        """Main execution loop."""
        if not self.authenticate():
            logger.error("Failed to authenticate, exiting")
            return

        # Load fields data
        fields_data = self.load_fields_from_csvs()
        logger.info(f"Loaded fields: {list(fields_data.keys())}")

        logger.info("Starting alpha miner...")

        # Start LLM producer thread
        producer = threading.Thread(
            target=self.llm_producer_worker,
            args=(fields_data,),
            daemon=True
        )
        producer.start()

        # Main consumer loop (event-driven, matching IQC approach)
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

                        mod_str = "+".join(factor.get("modules_used", []))
                        logger.info(f"Testing [{mod_str}]: {expression[:60]}...")
                        future = executor.submit(self.simulate_factor, factor)
                        running_tasks[future] = factor

                    if not running_tasks:
                        time.sleep(1)
                        continue

                    # Wait for any task to complete (event-driven, CPU efficient)
                    done, _ = concurrent.futures.wait(
                        running_tasks.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        factor = running_tasks.pop(future)
                        try:
                            result = future.result()
                            if not self.process_result(factor, result):
                                return  # Fatal error
                        except Exception as e:
                            logger.error(f"Task exception: {e}")
                            self.stats["failed"] += 1

                    # Print stats periodically
                    if self.stats["tested"] % 15 == 0 and self.stats["tested"] > 0:
                        rescue_count = self.alpha_db.count_rescue_pool()
                        logger.info(
                            f"Stats: tested={self.stats['tested']} "
                            f"passed={self.stats['passed']} "
                            f"failed={self.stats['failed']} "
                            f"rescued={self.stats['rescued']} "
                            f"flipped={self.stats['flipped']} "
                            f"rescue_pool={rescue_count} "
                            f"best_sharpe={self.stats['best_sharpe']:.2f} | "
                            f"Module weights: {self.module_stats}"
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
    parser.add_argument(
        "--member-id",
        type=str,
        default="default",
        help="Member ID for shared pool (prevents file conflicts in team)"
    )
    args = parser.parse_args()

    miner = AlphaMiner(llm_provider=args.llm, member_id=args.member_id)
    miner.run(max_workers=args.workers)


if __name__ == "__main__":
    main()
