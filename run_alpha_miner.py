"""
WorldQuant Brain Alpha Miner

Simplified alpha mining pipeline using direct API calls.
Supports both Ollama and DeepSeek for LLM-based alpha generation.
"""

import os
import sys
import time
import json
import re
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
from core.notifier import get_notifier

logger = setup_logger(__name__)

# Constants
BASE_URL = "https://api.worldquantbrain.com"
MIN_SHARPE = 1.25
MIN_FITNESS = 1.0
RESCUE_THRESHOLD = 1.7

# Parameter sweep settings for check failures
SETTINGS_SWEEP = [
    {"neutralization": "INDUSTRY", "truncation": 0.1, "decay": 5, "delay": 1},
    {"neutralization": "INDUSTRY", "truncation": 0.1, "decay": 5, "delay": 0},
    {"neutralization": "SUBINDUSTRY", "truncation": 0.15, "decay": 10, "delay": 1},
    {"neutralization": "SUBINDUSTRY", "truncation": 0.15, "decay": 10, "delay": 0},
    {"neutralization": "SECTOR", "truncation": 0.08, "decay": 20, "delay": 0},
    {"neutralization": "SECTOR", "truncation": 0.08, "decay": 20, "delay": 1},
    {"neutralization": "MARKET", "truncation": 0.2, "decay": 40, "delay": 1},
    {"neutralization": "MARKET", "truncation": 0.2, "decay": 40, "delay": 0},
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
FIELDS_DIR_DELAY1 = os.path.join(DATA_DIR, "fields_delay1")
FIELDS_DIR_DELAY0 = os.path.join(DATA_DIR, "fields_delay0")
OPERATORS_DIR = os.path.join(DATA_DIR, "operators")
SHARED_POOL_DIR = os.path.join(DATA_DIR, "shared_pool")


def clean_expression(expr: str) -> str:
    """Fix common LLM mistakes in expressions."""
    # WorldQuant Brain 使用函数形式的逻辑运算符：and(x,y), or(x,y), not(x)
    # 不是中缀形式：x and y, x & y
    # 暂时保留原样，因为简单的正则替换无法正确处理嵌套逻辑
    # TODO: 添加更复杂的解析逻辑来转换中缀形式为函数形式
    return expr

# Target field files to load (API-fetched dataset files)
TARGET_FIELD_FILES = [
    "analyst4.csv",
    "fundamental2.csv",
    "fundamental6.csv",
    "model16.csv",
    "model51.csv",
    "model77.csv",
    "news12.csv",
    "news18.csv",
    "option8.csv",
    "option9.csv",
    "pv1.csv",
    "pv13.csv",
    "sentiment1.csv",
    "socialmedia12.csv",
    "socialmedia8.csv",
    "univ1.csv",
]


class AlphaMiner:
    """Main alpha mining engine with direct API approach."""

    def __init__(self, llm_provider: str = "auto", member_id: str = "default",
                 username: str = None, password: str = None,
                 delay0_prob: float = 0.5):
        self.llm_client = get_llm_client(llm_provider)
        self.alpha_db = get_alpha_db()
        self.quota = get_submission_quota()
        self.member_id = member_id
        self.notifier = get_notifier()
        self._username = username
        self._password = password
        self.delay0_prob = delay0_prob

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
        self.module_stats = {
            "ANALYST4": {"tried": 0, "success": 0},
            "FUNDAMENTAL2": {"tried": 0, "success": 0},
            "FUNDAMENTAL6": {"tried": 0, "success": 0},
            "MODEL16": {"tried": 0, "success": 0},
            "MODEL51": {"tried": 0, "success": 0},
            "MODEL77": {"tried": 0, "success": 0},
            "NEWS12": {"tried": 0, "success": 0},
            "NEWS18": {"tried": 0, "success": 0},
            "OPTION8": {"tried": 0, "success": 0},
            "OPTION9": {"tried": 0, "success": 0},
            "PV1": {"tried": 0, "success": 0},
            "PV13": {"tried": 0, "success": 0},
            "SENTIMENT1": {"tried": 0, "success": 0},
            "SOCIALMEDIA12": {"tried": 0, "success": 0},
            "SOCIALMEDIA8": {"tried": 0, "success": 0},
            "UNIV1": {"tried": 0, "success": 0},
        }

        # Operator knowledge base
        self.operator_arity = self._load_operator_knowledge()

        # Create directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(FIELDS_DIR_DELAY1, exist_ok=True)
        os.makedirs(FIELDS_DIR_DELAY0, exist_ok=True)
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

    def validate_expression_variables(self, expression: str, available_fields: set) -> bool:
        """验证表达式中的变量是否都存在于可用字段列表中"""
        # 已知的运算符和内置变量
        known_operators = set(self.operator_arity.keys()) if self.operator_arity else set()
        builtin_vars = {
            'returns', 'volume', 'close', 'open', 'high', 'low', 'vwap',
            'adv20', 'adv50', 'adv120', 'adv240',
            'market_cap', 'sector', 'industry', 'subindustry',
            'liabilities', 'assets', 'equity', 'debt_lt', 'debt_st',
        }

        # 提取表达式中的所有标识符（变量名）
        # 匹配字母开头，包含字母、数字、下划线的标识符
        identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression))

        # 过滤掉运算符和内置变量
        potential_fields = identifiers - known_operators - builtin_vars

        # 检查每个潜在字段是否在可用字段列表中
        unknown_fields = []
        for field in potential_fields:
            # 跳过数字开头的标识符（可能是常量）
            if field[0].isdigit():
                continue
            # 跳过常见的非字段标识符
            if field in {'true', 'false', 'null', 'nan', 'inf', 'e', 'pi'}:
                continue
            if field not in available_fields:
                unknown_fields.append(field)

        if unknown_fields:
            logger.warning(f"Expression contains unknown fields: {unknown_fields}")
            return False

        return True

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

    def load_fields_from_csvs(self, fields_dir: str = None) -> Dict[str, List[Dict]]:
        """Load ALL field data from specified CSV files (full pool for sampling)."""
        import pandas as pd

        # 默认使用 delay1 目录
        if fields_dir is None:
            fields_dir = FIELDS_DIR_DELAY1

        all_fields = {}

        if not os.path.exists(fields_dir):
            logger.info(f"Fields directory not found: {fields_dir}")
            return self._get_default_fields()

        # 获取目录中所有 CSV 文件
        csv_files = [f for f in os.listdir(fields_dir) if f.endswith('.csv')]

        for file in csv_files:
            filepath = os.path.join(fields_dir, file)
            try:
                df = pd.read_csv(filepath)
                if 'Field' in df.columns and 'Description' in df.columns:
                    # 选择需要的列
                    columns_to_keep = ['Field', 'Description']
                    if 'Type' in df.columns:
                        columns_to_keep.append('Type')
                    if 'Alphas' in df.columns:
                        columns_to_keep.append('Alphas')

                    fields = df[columns_to_keep].to_dict(orient='records')

                    # 填充默认值
                    for field in fields:
                        if 'Type' not in field:
                            field['Type'] = 'MATRIX'
                        if 'Alphas' not in field:
                            field['Alphas'] = 0

                    category = file.replace(".csv", "").upper()
                    all_fields[category] = fields
                    logger.info(f"Loaded {len(fields)} fields from {file}")
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")

        if not all_fields:
            logger.info("No target CSV files found, using default fields")
            all_fields = self._get_default_fields()

        return all_fields

    def _get_default_fields(self) -> Dict[str, List[Dict]]:
        """Get default field data for alpha generation."""
        return {
            "PV13": [
                {"Field": "close", "Description": "收盘价"},
                {"Field": "open", "Description": "开盘价"},
                {"Field": "high", "Description": "最高价"},
                {"Field": "low", "Description": "最低价"},
                {"Field": "volume", "Description": "成交量"},
                {"Field": "returns", "Description": "收益率"},
                {"Field": "vwap", "Description": "成交量加权平均价"}
            ],
            "FUNDAMENTAL6": [
                {"Field": "market_cap", "Description": "市值"},
                {"Field": "pe_ratio", "Description": "市盈率"},
                {"Field": "pb_ratio", "Description": "市净率"},
                {"Field": "roe", "Description": "净资产收益率"}
            ]
        }

    # ==========================================
    # Dynamic Module Weights
    # ==========================================

    @staticmethod
    def log_minmax_softmax(values: list, temperature: float = 0.12) -> list:
        """Log + MinMax + Softmax 权重转换。

        1. Log(1+x) 压缩极端值
        2. MinMax 归一化到 [0, 1]
        3. Softmax with temperature
        """
        import math
        log_values = [math.log1p(v) for v in values]
        min_val = min(log_values)
        max_val = max(log_values)
        if max_val - min_val == 0:
            return [1.0 / len(values)] * len(values)
        normalized = [(x - min_val) / (max_val - min_val) for x in log_values]
        scaled = [x / temperature for x in normalized]
        max_scaled = max(scaled)
        exp_values = [math.exp(x - max_scaled) for x in scaled]
        sum_exp = sum(exp_values)
        return [v / sum_exp for v in exp_values]

    def get_dynamic_modules(self, fields_pool: Dict, sample_size: int = 15) -> tuple:
        """
        Select 1-2 modules using Log+MinMax+Softmax weighting,
        then sample fields using the same method.
        Returns (selected_fields, modules_used)
        """
        # 计算每个数据集的总 alpha 数量
        module_alpha_counts = {}
        for mod, fields in fields_pool.items():
            total_alphas = sum(f.get('Alphas', 0) for f in fields)
            module_alpha_counts[mod] = total_alphas

        # 使用 Log+MinMax+Softmax 计算数据集权重
        modules = list(fields_pool.keys())
        raw_counts = [module_alpha_counts.get(mod, 0) for mod in modules]

        if sum(raw_counts) == 0:
            weights = [1.0] * len(modules)
        else:
            weights = self.log_minmax_softmax(raw_counts, temperature=0.12)

        # Select 1 or 2 modules
        num_to_select = random.choice([1, 2])
        selected = random.choices(modules, weights=weights, k=num_to_select)
        selected = list(set(selected))

        # 在选定的数据集中，使用 Log+MinMax+Softmax 选择字段
        selected_fields = {}
        for mod in selected:
            pool = fields_pool.get(mod, [])
            if not pool:
                continue

            raw_weights = [f.get('Alphas', 0) for f in pool]
            if sum(raw_weights) == 0:
                field_weights = [1.0] * len(pool)
            else:
                field_weights = self.log_minmax_softmax(raw_weights, temperature=0.12)

            n = min(sample_size, len(pool))
            selected_fields[mod] = random.choices(pool, weights=field_weights, k=n)

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
        # 根据概率选择 delay0 或 delay1 字段
        if random.random() < self.delay0_prob:
            fields_data = self.fields_delay0
            delay = 0
            logger.info("Selected delay=0 fields")
        else:
            fields_data = self.fields_delay1
            delay = 1
            logger.info("Selected delay=1 fields")

        # Use dynamic module selection
        target_fields, modules_used = self.get_dynamic_modules(fields_data)
        mod_names = "+".join(modules_used)

        logger.info(f"Generating alphas for modules: {mod_names}")

        # 分离 MATRIX 和 VECTOR 字段
        matrix_fields = {}
        vector_fields = {}
        for mod, fields in target_fields.items():
            matrix_list = [f for f in fields if f.get('Type', 'MATRIX') == 'MATRIX']
            vector_list = [f for f in fields if f.get('Type', 'MATRIX') == 'VECTOR']
            if matrix_list:
                matrix_fields[mod] = matrix_list
            if vector_list:
                vector_fields[mod] = vector_list

        # 构建字段说明
        field_description = "【MATRIX 字段】（可正常使用所有运算符）:\n"
        field_description += json.dumps(matrix_fields, ensure_ascii=False)

        if vector_fields:
            field_description += "\n\n【VECTOR 字段】（event 类型，有严格限制）:\n"
            field_description += json.dumps(vector_fields, ensure_ascii=False)
            field_description += """
VECTOR 字段使用限制：
- ❌ 绝对不能用 > < >= <= 比较！
- ❌ 不能参与算术运算（+,-,*,/）
- ❌ 不能用 ts_delta, ts_mean, ts_sum, rank, sign 等运算符
- ✓ 只能用 == 或 != 判断：if_else(field == 1, x, y)
- ✓ 或用 sign() 转换后再比较：if_else(sign(field) == 1, x, y)
- ✓ 或直接作为 trade_when 的条件：trade_when(field, x, y)"""

        prompt = f"""请利用以下提供的数据字段，生成 5 个具有爆发力的全新因子。

【重要规则】
1. 只能使用下面列出的字段名，绝对不能使用其他字段名！
2. 不能使用 total_assets, book_value_per_share, return_on_equity 等不存在的字段
3. 如果需要总资产，用 assets；如果需要权益，用 equity；如果需要长期负债，用 debt_lt
4. 逻辑运算符必须使用函数形式：and(x,y), or(x,y), not(x)，禁止使用 & | ~ 或中缀形式
5. 当前数据字段为 delay={delay}，所有生成的因子 settings 中 delay 必须设置为 {delay}

{field_description}"""

        results = self.llm_client.generate_alphas(DEFAULT_SYSTEM_PROMPT, prompt)

        if results:
            self.notifier.record_llm_success()
        else:
            self.notifier.record_llm_error()

        # 构建可用字段集合（用于验证）
        available_fields = set()
        for mod, fields in target_fields.items():
            for field in fields:
                available_fields.add(field['Field'])

        # Clean expressions, validate variables, and tag with modules used
        valid_results = []
        for res in results:
            expression = res.get('expression', '')
            if not expression:
                continue

            # 验证变量
            if not self.validate_expression_variables(expression, available_fields):
                logger.warning(f"Skipping expression with unknown variables: {expression[:60]}...")
                continue

            res['expression'] = clean_expression(expression)
            res['modules_used'] = modules_used
            res['delay'] = delay  # 添加 delay 值
            valid_results.append(res)

        return valid_results

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

        if results:
            self.notifier.record_llm_success()
        else:
            self.notifier.record_llm_error()

        # Clean expressions and tag as crossover
        for res in results:
            res['expression'] = clean_expression(res.get('expression', ''))
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
            "delay": factor.get("delay", 1),  # 使用因子中的 delay 值
            "decay": 0,
            "neutralization": "INDUSTRY",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False
        }

        # Apply custom settings if provided (delay 由系统控制，不由 LLM 覆盖)
        actual_delay = factor.get("delay", 1)
        if isinstance(factor.get("settings"), dict):
            for k, v in factor["settings"].items():
                if k in settings and k != "delay":
                    settings[k] = v
        # 再次确保 delay 使用系统设定的值，不被 LLM 输出的 settings 覆盖
        settings["delay"] = actual_delay

        payload = {
            "type": "REGULAR",
            "settings": settings,
            "regular": expression
        }

        try:
            # Submit simulation with retry for concurrent limit
            for attempt in range(3):
                resp = self.session.post(
                    f"{BASE_URL}/simulations",
                    json=payload,
                    verify=False,
                    timeout=30
                )

                # Handle auth failures
                if resp.status_code in [401, 403] or "Incorrect authentication" in resp.text:
                    return {"error": "AUTH_FAILED"}

                # Handle concurrent limit
                if "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in resp.text:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"Concurrent limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if resp.status_code != 201:
                    return {"error": f"Simulation failed: {resp.text[:200]}"}

                break
            else:
                return {"error": "Max retries exceeded"}

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
                self.notifier.record_auth_failure()
                logger.warning("Authentication expired, attempting re-login...")
                if self.authenticate():
                    self.notifier.record_auth_success()
                    self.test_queue.put(factor)  # Re-queue factor
                else:
                    logger.error("Re-login failed, stopping")
                    self.notifier.notify_fatal(
                        "鉴权失败，重新登录也未成功。矿机已停止。",
                        member_id=self.member_id,
                    )
                    return False
            return True

        # Extract metrics (with None value protection)
        sharpe = result.get("sharpe", 0) or 0
        fitness = result.get("fitness", 0) or 0
        alpha_id = result.get("alpha_id")
        turnover = result.get("turnover", 0) or 0
        margin = result.get("margin", 0) or 0

        flipped_tag = " [FLIPPED]" if factor.get("flipped_from") else ""
        logger.info(
            f"Result: S={sharpe:.2f} F={fitness:.2f} T={turnover:.2f} M={margin:.2f}{flipped_tag} | "
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
                logger.info(f"Found alpha! S={sharpe:.2f} F={fitness:.2f} (pending)")
                self.stats["passed"] += 1

                # 飞书通知
                self.notifier.notify_alpha(
                    alpha_id=alpha_id or "N/A",
                    sharpe=sharpe,
                    fitness=fitness,
                    turnover=turnover,
                    expression=expression,
                    member_id=self.member_id,
                )

                # Save to database as pending (will become unsubmitted after correlation check)
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
                    status="pending",
                )

        # Reverse factor detection (Sharpe < -0.8)
        elif sharpe < -0.8:
            logger.info(f"Found reverse factor (S={sharpe:.2f}), flipping sign...")
            self.stats["flipped"] += 1

            # Create flipped version
            flipped_factor = factor.copy()
            flipped_factor['expression'] = f"-1 * ({expression})"
            flipped_factor['flipped_from'] = expression
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
                        status="pending",
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

        if results:
            self.notifier.record_llm_success()
        else:
            self.notifier.record_llm_error()

        # Clean expressions and tag as rescue
        for res in results:
            res['expression'] = clean_expression(res.get('expression', ''))
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

        if variants:
            self.notifier.record_llm_success()
        else:
            self.notifier.record_llm_error()

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

        # Load both delay0 and delay1 fields
        self.fields_delay1 = self.load_fields_from_csvs(FIELDS_DIR_DELAY1)
        self.fields_delay0 = self.load_fields_from_csvs(FIELDS_DIR_DELAY0)
        logger.info(f"Loaded delay1 fields: {list(self.fields_delay1.keys())}")
        logger.info(f"Loaded delay0 fields: {list(self.fields_delay0.keys())}")
        logger.info(f"Delay0 probability: {self.delay0_prob}")

        # 默认使用 delay1 字段
        fields_data = self.fields_delay1

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

                        # 每 100 个因子发送汇总通知（使用 DB 全量统计）
                        if self.stats["tested"] % 100 == 0:
                            db_stats = self.alpha_db.get_all_time_stats()
                            self.notifier.notify_summary(
                                tested=db_stats["tested"],
                                passed=db_stats["passed"],
                                failed=db_stats["failed"],
                                best_sharpe=db_stats["best_sharpe"],
                                best_fitness=db_stats["best_fitness"],
                                rescue_pool=rescue_count,
                                member_id=self.member_id,
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
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="WQ username (overrides .env)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="WQ password (overrides .env)"
    )
    parser.add_argument(
        "--delay0-prob",
        type=float,
        default=0.5,
        help="Probability of mining delay=0 factors (default: 0.5)"
    )
    args = parser.parse_args()

    miner = AlphaMiner(
        llm_provider=args.llm,
        member_id=args.member_id,
        username=args.username,
        password=args.password,
        delay0_prob=args.delay0_prob,
    )
    miner.run(max_workers=args.workers)


if __name__ == "__main__":
    main()
