"""
Unified LLM Client for Alpha Generation

Supports both Ollama (local) and DeepSeek API for generating
WorldQuant Brain alpha expressions.
"""

import os
import json
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting Ollama and DeepSeek API."""

    def __init__(self, provider: str = "auto"):
        """
        Initialize LLM client.

        Args:
            provider: "ollama", "deepseek", or "auto" (tries DeepSeek first, falls back to Ollama)
        """
        self.provider = provider
        self._setup_provider()

    def _setup_provider(self):
        """Setup the LLM provider based on configuration."""
        if self.provider == "auto":
            # Try DeepSeek first (faster, no local GPU needed)
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_key:
                self.provider = "deepseek"
                logger.info("Using DeepSeek API")
            else:
                self.provider = "ollama"
                logger.info("Using Ollama (local)")

        if self.provider == "deepseek":
            self._setup_deepseek()
        elif self.provider == "ollama":
            self._setup_ollama()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _setup_deepseek(self):
        """Setup DeepSeek API client."""
        try:
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set in .env")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
                timeout=120.0
            )
            self.model = "deepseek-v4-pro"
            logger.info("DeepSeek API client initialized")
        except ImportError:
            raise ImportError("openai package required for DeepSeek: pip install openai")

    def _setup_ollama(self):
        """Setup Ollama client."""
        import requests
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

        # Test connection
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                logger.warning(f"Ollama not responding at {self.ollama_url}")
        except Exception as e:
            logger.warning(f"Cannot connect to Ollama: {e}")

    def generate_alphas(self, system_prompt: str, user_prompt: str, num_alphas: int = 5) -> List[Dict]:
        """
        Generate alpha expressions using the configured LLM.

        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User prompt with specific requirements
            num_alphas: Number of alphas to generate

        Returns:
            List of alpha dicts with 'expression', 'logic', and optional 'settings'
        """
        if self.provider == "deepseek":
            return self._generate_deepseek(system_prompt, user_prompt)
        else:
            return self._generate_ollama(system_prompt, user_prompt)

    def _generate_deepseek(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        """Generate using DeepSeek API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.choices[0].message.content
            return self._extract_json(content)
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return []

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        """Generate using Ollama API."""
        import requests

        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048
                    }
                },
                timeout=120
            )

            if response.status_code != 200:
                logger.error(f"Ollama error: {response.status_code} {response.text}")
                return []

            data = response.json()
            content = data.get("message", {}).get("content", "")
            return self._extract_json(content)
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return []

    def _extract_json(self, text: str) -> List[Dict]:
        """Extract JSON array from LLM response text."""
        # Try to find JSON array in the response
        match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {e}")

        # Try to find individual JSON objects
        objects = re.findall(r'\{[^{}]+\}', text)
        if objects:
            results = []
            for obj_str in objects:
                try:
                    obj = json.loads(obj_str)
                    if "expression" in obj:
                        results.append(obj)
                except json.JSONDecodeError:
                    continue
            if results:
                return results

        logger.warning("No valid JSON found in LLM response")
        return []


# Default system prompt for alpha generation
DEFAULT_SYSTEM_PROMPT = """你是一名 WorldQuant 顶级量化架构师。
核心纪律：
1. 只能输出纯 JSON 数组，不要有任何多余的 Markdown 或对话文字。
2. 绝对不允许使用 <WINDOW> 等占位符，必须填入具体的整数(如 5, 10, 20, 60)。
3. 必须使用 FASTEXPR 语法，将核心逻辑嵌套在此平滑外壳内：
   ts_decay_linear( zscore( 你的核心截面/时序逻辑 ), 10 )
   注意：衰减窗口至少为 10，换手率高的因子使用 20 或更大。
4. 不要在表达式中使用 group_neutralize，中性化由 settings 控制。
5. JSON 结构必须为: [{"logic": "描述", "expression": "代码", "settings": {"delay":<由用户提示词指定>, "neutralization":"INDUSTRY", "truncation":0.08, "pasteurization":"ON"}}]
   - neutralization 只能是以下值之一: "NONE", "INDUSTRY", "SUBINDUSTRY", "SECTOR", "MARKET"
   - 绝对不能使用 "STYLE", "COUNTRY" 等其他值！
6. 事件字段（如 nws_*, snt_*, scl_*_buzz*, rp_*, fnd6_*event*, anl4_* 等）是 VECTOR 类型：
   - ❌ WQ API 对事件字段几乎拒绝所有运算符：==、!=、sign()、trade_when() 等均不支持！
   - ❌ 不能参与算术运算（+,-,*,/）
   - ❌ 不能用 ts_delta, ts_mean, ts_sum, rank, sign, zscore 等任何运算符
   - ⚠️ 强烈建议：只使用 MATRIX 类型字段生成因子，忽略 VECTOR 字段！
   - ⚠️ 如果 prompt 中列出了 VECTOR 字段，不要使用它们。只用 MATRIX 字段。
7. 运算符参数必须严格遵守：
   - 单参数：rank(x), sign(x), abs(x), log(x), zscore(x), inverse(x), sqrt(x)
   - 时序单参数+窗口：ts_rank(x,d), ts_zscore(x,d), ts_mean(x,d), ts_std_dev(x,d), ts_sum(x,d), ts_delta(x,d), ts_delay(x,d), ts_decay_linear(x,d)
   - 时序双参数+窗口：ts_corr(x,y,d), ts_covariance(y,x,d)
   - 逻辑：if_else(condition, true_val, false_val), trade_when(condition, x, y)
   - 错误示例：rank(a,b) ❌ → 正确：rank(a/b) ✓
8. 绝对禁止使用字符串字面量！WorldQuant 不支持字符串比较。
   - ❌ 错误：if_else(field == "revision", x, y)
   - ❌ 错误：if_else(field > "value", x, y)
   - ✓ 正确：if_else(field > 0, x, y)
   - ✓ 正确：trade_when(field > threshold, x, y)
   - 所有条件必须是数值比较（>, <, ==, >=, <=, !=）
9. 绝对禁止在表达式中使用双引号或单引号！表达式只能包含数字、变量名和运算符。
10. 逻辑运算符必须使用函数形式，禁止使用中缀形式或符号：
    - 逻辑与：and(input1, input2) — 例如 and(x > 0, y > 0)
    - 逻辑或：or(input1, input2) — 例如 or(x > 0, y > 0)
    - 逻辑非：not(x) — 例如 not(x > 0)
    - ✓ 正确：if_else(and(x > 0, y > 0), a, b)
    - ❌ 错误：if_else(x > 0 and y > 0, a, b) — 禁止中缀形式
    - ❌ 错误：if_else(x > 0 & y > 0, a, b) — 禁止符号形式
11. 绝对禁止使用科学计数法！FASTEXPR 不支持 1e-6、1e-8 等写法。
    - ❌ 错误：x / (y + 1e-6)  → 报错 Unexpected character 'e'
    - ✓ 正确：x / (y + 0.000001)
    - 所有极小值必须写成小数形式，不能使用 e 记法。
"""


def get_llm_client(provider: str = "auto") -> LLMClient:
    """Get or create LLM client singleton."""
    if not hasattr(get_llm_client, '_instance'):
        get_llm_client._instance = LLMClient(provider=provider)
    return get_llm_client._instance
