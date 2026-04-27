import re
import random
import logging
from typing import List, Dict, Tuple, Optional
from itertools import product

logger = logging.getLogger(__name__)

PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")


class ExpressionCompiler:
    def __init__(
        self,
        operators: List[str] = None,
        fields: List[str] = None,
        windows: List[int] = None,
        groups: List[str] = None,
    ):
        self.operators = operators or [
            "ts_mean", "ts_std_dev", "ts_rank", "ts_sum",
            "rank", "zscore", "log", "sqrt",
            "divide", "subtract", "add", "multiply",
            "group_neutralize", "group_mean", "group_zscore",
        ]
        self.fields = fields or []
        self.windows = windows or [5, 20, 60, 120, 180, 252]
        self.groups = groups or ["sector", "industry", "subindustry"]

    def compile_template(self, template: str, replacements: Dict[str, List[str]]) -> List[str]:
        placeholders = PLACEHOLDER_PATTERN.findall(template)
        if not placeholders:
            return [template]

        value_lists = []
        for ph in placeholders:
            if ph in replacements:
                value_lists.append([str(v) for v in replacements[ph]])
            elif ph == "FIELD" or ph.startswith("FIELD"):
                value_lists.append(self.fields if self.fields else ["returns"])
            elif ph == "WINDOW":
                value_lists.append([str(w) for w in self.windows])
            elif ph == "GROUP":
                value_lists.append(self.groups)
            elif ph == "OP":
                value_lists.append(self.operators)
            else:
                value_lists.append(["1"])

        results = []
        for combo in product(*value_lists):
            expr = template
            for ph, val in zip(placeholders, combo):
                expr = expr.replace(f"{{{ph}}}", val, 1)
            results.append(expr)

        return results

    def compile_templates(
        self,
        templates: List[str],
        replacements: Dict[str, List[str]] = None,
        max_expressions: int = 500,
    ) -> List[str]:
        if replacements is None:
            replacements = {}

        all_exprs = []
        for template in templates:
            compiled = self.compile_template(template, replacements)
            all_exprs.extend(compiled)

        if len(all_exprs) > max_expressions:
            random.shuffle(all_exprs)
            all_exprs = all_exprs[:max_expressions]

        logger.info(f"Compiled {len(all_exprs)} expressions from {len(templates)} templates")
        return all_exprs

    def extract_skeleton(self, expression: str) -> str:
        skeleton = re.sub(r"\b\d+\.?\d*\b", "{NUM}", expression)
        field_pattern = re.compile(
            r"\b([a-z][a-z0-9_]*(?:_[a-z0-9_]+)+)\b", re.IGNORECASE
        )
        used_fields = set(field_pattern.findall(expression))
        for field in sorted(used_fields, key=len, reverse=True):
            if not field.startswith("ts_") and not field.startswith("group_"):
                skeleton = skeleton.replace(field, "{FIELD}")
        return skeleton

    def crossover(self, expr1: str, expr2: str) -> List[str]:
        parts1 = self._split_at_operator(expr1)
        parts2 = self._split_at_operator(expr2)

        if len(parts1) < 2 or len(parts2) < 2:
            return [expr1, expr2]

        children = []
        cut1 = random.randint(1, len(parts1) - 1)
        cut2 = random.randint(1, len(parts2) - 1)

        child1 = parts1[:cut1] + parts2[cut2:]
        child2 = parts2[:cut2] + parts1[cut1:]

        if child1:
            children.append(" ".join(child1))
        if child2:
            children.append(" ".join(child2))

        return children

    def mutate(
        self,
        expression: str,
        mutation_rate: float = 0.3,
    ) -> str:
        if random.random() > mutation_rate:
            return expression

        mutation_type = random.choice(["window", "field", "operator"])

        if mutation_type == "window":
            numbers = re.findall(r"\b(\d+)\b", expression)
            if numbers and self.windows:
                old_num = random.choice(numbers)
                new_num = str(random.choice(self.windows))
                expression = expression.replace(old_num, new_num, 1)

        elif mutation_type == "field" and self.fields:
            tokens = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", expression, re.IGNORECASE)
            if tokens:
                old_field = random.choice(tokens)
                new_field = random.choice(self.fields)
                expression = expression.replace(old_field, new_field, 1)

        elif mutation_type == "operator" and self.operators:
            ops_in_expr = re.findall(r"\b(ts_\w+|group_\w+|rank|zscore|log|sqrt)\b", expression)
            if ops_in_expr:
                old_op = random.choice(ops_in_expr)
                new_op = random.choice(self.operators)
                expression = expression.replace(old_op, new_op, 1)

        return expression

    def _split_at_operator(self, expr: str) -> List[str]:
        parts = re.split(r"(?<=\)),\s*(?=[a-z])", expr, flags=re.IGNORECASE)
        if len(parts) <= 1:
            parts = re.split(r"\),\s*", expr)
        return parts if len(parts) > 1 else [expr]

    @staticmethod
    def get_default_templates() -> List[str]:
        return [
            "group_neutralize(zscore(ts_mean({FIELD}, {WINDOW})), {GROUP})",
            "group_neutralize(zscore(ts_mean({FIELD}, {WINDOW}) - ts_mean({FIELD}, {WINDOW2})), {GROUP})",
            "group_neutralize(zscore(rank(divide({FIELD}, {FIELD2}))), {GROUP})",
            "group_neutralize(zscore(ts_rank(ts_mean({FIELD}, {WINDOW}), {RANK})), {GROUP})",
            "rank(ts_std_dev({FIELD}, {WINDOW}))",
            "zscore(ts_delta({FIELD}, {WINDOW}))",
            "group_neutralize(zscore({OP}({FIELD})), {GROUP})",
            "ts_corr({FIELD}, {FIELD2}, {WINDOW})",
            "rank(divide(ts_mean({FIELD}, {WINDOW}), ts_std_dev({FIELD}, {WINDOW})))",
            "group_neutralize(zscore(subtract(ts_rank({FIELD}, {WINDOW}), ts_rank({FIELD2}, {WINDOW}))), {GROUP})",
            # Elite Price-Volume Chassis (New)
            "group_zscore(ts_decay_linear(rank((high - low) / close) + rank(min({FIELD} / adv20, 5)) - rank(returns), {WINDOW}), {GROUP}) * signed_power(group_rank(adv20, {GROUP}), 2)",
            "group_zscore(ts_decay_linear(ts_corr(ts_rank((high - low) / close, {WINDOW}), ts_rank({FIELD}, {WINDOW}), {WINDOW}), {WINDOW2}), {GROUP}) * signed_power(group_rank(adv20, {GROUP}), 2)",
            # 捕捉量价背离或基本面与价格的背离
            "group_neutralize(zscore(ts_rank(ts_corr({FIELD}, {FIELD2}, {WINDOW}), {WINDOW2})), {GROUP})",
            "rank(divide({FIELD}, {FIELD2}))",
            # 专门处理高频换手，只在标准差极大时才触发信号，平时信号归零
            "group_neutralize(if_else(abs(zscore({FIELD})) > 1.5, zscore(ts_av_diff({FIELD}, {WINDOW})), 0), {GROUP})",
            # 分别处理两种动量（基本面动量和价格动量），并在最后加权融合
            "group_neutralize(multiply(rank(ts_delta({FIELD}, {WINDOW})), rank(ts_delta({FIELD2}, {WINDOW2}))), {GROUP})",
            "group_neutralize(zscore(rank(divide({FIELD}, {FIELD2}))), {GROUP})",
            # 捕捉长期均值回归
            "group_neutralize(zscore(subtract(ts_mean({FIELD}, {WINDOW}), {FIELD})), {GROUP})",
        ]
