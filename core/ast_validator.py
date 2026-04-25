import re
import logging
from typing import List, Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)

CATEGORICAL_SUFFIXES = [
    "_naicss", "_sector", "_industry", "_sic", "_exchange",
    "_country", "_currency", "_region", "_subindustry", "_city",
]

VALID_GROUP_ARGS = {"sector", "industry", "subindustry", "market"}

SAFE_OPERATORS = {
    "ts_mean", "ts_std_dev", "ts_rank", "ts_sum", "ts_product",
    "ts_delta", "ts_zscore", "ts_ir", "ts_skewness", "ts_kurtosis",
    "ts_min", "ts_max", "ts_arg_min", "ts_arg_max",
    "ts_returns", "ts_scale", "ts_quantile",
    "ts_corr", "ts_covariance", "ts_regression",
    "rank", "zscore", "log", "sqrt", "abs", "sign",
    "divide", "subtract", "add", "multiply",
    "group_neutralize", "group_mean", "group_zscore",
    "group_rank", "group_sum", "group_max", "group_min",
    "group_std_dev", "group_median",
    "signed_power", "reverse", "inverse", "normalize",
    "scale_down", "fraction", "quantile",
    "winsorize", "ts_backfill",
    "pasteurize", "nan_mask",
}

TS_OPERATORS = {
    "ts_mean", "ts_std_dev", "ts_rank", "ts_sum", "ts_product",
    "ts_delta", "ts_zscore", "ts_ir", "ts_skewness", "ts_kurtosis",
    "ts_min", "ts_max", "ts_arg_min", "ts_arg_max",
    "ts_returns", "ts_scale", "ts_quantile",
    "ts_corr", "ts_covariance", "ts_regression",
    "ts_decay_exp_window", "ts_moment", "ts_entropy",
    "ts_min_max_cps", "ts_min_max_diff", "ts_percentage",
}

BANNED_OPERATORS = {
    "if_else", "trade_when", "bucket", "equal", "greater", "less",
    "not_equal", "normalize",
}

VALID_WINDOWS = {5, 20, 60, 120, 180, 252}


class ASTValidator:
    def __init__(
        self,
        known_operators: Set[str] = None,
        known_fields: Set[str] = None,
        strict: bool = False,
    ):
        self.known_operators = known_operators or SAFE_OPERATORS
        self.known_fields = known_fields or set()
        self.strict = strict
        self.error_log: List[Dict] = []

    def validate(self, expression: str) -> Tuple[bool, List[str]]:
        errors = []

        if not expression or not expression.strip():
            return False, ["Empty expression"]

        expr = expression.strip()

        if "=" in expr and not expr.startswith("-"):
            errors.append("Contains assignment")
        if ";" in expr:
            errors.append("Contains semicolon/multi-statement")
        if expr.startswith("Comment:"):
            errors.append("Is a comment")

        paren_balance = 0
        for ch in expr:
            if ch == "(":
                paren_balance += 1
            elif ch == ")":
                paren_balance -= 1
            if paren_balance < 0:
                errors.append("Unmatched closing parenthesis")
                break
        if paren_balance != 0:
            errors.append(f"Unbalanced parentheses (diff={paren_balance})")

        for banned in BANNED_OPERATORS:
            if banned + "(" in expr:
                errors.append(f"Uses banned operator: {banned}")

        operators_found = self._extract_operators(expr)
        for op in operators_found:
            if self.strict and op not in self.known_operators:
                errors.append(f"Unknown operator: {op}")

        fields_found = self._extract_field_references(expr)
        for field in fields_found:
            if self._is_categorical_field(field):
                for ts_op in TS_OPERATORS:
                    pattern = ts_op + r"\s*\([^)]*" + re.escape(field)
                    if re.search(pattern, expr, re.IGNORECASE):
                        errors.append(
                            f"Categorical field '{field}' used in time series operator '{ts_op}'"
                        )
                        break

        if "group_neutralize(" in expr or "group_mean(" in expr or "group_zscore(" in expr:
            has_valid_group = bool(re.search(r"(?:sector|industry|subindustry|market)\s*\)", expr))
            if not has_valid_group:
                errors.append("Group operator with invalid group argument")

        depth = self._max_nesting_depth(expr)
        if depth > 4:
            errors.append(f"Too deep nesting: {depth} levels (max 4)")

        is_valid = len(errors) == 0
        if not is_valid:
            self.error_log.append({"expression": expr, "errors": errors})
        return is_valid, errors

    def validate_batch(self, expressions: List[str]) -> Tuple[List[str], List[Dict]]:
        valid = []
        invalid = []
        for expr in expressions:
            ok, errors = self.validate(expr)
            if ok:
                valid.append(expr)
            else:
                invalid.append({"expression": expr, "errors": errors})
        logger.info(
            f"AST validation: {len(valid)} valid, {len(invalid)} invalid out of {len(expressions)}"
        )
        return valid, invalid

    def _extract_operators(self, expr: str) -> List[str]:
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        matches = re.findall(pattern, expr)
        return [m for m in matches if not m[0].isupper()]

    def _extract_field_references(self, expr: str) -> List[str]:
        operators = self._extract_operators(expr)
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)
        fields = [t for t in tokens if t not in operators and t not in VALID_GROUP_ARGS]
        return fields

    def _is_categorical_field(self, field: str) -> bool:
        field_lower = field.lower()
        return any(field_lower.endswith(s) or s + "_" in field_lower for s in CATEGORICAL_SUFFIXES)

    def _max_nesting_depth(self, expr: str) -> int:
        max_depth = 0
        current_depth = 0
        for ch in expr:
            if ch == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif ch == ")":
                current_depth -= 1
        return max_depth

    def get_error_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for entry in self.error_log:
            for error in entry["errors"]:
                key = error.split(":")[0] if ":" in error else error
                stats[key] = stats.get(key, 0) + 1
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
