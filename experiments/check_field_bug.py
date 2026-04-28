import re

def _extract_field_references(expr: str):
    # Simplified version of what's in the code
    pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
    operators = re.findall(pattern, expr)
    operators = [m for m in operators if not m[0].isupper()]

    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr)
    VALID_GROUP_ARGS = {"sector", "industry", "subindustry", "market"}
    fields = [t for t in tokens if t not in operators and t not in VALID_GROUP_ARGS]
    return fields

expr = "(close - open) / (close + 1e-6)"
print(f"Fields found: {_extract_field_references(expr)}")


