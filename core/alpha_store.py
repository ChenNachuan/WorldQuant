import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

ALPHA_DIR = "alpha"


def _ensure_alpha_dir() -> None:
    os.makedirs(ALPHA_DIR, exist_ok=True)


def _date_file(date_str: str = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ALPHA_DIR, f"{date_str}.json")


def _load_date_file(filepath: str) -> Dict:
    if not os.path.exists(filepath):
        return {"date": os.path.splitext(os.path.basename(filepath))[0], "alphas": []}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Corrupted file {filepath}, starting fresh: {e}")
        return {"date": os.path.splitext(os.path.basename(filepath))[0], "alphas": []}


def _save_date_file(filepath: str, data: Dict) -> None:
    _ensure_alpha_dir()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_alpha_record(
    expression: str,
    alpha_data: Dict,
    source: str = "generator",
    settings: Dict = None,
) -> Dict:
    is_data = alpha_data.get("is", {})
    ts = alpha_data.get("dateCreated") or datetime.now().isoformat()
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts).isoformat()

    api_settings = alpha_data.get("settings", {})
    if settings is None and api_settings:
        settings = api_settings

    if settings is None:
        settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "INDUSTRY",
            "truncation": 0.01,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
        }

    record = {
        "basic": {
            "alpha_id": alpha_data.get("id", "unknown"),
            "expression": expression,
            "created_at": ts,
            "source": source,
            "grade": alpha_data.get("grade", is_data.get("grade", "UNKNOWN")),
        },
        "settings": {
            "instrument_type": settings.get("instrumentType", "EQUITY"),
            "region": settings.get("region", "USA"),
            "universe": settings.get("universe", "TOP3000"),
            "delay": settings.get("delay", 1),
            "decay": settings.get("decay", 0),
            "neutralization": settings.get("neutralization", "INDUSTRY"),
            "truncation": settings.get("truncation", 0.01),
            "pasteurization": settings.get("pasteurization", "ON"),
            "unit_handling": settings.get("unitHandling", "VERIFY"),
            "nan_handling": settings.get("nanHandling", "OFF"),
            "language": settings.get("language", "FASTEXPR"),
        },
        "backtest": {
            "fitness": is_data.get("fitness"),
            "sharpe": is_data.get("sharpe"),
            "turnover": is_data.get("turnover"),
            "returns": is_data.get("returns"),
            "margin": is_data.get("margin"),
            "long_count": is_data.get("longCount"),
            "short_count": is_data.get("shortCount"),
            "checks": is_data.get("checks", []),
        },
    }
    return record


def save_alpha(
    expression: str,
    alpha_data: Dict,
    source: str = "generator",
    date_str: str = None,
    settings: Dict = None,
) -> str:
    record = build_alpha_record(expression, alpha_data, source, settings)
    filepath = _date_file(date_str)
    data = _load_date_file(filepath)
    data["alphas"].append(record)
    _save_date_file(filepath, data)
    logger.info(f"Saved alpha {record['basic']['alpha_id']} to {filepath}")
    return filepath


def load_alphas_by_date(date_str: str) -> List[Dict]:
    filepath = _date_file(date_str)
    data = _load_date_file(filepath)
    return data.get("alphas", [])


def load_all_alphas() -> List[Dict]:
    _ensure_alpha_dir()
    all_alphas = []
    for filename in sorted(os.listdir(ALPHA_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(ALPHA_DIR, filename)
        data = _load_date_file(filepath)
        for alpha in data.get("alphas", []):
            alpha["_source_file"] = filepath
            all_alphas.append(alpha)
    return all_alphas


def load_unsubmitted_alphas() -> List[Dict]:
    alphas = load_all_alphas()
    result = []
    for alpha in alphas:
        checks = alpha.get("backtest", {}).get("checks", [])
        has_fail = any(c.get("result") == "FAIL" for c in checks)
        if not has_fail:
            result.append(alpha)
    return result


def count_alphas(date_str: str = None) -> int:
    if date_str:
        return len(load_alphas_by_date(date_str))
    return len(load_all_alphas())


def remove_alpha_by_expression(expression: str) -> bool:
    _ensure_alpha_dir()
    for filename in sorted(os.listdir(ALPHA_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(ALPHA_DIR, filename)
        data = _load_date_file(filepath)
        original_count = len(data.get("alphas", []))
        data["alphas"] = [
            a for a in data["alphas"]
            if a.get("basic", {}).get("expression") != expression
        ]
        if len(data["alphas"]) < original_count:
            _save_date_file(filepath, data)
            logger.info(f"Removed alpha '{expression[:60]}...' from {filepath}")
            return True
    logger.info(f"Alpha '{expression[:60]}...' not found in any date file")
    return False


def clear_alphas(date_str: str = None) -> None:
    if date_str:
        filepath = _date_file(date_str)
        if os.path.exists(filepath):
            backup = filepath.replace(".json", f"_backup_{int(time.time())}.json")
            os.rename(filepath, backup)
            logger.info(f"Backed up and cleared {filepath} -> {backup}")
        return

    _ensure_alpha_dir()
    for filename in os.listdir(ALPHA_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(ALPHA_DIR, filename)
        backup = filepath.replace(".json", f"_backup_{int(time.time())}.json")
        os.rename(filepath, backup)
    logger.info("Backed up and cleared all alpha date files")


def migrate_from_hopeful_alphas(hopeful_file: str = "hopeful_alphas.json") -> int:
    if not os.path.exists(hopeful_file):
        logger.info(f"Legacy file {hopeful_file} not found, nothing to migrate")
        return 0

    try:
        with open(hopeful_file, "r", encoding="utf-8") as f:
            old_alphas = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Cannot read {hopeful_file}: {e}")
        return 0

    if not isinstance(old_alphas, list):
        logger.error(f"Unexpected format in {hopeful_file}")
        return 0

    migrated = 0
    for entry in old_alphas:
        expression = entry.get("expression", "")
        ts = entry.get("timestamp", 0)
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else datetime.now().strftime("%Y-%m-%d")

        alpha_data = {
            "id": entry.get("alpha_id", "unknown"),
            "grade": entry.get("grade", "UNKNOWN"),
            "is": {
                "fitness": entry.get("fitness"),
                "sharpe": entry.get("sharpe"),
                "turnover": entry.get("turnover"),
                "returns": entry.get("returns"),
                "checks": entry.get("checks", []),
            },
        }

        save_alpha(expression, alpha_data, source="migration", date_str=date_str)
        migrated += 1

    backup = hopeful_file.replace(".json", f"_migrated_{int(time.time())}.json")
    os.rename(hopeful_file, backup)
    logger.info(f"Migrated {migrated} alphas from {hopeful_file} -> {backup}")
    return migrated
