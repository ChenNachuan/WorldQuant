"""
Alpha50 CSV Miner
=================
Read alpha expressions and their specific simulation settings from alpha50.csv,
backtest each one with its original settings plus parameter variations.
"""

import csv
import ast
import time
import json
import os
import sys
import random
import re
from typing import List, Dict, Optional

sys.path.insert(0, ".")

from core.alpha_db import get_alpha_db
from core.log_manager import setup_logger

import requests
from requests.auth import HTTPBasicAuth

logger = setup_logger(__name__, "pipeline")

# Map CSV setting keys to WQ Brain API keys
NEUTRALIZATION_MAP = {
    "None": "NONE",
    "Market": "MARKET",
    "Sector": "SECTOR",
    "Industry": "INDUSTRY",
    "Subindustry": "SUBINDUSTRY",
}

UNIVERSE_MAP = {
    "TOP200": "TOP200",
    "TOP500": "TOP500",
    "TOP1000": "TOP1000",
    "TOP3000": "TOP3000",
}


def parse_settings(settings_str: str) -> Dict:
    """Parse the settings dictionary string from CSV."""
    try:
        settings = ast.literal_eval(settings_str)
        return settings
    except Exception as e:
        logger.warning(f"Failed to parse settings: {e}")
        return {}


def build_simulation_data(formula: str, settings: Dict) -> Dict:
    """Build WQ Brain simulation request data from CSV settings."""
    neutralization = NEUTRALIZATION_MAP.get(
        settings.get("Neutralization", "INDUSTRY"), "INDUSTRY"
    )
    universe = settings.get("Universe", "TOP3000")
    decay = int(settings.get("Decay", 0))
    delay = int(settings.get("Delay", 1))
    truncation = float(settings.get("Truncation", 0.01))
    nan_handling = "ON" if settings.get("NaN_Handling", "Off") == "On" else "OFF"
    pasteurization = "ON" if settings.get("Pasteurization", "On") == "On" else "OFF"
    unit_handling = settings.get("Unit_Handling", "VERIFY").upper()

    return {
        "type": "REGULAR",
        "settings": {
            "instrumentType": "EQUITY",
            "region": settings.get("Region", "USA"),
            "universe": universe,
            "delay": delay,
            "decay": decay,
            "neutralization": neutralization,
            "truncation": truncation,
            "pasteurization": pasteurization,
            "unitHandling": unit_handling,
            "nanHandling": nan_handling,
            "language": "FASTEXPR",
            "visualization": False,
        },
        "regular": formula,
    }


def create_session() -> requests.Session:
    """Create authenticated session."""
    from core.api_session import get_session_manager
    return get_session_manager().session


def submit_simulation(sess: requests.Session, sim_data: Dict) -> Optional[str]:
    """Submit simulation and return progress URL."""
    try:
        resp = sess.post(
            "https://api.worldquantbrain.com/simulations",
            json=sim_data,
            timeout=(30, 120),
        )
        if resp.status_code == 401:
            logger.warning("Auth expired, re-authenticating...")
            from core.config import load_credentials
            username, password = load_credentials()
            sess.auth = HTTPBasicAuth(username, password)
            sess.post("https://api.worldquantbrain.com/authentication", timeout=(15, 30))
            resp = sess.post(
                "https://api.worldquantbrain.com/simulations",
                json=sim_data,
                timeout=(30, 120),
            )

        if resp.status_code != 201:
            logger.warning(f"Submit failed ({resp.status_code}): {resp.text[:200]}")
            return None

        return resp.headers.get("location")
    except Exception as e:
        logger.error(f"Submit error: {e}")
        return None


def monitor_simulation(sess: requests.Session, progress_url: str, timeout: int = 1800) -> Optional[Dict]:
    """Monitor simulation until completion."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = sess.get(progress_url, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                time.sleep(retry_after)
                continue
            if resp.status_code != 200:
                return None

            data = resp.json()
            status = data.get("status")

            if status == "COMPLETE":
                alpha_id = data.get("alpha")
                if alpha_id:
                    alpha_resp = sess.get(
                        f"https://api.worldquantbrain.com/alphas/{alpha_id}",
                        timeout=30,
                    )
                    if alpha_resp.status_code == 200:
                        return alpha_resp.json()
                return None

            if status == "ERROR":
                logger.warning(f"Simulation error: {json.dumps(data)[:300]}")
                return None

            retry_after = resp.headers.get("Retry-After")
            wait = int(float(retry_after)) if retry_after else 5
            time.sleep(wait)

        except Exception as e:
            logger.warning(f"Monitor error: {e}")
            time.sleep(5)

    logger.warning("Simulation timed out")
    return None


def generate_setting_variations(settings: Dict, formula: str) -> List[Dict]:
    """Generate variations by tweaking decay and neutralization."""
    variations = []

    # Original
    variations.append({"settings": settings, "formula": formula})

    # Vary decay
    original_decay = int(settings.get("Decay", 0))
    for decay_delta in [-5, 5, 10, -10]:
        new_decay = max(0, original_decay + decay_delta)
        if new_decay != original_decay:
            new_settings = dict(settings)
            new_settings["Decay"] = str(new_decay)
            variations.append({"settings": new_settings, "formula": formula})

    # Vary ts_mean window if present
    ts_mean_match = re.search(r"ts_mean\(([^,]+),\s*(\d+)\)", formula)
    if ts_mean_match:
        original_window = int(ts_mean_match.group(2))
        for new_window in [5, 10, 20, 30, 60]:
            if new_window != original_window:
                new_formula = formula.replace(
                    f"ts_mean({ts_mean_match.group(1)},{original_window})",
                    f"ts_mean({ts_mean_match.group(1)},{new_window})",
                ).replace(
                    f"ts_mean({ts_mean_match.group(1)}, {original_window})",
                    f"ts_mean({ts_mean_match.group(1)}, {new_window})",
                )
                if new_formula != formula:
                    variations.append({"settings": settings, "formula": new_formula})

    # Vary ts_zscore window if present
    ts_zscore_match = re.search(r"ts_zscore\(([^,]+),\s*(\d+)\)", formula)
    if ts_zscore_match:
        original_window = int(ts_zscore_match.group(2))
        for new_window in [10, 20, 30, 52, 120, 252]:
            if new_window != original_window:
                new_formula = formula.replace(
                    f"ts_zscore({ts_zscore_match.group(1)}, {original_window})",
                    f"ts_zscore({ts_zscore_match.group(1)}, {new_window})",
                ).replace(
                    f"ts_zscore({ts_zscore_match.group(1)},{original_window})",
                    f"ts_zscore({ts_zscore_match.group(1)},{new_window})",
                )
                if new_formula != formula:
                    variations.append({"settings": new_settings, "formula": new_formula})

    return variations


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Alpha50 CSV Miner")
    parser.add_argument("--csv", type=str, default="alpha50.csv", help="Path to CSV file")
    parser.add_argument("--variations", action="store_true", help="Also test parameter variations")
    parser.add_argument("--max-alphas", type=int, default=0, help="Max alphas to test (0=all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Alpha50 CSV Miner")
    logger.info("=" * 60)

    # Read CSV
    rows = []
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("formula"):
                rows.append(row)

    if args.max_alphas > 0:
        rows = rows[: args.max_alphas]

    logger.info(f"Loaded {len(rows)} alphas from {args.csv}")

    # Create session
    sess = create_session()

    successful = []
    failed = []

    for idx, row in enumerate(rows, 1):
        formula = row["formula"].strip().strip('"')
        settings = parse_settings(row.get("settingdict", "{}"))
        original_sharpe = float(row.get("Sharpe", 0))
        original_fitness = float(row.get("Fitness", 0))

        logger.info(f"\n[{idx}/{len(rows)}] Formula: {formula[:80]}")
        logger.info(f"  Original: sharpe={original_sharpe}, fitness={original_fitness}")
        logger.info(f"  Settings: {settings.get('Universe')}, decay={settings.get('Decay')}, "
                    f"neut={settings.get('Neutralization')}")

        # Generate variations
        if args.variations:
            test_cases = generate_setting_variations(settings, formula)
        else:
            test_cases = [{"settings": settings, "formula": formula}]

        # Track if original formula has field-level errors (skip variations if so)
        skip_variations = False

        for vi, test_case in enumerate(test_cases):
            if vi > 0 and skip_variations:
                logger.info(f"  Skipping remaining variations (field-level error)")
                break

            tc_formula = test_case["formula"]
            tc_settings = test_case["settings"]

            label = "ORIGINAL" if vi == 0 else f"VAR#{vi}"
            if vi > 0:
                logger.info(f"  [{label}] {tc_formula[:60]}... decay={tc_settings.get('Decay')}")

            sim_data = build_simulation_data(tc_formula, tc_settings)
            progress_url = submit_simulation(sess, sim_data)

            if not progress_url:
                failed.append((tc_formula, tc_settings, None, None, label))
                if vi == 0:
                    skip_variations = True
                continue

            alpha_data = monitor_simulation(sess, progress_url)

            if alpha_data:
                is_data = alpha_data.get("is", {})
                fitness = is_data.get("fitness")
                sharpe = is_data.get("sharpe")
                turnover = is_data.get("turnover")

                logger.info(f"  [{label}] Result: fitness={fitness}, sharpe={sharpe}, turnover={turnover}")

                db = get_alpha_db()
                db.save_alpha(tc_formula, alpha_data, source="alpha50_csv")

                success = (
                    fitness is not None and fitness >= 1.0
                    and sharpe is not None and sharpe >= 1.25
                )

                if success:
                    successful.append((tc_formula, tc_settings, fitness, sharpe, label))
                    logger.info(f"  ✅ [{label}] SUCCESS!")
                    alpha_id = alpha_data.get("id")
                    if alpha_id:
                        try:
                            sess.patch(
                                f"https://api.worldquantbrain.com/alphas/{alpha_id}",
                                json={"color": "green"},
                                timeout=15,
                            )
                        except Exception:
                            pass
                else:
                    failed.append((tc_formula, tc_settings, fitness, sharpe, label))

                # Alpha Flipper
                if not success and sharpe is not None and sharpe <= -1.0:
                    flipped = f"-1 * ({tc_formula})"
                    logger.info(f"  💡 Flipper: sharpe={sharpe:.2f}, testing inverted...")
                    flip_sim_data = build_simulation_data(flipped, tc_settings)
                    flip_url = submit_simulation(sess, flip_sim_data)
                    if flip_url:
                        flip_data = monitor_simulation(sess, flip_url)
                        if flip_data:
                            flip_is = flip_data.get("is", {})
                            flip_fitness = flip_is.get("fitness")
                            flip_sharpe = flip_is.get("sharpe")
                            logger.info(f"  💡 Flipped result: fitness={flip_fitness}, sharpe={flip_sharpe}")
                            db = get_alpha_db()
                            db.save_alpha(flipped, flip_data, source="alpha50_flipped")
                            if flip_fitness and flip_fitness >= 1.0 and flip_sharpe and flip_sharpe >= 1.25:
                                successful.append((flipped, tc_settings, flip_fitness, flip_sharpe, "FLIPPED"))
                                logger.info(f"  ✅ FLIPPED SUCCESS!")
            else:
                failed.append((tc_formula, tc_settings, None, None, label))
                logger.info(f"  ❌ [{label}] Simulation failed")
                # If original fails (likely unknown field), skip variations
                if vi == 0:
                    skip_variations = True

            time.sleep(2)

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("FINAL REPORT - Alpha50 CSV Mining")
    logger.info(f"{'='*60}")
    logger.info(f"Total tested: {len(successful) + len(failed)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        logger.info("\n🏆 Successful Alphas:")
        sorted_alphas = sorted(successful, key=lambda x: x[2] or 0, reverse=True)
        for i, (expr, settings, fitness, sharpe, label) in enumerate(sorted_alphas[:20], 1):
            logger.info(f"  {i}. [{label}] fitness={fitness:.4f} sharpe={sharpe} | {expr[:60]}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "successful": [
            {"expr": e, "fitness": f, "sharpe": s, "label": l,
             "universe": st.get("Universe"), "decay": st.get("Decay")}
            for e, st, f, s, l in successful
        ],
        "failed_count": len(failed),
        "total_tested": len(successful) + len(failed),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_file = os.path.join("results", f"alpha50_results_{int(time.time())}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

