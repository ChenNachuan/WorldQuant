"""
101 Formulaic Alphas Miner
=========================
Extract alphas from the Kakushadze paper, translate to WQ Brain FASTEXPR syntax,
and backtest with parameter variations.
"""

import re
import time
import json
import os
import sys
import random
import itertools
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, ".")

from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_db import get_alpha_db
from core.region_config import get_region_config
from core.log_manager import setup_logger

logger = setup_logger(__name__, "pipeline")

# ---------------------------------------------------------------------------
# The 101 Formulaic Alphas translated to WQ Brain FASTEXPR syntax
# ---------------------------------------------------------------------------
# Key translations:
#   delay(x, d)           -> ts_delay(x, d)
#   delta(x, d)           -> ts_delta(x, d)
#   correlation(x, y, d)  -> ts_corr(x, y, d)
#   covariance(x, y, d)   -> ts_covariance(x, y, d)
#   stddev(x, d)          -> ts_std_dev(x, d)
#   ts_rank(x, d)         -> ts_rank(x, d)       (same)
#   Ts_ArgMax(x, d)       -> ts_arg_max(x, d)
#   Ts_ArgMin(x, d)       -> ts_arg_min(x, d)
#   ts_min(x, d)          -> ts_min(x, d)         (same)
#   ts_max(x, d)          -> ts_max(x, d)         (same)
#   sum(x, d)             -> ts_sum(x, d)
#   product(x, d)         -> ts_product(x, d)
#   decay_linear(x, d)    -> ts_decay_linear(x, d)
#   SignedPower(x, a)     -> signed_power(x, a)
#   sign(x)               -> sign(x)
#   scale(x)              -> scale(x)
#   rank(x)               -> rank(x)
#   log(x)                -> log(x)
#   abs(x)                -> abs(x)
#   IndNeutralize(x, IndClass.sector)      -> group_neutralize(x, sector)
#   IndNeutralize(x, IndClass.industry)    -> group_neutralize(x, industry)
#   IndNeutralize(x, IndClass.subindustry) -> group_neutralize(x, subindustry)
#   adv{N}                -> adv{N}  (already supported)
#   cap                   -> market_cap
#   returns               -> returns
#   open, close, high, low, volume, vwap -> same
# ---------------------------------------------------------------------------

# We focus on alphas that are most likely to translate cleanly to FASTEXPR.
# Some alphas use ternary operators (? :) which map to WQ Brain's
# "less(x, y) * a + (1 - less(x, y)) * b" pattern, but many are simpler.

ALPHA_101_EXPRESSIONS = [
    # --- Simple, clean alphas (high probability of working) ---
    # Alpha#3
    "-1 * ts_corr(rank(open), rank(volume), 10)",
    # Alpha#4
    "-1 * ts_rank(rank(low), 9)",
    # Alpha#5
    "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(rank(close - vwap)))",
    # Alpha#6
    "-1 * ts_corr(open, volume, 10)",
    # Alpha#12
    "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))",
    # Alpha#13
    "-1 * rank(ts_covariance(rank(close), rank(volume), 5))",
    # Alpha#14
    "-1 * rank(ts_delta(returns, 3)) * ts_corr(open, volume, 10)",
    # Alpha#15
    "-1 * ts_sum(rank(ts_corr(rank(high), rank(volume), 3)), 3)",
    # Alpha#16
    "-1 * rank(ts_covariance(rank(high), rank(volume), 5))",
    # Alpha#22
    "-1 * ts_delta(ts_corr(high, volume, 5), 5) * rank(ts_std_dev(close, 20))",
    # Alpha#25
    "rank(-1 * returns * adv20 * vwap * (high - close))",
    # Alpha#26
    "-1 * ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)",
    # Alpha#28
    "scale(ts_corr(adv20, low, 5) + (high + low) / 2 - close)",
    # Alpha#33
    "rank(-1 * (1 - open / close))",
    # Alpha#34
    "rank(1 - rank(ts_std_dev(returns, 2) / ts_std_dev(returns, 5)) + 1 - rank(ts_delta(close, 1)))",
    # Alpha#37
    "rank(ts_corr(ts_delay(open - close, 1), close, 200)) + rank(open - close)",
    # Alpha#38
    "-1 * rank(ts_rank(close, 10)) * rank(close / open)",
    # Alpha#40
    "-1 * rank(ts_std_dev(high, 10)) * ts_corr(high, volume, 10)",
    # Alpha#41
    "power(high * low, 0.5) - vwap",
    # Alpha#42
    "rank(vwap - close) / rank(vwap + close)",
    # Alpha#43
    "ts_rank(volume / adv20, 20) * ts_rank(-1 * ts_delta(close, 7), 8)",
    # Alpha#44
    "-1 * ts_corr(high, rank(volume), 5)",
    # Alpha#50
    "-1 * ts_max(rank(ts_corr(rank(volume), rank(vwap), 5)), 5)",
    # Alpha#53
    "-1 * ts_delta(((close - low) - (high - close)) / (close - low), 9)",
    # Alpha#54
    "-1 * (low - close) * power(open, 5) / ((low - high) * power(close, 5))",
    # Alpha#55
    "-1 * ts_corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)",
    # Alpha#2
    "-1 * ts_corr(rank(ts_delta(log(volume), 2)), rank((close - open) / open), 6)",
    # Alpha#84
    "signed_power(ts_rank(vwap - ts_max(vwap, 15), 21), ts_delta(close, 5))",

    # --- Alphas with group_neutralize (indneutralize) ---
    # Alpha#48 (simplified)
    "group_neutralize(ts_corr(ts_delta(close, 1), ts_delta(ts_delay(close, 1), 1), 250) * ts_delta(close, 1) / close, subindustry)",
    # Alpha#58 (simplified)
    "-1 * ts_rank(ts_decay_linear(ts_corr(group_neutralize(vwap, sector), volume, 4), 8), 6)",
    # Alpha#59 (simplified)
    "-1 * ts_rank(ts_decay_linear(ts_corr(group_neutralize(vwap, industry), volume, 4), 16), 8)",
    # Alpha#63 (simplified)
    "-1 * rank(ts_decay_linear(ts_delta(group_neutralize(close, industry), 2), 8))",

    # --- Alphas with ts_decay_linear ---
    # Alpha#57 (simplified)
    "-1 * (close - vwap) / ts_decay_linear(rank(ts_arg_max(close, 30)), 2)",
    # Alpha#72 (simplified)
    "rank(ts_decay_linear(ts_corr((high + low) / 2, adv40, 9), 10)) / rank(ts_decay_linear(ts_corr(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))",
    # Alpha#77
    "min(rank(ts_decay_linear((high + low) / 2 + high - vwap - high, 20)), rank(ts_decay_linear(ts_corr((high + low) / 2, adv40, 3), 6)))",

    # --- Alphas with rank/correlation combinations ---
    # Alpha#75
    "less(rank(ts_corr(vwap, volume, 4)), rank(ts_corr(rank(low), rank(adv50), 12)))",
    # Alpha#83 (simplified)
    "rank(ts_delay((high - low) / (ts_sum(close, 5) / 5), 2)) * rank(rank(volume)) / ((high - low) / (ts_sum(close, 5) / 5) / (vwap - close))",

    # --- Extra simplified versions for diversity ---
    "rank(ts_corr(close, volume, 10))",
    "rank(ts_delta(close, 5)) * rank(ts_corr(volume, close, 20))",
    "-1 * rank(ts_std_dev(returns, 20))",
    "ts_rank(ts_corr(vwap, volume, 20), 10)",
    "group_neutralize(rank(ts_delta(close, 5)), sector)",
    "group_neutralize(ts_corr(close, volume, 10), industry)",
    "-1 * ts_rank(ts_corr(rank(close), rank(volume), 5), 10)",
    "rank(ts_decay_linear(ts_delta(vwap, 5), 10))",
    "ts_rank(volume / adv20, 10) * ts_rank(ts_delta(returns, 5), 20)",
    "group_neutralize(rank(ts_av_diff(close, 20)), sector)",
    "rank(divide(ts_delta(close, 20), ts_std_dev(close, 20)))",
    "group_zscore(ts_corr(volume, returns, 20), industry)",
    "-1 * ts_corr(rank(open), rank(adv20), 10)",
    "group_neutralize(zscore(ts_mean(returns, 60)), sector)",
    "rank(ts_corr(ts_rank(close, 10), ts_rank(volume, 10), 10))",
]

# Parameter variation windows
WINDOWS = [5, 10, 15, 20, 30, 60, 120, 252]


def generate_variations(expr: str, max_variations: int = 5) -> List[str]:
    """Generate parameter variations of an expression by swapping numeric windows."""
    variations = [expr]  # always include original

    # Find all integer parameters that look like time windows
    # Match numbers that are arguments (after comma or opening paren)
    pattern = r',\s*(\d+)\s*\)'
    matches = list(re.finditer(pattern, expr))

    if not matches:
        return variations

    for _ in range(max_variations - 1):
        new_expr = expr
        # For each match, randomly decide whether to vary it
        for match in reversed(matches):
            original_val = int(match.group(1))
            # Pick a nearby window value
            candidates = [w for w in WINDOWS if w != original_val]
            if candidates:
                new_val = random.choice(candidates)
                # Replace just the number within the matched ", N)" pattern
                start = match.start(1)
                end = match.end(1)
                new_expr = new_expr[:start] + str(new_val) + new_expr[end:]

        if new_expr != expr and new_expr not in variations:
            variations.append(new_expr)

    return variations


def monitor_simulation(sess, progress_url: str, timeout: int = 1800) -> Optional[Dict]:
    """Monitor a simulation until completion."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = sess.get(progress_url, timeout=30)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                logger.info(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if resp.status_code != 200:
                logger.warning(f"Unexpected status {resp.status_code}: {resp.text[:100]}")
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
                error_msg = data.get("error", data.get("message", "unknown"))
                logger.warning(f"Simulation error: {error_msg}")
                return None

            retry_after = resp.headers.get("Retry-After")
            wait = int(float(retry_after)) if retry_after else 5
            time.sleep(wait)

        except Exception as e:
            logger.warning(f"Monitor error: {e}")
            time.sleep(5)

    logger.warning("Simulation monitoring timed out")
    return None


def submit_and_test(generator: AlphaGenerator, expr: str) -> Optional[Dict]:
    """Submit an expression for simulation and return alpha data if successful."""
    result = generator._test_alpha_impl(expr)

    if result.get("status") == "error":
        error_msg = result.get("message", "")
        logger.warning(f"Submit error: {error_msg[:100]}")
        return None

    progress_url = result.get("result", {}).get("progress_url")
    if not progress_url:
        logger.warning("No progress URL received")
        return None

    return monitor_simulation(generator.sess, progress_url)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="101 Formulaic Alphas Miner")
    parser.add_argument("--region", type=str, default="USA", help="Region key")
    parser.add_argument("--variations", type=int, default=3,
                        help="Number of parameter variations per alpha (default: 3)")
    parser.add_argument("--max-alphas", type=int, default=0,
                        help="Max alphas to test (0 = all)")
    parser.add_argument("--concurrent", type=int, default=1,
                        help="Max concurrent simulations")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("101 Formulaic Alphas Miner")
    logger.info("=" * 60)

    # Initialize generator (handles auth, session, etc.)
    generator = AlphaGenerator(max_concurrent=args.concurrent, region_key=args.region)
    generator.model_name = "qwen3-coder:latest"

    alphas_to_test = ALPHA_101_EXPRESSIONS
    if args.max_alphas > 0:
        alphas_to_test = alphas_to_test[:args.max_alphas]

    logger.info(f"Base alphas: {len(alphas_to_test)}")
    logger.info(f"Variations per alpha: {args.variations}")
    logger.info(f"Region: {args.region}")

    # Generate all expressions (base + variations)
    all_expressions = []
    for expr in alphas_to_test:
        variations = generate_variations(expr, max_variations=args.variations)
        all_expressions.extend(variations)

    # Deduplicate
    all_expressions = list(dict.fromkeys(all_expressions))
    logger.info(f"Total expressions to test (after dedup): {len(all_expressions)}")

    # Test each expression
    successful = []
    failed = []
    flipped_queue = []

    for i, expr in enumerate(all_expressions, 1):
        logger.info(f"[{i}/{len(all_expressions)}] Testing: {expr[:80]}...")

        try:
            alpha_data = submit_and_test(generator, expr)

            if alpha_data:
                is_data = alpha_data.get("is", {})
                fitness = is_data.get("fitness")
                sharpe = is_data.get("sharpe")
                turnover = is_data.get("turnover")

                logger.info(f"  Result: fitness={fitness}, sharpe={sharpe}, turnover={turnover}")

                # Save to SQLite
                db = get_alpha_db()
                db.save_alpha(expr, alpha_data, source="alpha101")

                success = (
                    fitness is not None and fitness >= 1.0
                    and sharpe is not None and sharpe >= 1.25
                )

                if success:
                    successful.append((expr, fitness, sharpe))
                    logger.info(f"  ✅ SUCCESS! fitness={fitness:.4f} sharpe={sharpe}")
                    # Mark green
                    alpha_id = alpha_data.get("id")
                    if alpha_id:
                        try:
                            generator.sess.patch(
                                f"https://api.worldquantbrain.com/alphas/{alpha_id}",
                                json={"color": "green"}, timeout=15
                            )
                        except Exception:
                            pass
                else:
                    failed.append((expr, fitness, sharpe))
                    logger.info(f"  ❌ Below threshold")

                    # Alpha Flipper: if sharpe is very negative, try flipping
                    if sharpe is not None and sharpe <= -1.0:
                        flipped = f"-1 * ({expr})"
                        if flipped not in all_expressions and flipped not in [e for e, _, _ in flipped_queue]:
                            flipped_queue.append((flipped, None, None))
                            logger.info(f"  💡 Queued flipped version for later testing")
            else:
                failed.append((expr, None, None))
                logger.info(f"  ❌ Simulation failed or timed out")

        except Exception as e:
            logger.error(f"  Error: {e}")
            failed.append((expr, None, None))

        time.sleep(2)  # Small delay between submissions

    # Test flipped alphas
    if flipped_queue:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {len(flipped_queue)} flipped alphas...")
        for i, (expr, _, _) in enumerate(flipped_queue, 1):
            logger.info(f"[Flip {i}/{len(flipped_queue)}] Testing: {expr[:80]}...")
            try:
                alpha_data = submit_and_test(generator, expr)
                if alpha_data:
                    is_data = alpha_data.get("is", {})
                    fitness = is_data.get("fitness")
                    sharpe = is_data.get("sharpe")
                    db = get_alpha_db()
                    db.save_alpha(expr, alpha_data, source="alpha101_flipped")
                    logger.info(f"  Flipped result: fitness={fitness}, sharpe={sharpe}")
                    if fitness is not None and fitness >= 1.0 and sharpe is not None and sharpe >= 1.25:
                        successful.append((expr, fitness, sharpe))
                        logger.info(f"  ✅ FLIPPED SUCCESS!")
                    else:
                        failed.append((expr, fitness, sharpe))
            except Exception as e:
                logger.error(f"  Error: {e}")
            time.sleep(2)

    # Final report
    logger.info(f"\n{'='*60}")
    logger.info("FINAL REPORT - 101 Formulaic Alphas Mining")
    logger.info(f"{'='*60}")
    logger.info(f"Total tested: {len(all_expressions) + len(flipped_queue)}")
    logger.info(f"Successful (fitness>=1.0 & sharpe>=1.25): {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        logger.info("\n🏆 Successful Alphas:")
        sorted_alphas = sorted(successful, key=lambda x: x[1] or 0, reverse=True)
        for i, (expr, fitness, sharpe) in enumerate(sorted_alphas, 1):
            logger.info(f"  {i}. fitness={fitness:.4f} sharpe={sharpe} | {expr[:80]}")

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "successful": [{"expr": e, "fitness": f, "sharpe": s} for e, f, s in successful],
        "failed": [{"expr": e, "fitness": f, "sharpe": s} for e, f, s in failed],
        "total_tested": len(all_expressions) + len(flipped_queue),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_file = os.path.join("results", f"alpha101_results_{int(time.time())}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

