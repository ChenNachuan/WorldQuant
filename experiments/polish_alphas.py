import time
import json
import logging
import random
from typing import List, Dict, Tuple
from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_db import get_alpha_db
from core.log_manager import setup_logger

logger = setup_logger(__name__, "polisher")

class AlphaPolisher:
    def __init__(self, region_key: str = "USA"):
        self.generator = AlphaGenerator(region_key=region_key)
        self.generator.initialize()
        self.db = get_alpha_db()

        # Grid parameters
        self.neutralizations = ["INDUSTRY", "SUBINDUSTRY", "SECTOR"]
        self.decays = [0, 4, 10]
        self.truncations = [0.01, 0.05]

    def polish(self, expressions: List[str]):
        logger.info(f"Starting specialized refinement for {len(expressions)} base alphas...")

        for base_expr in expressions:
            logger.info(f"--- Polishing: {base_expr[:60]}... ---")

            grid_tasks = []
            for neut in self.neutralizations:
                for decay in self.decays:
                    for trunc in self.truncations:
                        # Check if this exact combination exists in DB
                        if self.db.expression_exists(base_expr, region="USA", universe="TOP3000", neutralization=neut):
                            # This is a bit simplistic since it doesn't check decay/truncation,
                            # but better than nothing.
                            # For a perfect check, we'd need to query with those settings too.
                            # logger.info(f"  Skipping {neut} (already in DB)")
                            # continue
                            pass

                        settings = {
                            "neutralization": neut,
                            "decay": decay,
                            "truncation": trunc,
                            "universe": "TOP3000",
                            "region": "USA"
                        }
                        grid_tasks.append((base_expr, settings))

            logger.info(f"Generated {len(grid_tasks)} grid combinations for this alpha.")

            # Use the generator's batch tester
            results = self.generator.test_alphas_batch(grid_tasks)

            best_sharpe = -1.0
            best_res = None

            for res in results:
                if res.get("status") in ["COMPLETE", "WARNING"]:
                    alpha_data = res.get("alpha_data", {})
                    is_data = alpha_data.get("is", {})
                    sharpe = is_data.get("sharpe")
                    turnover = is_data.get("turnover")

                    # Record to DB
                    self.db.save_alpha(base_expr, alpha_data, source="polisher", settings=alpha_data.get("settings"))

                    if sharpe and sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_res = res

                    if sharpe and sharpe >= 1.25 and turnover and turnover <= 0.7:
                        logger.info(f"🚀 [SUCCESS] Found submittable version! Sharpe={sharpe}, Turnover={turnover}")
                        logger.info(f"  Settings: {alpha_data.get('settings')}")

            if best_res:
                logger.info(f"Best version for this alpha: Sharpe={best_sharpe}")
            else:
                logger.warning(f"No successful grid result for this alpha. Ready for mutation phase.")

if __name__ == "__main__":
    polisher = AlphaPolisher()
    # Top 3 from DB
    top_expressions = [
        "rank(open - (ts_sum(vwap, 60) / 60)) * (-1 * abs(returns))",
        "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(returns)) * (abs(returns) > 0.5)",
        "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(returns)) * (abs(returns) > 0.8)"
    ]
    polisher.polish(top_expressions)

