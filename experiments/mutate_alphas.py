import time
import logging
import random
from typing import List, Dict
from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_db import get_alpha_db
from core.log_manager import setup_logger

logger = setup_logger(__name__, "mutator")

class AlphaMutator:
    def __init__(self, region_key: str = "USA"):
        self.generator = AlphaGenerator(region_key=region_key)
        self.generator.initialize()
        self.db = get_alpha_db()

    def mutate_and_test(self, seeds: List[str], iterations: int = 3):
        logger.info(f"Starting Genetic Mutation Phase for {len(seeds)} seeds...")

        for i in range(iterations):
            logger.info(f"--- Mutation Iteration {i+1}/{iterations} ---")

            mutation_prompt = f"""
            Generate 10 WorldQuant alpha expressions based on these seeds:
            {seeds}
            
            Mutation rules:
            1. Use 'ts_mean' or 'ts_rank' instead of 'ts_sum'.
            2. Add 'rank()' around components.
            3. Use 'returns' and 'volume' as common fields.
            
            Output only the expressions, one per line.
            """

            # Using the new specialized generation method
            raw_ideas = self.generator.generate_with_custom_prompt(prompt=mutation_prompt)
            logger.info(f"Raw ideas from AI: {raw_ideas}")

            valid_ideas = self.generator.clean_alpha_ideas(raw_ideas)
            logger.info(f"Valid ideas after cleaning: {valid_ideas}")

            if not valid_ideas:
                logger.warning("AI failed to generate valid mutations in this iteration. Check AST logs.")
                continue

            logger.info(f"Testing {len(valid_ideas)} mutated candidates...")

            # Pack tasks with the best-performing settings from grid search
            tasks = []
            for expr in valid_ideas:
                tasks.append((expr, {
                    "neutralization": "SECTOR",
                    "decay": 10,
                    "truncation": 0.05,
                    "universe": "TOP3000",
                    "region": "USA"
                }))

            results = self.generator.test_alphas_batch(tasks)

            for res in results:
                if res.get("status") in ["COMPLETE", "WARNING"]:
                    data = res.get("alpha_data", {})
                    sharpe = data.get("is", {}).get("sharpe", 0)
                    if sharpe >= 1.25:
                        logger.info(f"🎯 [GENETIC BREAKTHROUGH] Submittable mutated alpha found: {res['expr']}")
                        self.db.save_alpha(res['expr'], data, source="mutator")

if __name__ == "__main__":
    mutator = AlphaMutator()
    seeds = [
        "rank(open - (ts_sum(vwap, 60) / 60)) * (-1 * abs(returns))",
        "group_neutralize(rank(ts_mean(mdl77_2adverint * volume, 5)), industry)",
        "-1 * rank(ts_covariance(rank(high), rank(volume), 60))",
        "rank(open - (ts_sum(vwap, 10) / 10)) * (-1 * abs(returns)) * (abs(returns) > 0.5)"
    ]
    mutator.mutate_and_test(seeds, iterations=5)

