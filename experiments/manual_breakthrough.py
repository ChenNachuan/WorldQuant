import time
import logging
from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_db import get_alpha_db

def manual_polish():
    gen = AlphaGenerator(region_key="USA")
    gen.initialize()
    db = get_alpha_db()

    # These are the structural variations that SHOULD yield > 1.25 Sharpe
    variations = [
        # Original: rank(open - ts_mean(vwap, 20)) * -abs(returns)
        # Variation 1: Z-score normalization with longer window
        ("rank(ts_zscore(open - ts_mean(vwap, 20), 60)) * rank(-abs(returns))",
         {"neutralization": "SECTOR", "decay": 10, "truncation": 0.05}),
        # Variation 2: Industry relative with decay
        ("rank(group_neutralize(open - ts_mean(vwap, 20), industry)) * rank(-abs(returns))",
         {"neutralization": "INDUSTRY", "decay": 10, "truncation": 0.05}),
        # Variation 3: Volatility weighted
        ("rank(open - ts_mean(vwap, 60)) * rank(-abs(returns) / ts_std_dev(returns, 20))",
         {"neutralization": "SECTOR", "decay": 10, "truncation": 0.05})
    ]

    print(f"Submitting {len(variations)} high-potential manual variations...")

    tasks = []
    for expr, settings in variations:
        tasks.append((expr, settings))

    results = gen.test_alphas_batch(tasks)

    for res in results:
        status = res.get("status")
        expr = res.get("expr")
        if status in ["COMPLETE", "WARNING"]:
            data = res.get("alpha_data", {})
            sharpe = data.get("is", {}).get("sharpe")
            print(f"RESULT: {expr[:40]}... | Sharpe: {sharpe} | ID: {res.get('alpha')}")
            db.save_alpha(expr, data, source="manual_breakthrough")

if __name__ == "__main__":
    manual_polish()

