import time
import sys

sys.path.insert(0, ".")

from core.alpha_generator_ollama import AlphaGenerator
from core.alpha_store import load_all_alphas
from core.log_manager import setup_logger

logger = setup_logger(__name__, "generator")

TARGET = 2
BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 30


def count_submittable() -> int:
    alphas = load_all_alphas()
    count = 0
    for a in alphas:
        checks = a.get("backtest", {}).get("checks", [])
        if not any(c.get("result") == "FAIL" for c in checks):
            count += 1
    return count


def main():
    logger.info("Initializing AlphaGenerator with deepseek-coder-v2:16b...")
    generator = AlphaGenerator(max_concurrent=1)
    generator.model_name = "deepseek-coder-v2:16b"
    generator.initial_model = "deepseek-coder-v2:16b"

    logger.info("Fetching data fields and operators...")
    data_fields = generator.get_data_fields()
    operators = generator.get_operators()

    if not data_fields or not operators:
        logger.error("Failed to fetch data fields or operators. Check credentials and network.")
        return

    batch_num = 0
    while True:
        submittable = count_submittable()
        logger.info(f"Current submittable alphas: {submittable}/{TARGET}")
        if submittable >= TARGET:
            logger.info(f"Target reached! {submittable} submittable alphas found.")
            break

        batch_num += 1
        logger.info(f"=== Batch #{batch_num} ===")

        try:
            ideas = generator.generate_alpha_ideas_with_ollama(data_fields, operators)
            if not ideas:
                logger.warning("No ideas generated, retrying...")
                time.sleep(10)
                continue

            logger.info(f"Generated {len(ideas)} alpha ideas")
            generator.test_alpha_batch(ideas)

            generator.operation_count += 1
            if generator.operation_count % generator.vram_cleanup_interval == 0:
                generator.cleanup_vram()

        except Exception as e:
            logger.error(f"Error in batch #{batch_num}: {e}")
            time.sleep(30)
            continue

        logger.info(f"Sleeping {SLEEP_BETWEEN_BATCHES}s before next batch...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    submittable = count_submittable()
    logger.info(f"Done! Total submittable alphas: {submittable}")


if __name__ == "__main__":
    main()
