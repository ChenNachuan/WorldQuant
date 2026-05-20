"""
Fetch data fields and operators from WorldQuant Brain API.

Saves results to:
- data/fields/{dataset}.csv — Data fields grouped by dataset
- data/operators/operators.csv — All operators

Usage:
    python fetch_fields.py
    python fetch_fields.py --region USA --universe TOP3000
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.data_fetcher import DataFetcher
from core.api_session import get_session
from core.log_manager import setup_logger

logger = setup_logger(__name__, "fetch_fields")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch WQ Brain data fields and operators"
    )
    parser.add_argument(
        "--region",
        default="USA",
        help="Region code (default: USA)"
    )
    parser.add_argument(
        "--universe",
        default="TOP3000",
        help="Universe (default: TOP3000)"
    )
    args = parser.parse_args()

    # Get authenticated session
    logger.info("Authenticating with WorldQuant Brain...")
    session = get_session()

    fetcher = DataFetcher(session=session)

    # Fetch and save fields
    logger.info(f"Fetching data fields for {args.region}/{args.universe}...")
    fields_by_dataset = fetcher.fetch_all_fields(
        region=args.region,
        universe=args.universe
    )
    saved_files = fetcher.save_fields_to_csv(fields_by_dataset)

    print(f"\n=== Data Fields ===")
    print(f"Saved {len(saved_files)} dataset files to data/fields/")
    for dataset, count in fetcher.get_field_summary().items():
        print(f"  {dataset}.csv: {count} fields")

    # Fetch and save operators
    logger.info("Fetching operators...")
    operators = fetcher.fetch_operators()
    csv_path = fetcher.save_operators_to_csv(operators)

    print(f"\n=== Operators ===")
    print(f"Saved {len(operators)} operators to {csv_path}")
    for category, count in fetcher.get_operator_summary().items():
        print(f"  {category}: {count} operators")

    print("\nDone!")


if __name__ == "__main__":
    main()
