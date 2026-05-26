"""
Fetch operators from WorldQuant Brain API.

Saves results to:
- data/fields_delay{delay}/{dataset}.csv — Data fields grouped by dataset
- data/operators/operators.csv — All operators

Usage:
    python fetch_fields.py              # Fetch delay=1 fields (default)
    python fetch_fields.py --delay 0    # Fetch delay=0 fields
    python fetch_fields.py --delay 1    # Fetch delay=1 fields
"""

import os
import sys
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from dotenv import load_dotenv
load_dotenv()

from core.data_fetcher import DataFetcher
from core.api_session import get_session


def main():
    parser = argparse.ArgumentParser(description="Fetch data fields from WorldQuant Brain")
    parser.add_argument(
        "--delay",
        type=int,
        default=1,
        choices=[0, 1],
        help="Delay type: 0 or 1 (default: 1)"
    )
    args = parser.parse_args()

    # Get authenticated session
    print("Authenticating with WorldQuant Brain...")
    session = get_session()

    fetcher = DataFetcher(session=session)

    # Fetch and save fields
    print(f"Fetching delay={args.delay} fields...")
    fields = fetcher.fetch_all_fields(delay=args.delay)
    field_files = fetcher.save_fields_to_csv(fields, delay=args.delay)

    print(f"\n=== Fields (delay={args.delay}) ===")
    print(f"Saved {len(field_files)} field files")
    for file in field_files:
        print(f"  {file}")

    # Fetch and save operators
    print("\nFetching operators...")
    operators = fetcher.fetch_operators()
    csv_path = fetcher.save_operators_to_csv(operators)

    print(f"\n=== Operators ===")
    print(f"Saved {len(operators)} operators to {csv_path}")
    for category, count in fetcher.get_operator_summary().items():
        print(f"  {category}: {count} operators")

    print("\nDone!")


if __name__ == "__main__":
    main()
