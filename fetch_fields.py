"""
Fetch operators from WorldQuant Brain API.

Saves results to:
- data/operators/operators.csv — All operators

Usage:
    python fetch_fields.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.data_fetcher import DataFetcher
from core.api_session import get_session


def main():
    # Get authenticated session
    print("Authenticating with WorldQuant Brain...")
    session = get_session()

    fetcher = DataFetcher(session=session)

    # Fetch and save operators
    print("Fetching operators...")
    operators = fetcher.fetch_operators()
    csv_path = fetcher.save_operators_to_csv(operators)

    print(f"\n=== Operators ===")
    print(f"Saved {len(operators)} operators to {csv_path}")
    for category, count in fetcher.get_operator_summary().items():
        print(f"  {category}: {count} operators")

    print("\nDone!")


if __name__ == "__main__":
    main()
