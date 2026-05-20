"""
Add alpha to database by ID.

Usage:
    python add_alpha.py <alpha_id>
    python add_alpha.py pwnbR9Gq akNmojM1
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import requests
from requests.auth import HTTPBasicAuth
from core.config import load_credentials
from core.alpha_db import get_alpha_db


def fetch_alpha_data(alpha_id: str) -> dict:
    """Fetch alpha data from WQ Brain API."""
    username, password = load_credentials()
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    session.post('https://api.worldquantbrain.com/authentication', verify=False, timeout=15)

    resp = session.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}', verify=False, timeout=30)
    if resp.status_code != 200:
        print(f"Error fetching alpha {alpha_id}: {resp.status_code}")
        return None

    data = resp.json()
    is_data = data.get('is', {})
    settings = data.get('settings', {})
    expression_data = data.get('regular', {})
    expression = expression_data.get('code', '') if isinstance(expression_data, dict) else str(expression_data)

    return {
        "expression": expression,
        "alpha_id": data.get("id", alpha_id),
        "sharpe": is_data.get("sharpe", 0),
        "fitness": is_data.get("fitness", 0),
        "turnover": is_data.get("turnover", 0),
        "margin": is_data.get("margin", 0),
        "returns": is_data.get("returns", 0),
        "pnl": is_data.get("pnl", 0),
        "long_count": is_data.get("longCount", 0),
        "short_count": is_data.get("shortCount", 0),
        "drawdown": is_data.get("drawdown", 0),
        "grade": data.get("grade", ""),
        "checks": is_data.get("checks", []),
        "region": settings.get("region", "USA"),
        "universe": settings.get("universe", "TOP3000"),
        "delay": settings.get("delay", 1),
        "decay": settings.get("decay", 0),
        "neutralization": settings.get("neutralization", "NONE"),
        "truncation": settings.get("truncation", 0.08),
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        sys.exit(0)

    db = get_alpha_db()
    alpha_ids = sys.argv[1:]

    for alpha_id in alpha_ids:
        print(f"\nFetching alpha {alpha_id}...")
        alpha_data = fetch_alpha_data(alpha_id)

        if alpha_data:
            print(f"  Expression: {alpha_data['expression'][:60]}...")
            print(f"  Sharpe: {alpha_data['sharpe']:.2f}")
            print(f"  Fitness: {alpha_data['fitness']:.2f}")

            db.add_alpha(**alpha_data, source="manual", status="submitted")
            print(f"  ✓ Added to database")
        else:
            print(f"  ✗ Failed to fetch alpha {alpha_id}")

    # Summary
    total = db.count_alphas()
    print(f"\nTotal alphas in database: {total}")


if __name__ == "__main__":
    main()
