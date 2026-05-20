"""
Submit alpha to WorldQuant Brain by ID.

Usage:
    python submit_alpha.py <alpha_id>
    python submit_alpha.py pwnbR9Gq akNmojM1
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


def submit_alpha(session: requests.Session, alpha_id: str) -> dict:
    """Submit alpha to WorldQuant Brain."""
    resp = session.post(
        f'https://api.worldquantbrain.com/alphas/{alpha_id}/submit',
        json={"type": "REGULAR"},
        verify=False,
        timeout=15
    )

    if resp.status_code == 201:
        return {"success": True}
    else:
        # Parse error response to get detailed check failures
        try:
            error_data = resp.json()
            checks = error_data.get("is", {}).get("checks", [])
            failed_checks = [c for c in checks if c.get("result") == "FAIL"]

            if failed_checks:
                details = []
                for check in failed_checks:
                    name = check.get("name", "UNKNOWN")
                    value = check.get("value", "N/A")
                    limit = check.get("limit", "N/A")
                    details.append(f"{name}: value={value}, limit={limit}")
                return {"success": False, "error": "Checks failed", "details": details}
            else:
                error_msg = resp.text[:200] if resp.text else "Unknown error"
                return {"success": False, "error": error_msg}
        except:
            error_msg = resp.text[:200] if resp.text else "Unknown error"
            return {"success": False, "error": error_msg}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        sys.exit(0)

    # Authenticate
    username, password = load_credentials()
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    resp = session.post('https://api.worldquantbrain.com/authentication', verify=False, timeout=15)

    if resp.status_code != 201:
        print(f"Authentication failed: {resp.text}")
        sys.exit(1)

    print("Authentication successful\n")

    db = get_alpha_db()
    alpha_ids = sys.argv[1:]

    for alpha_id in alpha_ids:
        print(f"Submitting alpha {alpha_id}...")
        result = submit_alpha(session, alpha_id)

        if result["success"]:
            print(f"  ✓ Submitted successfully")
            # Update status in database
            rows = db.update_alpha_status(alpha_id, "submitted")
            if rows > 0:
                print(f"  ✓ Database status updated")
            else:
                print(f"  ⚠ Alpha not found in database")
        else:
            print(f"  ✗ Submission failed: {result['error']}")
            # Show detailed check failures
            if "details" in result:
                for detail in result["details"]:
                    print(f"    - {detail}")
            # Delete alpha from database if submission failed
            rows = db.delete_alpha_by_alpha_id(alpha_id)
            if rows > 0:
                print(f"  ✓ Alpha deleted from database")

    # Summary
    print(f"\nDone. Total alphas in database: {db.count_alphas()}")


if __name__ == "__main__":
    main()
