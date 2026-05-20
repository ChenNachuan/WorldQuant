"""
WorldQuant Brain Data Field and Operator Fetcher

Fetches available data fields and operators from the WQ Brain API.
Saves results to organized directories:
- data/fields/{dataset}.csv — Data fields grouped by dataset
- data/operators/operators.csv — All operators
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Base directories
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FIELDS_DIR = os.path.join(BASE_DATA_DIR, "fields")
OPERATORS_DIR = os.path.join(BASE_DATA_DIR, "operators")


class DataFetcher:
    """Fetches data fields and operators from WorldQuant Brain API."""

    def __init__(self, session=None):
        """
        Initialize with an authenticated session.

        Args:
            session: WorldQuantSession instance (optional, will create if not provided)
        """
        # Create directories
        os.makedirs(FIELDS_DIR, exist_ok=True)
        os.makedirs(OPERATORS_DIR, exist_ok=True)

        # Use passed session or create a fresh one
        if session is not None:
            # Use the session from WorldQuantSession
            self._requests_session = session.session if hasattr(session, 'session') else session
        else:
            # Create fresh session
            import requests
            from requests.auth import HTTPBasicAuth
            from core.config import load_credentials

            username, password = load_credentials()
            self._requests_session = requests.Session()
            self._requests_session.auth = HTTPBasicAuth(username, password)

            # Authenticate
            resp = self._requests_session.post(
                'https://api.worldquantbrain.com/authentication',
                verify=False,
                timeout=15
            )
            if resp.status_code != 201:
                raise Exception(f"Authentication failed: {resp.text}")

    def fetch_all_fields(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        instrument_type: str = "EQUITY",
        delay: int = 1,
        limit: int = 100,
        timeout: int = 60
    ) -> List[Dict]:
        """
        Fetch all available data fields with pagination.

        Args:
            region: Region code (USA, CHN, EUR, ASI, etc.)
            universe: Universe (TOP3000, TOP1000, TOP500, etc.)
            instrument_type: Asset type (EQUITY, etc.)
            delay: Delay in days (1 = point-in-time)
            limit: Max fields to fetch per request
            timeout: Request timeout in seconds

        Returns:
            List of all field dicts
        """
        logger.info(f"Fetching data fields for {region}/{universe}...")

        base_url = "https://api.worldquantbrain.com/data-fields"
        all_fields = []
        offset = 0
        max_retries = 3

        while True:
            # Build params dict for clean URL construction
            params = {
                "instrumentType": instrument_type,
                "region": region,
                "universe": universe,
                "delay": delay,
                "limit": limit,
                "offset": offset
            }

            for attempt in range(max_retries):
                try:
                    logger.debug(f"Fetching offset={offset}...")
                    resp = self._requests_session.get(
                        base_url,
                        params=params,
                        verify=False,
                        timeout=timeout
                    )
                    logger.debug(f"Status: {resp.status_code}")

                    # Handle rate limiting
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 30))
                        logger.warning(f"Rate limited, waiting {retry_after}s...")
                        time.sleep(retry_after)
                        continue

                    if resp.status_code != 200:
                        logger.warning(f"Failed to fetch fields (status {resp.status_code}), retrying...")
                        time.sleep(5)
                        continue

                    data = resp.json()
                    results = data.get("results", [])
                    total_count = data.get("count", 0)

                    if not results:
                        logger.info(f"Finished fetching. Total: {len(all_fields)} fields")
                        return all_fields

                    all_fields.extend(results)
                    logger.info(f"  Fetched {len(all_fields)}/{total_count} fields...")

                    # Check if we've fetched all
                    if len(all_fields) >= total_count:
                        logger.info(f"Finished fetching. Total: {len(all_fields)} fields")
                        return all_fields

                    # Move to next page
                    offset += limit

                    # Polite delay to avoid rate limiting
                    time.sleep(2)
                    break  # Success, exit retry loop

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        logger.error(f"Failed to fetch fields after {max_retries} attempts")
                        return all_fields

        return all_fields

    def save_fields_to_csv(self, fields: List[Dict]) -> List[str]:
        """
        Save fields to CSV files grouped by dataset.

        Returns:
            List of saved file paths
        """
        # Group by dataset
        datasets = {}
        for field in fields:
            dataset_info = field.get("dataset", {})
            dataset_id = dataset_info.get("id", "unknown")
            dataset_name = dataset_info.get("name", dataset_id)

            if dataset_id not in datasets:
                datasets[dataset_id] = {
                    "name": dataset_name,
                    "fields": []
                }

            datasets[dataset_id]["fields"].append({
                "Field": field.get("id", ""),
                "Description": field.get("description", ""),
                "Type": field.get("type", ""),
                "Dataset": dataset_name,
                "Category": field.get("category", {}).get("name", "")
            })

        # Save each dataset to a separate CSV
        saved_files = []
        for dataset_id, dataset_info in datasets.items():
            dataset_fields = dataset_info.get("fields", [])
            dataset_name = dataset_info.get("name", dataset_id)

            if not dataset_fields:
                continue

            # Clean filename
            safe_name = dataset_id.replace("/", "_").replace(" ", "_").lower()
            csv_path = os.path.join(FIELDS_DIR, f"{safe_name}.csv")

            df = pd.DataFrame(dataset_fields)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            saved_files.append(csv_path)
            logger.info(f"Saved {len(dataset_fields)} fields to {csv_path}")

        return saved_files

    def fetch_operators(self) -> List[Dict]:
        """
        Fetch all available operators from WQ Brain API.

        Returns:
            List of operator dicts
        """
        logger.info("Fetching operators...")

        url = "https://api.worldquantbrain.com/operators"

        try:
            resp = self._requests_session.get(url, verify=False, timeout=60)

            if resp.status_code != 200:
                logger.warning(f"Failed to fetch operators: {resp.status_code}")
                return []

            # API returns a list directly
            operators_raw = resp.json()

            if not isinstance(operators_raw, list):
                logger.warning("Unexpected response format")
                return []

            operators = []
            for item in operators_raw:
                operators.append({
                    "Name": item.get("name", ""),
                    "Category": item.get("category", ""),
                    "Scope": ", ".join(item.get("scope", [])),
                    "Definition": item.get("definition", ""),
                    "Description": item.get("description", ""),
                    "Level": item.get("level", ""),
                    "Documentation": item.get("documentation", "")
                })

            logger.info(f"Fetched {len(operators)} operators")
            return operators

        except Exception as e:
            logger.error(f"Error fetching operators: {e}")
            return []

    def save_operators_to_csv(self, operators: List[Dict]) -> str:
        """
        Save operators to CSV file.

        Returns:
            Path to saved file
        """
        if not operators:
            logger.warning("No operators to save")
            return ""

        df = pd.DataFrame(operators)
        csv_path = os.path.join(OPERATORS_DIR, "operators.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved {len(operators)} operators to {csv_path}")

        return csv_path

    def fetch_and_save_all(
        self,
        region: str = "USA",
        universe: str = "TOP3000"
    ) -> Dict[str, List[str]]:
        """
        Fetch and save all data fields and operators.

        Returns:
            Dict with 'fields' and 'operators' file paths
        """
        result = {"fields": [], "operators": ""}

        # Fetch and save fields
        fields = self.fetch_all_fields(region=region, universe=universe)
        result["fields"] = self.save_fields_to_csv(fields)

        # Fetch and save operators
        operators = self.fetch_operators()
        result["operators"] = self.save_operators_to_csv(operators)

        return result

    def get_field_summary(self) -> Dict[str, int]:
        """Get summary of saved fields by file."""
        summary = {}

        if not os.path.exists(FIELDS_DIR):
            return summary

        for file in sorted(os.listdir(FIELDS_DIR)):
            if file.endswith(".csv"):
                category = file.replace(".csv", "")
                try:
                    df = pd.read_csv(os.path.join(FIELDS_DIR, file))
                    summary[category] = len(df)
                except Exception:
                    pass

        return summary

    def get_operator_summary(self) -> Dict[str, int]:
        """Get summary of saved operators by category."""
        csv_path = os.path.join(OPERATORS_DIR, "operators.csv")

        if not os.path.exists(csv_path):
            return {}

        try:
            df = pd.read_csv(csv_path)
            return df['Category'].value_counts().to_dict()
        except Exception:
            return {}


def fetch_and_save_all(
    region: str = "USA",
    universe: str = "TOP3000",
    session=None
) -> Dict[str, List[str]]:
    """
    Convenience function to fetch and save all data.

    Returns:
        Dict with 'fields' and 'operators' file paths
    """
    fetcher = DataFetcher(session=session)
    return fetcher.fetch_and_save_all(region=region, universe=universe)


if __name__ == "__main__":
    # Test the fetcher
    logging.basicConfig(level=logging.INFO)

    from api_session import get_session
    session = get_session()

    fetcher = DataFetcher(session)
    result = fetcher.fetch_and_save_all()

    print("\n=== Summary ===")
    print(f"Fields saved: {len(result['fields'])} files")
    print(f"Operators saved: {result['operators']}")

    print("\nField counts by dataset:")
    for dataset, count in fetcher.get_field_summary().items():
        print(f"  {dataset}.csv: {count} fields")

    print("\nOperator counts by category:")
    for category, count in fetcher.get_operator_summary().items():
        print(f"  {category}: {count}")
