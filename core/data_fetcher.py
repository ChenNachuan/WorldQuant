"""
WorldQuant Brain Data Field and Operator Fetcher

Fetches available data fields and operators from the WQ Brain API.
Saves results to organized directories:
- data/fields/{dataset}.csv — Data fields grouped by dataset
- data/operators/operators.csv — All operators

Fields are cached locally to avoid repeated API calls (429 rate limit).
Re-run this script manually when you want to refresh the field list.
"""

import os
import re
import json
import logging
import time
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Base directories
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FIELDS_DIR = os.path.join(BASE_DATA_DIR, "fields")
OPERATORS_DIR = os.path.join(BASE_DATA_DIR, "operators")

# Region -> Universe mapping
REGION_UNIVERSE_MAP: Dict[str, List[str]] = {
    "USA": ["TOP3000", "TOP1000", "TOP500"],
    "GLB": ["TOP3000"],
    "EUR": ["TOP2500", "TOP1200"],
    "ASI": ["MINVOL1M"],
    "CHN": ["TOP2000U"],
}


class DataFetcher:
    """Fetches data fields and operators from WorldQuant Brain API."""

    def __init__(self, session=None):
        os.makedirs(FIELDS_DIR, exist_ok=True)
        os.makedirs(OPERATORS_DIR, exist_ok=True)

        if session is not None:
            self._session = session.session if hasattr(session, "session") else session
        else:
            import requests
            from requests.auth import HTTPBasicAuth
            from core.config import load_credentials

            username, password = load_credentials()
            self._session = requests.Session()
            self._session.auth = HTTPBasicAuth(username, password)
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })

            resp = self._session.post(
                "https://api.worldquantbrain.com/authentication",
                verify=False,
                timeout=15,
            )
            if resp.status_code != 201:
                raise Exception(f"Authentication failed: {resp.text}")

    # ── Dataset discovery ────────────────────────────────────────────

    def fetch_datasets(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit: int = 50,
    ) -> List[Dict]:
        """Fetch all dataset IDs for a given region/universe/delay."""
        url = "https://api.worldquantbrain.com/data-sets"
        all_datasets = []
        offset = 0

        while True:
            params = {
                "delay": delay,
                "instrumentType": "EQUITY",
                "limit": limit,
                "offset": offset,
                "region": region,
                "universe": universe,
            }

            try:
                resp = self._session.get(url, params=params, verify=False, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch datasets: {resp.status_code}")
                    break

                data = resp.json()
                results = data.get("results", [])
                total = data.get("count", 0)

                if not results:
                    break

                for item in results:
                    ds_id = item.get("id") or item.get("dataset_id")
                    if ds_id:
                        all_datasets.append({
                            "id": ds_id,
                            "name": item.get("name", ds_id),
                            "fields": item.get("fieldCount", 0),
                        })

                logger.info(f"  Fetched {len(all_datasets)}/{total} datasets...")
                if len(all_datasets) >= total:
                    break

                offset += limit
                time.sleep(2)

            except Exception as e:
                logger.warning(f"Error fetching datasets: {e}")
                break

        logger.info(f"Found {len(all_datasets)} datasets for {region}/{universe}")
        return all_datasets

    # ── Field fetching (per dataset) ─────────────────────────────────

    def fetch_fields_for_dataset(
        self,
        dataset_id: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        limit: int = 50,
    ) -> List[Dict]:
        """Fetch all fields for a specific dataset."""
        url = "https://api.worldquantbrain.com/data-fields"
        all_fields = []
        offset = 0

        while True:
            params = {
                "instrumentType": "EQUITY",
                "region": region,
                "delay": delay,
                "universe": universe,
                "dataset.id": dataset_id,
                "limit": limit,
                "offset": offset,
            }

            try:
                resp = self._session.get(url, params=params, verify=False, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch fields for {dataset_id}: {resp.status_code}")
                    break

                data = resp.json()
                results = data.get("results", [])
                total = data.get("count", 0)

                if not results:
                    break

                all_fields.extend(results)
                if len(all_fields) >= total:
                    break

                offset += limit
                time.sleep(2)

            except Exception as e:
                logger.warning(f"Error fetching fields for {dataset_id}: {e}")
                break

        return all_fields

    # ── Full field fetch (all datasets) ──────────────────────────────

    def fetch_all_fields(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
    ) -> List[Dict]:
        """Fetch all fields by iterating through datasets."""
        logger.info(f"Fetching all fields for {region}/{universe}...")

        datasets = self.fetch_datasets(region=region, universe=universe, delay=delay)
        all_fields = []

        for i, ds in enumerate(datasets):
            ds_id = ds["id"]
            logger.info(f"  [{i+1}/{len(datasets)}] Fetching fields for dataset: {ds_id}")
            fields = self.fetch_fields_for_dataset(
                dataset_id=ds_id,
                region=region,
                universe=universe,
                delay=delay,
            )

            # Tag each field with dataset info
            for f in fields:
                f["_dataset_id"] = ds_id
                f["_dataset_name"] = ds.get("name", ds_id)

            all_fields.extend(fields)
            logger.info(f"    Got {len(fields)} fields (total: {len(all_fields)})")
            time.sleep(3)

        logger.info(f"Finished fetching. Total: {len(all_fields)} fields from {len(datasets)} datasets")
        return all_fields

    # ── Save fields to CSV ───────────────────────────────────────────

    def save_fields_to_csv(self, fields: List[Dict]) -> List[str]:
        """Save fields to CSV files grouped by dataset."""
        datasets: Dict[str, List[Dict]] = {}

        for field in fields:
            ds_id = field.get("_dataset_id", "unknown")
            ds_name = field.get("_dataset_name", ds_id)

            if ds_id not in datasets:
                datasets[ds_id] = {"name": ds_name, "fields": []}

            datasets[ds_id]["fields"].append({
                "Field": field.get("id", ""),
                "Description": field.get("description", ""),
                "Type": field.get("type", ""),
                "Dataset": ds_name,
                "Alphas": field.get("alphaCount", 0),  # 使用正确的字段名 alphaCount
            })

        saved_files = []
        for ds_id, ds_info in datasets.items():
            ds_fields = ds_info["fields"]
            if not ds_fields:
                continue

            safe_name = re.sub(r"[^a-z0-9_]+", "_", ds_id.lower()).strip("_")
            csv_path = os.path.join(FIELDS_DIR, f"{safe_name}.csv")

            df = pd.DataFrame(ds_fields)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            saved_files.append(csv_path)
            logger.info(f"Saved {len(ds_fields)} fields to {csv_path}")

        return saved_files

    # ── Operators ────────────────────────────────────────────────────

    def fetch_operators(self) -> List[Dict]:
        """Fetch all available operators."""
        logger.info("Fetching operators...")
        url = "https://api.worldquantbrain.com/operators"

        try:
            resp = self._session.get(url, verify=False, timeout=60)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch operators: {resp.status_code}")
                return []

            operators_raw = resp.json()
            if not isinstance(operators_raw, list):
                return []

            operators = []
            for item in operators_raw:
                operators.append({
                    "Name": item.get("name", ""),
                    "Category": item.get("category", ""),
                    "Scope": ", ".join(item.get("scope", [])),
                    "Definition": item.get("definition", ""),
                    "Description": item.get("description", ""),
                })

            logger.info(f"Fetched {len(operators)} operators")
            return operators

        except Exception as e:
            logger.error(f"Error fetching operators: {e}")
            return []

    def save_operators_to_csv(self, operators: List[Dict]) -> str:
        """Save operators to CSV file."""
        if not operators:
            return ""

        df = pd.DataFrame(operators)
        csv_path = os.path.join(OPERATORS_DIR, "operators.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(operators)} operators to {csv_path}")
        return csv_path

    # ── Combined fetch ───────────────────────────────────────────────

    def fetch_and_save_all(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
    ) -> Dict[str, object]:
        """Fetch and save all data fields and operators."""
        result = {"fields": [], "operators": ""}

        fields = self.fetch_all_fields(region=region, universe=universe)
        result["fields"] = self.save_fields_to_csv(fields)

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
                try:
                    df = pd.read_csv(os.path.join(FIELDS_DIR, file))
                    summary[file.replace(".csv", "")] = len(df)
                except Exception:
                    pass
        return summary

    def get_operator_summary(self) -> Dict[str, int]:
        """Get summary of saved operators by category."""
        summary = {}
        csv_path = os.path.join(OPERATORS_DIR, "operators.csv")
        if not os.path.exists(csv_path):
            return summary

        try:
            df = pd.read_csv(csv_path)
            if "Category" in df.columns:
                summary = df["Category"].value_counts().to_dict()
        except Exception:
            pass
        return summary


def fetch_and_save_all(
    region: str = "USA",
    universe: str = "TOP3000",
    session=None,
) -> Dict[str, object]:
    """Convenience function to fetch and save all data."""
    fetcher = DataFetcher(session=session)
    return fetcher.fetch_and_save_all(region=region, universe=universe)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_save_all()

    print("\n=== Summary ===")
    print(f"Fields saved: {len(result['fields'])} files")
    print(f"Operators saved: {result['operators']}")
