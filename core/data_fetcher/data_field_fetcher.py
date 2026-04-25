import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional
from time import sleep

logger = logging.getLogger(__name__)


class DataFieldFetcher:
    def __init__(self, session: requests.Session = None, cache_dir: str = None):
        self.session = session
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent.parent / "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_fields: Dict[str, List[Dict]] = {}

    def fetch_data_fields(
        self,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        force_refresh: bool = False,
    ) -> List[Dict]:
        cache_key = f"{region}_{universe}_{delay}"
        cache_file = self.cache_dir / f"data_fields_{cache_key}.json"

        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.data_fields[cache_key] = json.load(f)
                logger.info(
                    f"Loaded {len(self.data_fields[cache_key])} data fields from cache for {cache_key}"
                )
                return self.data_fields[cache_key]
            except Exception as e:
                logger.warning(f"Failed to load cached data fields: {e}")

        if not self.session:
            logger.error("No session available for fetching data fields")
            return self.data_fields.get(cache_key, [])

        try:
            logger.info(f"Fetching data fields for {cache_key} from API...")
            all_fields = []
            offset = 0
            page_size = 50

            while True:
                params = {
                    "instrumentType": "EQUITY",
                    "region": region,
                    "delay": delay,
                    "universe": universe,
                    "limit": page_size,
                    "offset": offset,
                }

                max_retries = 5
                response = None
                for attempt in range(max_retries):
                    try:
                        response = self.session.get(
                            "https://api.worldquantbrain.com/data-fields",
                            params=params,
                            timeout=30,
                        )
                        if response.status_code == 429:
                            retry_after = int(response.headers.get("Retry-After", 10))
                            logger.warning(f"Rate limited, waiting {retry_after}s...")
                            sleep(retry_after)
                            continue
                        break
                    except Exception as e:
                        logger.warning(f"Request error attempt {attempt+1}: {e}")
                        if attempt < max_retries - 1:
                            sleep(5)
                        else:
                            raise

                if response is None:
                    break

                if response.status_code == 401:
                    logger.warning("Session expired during data field fetch")
                    break

                if response.status_code != 200:
                    logger.error(f"Failed to fetch data fields: {response.status_code}")
                    break

                data = response.json()
                results = data.get("results", [])
                total_count = data.get("count", 0)

                all_fields.extend(results)
                offset += page_size

                logger.info(f"Fetched {len(all_fields)}/{total_count} fields...")

                if offset >= total_count or len(results) < page_size:
                    break

                sleep(1.0)

            filtered = [
                f
                for f in all_fields
                if f.get("region") == region
                and f.get("universe") == universe
                and f.get("delay") == delay
            ]

            if not filtered and all_fields:
                logger.warning(
                    f"No fields matched exact filter for {cache_key}, using unfiltered ({len(all_fields)} fields)"
                )
                filtered = all_fields

            self.data_fields[cache_key] = filtered
            self._save_cache(cache_key, cache_file)
            logger.info(
                f"Fetched and cached {len(filtered)} data fields for {cache_key}"
            )
            return filtered

        except Exception as e:
            logger.error(f"Error fetching data fields: {e}")
            return self.data_fields.get(cache_key, [])

    def _save_cache(self, cache_key: str, cache_file: Path):
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data_fields[cache_key], f, indent=2, ensure_ascii=False)
            logger.info(f"Cached data fields to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data fields: {e}")

    def clear_cache(self, region: str = None, universe: str = None, delay: int = None):
        if region and universe and delay is not None:
            cache_key = f"{region}_{universe}_{delay}"
            cache_file = self.cache_dir / f"data_fields_{cache_key}.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file}")
        else:
            for f in self.cache_dir.glob("data_fields_*.json"):
                f.unlink()
                logger.info(f"Cleared cache: {f}")

    def get_numeric_fields(self, cache_key: str = "USA_TOP3000_1") -> List[Dict]:
        fields = self.data_fields.get(cache_key, [])
        categorical_suffixes = [
            "_naicss",
            "_sector",
            "_industry",
            "_sic",
            "_exchange",
            "_country",
            "_currency",
            "_region",
            "_subindustry",
            "_city",
        ]
        result = []
        for field in fields:
            field_id = field.get("id", "").lower()
            if not any(field_id.endswith(s) or s + "_" in field_id for s in categorical_suffixes):
                result.append(field)
        return result

    def get_fields_by_dataset(self, cache_key: str, dataset: str) -> List[Dict]:
        fields = self.data_fields.get(cache_key, [])
        return [
            f for f in fields if f.get("dataset", {}).get("id") == dataset
        ]

    def get_fields_by_category(self, cache_key: str, category: str) -> List[Dict]:
        fields = self.data_fields.get(cache_key, [])
        return [
            f for f in fields if f.get("category", {}).get("id") == category
        ]

    def get_field_ids(self, cache_key: str = "USA_TOP3000_1") -> List[str]:
        return [f.get("id", "") for f in self.data_fields.get(cache_key, []) if f.get("id")]

    def get_all_datasets(self, cache_key: str = "USA_TOP3000_1") -> List[str]:
        datasets = set()
        for f in self.data_fields.get(cache_key, []):
            ds = f.get("dataset", {}).get("id")
            if ds:
                datasets.add(ds)
        return sorted(list(datasets))
