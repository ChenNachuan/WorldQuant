import json
import logging
import requests
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

OPERATOR_CACHE_FILE = Path(__file__).parent.parent.parent / "cache" / "operators.json"


class OperatorFetcher:
    def __init__(self, session: requests.Session = None, cache_dir: str = None):
        self.session = session
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent.parent / "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "operators.json"
        self.operators: List[Dict] = []

    def fetch_operators(self, force_refresh: bool = False) -> List[Dict]:
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.operators = json.load(f)
                logger.info(f"Loaded {len(self.operators)} operators from cache")
                return self.operators
            except Exception as e:
                logger.warning(f"Failed to load cached operators: {e}")

        if not self.session:
            logger.error("No session available for fetching operators")
            if self.operators:
                return self.operators
            return []

        try:
            logger.info("Fetching operators from WorldQuant Brain API...")
            response = self.session.get("https://api.worldquantbrain.com/operators", timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to fetch operators: {response.status_code}")
                return self.operators

            data = response.json()
            if isinstance(data, list):
                self.operators = data
            elif "results" in data:
                self.operators = data["results"]
            elif "items" in data:
                self.operators = data["items"]
            else:
                self.operators = []

            self._save_cache()
            logger.info(f"Fetched and cached {len(self.operators)} operators")
            return self.operators
        except Exception as e:
            logger.error(f"Error fetching operators: {e}")
            return self.operators

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.operators, f, indent=2, ensure_ascii=False)
            logger.info(f"Cached {len(self.operators)} operators to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving operator cache: {e}")

    def get_operators_by_category(self, category: str) -> List[Dict]:
        return [op for op in self.operators if op.get("category") == category]

    def get_operator_by_name(self, name: str) -> Optional[Dict]:
        for op in self.operators:
            if op.get("name") == name:
                return op
        return None

    def get_all_categories(self) -> List[str]:
        categories = set()
        for op in self.operators:
            cat = op.get("category")
            if cat:
                categories.add(cat)
        return sorted(list(categories))

    def get_operator_names(self) -> List[str]:
        return [op.get("name", "") for op in self.operators if op.get("name")]

    def get_operator_signatures(self) -> Dict[str, str]:
        result = {}
        for op in self.operators:
            name = op.get("name", "")
            definition = op.get("definition", "")
            if name and definition:
                result[name] = definition
        return result
