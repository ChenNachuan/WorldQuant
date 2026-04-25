import math
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SmartSearch:
    def __init__(self, data_fields: Dict[str, List[Dict]] = None):
        self.data_fields = data_fields or {}

    def update_fields(self, data_fields: Dict[str, List[Dict]]):
        self.data_fields = data_fields

    def search_data_fields(
        self,
        query: str,
        cache_key: str = "USA_TOP3000_1",
        limit: int = 20,
    ) -> List[Tuple[Dict, float]]:
        fields = self.data_fields.get(cache_key, [])
        if not fields:
            return []

        query_lower = query.lower()
        query_terms = query_lower.split()
        scored = []

        for field in fields:
            score = 0.0
            field_id = field.get("id", "").lower()
            field_desc = field.get("description", "").lower()
            field_cat = field.get("category", {}).get("name", "").lower()

            for term in query_terms:
                if term in field_id:
                    score += 3.0
                if term in field_desc:
                    score += 1.0
                if term in field_cat:
                    score += 0.5

            user_count = field.get("userCount", 0)
            alpha_count = field.get("alphaCount", 0)
            score += min(math.log1p(user_count) * 0.1, 1.0)
            score += min(math.log1p(alpha_count) * 0.05, 0.5)

            if score > 0:
                scored.append((field, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def multi_criteria_search(
        self,
        query: str,
        cache_key: str = "USA_TOP3000_1",
        criteria: Dict[str, float] = None,
        limit: int = 20,
    ) -> List[Tuple[Dict, float]]:
        if criteria is None:
            criteria = {"relevance": 0.4, "usage": 0.3, "coverage": 0.3}

        base_results = self.search_data_fields(query, cache_key, limit=limit * 2)
        scored = []

        for field, base_score in base_results:
            total_score = base_score * criteria.get("relevance", 0.4)

            if "usage" in criteria:
                user_count = field.get("userCount", 0)
                total_score += min(math.log1p(user_count) * 0.1, 1.0) * criteria["usage"]

            if "coverage" in criteria:
                coverage = field.get("coverage", 0.0)
                if isinstance(coverage, (int, float)):
                    total_score += coverage * criteria["coverage"]

            scored.append((field, total_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def statistical_ranking(
        self,
        fields: List[Dict],
        metric: str = "userCount",
    ) -> List[Tuple[Dict, float]]:
        if not fields:
            return []

        values = [f.get(metric, 0) for f in fields]
        if not values or all(v == 0 for v in values):
            return [(f, 0.0) for f in fields]

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0

        z_scores = []
        for field, value in zip(fields, values):
            z_score = (value - mean) / std_dev if std_dev > 0 else 0.0
            z_scores.append((field, z_score))

        z_scores.sort(key=lambda x: x[1], reverse=True)
        return z_scores

    def get_recommendations(
        self,
        context: Dict,
        cache_key: str = "USA_TOP3000_1",
        limit: int = 5,
    ) -> List[Dict]:
        fields = self.data_fields.get(cache_key, [])

        if "operators" in context:
            return sorted(
                fields,
                key=lambda f: f.get("alphaCount", 0),
                reverse=True,
            )[:limit]
        elif "categories" in context:
            cat_ids = context["categories"]
            return [
                f
                for f in fields
                if f.get("category", {}).get("id") in cat_ids
            ][:limit]

        return sorted(
            fields,
            key=lambda f: f.get("userCount", 0),
            reverse=True,
        )[:limit]
