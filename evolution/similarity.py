import re
import logging
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class TemplateSimilarity:
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self._seen_skeletons: Dict[str, str] = {}

    def normalize(self, expression: str) -> str:
        normalized = expression.strip()
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"\b\d+\.?\d*\b", "N", normalized)
        return normalized.lower()

    def extract_skeleton(self, expression: str) -> str:
        skeleton = expression.strip()
        skeleton = re.sub(r"\s+", " ", skeleton).strip()
        skeleton = re.sub(r"\b\d+\.?\d*\b", "{N}", skeleton)

        tokens = re.findall(r"[a-z][a-z0-9_]*(?:_[a-z0-9_]+)+", skeleton, re.IGNORECASE)
        operator_prefixes = ("ts_", "group_", "vec_")

        for token in sorted(set(tokens), key=len, reverse=True):
            if any(token.startswith(p) for p in operator_prefixes):
                continue
            skeleton = skeleton.replace(token, "{F}")

        return skeleton.lower()

    def jaccard_similarity(self, expr1: str, expr2: str) -> float:
        tokens1 = set(re.findall(r"[a-zA-Z_]\w*", expr1))
        tokens2 = set(re.findall(r"[a-zA-Z_]\w*", expr2))

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def structural_similarity(self, expr1: str, expr2: str) -> float:
        skel1 = self.extract_skeleton(expr1)
        skel2 = self.extract_skeleton(expr2)

        if skel1 == skel2:
            return 1.0

        ops1 = set(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr1))
        ops2 = set(re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expr2))

        depth1 = self._nesting_depth(expr1)
        depth2 = self._nesting_depth(expr2)

        op_sim = len(ops1 & ops2) / max(len(ops1 | ops2), 1)
        depth_sim = 1.0 - abs(depth1 - depth2) / max(depth1, depth2, 1)
        len_sim = 1.0 - abs(len(skel1) - len(skel2)) / max(len(skel1), len(skel2), 1)

        return op_sim * 0.5 + depth_sim * 0.3 + len_sim * 0.2

    def is_similar(self, expr1: str, expr2: str) -> bool:
        return self.structural_similarity(expr1, expr2) >= self.similarity_threshold

    def deduplicate(self, expressions: List[str]) -> List[str]:
        unique = []
        seen_skeletons: List[str] = []

        for expr in expressions:
            skeleton = self.extract_skeleton(expr)
            is_dup = False

            for seen in seen_skeletons:
                if self._skeleton_similarity(skeleton, seen) >= self.similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(expr)
                seen_skeletons.append(skeleton)

        removed = len(expressions) - len(unique)
        if removed > 0:
            logger.info(f"Deduplicated: {removed} similar expressions removed, {len(unique)} kept")

        return unique

    def find_clusters(self, expressions: List[str]) -> List[List[str]]:
        clusters: List[List[str]] = []
        assigned: Set[int] = set()

        for i, expr1 in enumerate(expressions):
            if i in assigned:
                continue

            cluster = [expr1]
            assigned.add(i)

            for j, expr2 in enumerate(expressions):
                if j in assigned:
                    continue
                if self.is_similar(expr1, expr2):
                    cluster.append(expr2)
                    assigned.add(j)

            clusters.append(cluster)

        logger.info(f"Found {len(clusters)} clusters from {len(expressions)} expressions")
        return clusters

    def _skeleton_similarity(self, skel1: str, skel2: str) -> float:
        if skel1 == skel2:
            return 1.0

        tokens1 = skel1.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()
        tokens2 = skel2.replace("(", " ( ").replace(")", " ) ").replace(",", " , ").split()

        if not tokens1 or not tokens2:
            return 0.0

        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    def _nesting_depth(self, expr: str) -> int:
        max_depth = 0
        current = 0
        for ch in expr:
            if ch == "(":
                current += 1
                max_depth = max(max_depth, current)
            elif ch == ")":
                current -= 1
        return max_depth
