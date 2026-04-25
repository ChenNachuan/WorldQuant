import json
import time
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class SelfOptimizer:
    def __init__(self, state_dir: str = None):
        if state_dir is None:
            state_dir = str(Path(__file__).parent.parent / "cache" / "optimizer")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "optimizer_state.json"

        self.operator_stats: Dict[str, Dict] = {}
        self.field_stats: Dict[str, Dict] = {}
        self.template_stats: Dict[str, Dict] = {}
        self.window_stats: Dict[int, Dict] = {}
        self.group_stats: Dict[str, Dict] = {}

        self.total_tested = 0
        self.total_successful = 0
        self._load_state()

    def record_result(
        self,
        expression: str,
        fitness: Optional[float],
        sharpe: Optional[float],
        turnover: Optional[float],
        success: bool,
    ):
        self.total_tested += 1
        if success:
            self.total_successful += 1

        import re

        operators = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)
        for op in operators:
            if op not in self.operator_stats:
                self.operator_stats[op] = {"tested": 0, "success": 0, "fitness_sum": 0.0}
            self.operator_stats[op]["tested"] += 1
            if success:
                self.operator_stats[op]["success"] += 1
            if fitness is not None:
                self.operator_stats[op]["fitness_sum"] += fitness

        numbers = re.findall(r"\b(\d+)\b", expression)
        for num_str in numbers:
            num = int(num_str)
            if num >= 3 and num <= 300:
                if num not in self.window_stats:
                    self.window_stats[num] = {"tested": 0, "success": 0}
                self.window_stats[num]["tested"] += 1
                if success:
                    self.window_stats[num]["success"] += 1

        for group in ["sector", "industry", "subindustry", "market"]:
            if group in expression:
                if group not in self.group_stats:
                    self.group_stats[group] = {"tested": 0, "success": 0}
                self.group_stats[group]["tested"] += 1
                if success:
                    self.group_stats[group]["success"] += 1

        if self.total_tested % 50 == 0:
            self._save_state()

    def get_operator_weights(self) -> Dict[str, float]:
        weights = {}
        for op, stats in self.operator_stats.items():
            if stats["tested"] > 0:
                success_rate = stats["success"] / stats["tested"]
                avg_fitness = stats["fitness_sum"] / max(stats["success"], 1)
                weights[op] = success_rate * 0.7 + min(avg_fitness / 2.0, 1.0) * 0.3
            else:
                weights[op] = 0.5
        return weights

    def get_recommended_operators(self, top_k: int = 10) -> List[str]:
        weights = self.get_operator_weights()
        sorted_ops = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [op for op, _ in sorted_ops[:top_k]]

    def get_recommended_windows(self, top_k: int = 4) -> List[int]:
        window_rates = {}
        for window, stats in self.window_stats.items():
            if stats["tested"] > 0:
                window_rates[window] = stats["success"] / stats["tested"]
        sorted_windows = sorted(window_rates.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_windows) < top_k:
            defaults = [5, 20, 60, 120, 252]
            for w in defaults:
                if w not in window_rates:
                    sorted_windows.append((w, 0.0))
        return [w for w, _ in sorted_windows[:top_k]]

    def get_recommended_groups(self) -> List[str]:
        group_rates = {}
        for group, stats in self.group_stats.items():
            if stats["tested"] > 0:
                group_rates[group] = stats["success"] / stats["tested"]
        sorted_groups = sorted(group_rates.items(), key=lambda x: x[1], reverse=True)
        return [g for g, _ in sorted_groups] if sorted_groups else ["sector", "industry"]

    def get_optimization_summary(self) -> Dict:
        success_rate = (
            self.total_successful / self.total_tested
            if self.total_tested > 0
            else 0.0
        )
        return {
            "total_tested": self.total_tested,
            "total_successful": self.total_successful,
            "success_rate": round(success_rate, 4),
            "top_operators": self.get_recommended_operators(5),
            "top_windows": self.get_recommended_windows(4),
            "top_groups": self.get_recommended_groups(),
        }

    def _save_state(self):
        try:
            state = {
                "operator_stats": self.operator_stats,
                "field_stats": self.field_stats,
                "template_stats": self.template_stats,
                "window_stats": {str(k): v for k, v in self.window_stats.items()},
                "group_stats": self.group_stats,
                "total_tested": self.total_tested,
                "total_successful": self.total_successful,
                "timestamp": time.time(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save optimizer state: {e}")

    def _load_state(self):
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.operator_stats = state.get("operator_stats", {})
            self.field_stats = state.get("field_stats", {})
            self.template_stats = state.get("template_stats", {})
            self.window_stats = {int(k): v for k, v in state.get("window_stats", {}).items()}
            self.group_stats = state.get("group_stats", {})
            self.total_tested = state.get("total_tested", 0)
            self.total_successful = state.get("total_successful", 0)
            logger.info(
                f"Loaded optimizer state: {self.total_tested} tested, {self.total_successful} successful"
            )
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
