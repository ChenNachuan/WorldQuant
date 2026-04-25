import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AlphaMetrics:
    expression: str
    fitness: Optional[float] = None
    sharpe: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


class QualityMonitor:
    def __init__(self, history_dir: str = None):
        if history_dir is None:
            history_dir = str(Path(__file__).parent.parent / "cache" / "quality")
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "quality_history.json"

        self.history: List[AlphaMetrics] = []
        self._load_history()

    def record(self, metrics: AlphaMetrics):
        self.history.append(metrics)
        if len(self.history) % 20 == 0:
            self._save_history()

    def record_batch(self, metrics_list: List[AlphaMetrics]):
        for m in metrics_list:
            self.history.append(m)
        self._save_history()

    def get_success_rate(self, hours: int = 24) -> float:
        cutoff = time.time() - hours * 3600
        recent = [m for m in self.history if m.timestamp >= cutoff]
        if not recent:
            return 0.0
        successful = sum(1 for m in recent if m.fitness is not None and m.fitness >= 1.0)
        return successful / len(recent)

    def get_avg_fitness(self, hours: int = 24) -> Optional[float]:
        cutoff = time.time() - hours * 3600
        recent = [m for m in self.history if m.timestamp >= cutoff and m.fitness is not None]
        if not recent:
            return None
        return sum(m.fitness for m in recent) / len(recent)

    def get_avg_sharpe(self, hours: int = 24) -> Optional[float]:
        cutoff = time.time() - hours * 3600
        recent = [m for m in self.history if m.timestamp >= cutoff and m.sharpe is not None]
        if not recent:
            return None
        return sum(m.sharpe for m in recent) / len(recent)

    def detect_degradation(self, window_hours: int = 6, threshold: float = 0.5) -> Dict:
        recent_rate = self.get_success_rate(window_hours)
        overall_rate = self.get_success_rate(72)

        is_degraded = False
        if overall_rate > 0 and recent_rate < overall_rate * threshold:
            is_degraded = True

        recent_fitness = self.get_avg_fitness(window_hours)
        overall_fitness = self.get_avg_fitness(72)

        fitness_degraded = False
        if (
            overall_fitness is not None
            and recent_fitness is not None
            and overall_fitness > 0
            and recent_fitness < overall_fitness * threshold
        ):
            fitness_degraded = True

        return {
            "is_degraded": is_degraded or fitness_degraded,
            "success_rate_recent": round(recent_rate, 4),
            "success_rate_overall": round(overall_rate, 4),
            "avg_fitness_recent": round(recent_fitness, 4) if recent_fitness else None,
            "avg_fitness_overall": round(overall_fitness, 4) if overall_fitness else None,
            "window_hours": window_hours,
        }

    def get_trend(self, hours: int = 24, bucket_hours: int = 4) -> List[Dict]:
        now = time.time()
        start = now - hours * 3600
        buckets = []

        for i in range(0, hours, bucket_hours):
            bucket_start = start + i * 3600
            bucket_end = bucket_start + bucket_hours * 3600

            bucket_metrics = [
                m
                for m in self.history
                if bucket_start <= m.timestamp < bucket_end
            ]

            count = len(bucket_metrics)
            successful = sum(
                1 for m in bucket_metrics if m.fitness is not None and m.fitness >= 1.0
            )
            fitnesses = [m.fitness for m in bucket_metrics if m.fitness is not None]
            avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else None

            buckets.append({
                "period_start": datetime.fromtimestamp(bucket_start).isoformat(),
                "count": count,
                "successful": successful,
                "success_rate": round(successful / count, 4) if count > 0 else 0.0,
                "avg_fitness": round(avg_fitness, 4) if avg_fitness else None,
            })

        return buckets

    def get_summary(self) -> Dict:
        total = len(self.history)
        successful = sum(1 for m in self.history if m.fitness is not None and m.fitness >= 1.0)
        fitnesses = [m.fitness for m in self.history if m.fitness is not None]
        sharpes = [m.sharpe for m in self.history if m.sharpe is not None]

        return {
            "total_alphas": total,
            "successful_alphas": successful,
            "overall_success_rate": round(successful / total, 4) if total > 0 else 0.0,
            "avg_fitness": round(sum(fitnesses) / len(fitnesses), 4) if fitnesses else None,
            "max_fitness": round(max(fitnesses), 4) if fitnesses else None,
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else None,
            "max_sharpe": round(max(sharpes), 4) if sharpes else None,
            "degradation": self.detect_degradation(),
            "last_24h_success_rate": round(self.get_success_rate(24), 4),
        }

    def _save_history(self):
        try:
            data = []
            for m in self.history[-1000:]:
                data.append({
                    "expression": m.expression,
                    "fitness": m.fitness,
                    "sharpe": m.sharpe,
                    "turnover": m.turnover,
                    "returns": m.returns,
                    "timestamp": m.timestamp,
                    "source": m.source,
                })
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save quality history: {e}")

    def _load_history(self):
        if not self.history_file.exists():
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data[-500:]:
                self.history.append(AlphaMetrics(**item))
            logger.info(f"Loaded {len(self.history)} quality history entries")
        except Exception as e:
            logger.warning(f"Failed to load quality history: {e}")
