import time
import logging
import threading
import uuid
from typing import Optional, Dict

from .alpha_db import get_alpha_db

logger = logging.getLogger(__name__)


class SimulationSlotManager:
    def __init__(
        self,
        max_concurrent: int = 4,
        daily_limit: int = 1000,
    ):
        self.max_concurrent = max_concurrent
        self.daily_limit = daily_limit
        self._lock = threading.Lock()
        self._daily_count = 0
        self._daily_reset_time = time.time()
        self.db = get_alpha_db()

    def acquire_slot(self, sim_id: str, timeout: float = 300.0) -> bool:
        self._check_daily_reset()

        with self._lock:
            if self._daily_count >= self.daily_limit:
                logger.warning(
                    f"Daily simulation limit reached: {self._daily_count}/{self.daily_limit}"
                )
                return False

        acquired = self.db.acquire_global_slot(sim_id, max_concurrent=self.max_concurrent, timeout=timeout)
        if acquired:
            with self._lock:
                self._daily_count += 1
            logger.debug(f"Global slot acquired: {sim_id}")
        else:
            logger.warning(f"Failed to acquire global slot within {timeout}s for {sim_id}")

        return acquired

    def release_slot(self, sim_id: str):
        self.db.release_global_slot(sim_id)
        logger.debug(f"Global slot released: {sim_id}")

    def _check_daily_reset(self):
        now = time.time()
        if now - self._daily_reset_time >= 86400:
            with self._lock:
                self._daily_count = 0
                self._daily_reset_time = now
            logger.info("Daily simulation counter reset")

    def get_status(self) -> Dict:
        return {
            "max_concurrent": self.max_concurrent,
            "daily_count": self._daily_count,
            "daily_limit": self.daily_limit,
            "daily_remaining": self.daily_limit - self._daily_count,
        }

    def set_max_concurrent(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        logger.info(f"Max concurrent simulations set to {max_concurrent}")

    def reset_daily_count(self):
        with self._lock:
            self._daily_count = 0
            self._daily_reset_time = time.time()
        logger.info("Daily simulation counter manually reset")

    class SlotContext:
        def __init__(self, manager: "SimulationSlotManager"):
            self.manager = manager
            self.sim_id = str(uuid.uuid4())

        def __enter__(self):
            acquired = self.manager.acquire_slot(self.sim_id)
            if not acquired:
                raise Exception("Failed to acquire simulation slot")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.manager.release_slot(self.sim_id)
            return False

    def slot_context(self) -> "SimulationSlotManager.SlotContext":
        return self.SlotContext(self)
