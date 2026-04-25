import time
import logging
import threading
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class SimulationSlotManager:
    def __init__(
        self,
        max_concurrent: int = 5,
        daily_limit: int = 1000,
    ):
        self.max_concurrent = max_concurrent
        self.daily_limit = daily_limit
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._daily_count = 0
        self._daily_reset_time = time.time()
        self._active_count = 0

    def acquire_slot(self, timeout: float = 300.0) -> bool:
        self._check_daily_reset()

        with self._lock:
            if self._daily_count >= self.daily_limit:
                logger.warning(
                    f"Daily simulation limit reached: {self._daily_count}/{self.daily_limit}"
                )
                return False

        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
                self._daily_count += 1
            logger.debug(
                f"Slot acquired. Active: {self._active_count}, Daily: {self._daily_count}/{self.daily_limit}"
            )
        else:
            logger.warning(f"Failed to acquire slot within {timeout}s")

        return acquired

    def release_slot(self):
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._semaphore.release()
        logger.debug(f"Slot released. Active: {self._active_count}")

    def _check_daily_reset(self):
        now = time.time()
        if now - self._daily_reset_time >= 86400:
            with self._lock:
                self._daily_count = 0
                self._daily_reset_time = now
            logger.info("Daily simulation counter reset")

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "active_count": self._active_count,
                "max_concurrent": self.max_concurrent,
                "daily_count": self._daily_count,
                "daily_limit": self.daily_limit,
                "available_slots": self.max_concurrent - self._active_count,
                "daily_remaining": self.daily_limit - self._daily_count,
            }

    def set_max_concurrent(self, max_concurrent: int):
        with self._lock:
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

        def __enter__(self):
            self.manager.acquire_slot()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.manager.release_slot()
            return False

    def slot_context(self) -> "SimulationSlotManager.SlotContext":
        return self.SlotContext(self)
