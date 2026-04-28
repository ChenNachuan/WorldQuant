"""
Async simulation poller using asyncio for efficient concurrent polling.

Replaces the sync `check_pending_results()` loop with cooperative
asyncio polling that can handle many concurrent simulations without
dedicating a thread to each one.

Uses `asyncio.to_thread()` to wrap the existing sync `requests.Session`
HTTP calls — no new HTTP library required.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PendingSim:
    """Track a pending simulation for async polling."""

    sim_id: str
    alpha: str
    progress_url: str
    start_time: float = field(default_factory=time.time)
    attempts: int = 0
    status: str = "pending"


class AsyncSimulationPoller:
    """Async poller for WorldQuant simulation results."""

    def __init__(
        self,
        api,  # SessionManager
        *,
        max_concurrent: int = 4,
        poll_interval_base: float = 5.0,
        poll_interval_max: float = 60.0,
        poll_backoff_factor: float = 1.5,
        timeout_per_sim: float = 1800.0,
    ):
        self._api = api
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._poll_interval_base = poll_interval_base
        self._poll_interval_max = poll_interval_max
        self._backoff = poll_backoff_factor
        self._timeout = timeout_per_sim
        self._pending: Dict[str, PendingSim] = {}

    def add(self, sim_id: str, alpha: str, progress_url: str):
        """Register a simulation for polling."""
        self._pending[sim_id] = PendingSim(
            sim_id=sim_id, alpha=alpha, progress_url=progress_url
        )
        logger.debug("Added sim %s for async polling", sim_id)

    def pending_count(self) -> int:
        return len(self._pending)

    async def poll_all(self) -> Dict[str, Dict]:
        """Poll all pending simulations concurrently until complete/timeout.

        Returns dict mapping sim_id -> result dict.
        """
        if not self._pending:
            return {}

        tasks = []
        for sim_id in list(self._pending.keys()):
            tasks.append(self._poll_one(sim_id))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        final: Dict[str, Dict] = {}
        sim_ids = list(self._pending.keys())
        for sim_id, result in zip(sim_ids, results_list):
            if isinstance(result, Exception):
                final[sim_id] = {
                    "status": "error",
                    "message": str(result),
                    "sim_id": sim_id,
                }
            else:
                final[sim_id] = result
            # Clean up from pending
            if sim_id in self._pending:
                del self._pending[sim_id]

        return final

    async def _poll_one(self, sim_id: str) -> Dict:
        sim = self._pending.get(sim_id)
        if sim is None:
            return {"status": "error", "message": f"Unknown sim {sim_id}"}

        async with self._semaphore:
            while True:
                elapsed = time.time() - sim.start_time
                if elapsed > self._timeout:
                    logger.warning(
                        "Simulation %s timed out after %.0fs", sim_id, elapsed
                    )
                    return {
                        "status": "timeout",
                        "sim_id": sim_id,
                        "alpha": sim.alpha,
                    }

                try:
                    # Use asyncio.to_thread for the sync HTTP call
                    resp = await asyncio.to_thread(
                        self._api.session.get, sim.progress_url
                    )

                    if resp.status_code == 429:
                        retry_after = float(
                            resp.headers.get("Retry-After", 30)
                        )
                        self._api.report_rate_limit(retry_after)
                        await asyncio.sleep(retry_after)
                        continue

                    data = resp.json()
                    status = data.get("status")

                    if status in ("COMPLETE", "WARNING"):
                        alpha_id = data.get("alpha")
                        if alpha_id:
                            alpha_resp = await asyncio.to_thread(
                                self._api.session.get,
                                f"https://api.worldquantbrain.com/alphas/{alpha_id}",
                            )
                            if alpha_resp.status_code == 200:
                                return {
                                    "status": status,
                                    "sim_id": sim_id,
                                    "alpha": sim.alpha,
                                    "alpha_data": alpha_resp.json(),
                                    "sim_result": data,
                                }
                        return {
                            "status": status,
                            "sim_id": sim_id,
                            "alpha": sim.alpha,
                        }

                    elif status == "ERROR":
                        return {
                            "status": "error",
                            "sim_id": sim_id,
                            "alpha": sim.alpha,
                            "message": data.get(
                                "error", data.get("message", "")
                            ),
                        }

                    # Exponential backoff for pending/running
                    sim.attempts += 1
                    delay = min(
                        self._poll_interval_base
                        * (self._backoff ** sim.attempts),
                        self._poll_interval_max,
                    )
                    delay += random.uniform(0, delay * 0.1)  # jitter
                    await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(
                        "Error polling %s: %s", sim_id, str(e)[:200]
                    )
                    sim.attempts += 1
                    if sim.attempts >= 10:
                        return {
                            "status": "error",
                            "sim_id": sim_id,
                            "message": str(e),
                        }
                    await asyncio.sleep(min(30, 5 * (2 ** sim.attempts)))