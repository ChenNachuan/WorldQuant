"""
Submission quota tracker with persistent state.

Tracks daily submission count and enforces a configurable limit.
Thread-safe for concurrent environments. Replaces the simplistic
`can_submit_today()` date-only check with counted quota management.
"""

import json
import os
import threading
import time
import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_QUOTA_FILE = os.path.join("data", "submission_log.json")


class SubmissionQuota:
    """Daily submission quota tracker with persistent state."""

    def __init__(
        self,
        daily_limit: int = 10,
        quota_file: str = DEFAULT_QUOTA_FILE,
    ):
        self.daily_limit = daily_limit
        self.quota_file = quota_file
        self._lock = threading.Lock()
        self._date: str = ""
        self._count: int = 0
        self._submitted_ids: List[str] = []
        self._load()

    # ── Public API ──────────────────────────────────────────────────

    def can_submit(self) -> bool:
        """Check if we can submit today (under daily limit)."""
        self._ensure_date()
        with self._lock:
            return self._count < self.daily_limit

    def remaining(self) -> int:
        """Number of submissions still available today."""
        self._ensure_date()
        with self._lock:
            return max(0, self.daily_limit - self._count)

    def record_submission(self, alpha_id: str):
        """Record a successful submission."""
        self._ensure_date()
        with self._lock:
            self._count += 1
            self._submitted_ids.append(alpha_id)
            self._save()

    def count_today(self) -> int:
        """Number of submissions already made today."""
        self._ensure_date()
        with self._lock:
            return self._count

    def last_submission_date(self) -> Optional[str]:
        """Return the date string of the last submission day, if any."""
        with self._lock:
            return self._date if self._count > 0 else None

    def is_already_submitted_today(self) -> bool:
        """True if any submission has already occurred today."""
        self._ensure_date()
        with self._lock:
            return self._count > 0

    # ── Internal ────────────────────────────────────────────────────

    def _ensure_date(self):
        today = datetime.now().date().isoformat()
        with self._lock:
            if self._date != today:
                self._date = today
                self._count = 0
                self._submitted_ids = []

    def _load(self):
        if os.path.exists(self.quota_file):
            try:
                with open(self.quota_file, "r") as f:
                    data = json.load(f)
                self._date = data.get("last_submission_date", "")
                self._count = data.get("submission_count", 0)
                self._submitted_ids = data.get("submitted_ids", [])
                self.daily_limit = data.get("daily_limit", self.daily_limit)
                self._ensure_date()
                logger.info(
                    "Loaded quota: %d/%d submissions today (%s)",
                    self._count,
                    self.daily_limit,
                    self._date,
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Could not load quota file: %s", e)

    def _save(self):
        os.makedirs(os.path.dirname(self.quota_file), exist_ok=True)
        data = {
            "last_submission_date": self._date,
            "submission_count": self._count,
            "submitted_ids": self._submitted_ids[-200:],  # keep last 200
            "daily_limit": self.daily_limit,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.quota_file, "w") as f:
            json.dump(data, f, indent=2)


# ── Global singleton ─────────────────────────────────────────────────

_quota_instance: Optional[SubmissionQuota] = None
_quota_lock = threading.Lock()


def get_submission_quota(daily_limit: int = 10) -> SubmissionQuota:
    """Return the process-wide singleton SubmissionQuota."""
    global _quota_instance
    with _quota_lock:
        if _quota_instance is None:
            _quota_instance = SubmissionQuota(daily_limit=daily_limit)
        return _quota_instance