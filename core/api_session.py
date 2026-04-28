"""
Unified API Session Manager for WorldQuant Brain.

Centralises authentication, proxy configuration, rate limiting,
and global 429 cooldown across all modules.  Every module that
talks to the WorldQuant Brain API should obtain its session
through get_session_manager() instead of constructing
requests.Session() directly.

Thread-safe — uses locks for login mutual-exclusion and global
rate-limit cooldown so that concurrent threads don't stampede.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

# ── Global singleton (per process) ────────────────────────────────────
_session_manager: Optional["SessionManager"] = None
_manager_lock = threading.Lock()


class SessionManager:
    """Unified API session with centralised auth, proxy, rate-limiting."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        *,
        auth_ttl_seconds: int = 1500,
        max_retries: int = 7,
        backoff_factor: float = 2.0,
        pool_connections: int = 20,
        pool_maxsize: int = 20,
    ):
        self._auth_ttl = auth_ttl_seconds
        self._last_auth_time: float = 0.0
        self._login_lock = threading.Lock()
        self._global_cooldown_lock = threading.Lock()
        self._global_cooldown_until: float = 0.0

        if username is None or password is None:
            from .config import load_credentials

            username, password = load_credentials()
        self._username = username
        self._password = password

        self.session = requests.Session()
        self.session.timeout = (30, 300)
        self._apply_proxy()
        self._apply_retry_adapter(
            max_retries, backoff_factor, pool_connections, pool_maxsize
        )
        self._authenticate()

    # ── Public API ────────────────────────────────────────────────────

    def ensure_authenticated(self):
        """Re-auth if token may have expired (called before API calls)."""
        if time.time() - self._last_auth_time > self._auth_ttl:
            self._authenticate()

    def wait_if_rate_limited(self):
        """Block if the session is in a global 429 cooldown window."""
        with self._global_cooldown_lock:
            remaining = self._global_cooldown_until - time.time()
        if remaining > 0:
            logger.info(
                "Global rate-limit cooldown active, sleeping %.1fs", remaining
            )
            time.sleep(remaining)

    def report_rate_limit(self, retry_after: float = 30.0):
        """Call when any thread receives a 429; sets global cooldown."""
        with self._global_cooldown_lock:
            self._global_cooldown_until = max(
                self._global_cooldown_until,
                time.time() + retry_after,
            )
            logger.info(
                "Rate limit reported, cooldown until %s",
                datetime.fromtimestamp(self._global_cooldown_until).strftime(
                    "%H:%M:%S"
                ),
            )

    def request_with_retry(
        self, method: str, url: str, max_attempts: int = 3, **kwargs
    ):
        """Make an API request with auth-aware retry.

        Handles 401/403 (re-auth), 429 (global cooldown), and
        transient network errors with exponential backoff.
        """
        for attempt in range(max_attempts):
            self.wait_if_rate_limited()
            self.ensure_authenticated()

            try:
                resp = self.session.request(method, url, **kwargs)

                if resp.status_code == 429:
                    retry_after = float(
                        resp.headers.get("Retry-After", 30)
                    )
                    self.report_rate_limit(retry_after)
                    if attempt < max_attempts - 1:
                        time.sleep(retry_after)
                    continue

                if resp.status_code in (401, 403):
                    logger.warning(
                        "Auth expired (HTTP %d), re-authenticating…",
                        resp.status_code,
                    )
                    self._authenticate()
                    if attempt < max_attempts - 1:
                        continue

                # Also detect auth failures in 400 bodies
                if (
                    resp.status_code == 400
                    and "authentication credentials"
                    in resp.text.lower()
                ):
                    logger.warning(
                        "Auth credentials rejected, re-authenticating…"
                    )
                    self._authenticate()
                    if attempt < max_attempts - 1:
                        continue

                return resp

            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Connection error attempt %d/%d: %s",
                        attempt + 1,
                        max_attempts,
                        str(e)[:100],
                    )
                    self._apply_proxy()
                    time.sleep(5)
                else:
                    raise

            except requests.exceptions.Timeout as e:
                if attempt < max_attempts - 1:
                    logger.warning(
                        "Timeout attempt %d/%d: %s",
                        attempt + 1,
                        max_attempts,
                        str(e)[:100],
                    )
                    time.sleep(5 * (2**attempt))
                else:
                    raise

        # Should not reach here, but return last response if we do
        return resp  # type: ignore[possibly-undefined]

    # ── Internal ──────────────────────────────────────────────────────

    def _apply_proxy(self):
        from .config import fix_session_proxy

        fix_session_proxy(self.session)

    def _apply_retry_adapter(
        self,
        max_retries: int,
        backoff_factor: float,
        pool_connections: int,
        pool_maxsize: int,
    ):
        retry_strategy = requests.adapters.Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PATCH"],
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _authenticate(self):
        with self._login_lock:
            # Double-check inside lock — another thread may have refreshed
            if time.time() - self._last_auth_time < self._auth_ttl:
                return

            self.session.auth = HTTPBasicAuth(
                self._username, self._password
            )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = self.session.post(
                        "https://api.worldquantbrain.com/authentication",
                        timeout=(15, 30),
                    )
                    if resp.status_code == 201:
                        self._last_auth_time = time.time()
                        logger.info(
                            "Authenticated with WorldQuant Brain"
                        )
                        return
                    raise RuntimeError(
                        f"Auth failed: {resp.status_code} {resp.text[:200]}"
                    )
                except (
                    requests.exceptions.SSLError,
                    requests.exceptions.ConnectionError,
                ) as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Auth attempt %d/%d: %s",
                            attempt + 1,
                            max_retries,
                            str(e)[:100],
                        )
                        self._apply_proxy()
                        time.sleep(5)
                    else:
                        raise


# ── Factory / singleton ──────────────────────────────────────────────


def get_session_manager(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> SessionManager:
    """Return the process-wide singleton SessionManager.

    On first call, creates the singleton from .env credentials.
    Subsequent calls return the same instance regardless of arguments.
    """
    global _session_manager
    with _manager_lock:
        if _session_manager is None:
            _session_manager = SessionManager(
                username=username, password=password
            )
        return _session_manager


def create_session(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> SessionManager:
    """Create a *new* SessionManager (not the singleton).

    Use when callers genuinely need independent sessions (rare).
    """
    return SessionManager(username=username, password=password)
