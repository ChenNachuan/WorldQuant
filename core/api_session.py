"""
Simple API Session Manager for WorldQuant Brain.

Simplified version matching the IQC approach - direct requests.Session
with basic authentication and retry logic.
"""

import os
import logging
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://api.worldquantbrain.com"


class WorldQuantSession:
    """Simple WorldQuant Brain API session."""

    def __init__(self, username: str = None, password: str = None):
        """Initialize session with credentials from .env if not provided."""
        if username is None or password is None:
            from .config import load_credentials
            username, password = load_credentials()

        self.username = username
        self.password = password
        self.session = None
        self._authenticate()

    def _authenticate(self):
        """Create authenticated session."""
        logger.info("Authenticating with WorldQuant Brain...")

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        })
        self.session.auth = HTTPBasicAuth(self.username, self.password)

        try:
            resp = self.session.post(
                f"{BASE_URL}/authentication",
                verify=False,
                timeout=15
            )
            if resp.status_code == 201:
                logger.info("Authentication successful")
                return True
            else:
                raise Exception(f"Authentication failed: {resp.text}")
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    def ensure_authenticated(self):
        """Re-authenticate if session may have expired."""
        # Simple check - if session exists, assume it's valid
        # The caller should handle 401/403 responses and re-auth if needed
        if self.session is None:
            self._authenticate()

    def request(self, method: str, url: str, **kwargs):
        """Make an API request with automatic retry on auth failure."""
        # Ensure URL is absolute
        if not url.startswith("http"):
            url = f"{BASE_URL}{url}"

        # Set defaults
        kwargs.setdefault("verify", False)
        kwargs.setdefault("timeout", 30)

        try:
            resp = self.session.request(method, url, **kwargs)

            # Handle auth failures
            if resp.status_code in [401, 403]:
                logger.warning("Auth expired, re-authenticating...")
                self._authenticate()
                resp = self.session.request(method, url, **kwargs)

            return resp

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def get(self, url: str, **kwargs):
        """Make a GET request."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        """Make a POST request."""
        return self.request("POST", url, **kwargs)

    def submit_simulation(self, expression: str, settings: dict = None) -> dict:
        """Submit a factor for backtesting."""
        default_settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "NONE",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "OFF",
            "language": "FASTEXPR",
            "visualization": False
        }

        if settings:
            default_settings.update(settings)

        payload = {
            "type": "REGULAR",
            "settings": default_settings,
            "regular": expression
        }

        resp = self.post(f"{BASE_URL}/simulations", json=payload)

        if resp.status_code == 201:
            sim_id = resp.headers.get("Location", "").split("/")[-1]
            return {"success": True, "sim_id": sim_id}
        else:
            return {"success": False, "error": resp.text[:200]}

    def poll_simulation(self, sim_id: str, timeout: int = 600) -> dict:
        """Poll simulation until complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            resp = self.get(f"{BASE_URL}/simulations/{sim_id}")

            if resp.status_code != 200:
                time.sleep(3)
                continue

            data = resp.json()
            status = data.get("status")

            if status in ["FINISHED", "WARNING", "COMPLETE", "COMPLETED"]:
                alpha_id = data.get("alpha")
                if alpha_id:
                    # Get alpha performance
                    alpha_resp = self.get(f"{BASE_URL}/alphas/{alpha_id}")
                    if alpha_resp.status_code == 200:
                        perf = alpha_resp.json().get("is", {})
                        return {
                            "success": True,
                            "alpha_id": alpha_id,
                            "sharpe": perf.get("sharpe", 0),
                            "fitness": perf.get("fitness", 0),
                            "turnover": perf.get("turnover", 0),
                            "margin": perf.get("margin", 0),
                            "message": data.get("message", "Success")
                        }
                return {"success": True, "alpha_id": alpha_id}

            elif status in ["ERROR", "FAILED"]:
                return {"success": False, "error": data.get("message", "Unknown error")}

            time.sleep(3)

        return {"success": False, "error": "Simulation timeout"}

    def submit_alpha(self, alpha_id: str) -> bool:
        """Submit an alpha to WorldQuant Brain."""
        resp = self.post(
            f"{BASE_URL}/alphas/{alpha_id}/submit",
            json={"type": "REGULAR"}
        )
        return resp.status_code == 201


# Global session instance
_session: Optional[WorldQuantSession] = None


def get_session(username: str = None, password: str = None) -> WorldQuantSession:
    """Get or create the global session instance."""
    global _session
    if _session is None:
        _session = WorldQuantSession(username, password)
    return _session


def reset_session():
    """Reset the global session (for testing or re-authentication)."""
    global _session
    _session = None
