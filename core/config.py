import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def load_credentials() -> tuple[str, str]:
    username = os.getenv("WQ_USERNAME")
    password = os.getenv("WQ_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "WQ_USERNAME and WQ_PASSWORD must be set in .env file"
        )
    return username, password


def get_ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434")


def get_default_model() -> str:
    """Get the default Ollama model name, with env/file override.

    Checks (in order):
      1. WQ_DEFAULT_MODEL env var
      2. data/model_config.json file
      3. fallback: 'qwen3.5:35b'
    """
    env_model = os.getenv("WQ_DEFAULT_MODEL")
    if env_model:
        return env_model

    config_path = os.path.join("data", "model_config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path) as f:
                data = json.load(f)
            model = data.get("default_model")
            if model:
                return model
        except Exception:
            pass

    return "qwen3.5:35b"


def set_default_model(model_name: str):
    """Persist the default model to a data file (never modifies source code)."""
    import json
    from datetime import datetime
    config_path = os.path.join("data", "model_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(
            {
                "default_model": model_name,
                "updated_at": datetime.now().isoformat(),
            },
            f,
        )


def fix_session_proxy(sess: requests.Session) -> None:
    for k in ['all_proxy', 'ALL_PROXY']:
        if os.environ.pop(k, None):
            logger.info(f"Removed {k} from environment to avoid SOCKS conflicts")

    http_proxy = (
        os.environ.get('HTTPS_PROXY', '')
        or os.environ.get('https_proxy', '')
        or os.environ.get('HTTP_PROXY', '')
        or os.environ.get('http_proxy', '')
    )

    if http_proxy and http_proxy.startswith('http'):
        sess.proxies = {
            'http': http_proxy,
            'https': http_proxy,
        }
        logger.info(f"Set HTTP proxy for session: {http_proxy}")
    elif http_proxy and 'socks' in http_proxy.lower():
        logger.warning(f"Proxy {http_proxy} is SOCKS-based, which may cause SSL errors. "
                       f"Consider setting HTTPS_PROXY to an HTTP proxy (e.g. http://127.0.0.1:7897)")
        sess.proxies = {
            'http': http_proxy,
            'https': http_proxy,
        }
    else:
        logger.info("No HTTP proxy configured, using direct connection")

    try:
        import certifi
        sess.verify = certifi.where()
        logger.info(f"Using certifi SSL certificates: {certifi.where()}")
    except ImportError:
        logger.warning("certifi not installed, using default SSL certificates")

    # Robust retry strategy (borrowed from forum Super Alpha framework)
    # Auto-retries on 429 (rate limit), 500, 502, 503, 504
    retry_strategy = requests.adapters.Retry(
        total=7,
        backoff_factor=2,  # exponential: 2s, 4s, 8s, 16s, 32s, 64s, 128s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PATCH"],
    )
    adapter = requests.adapters.HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20,
    )
    sess.mount('https://', adapter)
    sess.mount('http://', adapter)
