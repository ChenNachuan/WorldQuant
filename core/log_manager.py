import logging
import os
from datetime import datetime


LOG_DIR = "log"

MODULES = [
    "generator",
    "miner",
    "submitter",
    "orchestrator",
    "dashboard",
    "infrastructure",
]


def _ensure_log_dir(module: str) -> str:
    module_dir = os.path.join(LOG_DIR, module)
    os.makedirs(module_dir, exist_ok=True)
    return module_dir


def get_log_file(module: str) -> str:
    module_dir = _ensure_log_dir(module)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(module_dir, f"{date_str}.log")


def setup_logger(name: str, module: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(get_log_file(module), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
