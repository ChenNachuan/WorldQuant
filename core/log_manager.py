import logging
import os
from datetime import datetime


LOG_DIR = "log"


class DailyFileHandler(logging.FileHandler):
    """FileHandler that creates a new log file each day."""

    def __init__(self, log_dir: str, encoding: str = "utf-8"):
        self.log_dir = log_dir
        self._current_date = None
        self._encoding = encoding
        os.makedirs(log_dir, exist_ok=True)
        filepath = self._get_filepath()
        super().__init__(filepath, encoding=encoding)

    def _get_filepath(self) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"{date_str}.log")

    def emit(self, record):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self.stream.close()
            self.baseFilename = self._get_filepath()
            self.stream = self._open()
        super().emit(record)


def setup_logger(name: str, module: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = DailyFileHandler(LOG_DIR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
