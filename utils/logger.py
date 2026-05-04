# app/utils/logger.py

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Load config.yaml (for fallback if env not set)
# ─────────────────────────────────────────────────────────────
import yaml

CONFIG_PATH = Path.cwd() / "config.yaml"

CONFIG = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}

# ─────────────────────────────────────────────────────────────
# Environment / Config Resolution
# ─────────────────────────────────────────────────────────────
LOG_DIR = os.getenv(
    "LOG_DIR_PATH",
    CONFIG.get("logging", {}).get("log_dir", "./logs")
)

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    CONFIG.get("logging", {}).get("level", "INFO")
).upper()

MAX_BYTES = 10 * 1024 * 1024  # 10MB


# ─────────────────────────────────────────────────────────────
# Ensure log directory exists
# ─────────────────────────────────────────────────────────────
def _ensure_log_dir():
    path = Path(LOG_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────
# Custom Handler: Date + Size based rotation
# ─────────────────────────────────────────────────────────────
class DateSizeRotatingHandler(RotatingFileHandler):
    """
    Combines:
    - Daily log separation
    - Size-based rotation (10MB)
    - Incremental suffix: 01, 02, ...
    """

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.current_date = self._today_str()
        base_filename = self._build_filename(self.current_date, None)
        super().__init__(
            filename=str(base_filename),
            maxBytes=MAX_BYTES,
            backupCount=0,  # unlimited
            encoding="utf-8"
        )

    def _today_str(self):
        return datetime.now().strftime("%d%m%Y")

    def _build_filename(self, date_str, index):
        if index is None:
            return self.log_dir / f"ai_review_{date_str}.log"
        return self.log_dir / f"ai_review_{date_str}{index:02d}.log"

    def _get_next_index(self, date_str):
        existing = list(self.log_dir.glob(f"ai_review_{date_str}*.log"))

        max_index = 0
        for f in existing:
            name = f.stem  # ai_review_ddMMyyyy or ai_review_ddMMyyyy01
            suffix = name.replace(f"ai_review_{date_str}", "")
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))

        return max_index + 1

    def shouldRollover(self, record):
        # Date change
        today = self._today_str()
        if today != self.current_date:
            return True

        # Size check
        if self.stream:
            self.stream.seek(0, 2)
            if self.stream.tell() >= self.maxBytes:
                return True

        return False

    def doRollover(self):
        try:
            self.stream.close()
        except Exception:
            pass

        today = self._today_str()

        # If date changed → new base file
        if today != self.current_date:
            self.current_date = today
            new_file = self._build_filename(today, None)
        else:
            # Same day → increment suffix
            idx = self._get_next_index(today)
            new_file = self._build_filename(today, idx)

        self.baseFilename = str(new_file)
        self.stream = self._open()


# ─────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────
FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S,%f"


class MillisecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{int(record.msecs):03d}"


# ─────────────────────────────────────────────────────────────
# Logger Setup
# ─────────────────────────────────────────────────────────────
def get_logger(name: str = "app"):
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)  # always capture everything

    log_dir_path = _ensure_log_dir()

    # File Handler (DEBUG level)
    file_handler = DateSizeRotatingHandler(log_dir_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(MillisecondFormatter(FORMAT))

    # Console Handler (INFO+)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(MillisecondFormatter(FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# ─────────────────────────────────────────────────────────────
# Exception helper (full stack trace logging)
# ─────────────────────────────────────────────────────────────
def log_exception(logger: logging.Logger, msg: str):
    logger.error(msg, exc_info=True)