# spotlite/logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from spotlite.config import get_config


def setup_logging(force_debug=False):
    logging_cfg = get_config().get("logging", {}).copy()

    # --debug mode overrides config
    if force_debug:
        logging_cfg["level"] = "DEBUG"
        logging_cfg["save_to_file"] = True

    level = getattr(logging, logging_cfg.get(
        "level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(level)

    if logger.handlers:
        return logger

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    # console handler
    if logging_cfg.get("console", True):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file handler
    if logging_cfg.get("save_to_file", False):
        file_path = Path(logging_cfg.get("file_path", "logs/app.log"))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            file_path,
            maxBytes=logging_cfg.get("file_max_MB", 5) * 1024 * 1024,
            backupCount=logging_cfg.get("file_backup_count", 3),
            encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
