import os
import logging
import datetime

from pathlib import Path
from config import DB_LOG_PATH


def get_logger(name: str) -> logging.Logger:
    LOG_PATH = Path(str(DB_LOG_PATH).split(".log")[0] + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".log")
    os.makedirs(LOG_PATH.parent, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
