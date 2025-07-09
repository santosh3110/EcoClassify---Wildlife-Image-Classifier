import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name="ecoclassifyLogger", log_file="running_logs.log") -> logging.Logger:
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Log format
    logging_format = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

    # Create logger
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if already set
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(logging_format))

    # Console Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(logging_format))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
