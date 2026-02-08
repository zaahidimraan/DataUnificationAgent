import logging
import os
from logging.handlers import RotatingFileHandler
from config import Config

def setup_logger(name="data_agent"):
    """
    Configures a logger that writes to both file and console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 1. File Handler (Rotates after 1MB)
        log_file = os.path.join(Config.LOG_FOLDER, 'app.log')
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 2. Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

# Initialize the global logger
logger = setup_logger()