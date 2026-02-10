import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from config import Config

class SafeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode encoding errors gracefully."""
    
    def format(self, record):
        result = super().format(record)
        # If we're on Windows and can't encode properly, remove problematic characters
        if sys.platform == 'win32':
            try:
                # Try to encode to cp1252 (Windows console default)
                result.encode('cp1252')
            except UnicodeEncodeError:
                # Remove emoji and special Unicode characters
                result = result.encode('ascii', errors='ignore').decode('ascii')
        return result

def setup_logger(name="data_agent"):
    """
    Configures a logger that writes to both file and console with UTF-8 encoding.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # 1. File Handler (Rotates after 1MB) with UTF-8 encoding
        log_file = os.path.join(Config.LOG_FOLDER, 'app.log')
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=10, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 2. Console Handler with UTF-8 encoding
        # Reconfigure stdout to use UTF-8 encoding for Windows console
        if sys.platform == 'win32':
            try:
                # Set Windows console to UTF-8 mode
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                # Python 3.7+ can reconfigure the console encoding
                if hasattr(sys.stdout, 'reconfigure'):
                    sys.stdout.reconfigure(encoding='utf-8')
                    sys.stderr.reconfigure(encoding='utf-8')
            except Exception as e:
                # If setting UTF-8 fails, we'll use the safe formatter
                pass
        
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Use SafeFormatter for console to handle encoding issues gracefully
        console_formatter = SafeFormatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger

# Initialize the global logger
logger = setup_logger()

def log_info(message):
    """Log an info message to both file and console."""
    logger.info(message)

def log_error(message):
    """Log an error message to both file and console."""
    logger.error(message)

def log_warning(message):
    """Log a warning message to both file and console."""
    logger.warning(message)

def log_debug(message):
    """Log a debug message to both file and console."""
    logger.debug(message)

def setup_flask_logging(app):
    """
    Configure Flask app logging to use the same logger that writes to file and console.
    This ensures Flask's built-in logs also go to the log file.
    """
    flask_logger = setup_logger("flask_app")
    app.logger.handlers = flask_logger.handlers
    app.logger.setLevel(logging.INFO)
    return app.logger