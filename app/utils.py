"""
Helper functions for logging and configuration.
"""
import os
import logging
from datetime import datetime
from flask import current_app

def allowed_file(filename):
    """
    Check if a file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def setup_logging(app):
    """
    Configure logging for the application.
    
    Args:
        app: Flask application instance
    """
    log_dir = app.config['LOG_FOLDER']
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"app_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    app.logger.setLevel(logging.INFO)
    app.logger.info("Application logging initialized")

def log_info(message):
    """Log an info message."""
    current_app.logger.info(message)

def log_error(message):
    """Log an error message."""
    current_app.logger.error(message)

def log_warning(message):
    """Log a warning message."""
    current_app.logger.warning(message)
