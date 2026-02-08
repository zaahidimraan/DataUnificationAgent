"""
Flask App Factory for Data Unification Agent.
"""
import os
from flask import Flask
from config import config

def create_app(config_name='default'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Ensure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
    
    # Register blueprints/routes
    from app import routes
    app.register_blueprint(routes.bp)
    
    # Initialize logging
    from app.utils import setup_logging
    setup_logging(app)
    
    return app
