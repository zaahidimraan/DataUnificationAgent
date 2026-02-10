from flask import Flask
from config import Config
from app.utils import setup_flask_logging, log_info

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Setup Logging for Flask
    setup_flask_logging(app)
    log_info("=" * 60)
    log_info("🚀 DATA UNIFICATION AGENT STARTED")
    log_info("=" * 60)

    # Register Blueprints (Routes)
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app