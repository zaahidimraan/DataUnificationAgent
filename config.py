import os

class Config:
    # Essential for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'super-secret-key-change-in-prod'
    
    # Get the base directory (project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Define Paths
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    LOG_FOLDER = os.path.join(BASE_DIR, 'logs')  # <--- This was missing!
    
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

    @staticmethod
    def init_app(app):
        """Create required folders if they don't exist."""
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.LOG_FOLDER]:
            os.makedirs(folder, exist_ok=True)