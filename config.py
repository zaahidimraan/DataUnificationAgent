import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    
    # Get the base directory (project root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
    
    # Ensure directories exist at app startup
    @staticmethod
    def init_folders():
        """Create required folders if they don't exist."""
        folders = [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.LOG_FOLDER]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Folder ready: {folder}")

# Initialize folders when config is loaded
Config.init_folders()