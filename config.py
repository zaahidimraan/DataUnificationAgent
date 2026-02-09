import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
    
    # ADDED 'csv' to allowed extensions
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
    
    @staticmethod
    def init_folders():
        folders = [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER, Config.LOG_FOLDER]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

Config.init_folders()