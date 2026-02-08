import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-12345'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')
    LOG_FOLDER = os.path.join(os.getcwd(), 'logs')
    ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

    # Ensure directories exist
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, LOG_FOLDER]:
        os.makedirs(folder, exist_ok=True)