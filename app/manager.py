import os
import shutil
import uuid
import pandas as pd
from flask import current_app

class SessionFileManager:
    """
    Manages files per user session to ensure isolation and persistence.
    Structure: uploads/{session_id}/{filename}
    """
    
    @staticmethod
    def get_session_dir(session_id):
        """Get or create the directory for a specific session."""
        base_dir = current_app.config['UPLOAD_FOLDER']
        session_dir = os.path.join(base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    @staticmethod
    def save_file(session_id, file_obj, filename):
        """Saves a file to the session directory."""
        session_dir = SessionFileManager.get_session_dir(session_id)
        file_path = os.path.join(session_dir, filename)
        file_obj.save(file_path)
        return file_path

    @staticmethod
    def list_files(session_id):
        """Returns a list of files currently in the session."""
        session_dir = SessionFileManager.get_session_dir(session_id)
        if not os.path.exists(session_dir):
            return []
        return [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f))]

    @staticmethod
    def delete_file(session_id, filename):
        """Removes a specific file from the session."""
        session_dir = SessionFileManager.get_session_dir(session_id)
        file_path = os.path.join(session_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    @staticmethod
    def clear_session(session_id):
        """Resets the session by deleting all files."""
        session_dir = SessionFileManager.get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            return True
        return False
        
    @staticmethod
    def get_all_file_paths(session_id):
        """Helper to get full paths for the Agent."""
        session_dir = SessionFileManager.get_session_dir(session_id)
        files = SessionFileManager.list_files(session_id)
        return [os.path.join(session_dir, f) for f in files]