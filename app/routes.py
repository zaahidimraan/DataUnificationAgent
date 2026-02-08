"""
API Endpoints for the Data Unification Agent.
"""
import os
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from app.services import DataUnificationAgent
from app.utils import allowed_file, log_info, log_error

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html')

@bp.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_paths.append(filepath)
                log_info(f"File uploaded: {filename}")
            else:
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
        
        return jsonify({
            'message': 'Files uploaded successfully',
            'files': [os.path.basename(p) for p in uploaded_paths],
            'count': len(uploaded_paths)
        }), 200
        
    except Exception as e:
        log_error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/unify', methods=['POST'])
def unify_data():
    """Process and unify uploaded data files."""
    try:
        data = request.get_json()
        file_names = data.get('files', [])
        
        if not file_names:
            return jsonify({'error': 'No files specified'}), 400
        
        # Get full paths
        file_paths = [
            os.path.join(current_app.config['UPLOAD_FOLDER'], f) 
            for f in file_names
        ]
        
        # Process with agent
        agent = DataUnificationAgent()
        result = agent.unify_data(file_paths)
        
        if result['success']:
            log_info(f"Data unified successfully: {result['output_file']}")
            return jsonify(result), 200
        else:
            log_error(f"Unification failed: {result.get('error')}")
            return jsonify(result), 500
            
    except Exception as e:
        log_error(f"Unification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/download/<filename>')
def download_file(filename):
    """Download a processed file."""
    try:
        filepath = os.path.join(current_app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(filepath):
            log_info(f"File downloaded: {filename}")
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        log_error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/api/status')
def status():
    """Check application status."""
    return jsonify({
        'status': 'running',
        'version': '1.0.0'
    }), 200
