import os
import shutil
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.services import UnificationAgent

main_bp = Blueprint('main', __name__)

# Single global instance for simplicity (as requested)
agent = UnificationAgent()

@main_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main_bp.route('/process', methods=['POST'])
def process_files():
    """
    Handles Upload AND Processing in one step.
    """
    if 'files' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))
    
    files = request.files.getlist('files')
    
    # 1. Setup a clean workspace
    upload_dir = current_app.config['UPLOAD_FOLDER']
    output_dir = current_app.config['OUTPUT_FOLDER']
    
    # Clean previous run (Simple mode: One user, one state)
    for folder in [upload_dir, output_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    saved_paths = []
    
    # 2. Save Files
    for file in files:
        if file.filename == '': continue
        if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls', 'csv'}:
            filename = secure_filename(file.filename)
            path = os.path.join(upload_dir, filename)
            file.save(path)
            saved_paths.append(path)

    if not saved_paths:
        flash('No valid Excel/CSV files uploaded.', 'warning')
        return redirect(url_for('main.index'))

    # 3. Run Agent Automatically
    try:
        success, result = agent.unify_data(saved_paths, output_dir)
        
        if success:
            # Result is the filename
            flash("✅ Unification Complete! Your master file is ready.", "success")
            return render_template('index.html', download_file=result)
        else:
            # Result is the error message
            flash(f"❌ Error: {result}", "danger")
            
    except Exception as e:
        flash(f"System Error: {str(e)}", "danger")

    return redirect(url_for('main.index'))

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)