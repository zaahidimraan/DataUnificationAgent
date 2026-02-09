import os
import shutil
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.services import UnificationAgent

main_bp = Blueprint('main', __name__)
agent = UnificationAgent()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@main_bp.route('/process', methods=['POST'])
def process_files():
    if 'files' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))
    
    files = request.files.getlist('files')
    
    # Clean workspace
    upload_dir = current_app.config['UPLOAD_FOLDER']
    output_dir = current_app.config['OUTPUT_FOLDER']
    
    for folder in [upload_dir, output_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    saved_paths = []
    
    for file in files:
        if file.filename == '': continue
        
        # Validation
        if not allowed_file(file.filename):
            flash(f"Skipped '{file.filename}': Invalid file type.", "warning")
            continue

        try:
            filename = secure_filename(file.filename)
            path = os.path.join(upload_dir, filename)
            file.save(path)
            saved_paths.append(path)
        except Exception as e:
            flash(f"Error saving {file.filename}", "danger")

    if not saved_paths:
        flash('No valid files uploaded.', 'danger')
        return redirect(url_for('main.index'))

    # Run Agent
    try:
        success, result = agent.unify_data(saved_paths, output_dir)
        
        if success:
            flash("✅ Data Unification Complete!", "success")
            return render_template('index.html', download_file=result)
        else:
            flash(f"❌ Processing Error: {result}", "danger")
            
    except Exception as e:
        flash(f"System Error: {str(e)}", "danger")

    return redirect(url_for('main.index'))

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)