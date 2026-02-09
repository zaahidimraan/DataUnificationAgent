import os
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from app.services import IntelligentDataAgent

main_bp = Blueprint('main', __name__)
agent = IntelligentDataAgent() # Persistent Agent Instance

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@main_bp.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))
    
    files = request.files.getlist('files')
    saved_paths = []
    errors = []

    for file in files:
        if file.filename == '':
            errors.append('Empty filename')
            continue
        
        # Validate file extension
        if not allowed_file(file.filename):
            errors.append(f"'{file.filename}' - Invalid file type. Only .xlsx, .xls, .csv allowed")
            continue
        
        try:
            filename = secure_filename(file.filename)
            path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            # Verify file actually exists before proceeding
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                errors.append(f"'{filename}' - File save failed or empty")
                continue
            
            saved_paths.append(path)
        except Exception as e:
            errors.append(f"'{file.filename}' - Error: {str(e)}")
    
    # Show any errors
    for error in errors:
        flash(error, 'warning')
    
    if not saved_paths:
        flash('No valid files were uploaded', 'danger')
        return redirect(url_for('main.index'))
    
    # Trigger Ingestion
    try:
        logs = agent.ingest_files(saved_paths)
        session['logs'] = logs # Store logs to show user
        flash(f"Successfully ingested {len(files)} files. Ready for queries!", "success")
    except Exception as e:
        flash(f"Error during ingestion: {str(e)}", "danger")

    return redirect(url_for('main.index'))

@main_bp.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query_text')
    
    if not user_query:
        return redirect(url_for('main.index'))

    try:
        # UNPACKING FIX: Capture both success status and message
        success, result_text = agent.query_data(user_query)
        
        if not success:
            # If agent failed (e.g., no data loaded), show warning
            flash(f"Agent Error: {result_text}", "warning")
            return render_template('index.html', logs=session.get('logs', []))

        # If success, check for the file
        output_file = 'query_result.xlsx'
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_file)
        
        if os.path.exists(output_path):
            return render_template('index.html', 
                                   logs=session.get('logs', []), 
                                   answer=result_text, 
                                   download_file=output_file)
        else:
            # Agent claimed success but didn't save the file
            flash(f"Agent replied: {result_text} (But no file was generated)", "warning")
            
    except Exception as e:
        flash(f"System Error: {str(e)}", "danger")

    return render_template('index.html', logs=session.get('logs', []))

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)