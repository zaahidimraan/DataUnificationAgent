import os
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from app.services import IntelligentDataAgent

main_bp = Blueprint('main', __name__)
agent = IntelligentDataAgent() # Persistent Agent Instance

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@main_bp.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No file part')
        return redirect(url_for('main.index'))
    
    files = request.files.getlist('files')
    saved_paths = []

    for file in files:
        if file.filename == '': continue
        filename = secure_filename(file.filename)
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        saved_paths.append(path)
    
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
        # The agent executes the logic and saves 'query_result.xlsx'
        result_text = agent.query_data(user_query)
        
        output_file = 'query_result.xlsx'
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_file)
        
        if os.path.exists(output_path):
            return render_template('index.html', 
                                   logs=session.get('logs', []), 
                                   answer=result_text, 
                                   download_file=output_file)
        else:
            flash(f"Agent reply: {result_text} (No file generated)", "warning")
            
    except Exception as e:
        flash(f"Query Error: {str(e)}", "danger")

    return render_template('index.html', logs=session.get('logs', []))

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)