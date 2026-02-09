import os
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from app.services import IntelligentDataAgent

main_bp = Blueprint('main', __name__)

# Helper to get the Active Agent based on Session
def get_active_agent():
    # 1. Get API Key
    api_keys = session.get('api_keys', [])
    active_index = session.get('active_key_index', 0)
    
    if not api_keys:
        # Fallback to .env if no keys in session
        env_key = os.getenv("GOOGLE_API_KEY")
        if env_key:
            current_key = env_key
        else:
            raise ValueError("No API Key found. Please add one in Settings.")
    else:
        # Safe index access
        if active_index >= len(api_keys): active_index = 0
        current_key = api_keys[active_index]

    # 2. Get Model
    current_model = session.get('active_model', 'gemini-2.0-flash-exp')

    return IntelligentDataAgent(api_key=current_key, model_name=current_model)

@main_bp.route('/', methods=['GET'])
def index():
    # Initialize session defaults if not present
    if 'api_keys' not in session: session['api_keys'] = []
    if 'active_key_index' not in session: session['active_key_index'] = 0
    if 'active_model' not in session: session['active_model'] = 'gemini-2.0-flash-exp'
    
    return render_template('index.html', 
                           api_keys=session['api_keys'], 
                           active_index=session['active_key_index'],
                           active_model=session['active_model'])

@main_bp.route('/settings/update', methods=['POST'])
def update_settings():
    # 1. Handle Model Change
    selected_model = request.form.get('model_name')
    if selected_model:
        session['active_model'] = selected_model
    
    # 2. Handle API Key Add/Select
    action = request.form.get('action')
    
    if action == 'add_key':
        new_key = request.form.get('new_api_key')
        if new_key:
            keys = session.get('api_keys', [])
            keys.append(new_key.strip())
            session['api_keys'] = keys
            session['active_key_index'] = len(keys) - 1 # Auto-select new key
            flash("API Key added and selected.", "success")
            
    elif action == 'select_key':
        try:
            index = int(request.form.get('key_index'))
            session['active_key_index'] = index
            flash(f"Switched to API Key #{index+1}", "info")
        except:
            pass
            
    elif action == 'delete_key':
        try:
            index = int(request.form.get('key_index'))
            keys = session.get('api_keys', [])
            if 0 <= index < len(keys):
                keys.pop(index)
                session['api_keys'] = keys
                session['active_key_index'] = 0
                flash("API Key removed.", "warning")
        except:
            pass

    return redirect(url_for('main.index'))

@main_bp.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files: return redirect(url_for('main.index'))
    
    files = request.files.getlist('files')
    saved_paths = []

    for file in files:
        if file.filename == '': continue
        filename = secure_filename(file.filename)
        path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        saved_paths.append(path)
    
    try:
        # DYNAMIC AGENT INSTANTIATION
        agent = get_active_agent()
        logs = agent.ingest_files(saved_paths)
        session['logs'] = logs
        flash(f"Ingested {len(saved_paths)} files using {agent.model_name}.", "success")
    except Exception as e:
        flash(f"Ingestion Error: {str(e)}", "danger")

    return redirect(url_for('main.index'))

@main_bp.route('/query', methods=['POST'])
def query():
    user_query = request.form.get('query_text')
    if not user_query: return redirect(url_for('main.index'))

    try:
        # DYNAMIC AGENT INSTANTIATION
        agent = get_active_agent()
        success, result_text = agent.query_data(user_query)
        
        output_file = 'query_result.xlsx'
        output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_file)
        
        if os.path.exists(output_path):
            return render_template('index.html', 
                                   logs=session.get('logs', []), 
                                   answer=result_text, 
                                   download_file=output_file,
                                   api_keys=session.get('api_keys', []),
                                   active_index=session.get('active_key_index', 0),
                                   active_model=session.get('active_model'))
        else:
            flash(f"Agent ({agent.model_name}) replied: {result_text}", "warning")
            
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return render_template('index.html', 
                           logs=session.get('logs', []),
                           api_keys=session.get('api_keys', []),
                           active_index=session.get('active_key_index', 0),
                           active_model=session.get('active_model'))

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)