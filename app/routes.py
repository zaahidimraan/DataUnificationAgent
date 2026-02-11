import os
import shutil
import json
import pickle
import tempfile
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from app.services import UnificationGraphAgent
from app.utils import log_info, log_error, log_warning

main_bp = Blueprint('main', __name__)

# Store uploaded file paths temporarily between requests
_pending_uploads = {}

# Initialize the LangGraph Agent
agent = UnificationGraphAgent()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/', methods=['GET'])
def index():
    log_info("📄 Index page accessed")
    return render_template('index.html')

@main_bp.route('/process', methods=['POST'])
def process_files():
    log_info("🚀 Data unification process started")
    
    # Check if this is a modal response (one_to_many_choice without files)
    one_to_many_choice = request.form.get('one_to_many_choice', '')
    has_files = 'files' in request.files and len(request.files.getlist('files')) > 0
    
    # Check for target schema inputs
    target_schema_text = request.form.get('target_schema_text', '').strip()
    has_target_file = 'target_schema_file' in request.files and request.files['target_schema_file'].filename != ''
    
    # Initialize variables that might be set in either branch
    target_schema_file_path = None
    saved_paths = []
    output_dir = current_app.config['OUTPUT_FOLDER']
    
    if has_files:
        # FIRST SUBMISSION: New file upload
        log_info("📝 Processing: FIRST SUBMISSION (file upload)")
        
        files = request.files.getlist('files')
        log_info(f"📥 Received {len(files)} file(s)")
        
        # 1. Setup Workspace
        upload_dir = current_app.config['UPLOAD_FOLDER']
        output_dir = current_app.config['OUTPUT_FOLDER']
        
        log_info(f"📁 Cleaning upload and output directories")
        # Reset directories for single-user mode
        for folder in [upload_dir, output_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

        saved_paths = []
        
        # 2. Save Files
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                log_warning(f"⚠️  Skipped invalid file: {file.filename}")
                continue
                
            try:
                filename = secure_filename(file.filename)
                path = os.path.join(upload_dir, filename)
                file.save(path)
                saved_paths.append(path)
                log_info(f"✅ File saved: {filename}")
            except Exception as e:
                log_error(f"❌ Error saving {file.filename}: {str(e)}")
                flash(f"Error saving {file.filename}", "danger")

        if not saved_paths:
            log_error("❌ No valid files to process")
            flash('No valid files uploaded.', 'danger')
            return redirect(url_for('main.index'))

        log_info(f"✅ {len(saved_paths)} file(s) saved successfully")
        
        # Handle target schema file if provided
        target_schema_file_path = None
        if has_target_file:
            target_file = request.files['target_schema_file']
            if allowed_file(target_file.filename):
                try:
                    filename = "target_schema_" + secure_filename(target_file.filename)
                    target_schema_file_path = os.path.join(upload_dir, filename)
                    target_file.save(target_schema_file_path)
                    log_info(f"✅ Target schema file saved: {filename}")
                except Exception as e:
                    log_error(f"❌ Error saving target schema file: {str(e)}")
                    flash(f"Error saving target schema file", "warning")
            else:
                log_warning(f"⚠️  Invalid target schema file format")
                flash("Invalid target schema file format", "warning")
        
        # Log target schema mode
        if target_schema_text or target_schema_file_path:
            log_info("🎯 TARGET MODE ENABLED")
            if target_schema_text:
                log_info(f"   Input type: Text description")
            if target_schema_file_path:
                log_info(f"   Input type: Template file")
        
    elif one_to_many_choice:
        # SECOND SUBMISSION: Modal response with user choice
        log_info("📝 Processing: SECOND SUBMISSION (modal response)")
        log_info(f"👤 User selected: {one_to_many_choice}")
        
        # Retrieve stored file paths
        session_id = request.form.get('session_id', '')
        if not session_id or session_id not in _pending_uploads:
            log_error("❌ Session data not found - files were lost")
            flash('Session expired. Please upload files again.', 'danger')
            return redirect(url_for('main.index'))
        
        saved_paths = _pending_uploads[session_id]
        output_dir = current_app.config['OUTPUT_FOLDER']
        
        # Restore target schema info from session if it was provided
        target_schema_file_path = _pending_uploads.get(f"{session_id}_target_file", None)
        target_schema_text = _pending_uploads.get(f"{session_id}_target_text", "")
        
        log_info(f"📁 Restoring {len(saved_paths)} files from session")
        
    else:
        # No files and no choice
        log_error("❌ No files uploaded and no choice provided")
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))

    # 3. Run LangGraph Agent
    try:
        log_info("🔄 Starting LangGraph agent for data unification")
        
        # Pass target schema parameters to agent
        success, result, state = agent.run(
            saved_paths, 
            output_dir, 
            one_to_many_choice,
            target_schema_file=target_schema_file_path,
            target_schema_text=target_schema_text if target_schema_text else None
        )
        
        # Check if one-to-many was detected and user hasn't chosen yet
        if not success and state.get("one_to_many_detected") and state.get("one_to_many_resolution") == "awaiting_user_choice":
            session_id = os.urandom(16).hex()
            _pending_uploads[session_id] = saved_paths
            
            # Store target schema info in session for restoration after modal
            if has_files:
                if target_schema_file_path:
                    _pending_uploads[f"{session_id}_target_file"] = target_schema_file_path
                if target_schema_text:
                    _pending_uploads[f"{session_id}_target_text"] = target_schema_text
            
            log_info("⏸️  One-to-many detected - showing user options")
            log_info(f"   Stored session: {session_id}")
            
            flash("⚠️  One-to-many relationships detected in your data!", "warning")
            return render_template('index.html', 
                                 show_one_to_many_modal=True,
                                 one_to_many_detected=True,
                                 session_id=session_id)
        
        if success:
            log_info("✅ LangGraph agent processing completed successfully")
            # ALWAYS expect single file output now
            output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
            log_info(f"📊 Generated {len(output_files)} output file(s)")
            
            # Check if target mode was used and if fallback occurred
            target_fallback = state.get('target_fallback_triggered', False)
            target_mode_used = state.get('target_mode_enabled', False)
            
            if 'master_unified_data.xlsx' in output_files:
                # Single file case
                log_info("✅ Single unified file generated: master_unified_data.xlsx")
                
                if target_mode_used:
                    if target_fallback:
                        flash("✅ Data unified successfully. ⚠️ Note: Target schema mapping failed after 3 attempts - used auto-generated schema instead.", "warning")
                    else:
                        flash("✅ Data unified successfully using your target schema!", "success")
                else:
                    flash("✅ Data unified successfully into single file.", "success")
                
                return render_template('index.html', 
                                     download_file='master_unified_data.xlsx', 
                                     single_file=True,
                                     target_mode_used=target_mode_used,
                                     target_fallback=target_fallback)
            else:
                log_info("✅ Processing complete")
                flash("✅ Processing complete. Check output folder.", "success")
                return render_template('index.html')
        else:
            log_error(f"❌ Agent processing failed: {result}")
            flash(f"❌ Processing Error: {result}", "danger")
            
    except Exception as e:
        log_error(f"❌ System Error: {str(e)}")
        flash(f"System Error: {str(e)}", "danger")

    return redirect(url_for('main.index'))

@main_bp.route('/download/<filename>')
def download_file(filename):
    log_info(f"📥 Download requested for: {filename}")
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)