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
        
        log_info(f"📁 Restoring {len(saved_paths)} files from session")
        
    else:
        # No files and no choice
        log_error("❌ No files uploaded and no choice provided")
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))

    # Continue with processing
    output_dir = current_app.config['OUTPUT_FOLDER']

    # 3. Run LangGraph Agent
    try:
        log_info("🔄 Starting LangGraph agent for data unification")
        success, result, state = agent.run(saved_paths, output_dir, one_to_many_choice)
        
        # Check if one-to-many was detected and user hasn't chosen yet
        if not success and state.get("one_to_many_detected") and state.get("one_to_many_resolution") == "awaiting_user_choice":
            session_id = os.urandom(16).hex()
            _pending_uploads[session_id] = saved_paths
            
            log_info("⏸️  One-to-many detected - showing user options")
            log_info(f"   Stored session: {session_id}")
            
            flash("⚠️  One-to-many relationships detected in your data!", "warning")
            return render_template('index.html', 
                                 show_one_to_many_modal=True,
                                 one_to_many_detected=True,
                                 session_id=session_id)
        
        if success:
            log_info("✅ LangGraph agent processing completed successfully")
            # Check if single file or multiple files were generated
            output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
            log_info(f"📊 Generated {len(output_files)} output file(s)")
            
            if 'master_unified_data.xlsx' in output_files:
                # Single file case
                log_info("✅ Single unified file generated: master_unified_data.xlsx")
                flash("✅ Data unified successfully into single file.", "success")
                return render_template('index.html', download_file='master_unified_data.xlsx', single_file=True)
            elif len(output_files) > 1:
                # Multiple files case - create a zip
                import zipfile
                zip_path = os.path.join(output_dir, 'unified_data_output.zip')
                log_info(f"📦 Creating zip archive with {len(output_files)} files")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in output_files:
                        file_path = os.path.join(output_dir, file)
                        zipf.write(file_path, arcname=file)
                
                log_info("✅ Zip archive created successfully")
                flash("✅ Data processed. Multiple files generated (complex relationships detected).", "warning")
                return render_template('index.html', 
                                     download_file='unified_data_output.zip', 
                                     single_file=False,
                                     file_count=len(output_files))
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