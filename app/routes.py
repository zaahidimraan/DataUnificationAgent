import os
import shutil
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.services import UnificationGraphAgent
from app.utils import log_info, log_error, log_warning

main_bp = Blueprint('main', __name__)

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
    
    if 'files' not in request.files:
        log_error("❌ No files uploaded")
        flash('No file part', 'danger')
        return redirect(url_for('main.index'))
    
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

    # 3. Run LangGraph Agent
    try:
        log_info("🔄 Starting LangGraph agent for data unification")
        # The agent now handles the loop internally
        success, result = agent.run(saved_paths, output_dir)
        
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