import os
from flask import Blueprint, render_template, request, current_app, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename
from app.services import DataUnificationAgent

main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    logs = []
    download_filename = None
    error = None

    if request.method == 'POST':
        # 1. Validation
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # 2. Trigger the Intelligent Agent
            try:
                agent = DataUnificationAgent(upload_path)
                
                # Run the logic (Phase 2 code)
                unified_df, execution_logs = agent.run()
                logs = execution_logs # Pass logs to UI
                
                if unified_df is not None:
                    # 3. Save Output
                    output_filename = f"Unified_{filename}"
                    if output_filename.endswith('.csv'):
                        output_filename = output_filename.replace('.csv', '.xlsx')
                    else:
                        # Ensure extension is .xlsx
                        base = os.path.splitext(output_filename)[0]
                        output_filename = base + ".xlsx"

                    output_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_filename)
                    unified_df.to_excel(output_path, index=False)
                    
                    download_filename = output_filename
                    flash('File processed successfully!', 'success')
                else:
                    error = "Agent could not merge data. Check logs."
            
            except Exception as e:
                error = f"Critical Error: {str(e)}"
                logs.append(str(e))

    return render_template('index.html', logs=logs, download_filename=download_filename, error=error)

@main_bp.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename, as_attachment=True)