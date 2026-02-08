"""
Core Logic for the Data Unification Agent.
"""
import os
import pandas as pd
from datetime import datetime
from flask import current_app
from app.utils import log_info, log_error

class DataUnificationAgent:
    """Agent responsible for unifying data from multiple sources."""
    
    def __init__(self):
        """Initialize the Data Unification Agent."""
        self.supported_formats = ['csv', 'json', 'xlsx', 'xls']
    
    def read_file(self, filepath):
        """
        Read a file and return a pandas DataFrame.
        
        Args:
            filepath (str): Path to the file to read
            
        Returns:
            pd.DataFrame: Data from the file
        """
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.csv':
                return pd.read_csv(filepath)
            elif file_ext == '.json':
                return pd.read_json(filepath)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        except Exception as e:
            log_error(f"Error reading file {filepath}: {str(e)}")
            raise
    
    def unify_data(self, file_paths):
        """
        Unify data from multiple files into a single dataset.
        
        Args:
            file_paths (list): List of file paths to unify
            
        Returns:
            dict: Result with success status and output file path or error
        """
        try:
            log_info(f"Starting data unification for {len(file_paths)} files")
            
            # Read all files
            dataframes = []
            for filepath in file_paths:
                log_info(f"Reading file: {os.path.basename(filepath)}")
                df = self.read_file(filepath)
                dataframes.append(df)
            
            # Unify data (concatenate with outer join)
            unified_df = pd.concat(dataframes, ignore_index=True, sort=False)
            
            # Generate output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'unified_data_{timestamp}.csv'
            output_path = os.path.join(
                current_app.config['OUTPUT_FOLDER'], 
                output_filename
            )
            
            # Save unified data
            unified_df.to_csv(output_path, index=False)
            
            log_info(f"Data unified successfully: {output_filename}")
            
            return {
                'success': True,
                'output_file': output_filename,
                'rows': len(unified_df),
                'columns': len(unified_df.columns),
                'column_names': list(unified_df.columns)
            }
            
        except Exception as e:
            error_msg = f"Data unification failed: {str(e)}"
            log_error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
