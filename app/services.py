import pandas as pd
import numpy as np
import re
import os
import Levenshtein
from sklearn.cluster import AffinityPropagation
from app.utils import setup_logger

# Initialize Logger
logger = setup_logger("agent_service")

class DataUnificationAgent:
    def __init__(self, file_path):
        self.file_path = file_path
        self.xl = None
        self.dfs = {}          # {sheet_name: dataframe}
        self.id_map = {}       # {sheet_name: id_column_name}
        self.col_mapping = {}  # {old_col_name: new_unified_name}
        self.logs = []         # Internal audit log for the user

    def log(self, message):
        """Helper to log to both system and user audit trail."""
        logger.info(message)
        self.logs.append(message)

    def load_data(self):
        """Step 1: Ingest Data with Error Handling."""
        try:
            self.log(f"Loading file: {os.path.basename(self.file_path)}")
            # Handle both Excel and CSV
            if self.file_path.endswith('.csv'):
                self.dfs['Sheet1'] = pd.read_csv(self.file_path)
            else:
                self.xl = pd.ExcelFile(self.file_path)
                for sheet in self.xl.sheet_names:
                    try:
                        df = self.xl.parse(sheet)
                        if not df.empty:
                            # Clean up: strip whitespace from headers
                            df.columns = df.columns.astype(str).str.strip()
                            self.dfs[sheet] = df
                            self.log(f"  -> Loaded sheet '{sheet}' with {len(df)} rows.")
                        else:
                            self.log(f"  -> Skipped empty sheet '{sheet}'.")
                    except Exception as e:
                        logger.error(f"Failed to parse sheet {sheet}: {str(e)}")
            
            if not self.dfs:
                raise ValueError("No valid data found in file.")
                
        except Exception as e:
            logger.critical(f"Fatal error loading file: {str(e)}")
            raise e

    def _detect_id_column(self, df, sheet_name):
        """
        Step 2: The Detective (Heuristic ID Finder).
        Returns the name of the column that is most likely the ID.
        """
        best_col = None
        max_score = -1
        
        # Keywords to boost score
        id_keywords = ['id', 'code', 'ref', 'no', 'sku', 'key', 'identifier']

        for col in df.columns:
            score = 0
            col_lower = str(col).lower()
            
            # 1. Name Match (Medium weight)
            if any(k in col_lower for k in id_keywords):
                score += 40
            
            # 2. Uniqueness (Heavy weight) - The most important factor
            n_unique = df[col].nunique()
            n_total = len(df)
            if n_total == 0: continue
            
            uniqueness_ratio = n_unique / n_total
            
            if uniqueness_ratio == 1.0:
                score += 100  # Perfect unique key
            elif uniqueness_ratio > 0.9:
                score += 50   # Near unique
            else:
                score -= 20   # Penalize non-unique columns

            # 3. Data Type (Prefer strings/alphanumeric over floats)
            # Sample the first non-null value
            first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            if isinstance(first_valid, str):
                if re.search(r'\d', first_valid) and re.search(r'[a-zA-Z]', first_valid):
                    score += 20 # Alphanumeric bonus (e.g., "FLAT-01")

            if score > max_score:
                max_score = score
                best_col = col
        
        # Fallback: If score is too low, we might return None or the first column
        if max_score < 30:
            self.log(f"  -> Warning: Low confidence ID detection for '{sheet_name}'. Best guess: {best_col}")
        
        return best_col

    def _cluster_headers(self):
        """
        Step 3: The Linguist (Clustering).
        Maps similar headers (Price, Cost) to a single standard name.
        """
        self.log("Analyze column headers for semantic mapping...")
        
        all_headers = set()
        for df in self.dfs.values():
            all_headers.update(df.columns)
        
        headers_list = list(all_headers)
        if not headers_list: return

        # Calculate Levenshtein Distance Matrix
        words = np.asarray(headers_list)
        try:
            similarity_matrix = -1 * np.array([[Levenshtein.distance(w1, w2) for w1 in words] for w2 in words])
            
            # Affinity Propagation (No need to specify number of clusters)
            affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=42)
            affprop.fit(similarity_matrix)
            
            for cluster_id in np.unique(affprop.labels_):
                cluster_members = words[np.nonzero(affprop.labels_ == cluster_id)]
                if len(cluster_members) > 1:
                    # Pick the shortest name as the "Standard"
                    standard_name = min(cluster_members, key=len)
                    self.log(f"  -> Mapping {cluster_members} to '{standard_name}'")
                    for member in cluster_members:
                        self.col_mapping[member] = standard_name
        except Exception as e:
            logger.error(f"Clustering failed: {e}. Proceeding without mapping.")

    def run(self):
        """
        Step 4: The Judge (Execution Pipeline).
        """
        try:
            self.load_data()
            self._cluster_headers()
            
            processed_dfs = []
            global_id_set = set()
            
            for sheet_name, df in self.dfs.items():
                # A. Apply Column Mapping
                if self.col_mapping:
                    df = df.rename(columns=self.col_mapping)

                # B. Detect ID
                id_col = self._detect_id_column(df, sheet_name)
                if not id_col:
                    self.log(f"Skipping sheet '{sheet_name}': No ID column detected.")
                    continue
                
                self.log(f"Sheet '{sheet_name}': ID detected as '{id_col}'")
                
                # C. Normalize ID Column Name
                # Rename the detected ID column to 'Global_ID' for merging
                df = df.rename(columns={id_col: 'Global_ID'})
                df['Global_ID'] = df['Global_ID'].astype(str) # Ensure string for matching

                # D. Conflict Resolution (The "Suffix Rule")
                # Check if ANY ID in this sheet already exists in our global tracker
                current_ids = set(df['Global_ID'])
                
                # Intersection check
                conflicts = current_ids.intersection(global_id_set)
                
                if conflicts:
                    self.log(f"  !! Conflict detected in '{sheet_name}' ({len(conflicts)} overlaps). Appending suffix.")
                    # Apply suffix to ALL IDs in this sheet to ensure consistency
                    df['Global_ID'] = df['Global_ID'] + "_" + sheet_name
                
                # Update global tracker
                global_id_set.update(df['Global_ID'])
                
                # Add Metadata
                df['Source_Sheet'] = sheet_name
                processed_dfs.append(df)

            # E. Final Merge
            if processed_dfs:
                self.log("Merging all processed sheets...")
                # ignore_index=True resets the index, but we keep Global_ID as a column
                master_df = pd.concat(processed_dfs, ignore_index=True)
                
                # Reorder columns: Global_ID first
                cols = ['Global_ID', 'Source_Sheet'] + [c for c in master_df.columns if c not in ['Global_ID', 'Source_Sheet']]
                master_df = master_df[cols]
                
                return master_df, self.logs
            else:
                self.log("No data could be processed.")
                return None, self.logs

        except Exception as e:
            logger.critical(f"Process failed: {str(e)}")
            self.log(f"CRITICAL ERROR: {str(e)}")
            return None, self.logs