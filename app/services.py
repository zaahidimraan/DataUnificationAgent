import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from app.utils import setup_logger

logger = setup_logger("unification_engine")

class UnificationAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            convert_system_message_to_human=True
        )

    def _load_dataframe(self, path):
        """
        Smart Loader: Handles CSV and Excel (multiple sheets).
        Returns: Dict { 'filename_sheetname': dataframe }
        """
        loaded_dfs = {}
        filename = os.path.basename(path)
        name_only, ext = os.path.splitext(filename)
        ext = ext.lower()

        try:
            if ext == '.csv':
                # Read CSV
                df = pd.read_csv(path)
                key = f"{name_only}_csv"
                loaded_dfs[key] = df
                
            elif ext in ['.xlsx', '.xls']:
                # Read Excel (All Sheets)
                xl = pd.ExcelFile(path)
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    # Create a unique key for the agent to reference
                    key = f"{name_only}_{sheet}".replace(" ", "_")
                    loaded_dfs[key] = df
            else:
                logger.warning(f"Unsupported file type: {filename}")

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

        return loaded_dfs

    def unify_data(self, file_paths, output_folder):
        """
        1. Loads only headers/samples.
        2. Sends samples to LLM.
        3. Executes merge.
        """
        # --- Step 1: Load Dataframes ---
        master_registry = {} # Dict of {name: df}
        
        for path in file_paths:
            dfs = self._load_dataframe(path)
            master_registry.update(dfs)

        if not master_registry:
            return False, "No valid data found (CSV or Excel) in uploaded files."

        # Cleaning: Strip whitespace from all column headers
        for name, df in master_registry.items():
            df.columns = df.columns.astype(str).str.strip()

        # --- Step 2: Create Samples for LLM ---
        # We manually construct the context to ensure ONLY samples are considered.
        context_summary = ""
        for name, df in master_registry.items():
            # Taking only top 3 rows as sample
            sample = df.head(3).to_markdown(index=False) 
            context_summary += f"\n--- Table Name: {name} ---\nColumns: {list(df.columns)}\nSample Data:\n{sample}\n"

        # --- Step 3: Initialize Agent ---
        # We pass the list of dataframes. The agent functions map these to variables.
        df_list = list(master_registry.values())
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            df_list,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        output_file = 'master_unified_data.xlsx'
        output_path = os.path.join(output_folder, output_file)

        # --- Step 4: The 'Sample-Only' Prompt ---
        # We explicitly tell the LLM to use the context_summary we built.
        prompt = f"""
        You are a Data Architech. You have access to {len(df_list)} tables.
        
        I have analyzed the files and extracted these SAMPLES (Top 3 rows only):
        {context_summary}

        YOUR TASK:
        1. Look at the 'Columns' and 'Sample Data' above to identify how these tables relate.
        2. Find common keys (like 'ID', 'Email', 'Reference').
        3. Write Pandas code to MERGE these tables into a single Master DataFrame.
           - Start with the main 'Entity' table (like Users, Products, Properties).
           - Join 'Transaction/History' tables to it using the keys identified.
           - If tables have identical structures (e.g. data from Jan, data from Feb), append them first.
        4. Save the final merged dataframe to: '{output_path}'
        5. Return "SUCCESS_DONE" if the file is saved.
        """

        try:
            response = agent.invoke(prompt)
            
            if os.path.exists(output_path):
                return True, output_file
            else:
                return False, f"Agent finished but file was not saved. Output: {response['output']}"
                
        except Exception as e:
            return False, f"Unification failed: {str(e)}"