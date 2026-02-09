import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from app.utils import setup_logger

logger = setup_logger("unification_engine")

class UnificationAgent:
    def __init__(self):
        # Using Gemini 2.5 Flash for speed and reasoning
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            convert_system_message_to_human=True
        )

    def unify_data(self, file_paths, output_folder):
        """
        Loads all files and automatically merges them into a master file.
        """
        df_list = []
        df_names = []
        
        # 1. Load all Excel sheets into DataFrames
        for path in file_paths:
            try:
                xl = pd.ExcelFile(path)
                filename = os.path.basename(path)
                
                for sheet in xl.sheet_names:
                    df = xl.parse(sheet)
                    if df.empty: continue
                    
                    # Basic cleaning
                    df.columns = df.columns.astype(str).str.strip()
                    
                    # Store DF and a helpful name for the LLM
                    df_list.append(df)
                    df_names.append(f"{filename} - Sheet: {sheet}")
                    
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                return False, f"Failed to read file: {os.path.basename(path)}"

        if not df_list:
            return False, "No valid data found in uploaded files."

        # 2. Create the Pandas Agent
        agent = create_pandas_dataframe_agent(
            self.llm,
            df_list,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        output_file = 'master_unified_data.xlsx'
        output_path = os.path.join(output_folder, output_file)

        # 3. The "Auto-Pilot" Prompt
        # We give the LLM the list of sheet names so it understands the context (e.g., "Flats" vs "Repairs")
        context_str = "\n".join(df_names)
        
        prompt = f"""
        You are an Autonomous Data Unification Bot.
        
        I have loaded {len(df_list)} data tables. Here are their sources:
        {context_str}
        
        YOUR MISSION:
        1. Analyze the columns to find common Identifiers (e.g., 'ID', 'Ref', 'Code').
        2. Merge ALL these tables into a single 'Master DataFrame'. 
           - If there are multiple master lists (like Flats and Villas), append/concat them first.
           - Then merge that combined list with any Transaction/History/Repair logs.
        3. Ensure no data is lost (use outer joins if necessary).
        4. **CRITICAL**: Save the final result to an Excel file at: '{output_path}'
        5. If successful, return the string: "SUCCESS_DONE"
        """

        try:
            response = agent.invoke(prompt)
            
            if os.path.exists(output_path):
                return True, output_file
            else:
                return False, f"Agent finished but file was not saved. Output: {response['output']}"
                
        except Exception as e:
            return False, f"Unification failed: {str(e)}"