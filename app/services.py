import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from app.utils import setup_logger

logger = setup_logger("intelligent_agent")

# Global storage (In a real app, use Redis/Database)
DATA_MEMORY = {}

class IntelligentDataAgent:
    def __init__(self, api_key, model_name="gemini-2.0-flash-exp"):
        """
        Initialize the agent with a dynamic API Key and Model.
        """
        self.api_key = api_key
        self.model_name = model_name
        
        # Initialize the LLM with the specific key and model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0,
            convert_system_message_to_human=True
        )

    def ingest_files(self, file_paths):
        """
        Phase 1: Ingest, Detect IDs, and Register Dataframes.
        """
        global DATA_MEMORY
        DATA_MEMORY.clear()
        metadata_log = []

        if not file_paths:
            return ["❌ No files provided for ingestion"]

        for path in file_paths:
            if not os.path.exists(path):
                continue
            
            filename = os.path.basename(path)
            try:
                # Handle CSV and Excel
                if filename.endswith('.csv'):
                    xl = None
                    # dict of {filename: df} for consistency
                    dfs = {filename: pd.read_csv(path)}
                else:
                    xl = pd.ExcelFile(path)
                    dfs = {sheet: xl.parse(sheet) for sheet in xl.sheet_names}

                for sheet_name, df in dfs.items():
                    if df.empty: continue
                    
                    # Clean headers
                    df.columns = df.columns.astype(str).str.strip()

                    # Ask Gemini to find the ID
                    head_sample = df.head(3).to_markdown(index=False)
                    prompt = f"""
                    Analyze this data sample from '{filename}' (Sheet: '{sheet_name}'):
                    {head_sample}
                    
                    Task: Identify the unique 'Identifier' column (Primary Key).
                    Return ONLY the column name string.
                    """
                    
                    try:
                        response = self.llm.invoke(prompt)
                        id_col = response.content.strip()
                        
                        if id_col not in df.columns:
                            id_col = df.columns[0] # Fallback

                        # Store in Memory
                        unique_name = f"{filename}_{sheet_name}"
                        DATA_MEMORY[unique_name] = df
                        metadata_log.append(f"✅ Loaded '{unique_name}'. ID: '{id_col}'")
                        
                    except Exception as e:
                        logger.error(f"LLM Error on {sheet_name}: {e}")
                        metadata_log.append(f"⚠️ Error analyzing '{sheet_name}': {e}")

            except Exception as e:
                metadata_log.append(f"❌ Failed to read {filename}: {e}")
        
        return metadata_log

    def query_data(self, user_query):
        """
        Phase 2: The Analyst.
        Returns: (Success_Bool, Result_Text_or_Error)
        """
        if not DATA_MEMORY:
            return False, "No data loaded. Please upload files first."

        df_list = list(DATA_MEMORY.values())
        
        agent = create_pandas_dataframe_agent(
            self.llm,
            df_list,
            verbose=True,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )

        output_path = os.path.join(os.getcwd(), 'outputs', 'query_result.xlsx')
        
        full_prompt = f"""
        You are an Expert Data Analyst using {self.model_name}.
        You have access to {len(df_list)} tables.
        
        User Query: "{user_query}"
        
        Steps:
        1. Identify common keys across tables.
        2. Merge/Aggregate data to answer the query.
        3. **CRITICAL**: Save the result to Excel at: '{output_path}'
        4. If successful, reply exactly: "SUCCESS: File generated."
        """
        
        try:
            response = agent.invoke(full_prompt)
            return True, response['output']
        except Exception as e:
            return False, f"Query failed: {str(e)}"