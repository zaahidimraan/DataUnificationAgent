import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from app.utils import setup_logger

logger = setup_logger("intelligent_agent")

# Global storage for the session (In production, use Redis/SQL)
# Structure: {'sheet_name': dataframe}
DATA_MEMORY = {} 

class IntelligentDataAgent:
    def __init__(self):
        # Initialize Gemini 2.5 Flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            convert_system_message_to_human=True
        )

    def ingest_files(self, file_paths):
        """
        Phase 1: Ingest, Detect IDs, and Register Dataframes.
        """
        global DATA_MEMORY
        DATA_MEMORY.clear() # Clear old session data
        metadata_log = []

        if not file_paths:
            metadata_log.append("❌ No files provided for ingestion")
            return metadata_log

        for path in file_paths:
            # Validate file exists
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                metadata_log.append(f"❌ File not found: {path}")
                continue
            
            filename = os.path.basename(path)
            
            try:
                xl = pd.ExcelFile(path)
            except Exception as e:
                logger.error(f"Failed to read Excel file {path}: {e}")
                metadata_log.append(f"❌ Failed to read {filename}: {str(e)}")
                continue
            
            for sheet in xl.sheet_names:
                df = xl.parse(sheet)
                if df.empty: continue

                # Clean headers
                df.columns = df.columns.astype(str).str.strip()
                
                # --- INTELLIGENCE STEP 1: Identify the ID Column ---
                # We ask Gemini to look at the headers and first 3 rows to find the ID.
                head_sample = df.head(3).to_markdown(index=False)
                
                prompt = f"""
                Analyze this dataframe sample from file '{filename}', sheet '{sheet}':
                {head_sample}

                Task: Identify the unique 'Identifier' column (Primary Key). 
                - It is likely named 'ID', 'Code', 'Ref', 'No', or similar.
                - It must be unique per row if this is a master table, or a foreign key if this is a history table.
                
                Return ONLY the column name string. Nothing else.
                """
                
                try:
                    response = self.llm.invoke(prompt)
                    id_col = response.content.strip()
                    
                    # Verify if LLM was correct
                    if id_col not in df.columns:
                        logger.warning(f"LLM hallucinated column {id_col}. Using default.")
                        id_col = df.columns[0] # Fallback
                    
                    # Store in Memory with a unique name
                    # e.g., 'properties_Flat'
                    unique_sheet_name = f"{os.path.splitext(filename)[0]}_{sheet}"
                    DATA_MEMORY[unique_sheet_name] = df
                    
                    metadata_log.append(f"✅ Loaded '{unique_sheet_name}' ({len(df)} rows). ID column: '{id_col}'")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {sheet}: {e}")
                    metadata_log.append(f"⚠️ Error processing sheet '{sheet}': {str(e)}")
        
        # Final summary
        if DATA_MEMORY:
            metadata_log.append(f"\n✅ Total tables loaded: {len(DATA_MEMORY)}")
        else:
            metadata_log.append("❌ No data was successfully loaded. Check file formats and content.")
        
        return metadata_log

    def query_data(self, user_query):
            """
            Phase 2: The Analyst.
            Returns: (Success_Bool, Message_String)
            """
            if not DATA_MEMORY:
                return False, "No data loaded. Please upload files first."

            # Prepare the list of dataframes
            df_list = list(DATA_MEMORY.values())
            
            # Create Pandas Agent
            agent = create_pandas_dataframe_agent(
                self.llm,
                df_list,
                verbose=True,
                allow_dangerous_code=True,
                agent_executor_kwargs={"handle_parsing_errors": True}
            )

            output_path = os.path.join(os.getcwd(), 'outputs', 'query_result.xlsx')
            
            full_prompt = f"""
            You are an Expert Data Analyst. 
            You have access to {len(df_list)} dataframes.
            
            User Request: "{user_query}"
            
            COMMANDS:
            1. Find the common Identifier column across sheets.
            2. Merge the necessary dataframes to answer the query.
            3. **CRITICAL**: You MUST save the resulting dataframe to an Excel file at this path: '{output_path}'.
            4. If successful, simply say "SUCCESS: File generated."
            """
            
            try:
                response = agent.invoke(full_prompt)
                # Return True (Success) and the text response
                return True, response['output']
                
            except Exception as e:
                # Return False (Failure) and the error message
                return False, f"Query failed: {str(e)}"