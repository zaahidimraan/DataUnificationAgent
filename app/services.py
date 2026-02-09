import pandas as pd
import os
import operator
from typing import Annotated, List, Dict, TypedDict, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from app.utils import setup_logger

logger = setup_logger("langgraph_agent")

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    # Inputs
    file_paths: List[str]
    dfs_sample_str: str  # String representation of headers & first 3 rows
    output_folder: str
    
    # Internal Reasoning State
    strategy: str        # The current plan
    identifiers: str     # Identified Keys (e.g., "ID in Sheet A matches Ref in Sheet B")
    schema: str          # The proposed final column structure
    
    # Control Flow
    iteration: int       # Loop counter
    is_satisfied: bool   # Stopping condition
    final_code: str      # Python code to execute the merge
    execution_result: str

# --- THE AGENT CLASS ---
class UnificationGraphAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0
        )
        self.graph = self._build_graph()

    def _load_samples(self, file_paths):
        """Helper to load lightweight samples for the LLM."""
        context = ""
        for path in file_paths:
            try:
                filename = os.path.basename(path)
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    sample = df.head(3).to_markdown(index=False)
                    context += f"\nFILE: {filename}\nTYPE: CSV\nCOLUMNS: {list(df.columns)}\nSAMPLE:\n{sample}\n"
                else:
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet)
                        sample = df.head(3).to_markdown(index=False)
                        context += f"\nFILE: {filename}\nSHEET: {sheet}\nCOLUMNS: {list(df.columns)}\nSAMPLE:\n{sample}\n"
            except Exception as e:
                logger.error(f"Error loading sample {path}: {e}")
        return context

    # --- NODES ---

    def node_strategy_maker(self, state: AgentState):
        """Analyzes state and decides plan."""
        iteration = state.get("iteration", 0)
        current_schema = state.get("schema", "None yet")
        
        prompt = f"""
        You are the STRATEGY MAKER for a Data Unification Task.
        
        DATA CONTEXT:
        {state['dfs_sample_str']}
        
        CURRENT STATUS:
        - Iteration: {iteration}/3
        - Current Proposed Schema: {current_schema}
        
        YOUR TASK:
        1. Analyze the data relationships.
        2. if 'Current Proposed Schema' is "None yet" or seems incomplete/wrong, propose a HIGH-LEVEL MERGE STRATEGY.
        3. If the schema looks solid and covers all data needs, Output "SATISFIED".
        
        OUTPUT FORMAT:
        Start your response with either "SATISFIED" or "PLAN: [Your detailed strategy here]".
        """
        response = self.llm.invoke(prompt).content
        
        is_satisfied = False
        strategy_text = response
        
        if "SATISFIED" in response.upper() or iteration >= 3:
            is_satisfied = True
        
        return {
            "strategy": strategy_text,
            "is_satisfied": is_satisfied,
            "iteration": iteration + 1
        }

    def node_identifier(self, state: AgentState):
        """Finds linking keys."""
        prompt = f"""
        You are the IDENTIFIER.
        
        STRATEGY: {state['strategy']}
        DATA CONTEXT: {state['dfs_sample_str']}
        
        YOUR TASK:
        Identify the Primary Keys and Foreign Keys that link these files.
        - Look for exact name matches (e.g., 'id' == 'id')
        - Look for semantic matches (e.g., 'Employee_ID' == 'Emp_Ref')
        
        Output a clear list of mappings.
        """
        response = self.llm.invoke(prompt).content
        return {"identifiers": response}

    def node_schema_maker(self, state: AgentState):
        """Defines the target table structure."""
        prompt = f"""
        You are the SCHEMA MAKER.
        
        STRATEGY: {state['strategy']}
        IDENTIFIED KEYS: {state['identifiers']}
        DATA CONTEXT: {state['dfs_sample_str']}
        
        YOUR TASK:
        Define the Final Schema for the Unified Master File.
        List exactly which columns from which files will be included.
        Resolve naming conflicts (e.g., rename 'Cost' to 'Repair_Cost' and 'Purchase_Cost').
        """
        response = self.llm.invoke(prompt).content
        return {"schema": response}

    def node_code_generator(self, state: AgentState):
        """Writes the Python code to perform the merge."""
        output_path = os.path.join(state['output_folder'], 'master_unified_data.xlsx').replace("\\", "/")
        
        prompt = f"""
        You are the CODE GENERATOR.
        
        Final Schema Plan: {state['schema']}
        Identifiers: {state['identifiers']}
        File Paths: {state['file_paths']}
        
        YOUR TASK:
        Write a robust Python script to:
        1. Load the files specified in 'File Paths' using pandas.
        2. Perform the merges/joins as described in the Schema Plan.
        3. Handle missing values (NaN) gracefully.
        4. Save the final dataframe to: '{output_path}'
        
        IMPORTANT:
        - Return ONLY the Python code. 
        - Do not use markdown blocks like ```python. Just the code.
        - Ensure you import pandas as pd.
        """
        code = self.llm.invoke(prompt).content.strip().replace("```python", "").replace("```", "")
        return {"final_code": code}

    def node_executor(self, state: AgentState):
        """Executes the generated code."""
        code = state['final_code']
        try:
            # Dangerous execution - in prod use a sandbox
            exec_globals = {}
            exec(code, exec_globals)
            return {"execution_result": "SUCCESS"}
        except Exception as e:
            logger.error(f"Execution Error: {e}")
            return {"execution_result": f"FAILED: {str(e)}"}

    # --- GRAPH BUILDER ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("strategy_maker", self.node_strategy_maker)
        workflow.add_node("identifier", self.node_identifier)
        workflow.add_node("schema_maker", self.node_schema_maker)
        workflow.add_node("code_generator", self.node_code_generator)
        workflow.add_node("executor", self.node_executor)
        
        # Define Edges
        workflow.set_entry_point("strategy_maker")
        
        # Conditional Logic: Loop or Finish
        def check_satisfaction(state):
            if state["is_satisfied"]:
                return "code_generator"
            else:
                return "identifier"

        workflow.add_conditional_edges(
            "strategy_maker",
            check_satisfaction,
            {
                "code_generator": "code_generator",
                "identifier": "identifier"
            }
        )
        
        workflow.add_edge("identifier", "schema_maker")
        workflow.add_edge("schema_maker", "strategy_maker") # Loop back to check
        workflow.add_edge("code_generator", "executor")
        workflow.add_edge("executor", END)
        
        return workflow.compile()

    def run(self, file_paths, output_folder):
        """Main entry point."""
        # 1. Prepare Context
        samples = self._load_samples(file_paths)
        
        # 2. Initialize State
        initial_state = {
            "file_paths": file_paths,
            "dfs_sample_str": samples,
            "output_folder": output_folder,
            "iteration": 0,
            "schema": "None yet",
            "is_satisfied": False
        }
        
        # 3. Run Graph
        final_state = self.graph.invoke(initial_state)
        
        # 4. Check Result
        if final_state["execution_result"] == "SUCCESS":
            return True, "master_unified_data.xlsx"
        else:
            return False, final_state["execution_result"]