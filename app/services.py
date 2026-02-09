import pandas as pd
import os
import json
from typing import Annotated, List, Dict, TypedDict, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.utils import setup_logger

logger = setup_logger("langgraph_agent")

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    # Inputs
    file_paths: List[str]
    dfs_sample_str: str  
    output_folder: str
    
    # Phase 1: Identification State
    identifiers: str
    id_feedback: str
    id_confidence: float
    id_retries: int
    
    # Phase 2: Schema State
    schema: str
    schema_feedback: str
    schema_confidence: float
    schema_retries: int
    
    # Phase 3: Execution State
    final_code: str
    execution_result: str
    execution_error: str
    execution_retries: int

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
        logger.info("📁 Loading file samples for analysis...")
        context = ""
        file_count = 0
        sheet_count = 0
        
        for path in file_paths:
            try:
                filename = os.path.basename(path)
                # Handle CSV
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    sample = df.head(3).to_markdown(index=False)
                    context += f"\nFILE: {filename}\nTYPE: CSV\nCOLUMNS: {list(df.columns)}\nSAMPLE:\n{sample}\n"
                    file_count += 1
                    logger.info(f"  ✓ Loaded CSV: {filename} ({len(df.columns)} columns, {len(df)} rows)")
                # Handle Excel
                elif path.endswith(('.xlsx', '.xls')):
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet)
                        sample = df.head(3).to_markdown(index=False)
                        context += f"\nFILE: {filename}\nSHEET: {sheet}\nCOLUMNS: {list(df.columns)}\nSAMPLE:\n{sample}\n"
                        sheet_count += 1
                        logger.info(f"  ✓ Loaded Excel: {filename} - Sheet '{sheet}' ({len(df.columns)} columns, {len(df)} rows)")
                    file_count += 1
            except Exception as e:
                logger.error(f"  ✗ Error loading sample {path}: {e}")
        
        logger.info(f"📊 Sample loading complete: {file_count} files, {sheet_count} sheets analyzed")
        return context

    # ============================================================
    # PHASE 1: IDENTIFICATION LOOP
    # ============================================================

    def node_identifier(self, state: AgentState):
        """Identifies linking keys (single or composite)."""
        retry_count = state.get("id_retries", 0)
        previous_feedback = state.get("id_feedback", "")
        
        logger.info("="*60)
        logger.info("🔍 PHASE 1: IDENTIFICATION")
        logger.info(f"Attempt #{retry_count + 1}")
        if previous_feedback:
            logger.info(f"📝 Addressing feedback from previous attempt")
        logger.info("="*60)
        
        feedback_context = ""
        if previous_feedback:
            feedback_context = f"\n**PREVIOUS ATTEMPT FEEDBACK (Retry {retry_count}):**\n{previous_feedback}\n\nYou MUST address these concerns in your new proposal."
        
        prompt = f"""You are a relational database expert. Analyze the data and identify linking keys for unification.

DATA CONTEXT:
{state['dfs_sample_str']}

TASK:
1. Identify the entity with the MAXIMUM number of identifier columns - this defines your composite key structure dimensions
2. For entities with fewer identifiers, map them to this structure using constant padding (e.g., '0' or 'NA')
3. If different entity types might have overlapping ID values, propose a prefix strategy to prevent collisions
4. Be specific about which source column maps to which key position for every file/sheet

{feedback_context}

OUTPUT FORMAT (for EACH file/sheet):
FILE: <filename>
SHEET: <sheet>
Key_Mappings: <describe how source columns map to the composite key structure>
Prefix: <prefix if needed, or 'None'>
---
"""
        
        logger.info("🤖 Calling LLM to identify keys...")
        response = self.llm.invoke(prompt).content
        logger.info("✅ Identifier proposal generated")
        logger.info(f"📋 Proposal length: {len(response)} characters")
        
        return {
            "identifiers": response,
            "id_retries": retry_count + 1
        }

    def node_id_evaluator(self, state: AgentState):
        """Evaluates the identification proposal with a confidence score."""
        logger.info("⚖️  Evaluating identification proposal...")
        
        system_prompt = """You are a Senior Data Architect. Evaluate identification strategies with strict standards. Only give high confidence (90+) if flawless."""
        
        evaluation_prompt = f"""Evaluate this identification strategy:

PROPOSED IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

CRITERIA (Total 100 points):
1. Completeness: All files/sheets mapped? (30 pts)
2. Consistency: Uniform composite key structure? (25 pts)
3. Collision Prevention: No ID conflicts? (25 pts)
4. Clarity: Unambiguous and implementable? (20 pts)

SCORING: 90-100=Production ready, 70-89=Minor issues, 50-69=Needs work, <50=Major flaws

OUTPUT (JSON only):
{{
  "confidence_score": <0-100>,
  "feedback_text": "<Specific issues if score < 90>"
}}
"""
        
        # Create a combined prompt with system context
        full_prompt = f"{system_prompt}\n\n{evaluation_prompt}"
        response = self.llm.invoke(full_prompt).content
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            eval_result = json.loads(json_str)
            confidence = float(eval_result.get("confidence_score", 0))
            feedback = eval_result.get("feedback_text", "")
            
            logger.info(f"📊 Evaluation Score: {confidence}/100")
            if confidence >= 90:
                logger.info("✅ Identification approved! Moving to next phase.")
            else:
                logger.warning(f"⚠️  Identification needs improvement (Score: {confidence})")
                logger.warning(f"💬 Feedback: {feedback[:200]}...") if len(feedback) > 200 else logger.warning(f"💬 Feedback: {feedback}")
            
        except Exception as e:
            logger.error(f"❌ Failed to parse evaluator response: {e}")
            confidence = 0.0
            feedback = f"Evaluator response parsing failed: {response}"
        
        return {
            "id_confidence": confidence,
            "id_feedback": feedback
        }

    # ============================================================
    # PHASE 2: SCHEMA LOOP
    # ============================================================

    def node_schema_maker(self, state: AgentState):
        """Defines the target unified table structure."""
        retry_count = state.get("schema_retries", 0)
        previous_feedback = state.get("schema_feedback", "")
        
        logger.info("="*60)
        logger.info("📐 PHASE 2: SCHEMA DESIGN")
        logger.info(f"Attempt #{retry_count + 1}")
        if previous_feedback:
            logger.info(f"📝 Addressing feedback from previous attempt")
        logger.info("="*60)
        
        feedback_context = ""
        if previous_feedback:
            feedback_context = f"\n**PREVIOUS ATTEMPT FEEDBACK (Retry {retry_count}):**\n{previous_feedback}\n\nYou MUST fix these issues."
        
        prompt = f"""You are a Schema Architect. Design the unified master table schema.

APPROVED IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

REQUIREMENTS:
1. Define standardized key columns based on the composite key structure identified (name, type, logic for each)
2. Define Master Unique ID formula that concatenates all key columns with delimiter
3. List all value columns to retain from source files with null handling strategy
4. Specify data type conversions and transformations needed

{feedback_context}

OUTPUT FORMAT:
# COMPOSITE KEY STRUCTURE
<List each key column with name, type, and logic>
Master_UID: <formula using identified keys>

# VALUE COLUMNS
- <column>: <source mapping, type, null handling>

# TRANSFORMATIONS
- <any special rules>
"""
        
        logger.info("🤖 Calling LLM to design schema...")
        response = self.llm.invoke(prompt).content
        logger.info("✅ Schema proposal generated")
        logger.info(f"📋 Schema length: {len(response)} characters")
        
        return {
            "schema": response,
            "schema_retries": retry_count + 1
        }

    def node_schema_evaluator(self, state: AgentState):
        """Evaluates the schema proposal."""
        logger.info("⚖️  Evaluating schema design...")
        
        system_prompt = """You are a Database Schema Expert. Evaluate schemas with strict ETL standards. Reject schemas lacking clarity or with logical flaws."""
        
        evaluation_prompt = f"""Evaluate this schema design:

PROPOSED SCHEMA:
{state['schema']}

IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

CRITERIA (Total 100 points):
1. Key Design: Composite key properly defined? (25 pts)
2. UID Formula: Correct and collision-free? (25 pts)
3. Column Coverage: All important columns included? (20 pts)
4. Type Safety: Data types/transformations specified? (15 pts)
5. Null Handling: Missing data strategy clear? (15 pts)

SCORING: 90-100=Production ready, 70-89=Minor improvements, 50-69=Gaps, <50=Redesign

OUTPUT (JSON only):
{{
  "confidence_score": <0-100>,
  "feedback_text": "<Specific issues if score < 90>"
}}
"""
        
        full_prompt = f"{system_prompt}\n\n{evaluation_prompt}"
        response = self.llm.invoke(full_prompt).content
        
        # Parse JSON response
        try:
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            eval_result = json.loads(json_str)
            confidence = float(eval_result.get("confidence_score", 0))
            feedback = eval_result.get("feedback_text", "")
            
            logger.info(f"📊 Evaluation Score: {confidence}/100")
            if confidence >= 90:
                logger.info("✅ Schema approved! Moving to code generation.")
            else:
                logger.warning(f"⚠️  Schema needs improvement (Score: {confidence})")
                logger.warning(f"💬 Feedback: {feedback[:200]}...") if len(feedback) > 200 else logger.warning(f"💬 Feedback: {feedback}")
            
        except Exception as e:
            logger.error(f"❌ Failed to parse schema evaluator response: {e}")
            confidence = 0.0
            feedback = f"Evaluator response parsing failed: {response}"
        
        return {
            "schema_confidence": confidence,
            "schema_feedback": feedback
        }

    # ============================================================
    # PHASE 3: EXECUTION & RECOVERY
    # ============================================================

    def node_code_generator(self, state: AgentState):
        """Generates executable Python code based on approved design."""
        logger.info("="*60)
        logger.info("💻 PHASE 3: CODE GENERATION")
        logger.info("="*60)
        
        safe_output_path = os.path.join(state['output_folder'], 'master_unified_data.xlsx').replace("\\", "/")
        
        prompt = f"""Generate Python code to unify the data based on approved design.

IDENTIFIERS:
{state['identifiers']}

SCHEMA:
{state['schema']}

FILES:
{state['file_paths']}

CRITICAL: Analyze data relationships first. Determine if single-file merge is appropriate.

DECISION CRITERIA:
- Single File: If all data shares same granularity (one-to-one relationships)
- Multiple Files: If data has different granularities (one-to-many, many-to-many relationships)

IF SINGLE FILE IS POSSIBLE:
1. Load all dataframes (handle CSV/Excel, iterate through sheets)
2. Normalize keys: Create standardized key columns for each dataframe
   - Use .get() for optional columns, fill with constant if missing
   - Convert all keys to string: .astype(str).str.strip()
   - Fill NaN values: .fillna('<constant>', inplace=True)
   - Apply prefixes if specified
3. Create MASTER_UID: Concatenate all key columns with '_' delimiter
4. Select columns: Keep [MASTER_UID, key_columns, value_columns]
5. Merge: Use pd.concat() to stack OR pd.merge() on MASTER_UID
6. Save: final_df.to_excel('{safe_output_path}', index=False)
7. Print "SUCCESS: Unified data saved to master_unified_data.xlsx"

IF SINGLE FILE IS IMPOSSIBLE (Different Granularities):
1. Create normalized structure with multiple related tables
2. Save as: master.xlsx, transactions.xlsx, details.xlsx, etc. in same output folder
3. Create a relationships.txt file explaining the schema and foreign keys
4. Print "SUCCESS: Data normalized into multiple files due to complex relationships. See relationships.txt for schema."

REQUIREMENTS:
- Detect data granularity conflicts (e.g., master records vs transaction history)
- Handle missing columns gracefully
- Convert all keys to string before operations
- Ensure no data loss
- Include error handling
- Be explicit about single vs multi-file output

OUTPUT: Python code only, no explanations.
"""
        
        logger.info("🤖 Calling LLM to generate Python code...")
        response = self.llm.invoke(prompt).content.strip()
        
        # Clean code blocks
        code = response.replace("```python", "").replace("```", "").strip()
        
        logger.info("✅ Code generated successfully")
        logger.info(f"📋 Code length: {len(code)} characters, {code.count('def')} functions")
        
        return {"final_code": code}

    def node_executor(self, state: AgentState):
        """Executes the generated code with error analysis."""
        code = state['final_code']
        exec_retries = state.get("execution_retries", 0)
        
        logger.info("="*60)
        logger.info("⚡ EXECUTING GENERATED CODE")
        logger.info(f"Execution attempt #{exec_retries + 1}")
        logger.info("="*60)
        
        try:
            logger.info("🔄 Running Python code...")
            # Execute in isolated namespace
            exec_globals = {}
            
            # Get initial files in output folder
            output_folder = state['output_folder']
            files_before = set(os.listdir(output_folder)) if os.path.exists(output_folder) else set()
            
            exec(code, exec_globals)
            
            # Check what files were created
            files_after = set(os.listdir(output_folder)) if os.path.exists(output_folder) else set()
            new_files = files_after - files_before
            
            # Check for standard output file OR any new files created
            standard_output = os.path.join(output_folder, 'master_unified_data.xlsx')
            
            if os.path.exists(standard_output):
                # Single file merge case
                file_size = os.path.getsize(standard_output) / 1024  # KB
                logger.info(f"✅ Code executed successfully!")
                logger.info(f"📁 Output file created: master_unified_data.xlsx ({file_size:.2f} KB)")
                return {
                    "execution_result": "SUCCESS",
                    "execution_error": "",
                    "execution_retries": exec_retries + 1
                }
            elif new_files:
                # Multiple files case (normalized structure)
                logger.info(f"✅ Code executed successfully!")
                logger.info(f"📁 Output files created: {len(new_files)} files")
                for f in sorted(new_files):
                    file_path = os.path.join(output_folder, f)
                    file_size = os.path.getsize(file_path) / 1024
                    logger.info(f"   - {f} ({file_size:.2f} KB)")
                
                # Create a summary file listing for download
                summary_path = os.path.join(output_folder, 'OUTPUT_SUMMARY.txt')
                with open(summary_path, 'w', encoding='utf-8') as sf:
                    sf.write("DATA UNIFICATION RESULTS\n")
                    sf.write("=" * 50 + "\n\n")
                    sf.write(f"Total files generated: {len(new_files)}\n\n")
                    sf.write("Files:\n")
                    for f in sorted(new_files):
                        sf.write(f"  - {f}\n")
                    sf.write("\n")
                    if 'relationships.txt' in new_files:
                        sf.write("⚠️  Data was split into multiple files due to complex relationships.\n")
                        sf.write("    See 'relationships.txt' for schema details.\n")
                
                return {
                    "execution_result": "SUCCESS",
                    "execution_error": "",
                    "execution_retries": exec_retries + 1
                }
            else:
                logger.error("❌ Code executed but no output files created")
                return {
                    "execution_result": "FAILED",
                    "execution_error": "Code executed but no output files created",
                    "execution_retries": exec_retries + 1
                }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Execution Error: {error_msg}")
            
            # Analyze error type
            data_errors = ['KeyError', 'MergeError', 'Column', 'key', 'merge', 'index']
            is_data_error = any(keyword.lower() in error_msg.lower() for keyword in data_errors)
            
            if is_data_error:
                logger.error("🔍 Error Type: DATA STRUCTURE ISSUE (will retry identification)")
            else:
                logger.error("🔍 Error Type: CODE SYNTAX/LOGIC ISSUE (will regenerate code)")
            
            return {
                "execution_result": "FAILED",
                "execution_error": error_msg,
                "execution_retries": exec_retries + 1,
                "error_is_data_related": is_data_error
            }

    # ============================================================
    # GRAPH BUILDER
    # ============================================================
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("identifier", self.node_identifier)
        workflow.add_node("id_evaluator", self.node_id_evaluator)
        workflow.add_node("schema_maker", self.node_schema_maker)
        workflow.add_node("schema_evaluator", self.node_schema_evaluator)
        workflow.add_node("code_generator", self.node_code_generator)
        workflow.add_node("executor", self.node_executor)
        
        # Entry point
        workflow.set_entry_point("identifier")
        
        # Phase 1: Identification Loop
        workflow.add_edge("identifier", "id_evaluator")
        
        def route_after_id_eval(state):
            """Route based on ID confidence and retries."""
            confidence = state.get("id_confidence", 0)
            retries = state.get("id_retries", 0)
            
            if confidence >= 90 or retries >= 3:
                logger.info("")
                logger.info("🎯 PHASE 1 COMPLETE: Proceeding to Schema Design")
                logger.info(f"   Final confidence: {confidence}/100, Total attempts: {retries}")
                logger.info("")
                return "schema_maker"
            else:
                logger.info("")
                logger.info(f"🔄 PHASE 1 RETRY: Attempt {retries} - Score {confidence}/100 (Need 90+)")
                logger.info("")
                return "identifier"
        
        workflow.add_conditional_edges(
            "id_evaluator",
            route_after_id_eval,
            {
                "schema_maker": "schema_maker",
                "identifier": "identifier"
            }
        )
        
        # Phase 2: Schema Loop
        workflow.add_edge("schema_maker", "schema_evaluator")
        
        def route_after_schema_eval(state):
            """Route based on schema confidence and retries."""
            confidence = state.get("schema_confidence", 0)
            retries = state.get("schema_retries", 0)
            
            if confidence >= 90 or retries >= 3:
                logger.info("")
                logger.info("🎯 PHASE 2 COMPLETE: Proceeding to Code Generation")
                logger.info(f"   Final confidence: {confidence}/100, Total attempts: {retries}")
                logger.info("")
                return "code_generator"
            else:
                logger.info("")
                logger.info(f"🔄 PHASE 2 RETRY: Attempt {retries} - Score {confidence}/100 (Need 90+)")
                logger.info("")
                return "schema_maker"
        
        workflow.add_conditional_edges(
            "schema_evaluator",
            route_after_schema_eval,
            {
                "code_generator": "code_generator",
                "schema_maker": "schema_maker"
            }
        )
        
        # Phase 3: Execution & Recovery
        workflow.add_edge("code_generator", "executor")
        
        def route_after_execution(state):
            """Smart error recovery based on error type."""
            result = state.get("execution_result", "")
            retries = state.get("execution_retries", 0)
            error = state.get("execution_error", "")
            is_data_error = state.get("error_is_data_related", False)
            
            if result == "SUCCESS":
                logger.info("")
                logger.info("🎉 " + "="*50)
                logger.info("🎉 ALL PHASES COMPLETE - UNIFICATION SUCCESSFUL!")
                logger.info("🎉 " + "="*50)
                logger.info("")
                return "end"
            
            # Safety: prevent infinite loops
            if retries >= 3:
                logger.error("")
                logger.error("⛔ MAX RETRIES REACHED - STOPPING EXECUTION")
                logger.error(f"   Total execution attempts: {retries}")
                logger.error("")
                return "end"
            
            # Analyze error type for smart recovery
            if is_data_error:
                logger.warning("")
                logger.warning("🔄 RECOVERY MODE: Data structure error detected")
                logger.warning("   → Resetting to PHASE 1 (Identification) for fresh analysis")
                logger.warning("")
                # Reset retry counters for fresh analysis
                state["id_retries"] = 0
                state["id_feedback"] = f"CODE EXECUTION FAILED with data error: {error}\n\nRethink the identification strategy."
                return "identifier"
            else:
                logger.warning("")
                logger.warning("🔄 RECOVERY MODE: Code syntax/logic error detected")
                logger.warning("   → Regenerating code with corrected logic")
                logger.warning("")
                return "code_generator"
        
        workflow.add_conditional_edges(
            "executor",
            route_after_execution,
            {
                "end": END,
                "identifier": "identifier",
                "code_generator": "code_generator"
            }
        )
        
        return workflow.compile()

    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    def run(self, file_paths, output_folder):
        """Main entry point for the agent."""
        logger.info("")
        logger.info("🚀 " + "="*55)
        logger.info("🚀 MULTI-STAGE REFLEXION AGENT - STARTING")
        logger.info("🚀 " + "="*55)
        logger.info("")
        logger.info(f"📂 Files to process: {len(file_paths)}")
        for i, path in enumerate(file_paths, 1):
            logger.info(f"   {i}. {os.path.basename(path)}")
        logger.info("")
        
        samples = self._load_samples(file_paths)
        
        # Initialize complete state
        initial_state = {
            "file_paths": file_paths,
            "dfs_sample_str": samples,
            "output_folder": output_folder,
            
            # Phase 1 state
            "identifiers": "",
            "id_feedback": "",
            "id_confidence": 0.0,
            "id_retries": 0,
            
            # Phase 2 state
            "schema": "",
            "schema_feedback": "",
            "schema_confidence": 0.0,
            "schema_retries": 0,
            
            # Phase 3 state
            "final_code": "",
            "execution_result": "",
            "execution_error": "",
            "execution_retries": 0
        }
        
        logger.info("🎬 Graph execution starting...")
        logger.info("")
        
        final_state = self.graph.invoke(initial_state)
        
        logger.info("")
        logger.info("📊 FINAL STATISTICS:")
        logger.info(f"   Phase 1 (Identification) attempts: {final_state.get('id_retries', 0)}")
        logger.info(f"   Phase 2 (Schema) attempts: {final_state.get('schema_retries', 0)}")
        logger.info(f"   Phase 3 (Execution) attempts: {final_state.get('execution_retries', 0)}")
        logger.info("")
        
        if final_state["execution_result"] == "SUCCESS":
            logger.info("✅ " + "="*55)
            logger.info("✅ UNIFICATION COMPLETED SUCCESSFULLY!")
            logger.info("✅ " + "="*55)
            logger.info("")
            return True, "master_unified_data.xlsx"
        else:
            error_msg = final_state.get("execution_error", "Unknown error")
            logger.error("❌ " + "="*55)
            logger.error("❌ UNIFICATION FAILED")
            logger.error(f"❌ Error: {error_msg}")
            logger.error("❌ " + "="*55)
            logger.error("")
            return False, error_msg