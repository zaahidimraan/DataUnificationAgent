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
    
    # Validation State
    validation_passed: bool
    validation_errors: str
    
    # Phase 1: Identification State
    identifiers: str
    id_feedback: str
    id_confidence: float
    id_retries: int
    has_one_to_many: bool
    
    # Phase 1.5: One-to-Many Resolution State (NEW)
    one_to_many_detected: bool
    one_to_many_resolution: str  # 'auto_solve', 'aggregate_max', 'aggregate_min', 'aggregate_sum', 'aggregate_avg', 'multi_file'
    aggregation_strategy: str  # LLM's recommended strategy
    user_intent: str  # What user wants to achieve
    
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
    # PHASE 0: DATA VALIDATION
    # ============================================================

    def node_data_validator(self, state: AgentState):
        """Validates uploaded data for quality issues before processing."""
        logger.info("="*60)
        logger.info("🔍 PHASE 0: DATA VALIDATION")
        logger.info("="*60)
        
        errors = []
        warnings = []
        file_summaries = []
        
        for path in state['file_paths']:
            filename = os.path.basename(path)
            logger.info(f"Validating: {filename}")
            
            try:
                # Validate file size (max 50MB per file)
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                if file_size > 50:
                    errors.append(f"{filename}: File too large ({file_size:.1f}MB). Max 50MB per file.")
                    continue
                
                # Load and validate based on type
                if path.endswith('.csv'):
                    try:
                        # Try different encodings and delimiters
                        df = None
                        for encoding in ['utf-8', 'latin1', 'cp1252']:
                            try:
                                df = pd.read_csv(path, encoding=encoding)
                                break
                            except:
                                continue
                        
                        if df is None:
                            errors.append(f"{filename}: Cannot read CSV. Check file encoding.")
                            continue
                        
                        self._validate_dataframe(df, filename, filename, errors, warnings, file_summaries)
                        
                    except Exception as e:
                        errors.append(f"{filename}: CSV parsing error - {str(e)}")
                
                elif path.endswith(('.xlsx', '.xls')):
                    try:
                        xl = pd.ExcelFile(path)
                        
                        if len(xl.sheet_names) == 0:
                            errors.append(f"{filename}: No sheets found in Excel file.")
                            continue
                        
                        for sheet in xl.sheet_names:
                            df = xl.parse(sheet)
                            self._validate_dataframe(df, filename, sheet, errors, warnings, file_summaries)
                            
                    except Exception as e:
                        if 'Workbook is encrypted' in str(e) or 'password' in str(e).lower():
                            errors.append(f"{filename}: File is password-protected. Please remove protection.")
                        else:
                            errors.append(f"{filename}: Excel reading error - {str(e)}")
                else:
                    errors.append(f"{filename}: Unsupported format. Only .xlsx, .xls, .csv allowed.")
                    
            except Exception as e:
                errors.append(f"{filename}: Validation failed - {str(e)}")
        
        # Check total row count across all files (max 500K rows)
        total_rows = sum([s['rows'] for s in file_summaries])
        if total_rows > 500000:
            errors.append(f"Total data too large: {total_rows:,} rows. Max 500,000 rows across all files.")
        
        # Log validation results
        logger.info("")
        if file_summaries:
            logger.info(f"✅ Validated {len(file_summaries)} datasets, {total_rows:,} total rows")
        
        if warnings:
            logger.warning("⚠️  WARNINGS:")
            for w in warnings:
                logger.warning(f"   {w}")
        
        if errors:
            logger.error("")
            logger.error("❌ VALIDATION FAILED:")
            for e in errors:
                logger.error(f"   {e}")
            logger.error("")
            
            return {
                "validation_passed": False,
                "validation_errors": "\n".join(errors)
            }
        
        logger.info("✅ All validation checks passed")
        logger.info("")
        return {
            "validation_passed": True,
            "validation_errors": ""
        }
    
    def _validate_dataframe(self, df, filename, sheet, errors, warnings, file_summaries):
        """Helper to validate individual dataframe."""
        
        # Check if empty
        if len(df) == 0:
            warnings.append(f"{filename}/{sheet}: Empty dataset (0 rows)")
            return
        
        # Check if too wide
        if len(df.columns) > 500:
            errors.append(f"{filename}/{sheet}: Too many columns ({len(df.columns)}). Max 500 columns.")
            return
        
        # Check for duplicate column names
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            errors.append(f"{filename}/{sheet}: Duplicate column names: {duplicates}")
            return
        
        # Check for completely empty columns
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if len(empty_cols) > len(df.columns) * 0.5:
            warnings.append(f"{filename}/{sheet}: {len(empty_cols)} columns are completely empty")
        
        # Check for columns with only one unique value
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols and len(constant_cols) > 3:
            warnings.append(f"{filename}/{sheet}: {len(constant_cols)} columns have constant values")
        
        # Detect potential ID columns (for better error messages later)
        id_candidates = [col for col in df.columns if 
                        any(keyword in col.lower() for keyword in ['id', '_no', 'code', 'ref', 'key'])]
        
        # Check if ID columns have nulls
        for col in id_candidates:
            null_pct = df[col].isna().sum() / len(df) * 100
            if null_pct > 50:
                warnings.append(f"{filename}/{sheet}: Column '{col}' has {null_pct:.1f}% missing values")
            
            # Check for duplicates in potential ID columns
            if df[col].notna().any():
                non_null = df[col].dropna()
                dup_pct = (len(non_null) - non_null.nunique()) / len(non_null) * 100
                if dup_pct > 0 and 'id' in col.lower():
                    warnings.append(f"{filename}/{sheet}: '{col}' has {dup_pct:.1f}% duplicate values")
        
        file_summaries.append({
            'file': filename,
            'sheet': sheet,
            'rows': len(df),
            'cols': len(df.columns)
        })

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
            feedback_context = f"\nPREVIOUS FEEDBACK: {previous_feedback}\nFix these issues."
        
        prompt = f"""Analyze data and identify linking keys.

DATA:
{state['dfs_sample_str']}

TASK:
1. For EACH file/sheet, classify as MASTER (one per entity) or DETAIL (many per entity, like transaction history)
2. Find entity with most ID columns - this defines composite key structure
3. Map other entities to this structure (use '0' for missing keys)
4. Add prefix if IDs might conflict (e.g., 'F-' for Flats)

{feedback_context}

OUTPUT (for EACH file/sheet):
FILE: <name>
SHEET: <sheet>
TYPE: <MASTER or DETAIL>
KEYS: <how columns map to composite key>
PREFIX: <prefix or None>
---

SUMMARY: List MASTER files and DETAIL files separately.
"""
        
        logger.info("🤖 Calling LLM to identify keys...")
        response = self.llm.invoke(prompt).content
        logger.info("✅ Identifier proposal generated")
        logger.info(f"📋 Proposal length: {len(response)} characters")
        
        # Detect if one-to-many relationships exist
        has_one_to_many = False
        one_to_many_indicators = [
            'DETAIL', 'detail', 'transaction', 'history', 'many per entity', 
            'multiple records per', 'one-to-many', 'time-series', 'events', 'logs'
        ]
        
        if any(indicator in response for indicator in one_to_many_indicators):
            has_one_to_many = True
            logger.warning("⚠️  ONE-TO-MANY relationship detected in data structure")
        
        return {
            "identifiers": response,
            "id_retries": retry_count + 1,
            "has_one_to_many": has_one_to_many
        }

    def node_id_evaluator(self, state: AgentState):
        """Evaluates the identification proposal with a confidence score."""
        logger.info("⚖️  Evaluating identification proposal...")
        
        prompt = f"""You are a Senior Data Architect. Score this identification strategy (0-100).

PROPOSAL:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

SCORING (100 points):
1. All files mapped? (25 pts)
2. Consistent key structure? (20 pts)
3. No ID collisions? (20 pts)
4. Clear implementation? (15 pts)
5. Proper MASTER/DETAIL classification? (20 pts)

CRITICAL: If proposal doesn't classify MASTER vs DETAIL data, score < 70.

OUTPUT (JSON only):
{{
  "confidence_score": <0-100>,
  "feedback_text": "<Issues if < 90. Request MASTER/DETAIL classification if missing>"
}}
"""
        
        response = self.llm.invoke(prompt).content
        
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
            feedback_context = f"\nPREVIOUS FEEDBACK: {previous_feedback}\nFix these."
        
        prompt = f"""Design unified schema that preserves all data.

IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

CRITICAL: If data has MASTER + DETAIL files, they CANNOT merge into one file. State multi-file output.

REQUIREMENTS:
1. Analyze granularity: All same level or mixed?  
2. Define key columns and Master UID formula
3. List value columns to keep
4. State SINGLE FILE (if same granularity) or MULTI-FILE (if mixed)

{feedback_context}

OUTPUT:
# GRANULARITY
<All same OR mixed (master+detail)>

# STRATEGY
<Single file OR Multi-file with reason>

# KEYS
<Key columns + Master_UID formula>

# COLUMNS
<Value columns to retain>
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
        
        evaluation_prompt = f"""Evaluate schema quality.

SCHEMA:
{state['schema']}

DATA:
{state['dfs_sample_str']}

CHECK:
1. Keys handle all unique combinations?
2. All value columns included?
3. Strategy matches granularity (single vs multi-file)?
4. If one-to-many, is it preserved (not collapsed)?

SCORE: 0-100
- 90+: Approve
- <90: Reject with specific issues

OUTPUT (JSON):
{{
  "confidence_score": <number>,
  "feedback_text": "<Issues if rejected, else 'Approved'>"
}}
"""
        
        response = self.llm.invoke(evaluation_prompt).content
        
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
        
        # Include aggregation strategy if one-to-many detected
        aggregation_section = ""
        if state.get("one_to_many_detected"):
            strategy = state.get("aggregation_strategy", "MULTI_FILE")
            aggregation_section = f"""
ONE-TO-MANY AGGREGATION STRATEGY: {strategy}

When merging master and detail records:
- If {strategy}: Use pandas agg() to {strategy.replace('AGGREGATE_', '').lower()} detail values per master record
- If MULTI_FILE: Create separate files instead of merging

For aggregation:
  - AGGREGATE_MAX: df.groupby('MASTER_UID').max()
  - AGGREGATE_MIN: df.groupby('MASTER_UID').min()
  - AGGREGATE_SUM: df.groupby('MASTER_UID').sum()
  - AGGREGATE_AVG: df.groupby('MASTER_UID').mean()
  - AGGREGATE_COUNT: df.groupby('MASTER_UID').size()
"""
        
        prompt = f"""Generate Python code to implement the schema.

SCHEMA:
{state['schema']}{aggregation_section}

FILES:
{state['file_paths']}

RULES:
- NEVER use drop_duplicates(), deduplicate, or select first/last record (unless aggregating per strategy)
- Preserve ALL records UNLESS explicitly aggregating per strategy
- Handle missing columns with .get() and fillna('<missing>')
- Convert keys to string: astype(str).str.strip()
- Create MASTER_UID by concatenating keys with '_'

SINGLE FILE (if same granularity):
1. Load all dataframes (handle CSV/Excel sheets)
2. Normalize keys for each df
3. Create MASTER_UID column
4. Merge with pd.concat() or pd.merge(how='outer')
5. Verify: row count >= sum of inputs
6. Save to '{safe_output_path}'
7. Print "SUCCESS: Unified data saved to master_unified_data.xlsx"

MULTI-FILE (if different granularity):
1. Create separate files for each level (master, detail, etc.)
2. Save each to output folder
3. Create relationships.txt explaining structure
4. Print "SUCCESS: Data normalized into multiple files. See relationships.txt"

OUTPUT: Python code only, no markdown."""
        
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
    # TERMINATION NODES
    # ============================================================
    
    def node_validation_failed(self, state: AgentState):
        """Stops processing when validation fails."""
        logger.error("⛔ PROCESS STOPPED: DATA VALIDATION FAILED")
        logger.error("")
        return {
            "execution_result": "FAILED",
            "execution_error": state.get("validation_errors", "Data validation failed")
        }
    
    def node_one_to_many_resolver(self, state: AgentState):
        """Intelligent handler for one-to-many relationships with user choice."""
        logger.warning("")
        logger.warning("⚠️  " + "="*55)
        logger.warning("⚠️  ONE-TO-MANY RELATIONSHIP DETECTED")
        logger.warning("⚠️  " + "="*55)
        logger.warning("")
        logger.warning("📊 Your data contains different granularities:")
        logger.warning("   - Master/reference data (one record per entity)")
        logger.warning("   - Detail/transaction data (multiple records per entity)")
        logger.warning("")
        
        # Check if user has already provided resolution choice
        user_choice = state.get("one_to_many_resolution", "")
        
        if user_choice == "auto_solve":
            logger.info("🤖 AUTO-SOLVE MODE: LLM will intelligently aggregate data")
            logger.info("   Analyzing data structure to determine best aggregation...")
            
            # LLM decides best aggregation strategy
            prompt = f"""Analyze this data structure and recommend the BEST aggregation strategy.

IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

Your data has one-to-many relationships. Choose ONE strategy:
1. AGGREGATE_MAX: Keep maximum values from detail records
2. AGGREGATE_MIN: Keep minimum values from detail records  
3. AGGREGATE_SUM: Sum all detail record values
4. AGGREGATE_AVG: Average all detail record values
5. AGGREGATE_COUNT: Count detail records per master

Considering the data nature and business logic, choose the BEST ONE ONLY.

REQUIREMENT: You MUST recommend ONE and ONLY ONE strategy.

OUTPUT (JSON):
{{
  "recommended_strategy": "<AGGREGATE_MAX, AGGREGATE_MIN, AGGREGATE_SUM, AGGREGATE_AVG, or AGGREGATE_COUNT>",
  "reasoning": "<Why this is best for this data>"
}}
"""
            
            response = self.llm.invoke(prompt).content
            
            try:
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                strategy = result.get("recommended_strategy", "AGGREGATE_SUM")
                reasoning = result.get("reasoning", "")
                
                logger.info(f"✅ LLM Recommendation: {strategy}")
                logger.info(f"   Reasoning: {reasoning}")
                
            except Exception as e:
                logger.warning(f"⚠️  Could not parse LLM recommendation: {e}")
                strategy = "AGGREGATE_SUM"  # Fallback to safe option
            
            logger.info("✅ Auto-solve complete - proceeding with aggregation")
            return {
                "one_to_many_detected": True,
                "one_to_many_resolution": strategy,
                "aggregation_strategy": strategy,
                "user_intent": "auto_solve"
            }
        
        elif user_choice in ["aggregate_max", "aggregate_min", "aggregate_sum", "aggregate_avg", "aggregate_count"]:
            logger.info(f"👤 USER CHOICE RECEIVED: {user_choice}")
            logger.info(f"   Data will be aggregated using: {user_choice}")
            logger.info("✅ Proceeding with user selection")
            
            return {
                "one_to_many_detected": True,
                "one_to_many_resolution": user_choice,
                "aggregation_strategy": user_choice,
                "user_intent": user_choice
            }
        
        else:
            # No choice yet - signal to return to UI
            logger.warning("⏸️  WAITING FOR USER DECISION")
            logger.warning("   UI will show resolution options to user")
            
            return {
                "one_to_many_detected": True,
                "one_to_many_resolution": "awaiting_user_choice",
                "aggregation_strategy": ""
            }

    # ============================================================
    # GRAPH BUILDER
    # ============================================================
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("data_validator", self.node_data_validator)  # Phase 0
        workflow.add_node("validation_stop", self.node_validation_failed)  # Validation failure
        workflow.add_node("identifier", self.node_identifier)
        workflow.add_node("id_evaluator", self.node_id_evaluator)
        workflow.add_node("one_to_many_resolver", self.node_one_to_many_resolver)  # Smart resolution
        workflow.add_node("schema_maker", self.node_schema_maker)
        workflow.add_node("schema_evaluator", self.node_schema_evaluator)
        workflow.add_node("code_generator", self.node_code_generator)
        workflow.add_node("executor", self.node_executor)
        
        # Entry point: Validation first
        workflow.set_entry_point("data_validator")
        
        # Phase 0: Validation routing
        def route_after_validation(state):
            """Route based on validation results."""
            if state.get("validation_passed", False):
                logger.info("✅ Data validation passed - proceeding to Phase 1")
                return "identifier"
            else:
                return "validation_stop"
        
        workflow.add_conditional_edges(
            "data_validator",
            route_after_validation,
            {
                "identifier": "identifier",
                "validation_stop": "validation_stop"
            }
        )
        
        # Validation stop leads to END
        workflow.add_edge("validation_stop", END)
        
        # Phase 1: Identification Loop
        workflow.add_edge("identifier", "id_evaluator")
        
        def route_after_id_eval(state):
            """Route based on ID confidence, retries, and one-to-many detection."""
            confidence = state.get("id_confidence", 0)
            retries = state.get("id_retries", 0)
            has_one_to_many = state.get("has_one_to_many", False)
            
            # Check for one-to-many relationships first
            if has_one_to_many and (confidence >= 90 or retries >= 3):
                logger.info("")
                logger.info("🎯 ONE-TO-MANY RELATIONSHIP DETECTED - Routing to resolver")
                logger.info("")
                return "one_to_many_resolver"
            
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
                "identifier": "identifier",
                "one_to_many_resolver": "one_to_many_resolver"  # NEW route
            }
        )
        
        # Phase 1.5: One-to-Many Resolution
        def route_after_one_to_many_resolution(state):
            """Route based on one-to-many resolution strategy."""
            resolution = state.get("one_to_many_resolution", "")
            
            if resolution == "awaiting_user_choice":
                # User hasn't chosen yet - pause and show UI
                logger.warning("⏸️  PAUSED: Waiting for user input")
                logger.warning("    User should receive modal and resubmit with choice")
                return END  # Pause graph execution, UI will resubmit
            else:
                # User selection received or auto-solve completed
                logger.info("✅ Aggregation strategy confirmed - proceeding to schema design")
                return "schema_maker"
        
        workflow.add_conditional_edges(
            "one_to_many_resolver",
            route_after_one_to_many_resolution,
            {
                "schema_maker": "schema_maker",
                END: END
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
    
    def run(self, file_paths, output_folder, one_to_many_choice=""):
        """Main entry point for the agent.
        
        Args:
            file_paths: List of file paths to unify
            output_folder: Output directory for results
            one_to_many_choice: User's choice for one-to-many resolution ('auto_solve', 'aggregate_max', etc.)
        
        Returns:
            Tuple of (success: bool, message: str, final_state: dict)
        """
        logger.info("")
        logger.info("🚀 " + "="*55)
        logger.info("🚀 MULTI-STAGE REFLEXION AGENT - STARTING")
        logger.info("🚀 " + "="*55)
        logger.info("")
        logger.info(f"📂 Files to process: {len(file_paths)}")
        for i, path in enumerate(file_paths, 1):
            logger.info(f"   {i}. {os.path.basename(path)}")
        
        if one_to_many_choice:
            logger.info(f"👤 One-to-many resolution choice: {one_to_many_choice}")
        logger.info("")
        
        samples = self._load_samples(file_paths)
        
        # Initialize complete state
        initial_state = {
            "file_paths": file_paths,
            "dfs_sample_str": samples,
            "output_folder": output_folder,
            
            # Phase 0 validation state
            "validation_passed": False,
            "validation_errors": [],
            
            # Phase 1 state
            "identifiers": "",
            "id_feedback": "",
            "id_confidence": 0.0,
            "id_retries": 0,
            "has_one_to_many": False,
            
            # Phase 1.5: One-to-Many Resolution State (NEW)
            "one_to_many_detected": False,
            "one_to_many_resolution": one_to_many_choice or "",  # Pre-set if user provided choice
            "aggregation_strategy": "",
            "user_intent": one_to_many_choice or "",
            
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
        if final_state.get('one_to_many_detected'):
            logger.info(f"   One-to-many detected: {final_state.get('one_to_many_resolution', 'N/A')}")
        logger.info("")
        
        if final_state["execution_result"] == "SUCCESS":
            logger.info("✅ " + "="*55)
            logger.info("✅ UNIFICATION COMPLETED SUCCESSFULLY!")
            logger.info("✅ " + "="*55)
            logger.info("")
            return True, "master_unified_data.xlsx", final_state
        else:
            error_msg = final_state.get("execution_error", "Unknown error")
            logger.error("❌ " + "="*55)
            logger.error("❌ UNIFICATION FAILED")
            logger.error(f"❌ Error: {error_msg}")
            logger.error("❌ " + "="*55)
            logger.error("")
            return False, error_msg, final_state