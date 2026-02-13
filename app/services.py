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
    
    # Target Schema Mode State (NEW)
    target_mode_enabled: bool  # Is target mode active?
    target_schema_input: str  # Raw user input (columns from file or text description)
    target_schema_type: str  # 'file' or 'text'
    target_validation_retries: int  # How many times we've retried mapping
    target_validation_feedback: str  # Feedback from last validation
    target_fallback_triggered: bool  # Have we fallen back to auto mode?
    
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
    # Multiple API keys for rotation
    API_KEYS = [
        "AIzaSyA6F2LGibagOOVH3rZ_NE79Bx0-7akxBg4",
        "AIzaSyDGKzEn41rJ4poL8e4fhr2jBOLnkqFcX8o",
        "AIzaSyAVs62C1kvx07V10QTepLNu55An6mIw_Zw",
        "AIzaSyCdR3frcLjBt9V8tawCbwr9VYPeHYV1Wz0",
        "AIzaSyAcPP1BPRI6ENPosu7N4PSJok8wT9CPNNk"
    ]
    
    def __init__(self):
        self.current_key_index = 0
        self.llm = self._create_llm()
        self.graph = self._build_graph()
    
    def _create_llm(self):
        """Create LLM instance with current API key."""
        import os
        api_key = self.API_KEYS[self.current_key_index]
        os.environ['GOOGLE_API_KEY'] = api_key
        logger.info(f"🔑 Using API Key #{self.current_key_index + 1}")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key
        )
    
    def _rotate_api_key(self):
        """Rotate to next API key if available."""
        if self.current_key_index < len(self.API_KEYS) - 1:
            self.current_key_index += 1
            self.llm = self._create_llm()
            logger.warning(f"🔄 Rotated to API Key #{self.current_key_index + 1}")
            return True
        else:
            logger.error("❌ All API keys exhausted")
            return False
    
    def _invoke_llm_with_retry(self, prompt, max_key_retries=None):
        """Invoke LLM with automatic key rotation on failure.
        
        Args:
            prompt: The prompt to send to LLM
            max_key_retries: Maximum number of key rotations (None = try all keys)
        
        Returns:
            LLM response content
        
        Raises:
            Exception: If all keys fail
        """
        if max_key_retries is None:
            max_key_retries = len(self.API_KEYS) - 1
        
        keys_tried = 0
        original_key_index = self.current_key_index
        
        while keys_tried <= max_key_retries:
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a rate limit or quota error
                if any(keyword in error_str for keyword in ['quota', 'rate limit', 'resource exhausted', '429', 'limit exceeded']):
                    logger.warning(f"⚠️  API Key #{self.current_key_index + 1} limit reached: {str(e)[:100]}")
                    
                    # Try to rotate to next key
                    if self._rotate_api_key():
                        keys_tried += 1
                        continue
                    else:
                        # All keys exhausted
                        raise Exception(f"All {len(self.API_KEYS)} API keys exhausted. Last error: {e}")
                else:
                    # Non-quota error, raise immediately
                    raise e
        
        # Should not reach here, but just in case
        raise Exception(f"Failed to get LLM response after trying {keys_tried + 1} API keys")
    
    def _parse_target_schema(self, target_input, input_type):
        """Parse target schema from user input (file or text).
        
        Args:
            target_input: Either file path (for file type) or text description
            input_type: 'file' or 'text'
            
        Returns:
            String containing parsed column names and any metadata
        """
        logger.info(f"📋 Parsing target schema (type: {input_type})")
        
        if input_type == 'file':
            try:
                # Load file and extract column headers
                if target_input.endswith('.csv'):
                    df = pd.read_csv(target_input, nrows=0)  # Just read headers
                    columns = list(df.columns)
                elif target_input.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(target_input, nrows=0)  # Just read headers
                    columns = list(df.columns)
                else:
                    logger.error(f"❌ Unsupported target file format: {target_input}")
                    return None
                
                logger.info(f"✅ Extracted {len(columns)} columns from target file")
                logger.info(f"   Columns: {columns}")
                
                # Format as structured text for LLM
                result = f"TARGET COLUMNS ({len(columns)} total):\n"
                result += "\n".join([f"  - {col}" for col in columns])
                return result
                
            except Exception as e:
                logger.error(f"❌ Error parsing target file: {e}")
                return None
        
        elif input_type == 'text':
            # Text description - clean and format it
            logger.info(f"✅ Using text description as target schema")
            result = f"TARGET DESCRIPTION:\n{target_input}"
            return result
        
        else:
            logger.error(f"❌ Unknown target input type: {input_type}")
            return None

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
        response = self._invoke_llm_with_retry(prompt)
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
        
        response = self._invoke_llm_with_retry(prompt)
        
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
    # PHASE 2: SCHEMA LOOP (Auto Mode & Target Mode)
    # ============================================================
    
    def node_target_schema_mapper(self, state: AgentState):
        """Maps source data to user-defined target schema.
        
        This node is used when target_mode_enabled=True. It attempts to create
        a schema that matches the user's specified target columns/structure.
        """
        retry_count = state.get("target_validation_retries", 0)
        previous_feedback = state.get("target_validation_feedback", "")
        
        logger.info("="*60)
        logger.info("🎯 PHASE 2: TARGET-DRIVEN SCHEMA MAPPING")
        logger.info(f"Attempt #{retry_count + 1} / 3")
        if previous_feedback:
            logger.info(f"📝 Addressing feedback from previous attempt")
        logger.info("="*60)
        
        feedback_context = ""
        if previous_feedback:
            feedback_context = f"""
PREVIOUS VALIDATION FEEDBACK:
{previous_feedback}

CRITICAL: Address ALL issues mentioned above. Use the feedback to improve your mapping.
"""
        
        # Prepare aggregation context if one-to-many detected
        aggregation_context = ""
        if state.get("one_to_many_detected"):
            aggregation_context = f"""
ONE-TO-MANY RELATIONSHIP DETECTED:
- Aggregation strategy: {state.get('aggregation_strategy', 'AGGREGATE_SUM')}
- Apply this strategy when mapping numeric fields to target columns
- For text fields: use random selection from grouped values
"""
        
        prompt = f"""You are designing a unified data schema to match the USER'S TARGET SPECIFICATION.

TARGET SPECIFICATION (what user wants):
{state['target_schema_input']}

SOURCE DATA IDENTIFIERS:
{state['identifiers']}

SOURCE DATA SAMPLES:
{state['dfs_sample_str']}
{aggregation_context}
{feedback_context}

YOUR TASK:
Map source data columns to the target schema specified by the user. You MUST:

1. **Match target column names exactly** - Use the exact column names from the target specification
2. **Map all source data** - Ensure no source columns are lost if they're relevant
3. **Handle missing target columns** - If target asks for columns that don't exist in source:
   - Try to derive them (e.g., "total_sales" = sum of sale amounts)
   - If impossible to derive, mark as "UNMAPPABLE - <reason>"
4. **Respect aggregation strategy** - If one-to-many detected, apply the aggregation strategy
5. **Maintain data integrity** - Ensure keys/identifiers are properly mapped

OUTPUT FORMAT (use this exact structure):
# GRANULARITY
<Describe the data granularity: one row per what entity?>

# TARGET MODE
ENABLED - Mapping to user-defined target schema

# MAPPING STRATEGY
<How you'll map source → target, handling any gaps or transformations>

# TARGET COLUMNS
For each target column the user requested, provide:
- Column Name: <exact name from target>
- Source Mapping: <which source column(s) map to this>
- Transformation: <any calculation/aggregation needed>
- Status: MAPPED or UNMAPPABLE

# KEYS
<Which columns serve as identifiers + Master_UID formula>

# ADDITIONAL COLUMNS
<Any source columns not in target but should be preserved>

CRITICAL: If you cannot map certain target columns, be explicit about why and mark them as UNMAPPABLE.
"""
        
        logger.info("🤖 Calling LLM to map source data to target schema...")
        response = self._invoke_llm_with_retry(prompt)
        logger.info("✅ Target schema mapping proposal generated")
        logger.info(f"📋 Mapping length: {len(response)} characters")
        
        return {
            "schema": response,
            "target_validation_retries": retry_count + 1
        }
    
    def node_target_schema_validator(self, state: AgentState):
        """Validates the target schema mapping with strict checks.
        
        This validates whether the mapping successfully meets the target requirements.
        If validation fails 3 times, it triggers fallback to auto mode.
        """
        logger.info("⚖️  Validating target schema mapping...")
        
        prompt = f"""You are a Senior Data Quality Engineer. Validate this target schema mapping (0-100 score).

USER'S TARGET SPECIFICATION:
{state['target_schema_input']}

PROPOSED MAPPING:
{state['schema']}

SOURCE DATA:
{state['dfs_sample_str']}

VALIDATION CRITERIA (100 points total):
1. **Target Column Coverage (40 pts)**: Are ALL requested target columns included?
   - Full credit: All target columns mapped
   - Partial: Some mapped, others marked UNMAPPABLE with valid reason
   - Zero: Missing target columns without explanation

2. **Mapping Correctness (30 pts)**: Are mappings logically correct?
   - Do source columns match target semantics?
   - Are data types compatible?
   - Are transformations appropriate?

3. **Data Preservation (15 pts)**: Is source data properly handled?
   - No important source data lost
   - Aggregations applied correctly if needed

4. **Implementation Clarity (15 pts)**: Can this be implemented?
   - Clear instructions for code generation
   - Unambiguous mapping logic
   - Proper key/identifier handling

SCORING:
- 90-100: Excellent mapping, approve
- 70-89: Good but needs improvement, provide specific feedback
- 0-69: Poor mapping, major issues

If score < 90, provide SPECIFIC, ACTIONABLE feedback:
- Which target columns are missing or incorrectly mapped?
- What transformations are wrong or missing?
- How should the mapper improve the next attempt?

OUTPUT (JSON only):
{{
  "confidence_score": <0-100>,
  "target_columns_mapped": <number of target columns successfully mapped>,
  "target_columns_total": <total target columns user requested>,
  "unmappable_columns": ["<list of columns that can't be mapped>"],
  "feedback_text": "<Specific issues if score < 90, else 'Approved'>"
}}
"""
        
        response = self._invoke_llm_with_retry(prompt)
        
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
            mapped_count = eval_result.get("target_columns_mapped", 0)
            total_count = eval_result.get("target_columns_total", 0)
            unmappable = eval_result.get("unmappable_columns", [])
            
            logger.info(f"📊 Target Validation Score: {confidence}/100")
            logger.info(f"📊 Target Columns: {mapped_count}/{total_count} mapped")
            
            if unmappable:
                logger.warning(f"⚠️  Unmappable columns: {', '.join(unmappable)}")
            
            if confidence >= 90:
                logger.info("✅ Target mapping approved! Moving to code generation.")
            else:
                retries = state.get("target_validation_retries", 0)
                logger.warning(f"⚠️  Target mapping needs improvement (Score: {confidence})")
                logger.warning(f"💬 Feedback: {feedback[:200]}...") if len(feedback) > 200 else logger.warning(f"💬 Feedback: {feedback}")
                logger.warning(f"🔄 Retry count: {retries}/3")
            
        except Exception as e:
            logger.error(f"❌ Failed to parse target validator response: {e}")
            confidence = 0.0
            feedback = f"Validator response parsing failed: {response}"
        
        return {
            "schema_confidence": confidence,
            "target_validation_feedback": feedback
        }

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

CRITICAL: ALWAYS CREATE SINGLE FILE OUTPUT. Even if data has MASTER+DETAIL:
- For numeric fields: apply aggregation (sum, avg, min, max, count)
- For text fields: randomly select from aggregated rows (or first value for MAX/MIN)
- Result: ONE row per master record

REQUIREMENTS:
1. Analyze granularity: All same level or mixed?  
2. Define key columns and Master UID formula
3. List all value columns to keep (will be aggregated smartly)
4. ALWAYS state SINGLE FILE (even if granularity is mixed with master+detail)

{feedback_context}

OUTPUT:
# GRANULARITY
<All same level OR mixed (master+detail) - doesn't matter, output is single file>

# STRATEGY
<Always: Single file with smart aggregation per data type>

# KEYS
<Key columns + Master_UID formula>

# COLUMNS
<Value columns to retain>
"""
        
        logger.info("🤖 Calling LLM to design schema...")
        response = self._invoke_llm_with_retry(prompt)
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
        
        response = self._invoke_llm_with_retry(evaluation_prompt)
        
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
                strategy = state.get("aggregation_strategy", "AGGREGATE_SUM")
            
                # Map strategy to aggregation type
                agg_type = strategy.replace('AGGREGATE_', '').lower()
            
                aggregation_section = f"""
ONE-TO-MANY AGGREGATION STRATEGY: {strategy}

⚠️ CRITICAL RULES FOR SINGLE FILE OUTPUT:
1. ALWAYS create ONE file only, even if there's data loss
2. For NUMERIC fields:
    - Apply {agg_type} using groupby aggregation
    - {agg_type}.upper() = sum/avg/min/max/count values per master record
3. For TEXT/DATE fields:
    - If strategy is MAX/MIN: use that strategy
    - Otherwise: Use RANDOM selection from aggregated rows (random.choice on unique values)
    - Include complete row data with the selected value
4. For fields that can't be aggregated properly:
    - Leave empty or use first non-null value
5. DO NOT create multiple files
6. DO NOT preserve all detail records as separate rows
7. Result: ONE row per master record with aggregated/selected values

Aggregation mapping:
  - AGGREGATE_MAX: df.groupby('MASTER_UID').max() (or random for text)
  - AGGREGATE_MIN: df.groupby('MASTER_UID').min() (or random for text)
  - AGGREGATE_SUM: df.groupby('MASTER_UID').sum() (or random for text)
  - AGGREGATE_AVG: df.groupby('MASTER_UID').mean() (or random for text)
  - AGGREGATE_COUNT: Use df.groupby('MASTER_UID').size() for counts + random for text fields
"""
        
        prompt = f"""Generate Python code to implement the schema.

SCHEMA:
{state['schema']}{aggregation_section}

FILES:
{state['file_paths']}

CRITICAL REQUIREMENTS:
1. ALWAYS output EXACTLY ONE file to '{safe_output_path}'
2. NEVER create multiple files
3. NEVER preserve all detail rows as separate records
4. FOCUS: Create one unified master file with aggregated/merged data

For one-to-many data:
- Group by master record (MASTER_UID)
- Apply aggregation strategy to numeric columns
- For text/date: use random selection from grouped values (except MAX/MIN use that directly)
- Result: ONE row per unique MASTER_UID

Code structure:
1. Load all dataframes (handle CSV/Excel sheets)
2. Normalize keys for each df (convert to string, strip whitespace)
3. Create MASTER_UID column by concatenating key columns
4. Identify column data types:
    - Numeric (int, float) → apply aggregation
    - Text/String → random selection from grouped data
    - Date/Datetime → random selection or leave empty
5. Apply groupby with custom aggregation:
    - Numeric: {state.get("aggregation_strategy", "AGGREGATE_SUM").replace('AGGREGATE_', '').lower()}
    - Text: use lambda to random.choice from unique values
6. Merge all data into single dataframe
7. VERIFY: result has exactly one row per MASTER_UID
8. Save to '{safe_output_path}'
9. Print "SUCCESS: Unified data saved to master_unified_data.xlsx"

OUTPUT: Python code only, no markdown."""
        
        logger.info("🤖 Calling LLM to generate Python code...")
        response = self._invoke_llm_with_retry(prompt).strip()
        
        # Clean code blocks
        code = response.replace("```python", "").replace("```", "").strip()
        
        logger.info("✅ Code generated successfully")
        logger.info(f"📋 Code length: {len(code)} characters")
        logger.info("📋 Single-file output configured")
        
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
                # Multiple files case - ERROR! We require single file output
                logger.info(f"✅ Code executed successfully!")
                logger.error(f"❌ CONSTRAINT VIOLATION: Created {len(new_files)} files instead of single file")
                for f in sorted(new_files):
                    file_path = os.path.join(output_folder, f)
                    file_size = os.path.getsize(file_path) / 1024
                    logger.error(f"   - {f} ({file_size:.2f} KB)")
                
                return {
                    "execution_result": "FAILED",
                    "execution_error": f"System created {len(new_files)} files but single file output is required. Regenerating code with stricter constraints.",
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
            
            response = self._invoke_llm_with_retry(prompt)
            
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
    
    def node_target_fallback(self, state: AgentState):
        """Handles fallback from target mode to auto mode after 3 failed attempts.
        
        This node is reached when target validation has failed 3 times.
        It logs the failure and switches to auto schema generation mode.
        """
        logger.warning("")
        logger.warning("⚠️  " + "="*55)
        logger.warning("⚠️  TARGET MODE FALLBACK TRIGGERED")
        logger.warning("⚠️  " + "="*55)
        logger.warning("")
        logger.warning("📊 Unable to map source data to target schema after 3 attempts")
        logger.warning("")
        logger.warning("🔄 FALLING BACK TO AUTO MODE:")
        logger.warning("   - Disabling target mode")
        logger.warning("   - Using automatic schema generation")
        logger.warning("   - This ensures your data is still unified successfully")
        logger.warning("")
        logger.warning("💡 Recommendation: Review your target schema and source data compatibility")
        logger.warning("")
        
        return {
            "target_mode_enabled": False,
            "target_fallback_triggered": True,
            "schema_feedback": "Target mode failed. Starting fresh with auto schema generation.",
            "schema_retries": 0  # Reset for auto mode
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
        
        # Phase 2: Schema nodes (both auto and target mode)
        workflow.add_node("target_schema_mapper", self.node_target_schema_mapper)  # Target mode
        workflow.add_node("target_schema_validator", self.node_target_schema_validator)  # Target validation
        workflow.add_node("target_fallback", self.node_target_fallback)  # Fallback to auto
        workflow.add_node("schema_maker", self.node_schema_maker)  # Auto mode
        workflow.add_node("schema_evaluator", self.node_schema_evaluator)  # Auto validation
        
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
                
                # Check if target mode is enabled
                if state.get("target_mode_enabled", False):
                    logger.info("🎯 TARGET MODE ENABLED - Using target-driven schema mapping")
                    return "target_schema_mapper"
                else:
                    logger.info("🤖 AUTO MODE - Using automatic schema generation")
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
                "target_schema_mapper": "target_schema_mapper",
                "identifier": "identifier",
                "one_to_many_resolver": "one_to_many_resolver"  # NEW route
            }
        )
        
        # Phase 1.5: One-to-Many Resolution
        def route_after_one_to_many_resolution(state):
            """Route based on one-to-many resolution strategy and target mode."""
            resolution = state.get("one_to_many_resolution", "")
            
            if resolution == "awaiting_user_choice":
                # User hasn't chosen yet - pause and show UI
                logger.warning("⏸️  PAUSED: Waiting for user input")
                logger.warning("    User should receive modal and resubmit with choice")
                return END  # Pause graph execution, UI will resubmit
            else:
                # User selection received or auto-solve completed
                logger.info("✅ Aggregation strategy confirmed - proceeding to schema design")
                
                # Check if target mode is enabled
                if state.get("target_mode_enabled", False):
                    logger.info("🎯 TARGET MODE ENABLED - Using target-driven schema mapping")
                    return "target_schema_mapper"
                else:
                    logger.info("🤖 AUTO MODE - Using automatic schema generation")
                    return "schema_maker"
        
        workflow.add_conditional_edges(
            "one_to_many_resolver",
            route_after_one_to_many_resolution,
            {
                "schema_maker": "schema_maker",
                "target_schema_mapper": "target_schema_mapper",
                END: END
            }
        )
        
        # Phase 2: Schema Loop (Auto Mode)
        workflow.add_edge("schema_maker", "schema_evaluator")
        
        # Phase 2: Target Schema Mapping Loop
        workflow.add_edge("target_schema_mapper", "target_schema_validator")
        
        def route_after_target_validation(state):
            """Route based on target validation confidence and retries."""
            confidence = state.get("schema_confidence", 0)
            retries = state.get("target_validation_retries", 0)
            
            if confidence >= 90:
                logger.info("")
                logger.info("🎯 TARGET MAPPING APPROVED: Proceeding to Code Generation")
                logger.info(f"   Final confidence: {confidence}/100, Total attempts: {retries}")
                logger.info("")
                return "code_generator"
            elif retries >= 3:
                # Exceeded max retries - trigger fallback to auto mode
                logger.warning("")
                logger.warning("⚠️  TARGET MAPPING FAILED: Max retries (3) reached")
                logger.warning(f"   Final confidence: {confidence}/100")
                logger.warning("   Triggering fallback to auto mode...")
                logger.warning("")
                return "target_fallback"
            else:
                logger.info("")
                logger.info(f"🔄 TARGET MAPPING RETRY: Attempt {retries} - Score {confidence}/100 (Need 90+)")
                logger.info("")
                return "target_schema_mapper"
        
        workflow.add_conditional_edges(
            "target_schema_validator",
            route_after_target_validation,
            {
                "code_generator": "code_generator",
                "target_schema_mapper": "target_schema_mapper",
                "target_fallback": "target_fallback"
            }
        )
        
        # Fallback to auto mode after target failure
        workflow.add_edge("target_fallback", "schema_maker")
        
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
    
    def run(self, file_paths, output_folder, one_to_many_choice="", target_schema_file=None, target_schema_text=None):
        """Main entry point for the agent.
        
        Args:
            file_paths: List of file paths to unify
            output_folder: Output directory for results
            one_to_many_choice: User's choice for one-to-many resolution ('auto_solve', 'aggregate_max', etc.)
            target_schema_file: Path to template file with target column headers (optional)
            target_schema_text: Text description of target schema (optional)
        
        Returns:
            Tuple of (success: bool, message: str, final_state: dict)
        """
        logger.info("")
        logger.info("🚀 " + "="*55)
        logger.info("🚀 MULTI-STAGE REFLEXION AGENT - STARTING")
        
        # Determine if target mode is enabled
        target_mode_enabled = bool(target_schema_file or target_schema_text)
        target_schema_input = None
        target_schema_type = None
        
        if target_mode_enabled:
            logger.info("📋 MODE: TARGET-DRIVEN SCHEMA")
            if target_schema_file:
                logger.info(f"   Input: Template file ({os.path.basename(target_schema_file)})")
                target_schema_type = "file"
                # Parse the target schema
                target_schema_input = self._parse_target_schema(target_schema_file, "file")
                if not target_schema_input:
                    logger.error("❌ Failed to parse target schema file")
                    return False, "Invalid target schema file", {}
            elif target_schema_text:
                logger.info(f"   Input: Text description")
                target_schema_type = "text"
                target_schema_input = self._parse_target_schema(target_schema_text, "text")
            
            logger.info("   Fallback: Auto mode (if target mapping fails after 3 attempts)")
        else:
            logger.info("📋 MODE: AUTOMATIC SCHEMA GENERATION")
        
        logger.info("   - Always creates ONE master file")
        logger.info("   - Numeric fields aggregated per strategy")
        logger.info("   - Text fields randomly selected from detail records")
        logger.info("")
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
            
            # Phase 1.5: One-to-Many Resolution State
            "one_to_many_detected": False,
            "one_to_many_resolution": one_to_many_choice or "",  # Pre-set if user provided choice
            "aggregation_strategy": "",
            "user_intent": one_to_many_choice or "",
            
            # Target Schema Mode State (NEW)
            "target_mode_enabled": target_mode_enabled,
            "target_schema_input": target_schema_input or "",
            "target_schema_type": target_schema_type or "",
            "target_validation_retries": 0,
            "target_validation_feedback": "",
            "target_fallback_triggered": False,
            
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
        
        if final_state.get('target_mode_enabled') and not final_state.get('target_fallback_triggered'):
            logger.info(f"   Phase 2 (Target Mapping) attempts: {final_state.get('target_validation_retries', 0)}")
        elif final_state.get('target_fallback_triggered'):
            logger.info(f"   Phase 2 (Target Mapping) attempts: {final_state.get('target_validation_retries', 0)} - FAILED")
            logger.info(f"   Phase 2 (Auto Schema) attempts: {final_state.get('schema_retries', 0)} - FALLBACK")
        else:
            logger.info(f"   Phase 2 (Schema) attempts: {final_state.get('schema_retries', 0)}")
        
        logger.info(f"   Phase 3 (Execution) attempts: {final_state.get('execution_retries', 0)}")
        if final_state.get('one_to_many_detected'):
            logger.info(f"   One-to-many detected: {final_state.get('one_to_many_resolution', 'N/A')}")
        logger.info("")
        
        if final_state["execution_result"] == "SUCCESS":
            logger.info("✅ " + "="*55)
            logger.info("✅ UNIFICATION COMPLETED SUCCESSFULLY!")
            logger.info("")
            logger.info("📄 OUTPUT: master_unified_data.xlsx")
            
            if final_state.get('target_mode_enabled'):
                if final_state.get('target_fallback_triggered'):
                    logger.warning("   ⚠️  Target mode fallback: Used auto-generated schema")
                    logger.warning("       (Target mapping failed after 3 attempts)")
                else:
                    logger.info("   ✅ Target mode: Output matches user-defined schema")
            
            if final_state.get('one_to_many_detected'):
                agg_strategy = final_state.get('aggregation_strategy', 'N/A')
                logger.info(f"   - One-to-many data aggregated using: {agg_strategy}")
                logger.info("   - Numeric fields: aggregated per strategy")
                logger.info("   - Text fields: randomly selected from detail records")
            logger.info("   - Output: Single unified file with all data merged")
            logger.info("✅ " + "="*55)
            logger.info("")
            
            # Prepare success message
            success_msg = "master_unified_data.xlsx"
            if final_state.get('target_fallback_triggered'):
                success_msg += " (auto-generated schema - target mapping failed)"
            
            return True, success_msg, final_state
        else:
            error_msg = final_state.get("execution_error", "Unknown error")
            logger.error("❌ " + "="*55)
            logger.error("❌ UNIFICATION FAILED")
            logger.error(f"❌ Error: {error_msg}")
            logger.error("❌ " + "="*55)
            logger.error("")
            return False, error_msg, final_state