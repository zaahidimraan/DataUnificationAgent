import pandas as pd
import os
import json
import time
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
    one_to_many_resolution: str  # 'auto_solve', 'keep_all', 'aggregate_max', 'aggregate_min', 'aggregate_sum', 'aggregate_avg', 'multi_file'
    aggregation_strategy: str  # LLM's recommended strategy or 'KEEP_ALL'
    user_intent: str  # What user wants to achieve
    detail_files_info: str  # Per-file MASTER/DETAIL classification for UI display
    
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
    
    @staticmethod
    def _load_api_keys():
        """Load API keys from environment variables (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...)."""
        keys = []
        i = 1
        while True:
            key = os.environ.get(f"GOOGLE_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break
        if not keys:
            raise ValueError("No API keys found in environment. Set GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ... in .env")
        logger.info(f"🔑 Loaded {len(keys)} API key(s) from environment")
        return keys
    
    def __init__(self):
        self.API_KEYS = self._load_api_keys()
        self.current_key_index = 0
        self.llm = self._create_llm()
        self.graph = self._build_graph()
    
    def _create_llm(self):
        """Create LLM instance with current API key."""
        api_key = self.API_KEYS[self.current_key_index]
        os.environ['GOOGLE_API_KEY'] = api_key
        logger.info(f"🔑 Using API Key #{self.current_key_index + 1}")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
            timeout=120,
            max_retries=2
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
    
    def _invoke_llm_with_retry(self, prompt, max_key_retries=None, max_network_retries=3):
        """Invoke LLM with automatic key rotation and network error retry.
        
        Args:
            prompt: The prompt to send to LLM
            max_key_retries: Maximum number of key rotations (None = try all keys)
            max_network_retries: Maximum retries per key for network/transient errors
        
        Returns:
            LLM response content
        
        Raises:
            Exception: If all keys and retries fail
        """
        if max_key_retries is None:
            max_key_retries = len(self.API_KEYS) - 1
        
        # Network/transient error keywords that should trigger retry (not key rotation)
        network_error_keywords = [
            'socket', 'connection', 'timeout', 'timed out', 'unreachable',
            'disconnected', 'reset by peer', 'broken pipe', 'eof occurred',
            'remotedisconnected', 'connectionerror', 'networkerror',
            'server disconnected', 'without sending a response',
            'sslerror', 'temporary failure', 'name resolution',
            'getaddrinfo failed', 'max retries exceeded', 'connect timeout'
        ]
        
        keys_tried = 0
        
        while keys_tried <= max_key_retries:
            # Retry loop for network/transient errors on the SAME key
            for network_attempt in range(max_network_retries):
                try:
                    response = self.llm.invoke(prompt)
                    return response.content
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check if it's a rate limit or quota error → rotate key
                    if any(kw in error_str for kw in ['quota', 'rate limit', 'resource exhausted', '429', 'limit exceeded']):
                        logger.warning(f"⚠️  API Key #{self.current_key_index + 1} limit reached: {str(e)[:100]}")
                        break  # Break inner loop to rotate key
                    
                    # Check if it's a network/transient error → retry same key with backoff
                    elif any(kw in error_str for kw in network_error_keywords):
                        wait_time = (2 ** network_attempt) * 3  # 3s, 6s, 12s
                        logger.warning(
                            f"🌐 Network error on Key #{self.current_key_index + 1} "
                            f"(attempt {network_attempt + 1}/{max_network_retries}): {str(e)[:100]}"
                        )
                        if network_attempt < max_network_retries - 1:
                            logger.info(f"⏳ Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            # Recreate the LLM instance (fresh connection)
                            self.llm = self._create_llm()
                            continue
                        else:
                            logger.warning(f"🌐 Network retries exhausted on Key #{self.current_key_index + 1}, trying next key...")
                            break  # Break inner loop to try next key
                    
                    else:
                        # Unknown error, raise immediately
                        raise e
            
            # Try to rotate to next key
            if self._rotate_api_key():
                keys_tried += 1
                continue
            else:
                raise Exception(f"All {len(self.API_KEYS)} API keys exhausted after network retries. Check your internet connection.")
        
        raise Exception(f"Failed to get LLM response after trying {keys_tried + 1} API keys with network retries")
    
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
        logger.info(f"📝 Addressing feedback from previous attempt")
        logger.info("="*60)
        
        feedback_context = ""
        if previous_feedback:
            feedback_context = f"\nPREVIOUS FEEDBACK: {previous_feedback}\nFix these issues."
        
        prompt = f"""Analyze data and identify linking keys for unification.

DATA:
{state['dfs_sample_str']}

TASK - Identify how to link ALL files together:
1. Classify EACH file/sheet as:
   - MASTER: One record per entity (e.g., Property Master, Customer Master)
   - DETAIL: Multiple records per entity (e.g., Transactions, History, Logs)

2. Identify ALL possible ID/key columns in each file:
   - Look for: ID, Code, Number, Reference, Key columns
   - These will form the COMPOSITE KEY for unification

3. CRITICAL - Design cross-file key mapping:
   - Keys from DIFFERENT files can combine to form MASTER_UID
   - Example: File1 has 'PropertyID', File2 has 'BuildingID' + 'UnitID'
   - MASTER_UID = PropertyID (from File1) OR BuildingID + UnitID (from File2)
   - Show EXACT column names and which file they come from

4. Handle key name variations:
   - Same entity might have different column names (PropertyID vs Property_Code)
   - Map these equivalent keys explicitly

5. Add prefixes ONLY if IDs might collide:
   - Example: If FlatID and PropertyID both use numbers 1-100

{feedback_context}

OUTPUT (for EACH file/sheet):
FILE: <name>
SHEET: <sheet>
TYPE: <MASTER or DETAIL>
KEY_COLUMNS: <list exact column names that contain IDs>
KEY_MAPPING: <how these columns map to create unified key>
EQUIVALENT_KEYS: <columns from other files that represent same entity>
PREFIX: <prefix or None>
NOTES: <any special handling needed>
---

COMPOSITE KEY STRUCTURE:
<Define MASTER_UID formula showing which columns from which files combine>
Example: MASTER_UID = File1.PropertyID OR (File2.BuildingID + '_' + File2.UnitID)

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
        
        # Extract per-file MASTER/DETAIL classification for UI display
        detail_files_info = ""
        if has_one_to_many:
            detail_files_info = self._extract_file_classifications(response, state['file_paths'])
            if detail_files_info:
                logger.info(f"📋 File classifications extracted for user display")
        
        return {
            "identifiers": response,
            "id_retries": retry_count + 1,
            "has_one_to_many": has_one_to_many,
            "detail_files_info": detail_files_info
        }

    def _extract_file_classifications(self, identifier_response, file_paths):
        """Extract per-file MASTER/DETAIL classification from identifier LLM response.
        
        Returns a JSON string with per-file info for UI display.
        """
        try:
            filenames = [os.path.basename(p) for p in file_paths]
            classifications = []
            
            for filename in filenames:
                file_info = {"filename": filename, "type": "UNKNOWN", "description": ""}
                
                # Search in identifier response for this file's classification
                response_upper = identifier_response.upper()
                fname_upper = filename.upper()
                
                # Find the section about this file
                idx = response_upper.find(fname_upper)
                if idx != -1:
                    # Get surrounding context (500 chars after filename mention)
                    context = identifier_response[max(0, idx):idx + 500]
                    
                    if 'DETAIL' in context.upper():
                        file_info["type"] = "DETAIL"
                        file_info["description"] = "Multiple records per entity (e.g., history, transactions, logs)"
                    elif 'MASTER' in context.upper():
                        file_info["type"] = "MASTER"
                        file_info["description"] = "One record per entity (reference/master data)"
                    
                classifications.append(file_info)
            
            return json.dumps(classifications)
            
        except Exception as e:
            logger.warning(f"⚠️  Could not extract file classifications: {e}")
            return ""

    def node_id_evaluator(self, state: AgentState):
        """Evaluates the identification proposal with a confidence score."""
        logger.info("⚖️  Evaluating identification proposal...")
        
        prompt = f"""You are a Senior Data Architect. Score this identification strategy (0-100).

PROPOSAL:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}

SCORING (100 points):
1. All files mapped? (20 pts)
   - Every file/sheet has key column identification

2. Cross-file key mapping clear? (25 pts)
   - Shows which columns from different files represent same entity
   - Handles equivalent keys (PropertyID = BuildingID_UnitID)
   - MASTER_UID formula works across all files

3. Key structure consistency? (20 pts)
   - Keys can be uniquely combined
   - No ambiguity in composite key creation

4. No ID collisions? (15 pts)
   - Prefixes added where needed
   - Different entities won't have same ID

5. Proper MASTER/DETAIL classification? (20 pts)
   - Clear distinction between master and detail data

CRITICAL CHECKS:
- If proposal doesn't show cross-file key mapping: score < 70
- If MASTER_UID formula is unclear or same for all files: score < 70  
- If proposal doesn't classify MASTER vs DETAIL: score < 70

OUTPUT (JSON only):
{{
  "confidence_score": <0-100>,
  "feedback_text": "<Specific issues if < 90. Must include: which files lack key mapping, unclear MASTER_UID logic, missing classifications>"
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
            strategy = state.get('aggregation_strategy', 'AGGREGATE_SUM')
            if strategy == "KEEP_ALL":
                aggregation_context = f"""
ONE-TO-MANY RELATIONSHIP DETECTED - STRATEGY: KEEP ALL RECORDS
- The user wants ALL detail/transaction rows PRESERVED in the output
- DO NOT aggregate or collapse rows
- Each detail record (e.g., each repair, each transaction) must appear as its own row
- Master data columns should be repeated/joined for each detail row
- Output granularity: ONE ROW PER DETAIL RECORD (not per master entity)
- This means the output will have MULTIPLE rows per master entity
"""
            else:
                aggregation_context = f"""
ONE-TO-MANY RELATIONSHIP DETECTED - STRATEGY: {strategy}
- Apply {strategy.replace('AGGREGATE_', '').lower()} aggregation when mapping numeric fields to target columns
- For text/categorical fields (names, statuses, types): use MODE (most frequent value)
- For date fields: use latest (max) date by default
- For descriptive text fields (notes, comments, descriptions): concatenate unique values with ' | '
- For any other non-numeric: use first non-null value as fallback
- Output granularity: ONE ROW PER MASTER ENTITY
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
4. **Respect the data handling strategy**:
   - If KEEP_ALL: Preserve every detail row. DO NOT aggregate. Join master data to each detail row.
   - If AGGREGATE_*: Apply the specified aggregation to collapse detail rows per master entity.
5. **Analyze user's target text for intent clues**:
   - "complete history", "all records", "full log", "detailed" → confirms KEEP_ALL behavior
   - "summary", "total", "per property" → confirms aggregation behavior
   - If user's text contradicts the strategy, FOLLOW THE STRATEGY but note the conflict
6. **Maintain data integrity** - Ensure keys/identifiers are properly mapped

OUTPUT FORMAT (use this exact structure):
# GRANULARITY
<Describe the data granularity based on strategy:
  - KEEP_ALL: one row per detail record (multiple rows per master entity)
  - AGGREGATE: one row per master entity>

# DATA HANDLING STRATEGY
<KEEP_ALL or AGGREGATE with specific function>

# TARGET MODE
ENABLED - Mapping to user-defined target schema

# MAPPING STRATEGY
<How you'll map source → target, handling any gaps or transformations>

# TARGET COLUMNS
For each target column the user requested, provide:
- Column Name: <exact name from target>
- Source Mapping: <which source column(s) map to this>
- Transformation: <any calculation/aggregation needed, or "direct mapping" for KEEP_ALL>
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
        
        # Determine strategy context based on aggregation strategy
        strategy = state.get("aggregation_strategy", "")
        if strategy == "KEEP_ALL":
            strategy_context = """
DATA HANDLING STRATEGY: KEEP ALL RECORDS
- PRESERVE all detail/transaction rows in the output
- DO NOT aggregate or collapse rows
- Join master data to each detail row (left join)
- Output will have MULTIPLE rows per master entity
- Each detail record appears as its own row with master data attached
- Result: ONE file with ALL detail rows preserved
"""
        else:
            strategy_context = f"""
CRITICAL: ALWAYS CREATE SINGLE FILE OUTPUT. Even if data has MASTER+DETAIL:
- For numeric fields: apply aggregation (sum, avg, min, max, count)
- For text/categorical fields (names, statuses, types): use MODE (most frequent value)
- For date fields: use latest (max) date
- For descriptive text fields (notes, comments): concatenate unique values with ' | '
- For any other non-numeric: use first non-null value as fallback
- Result: ONE row per master record
"""
        
        prompt = f"""Design unified schema that preserves all data.

IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}
{strategy_context}

REQUIREMENTS:
1. Analyze granularity: All same level or mixed?  
2. Define key columns and Master UID formula - BE EXPLICIT:
   - Show which column from which file
   - If keys are equivalent across files, map them clearly
   - Formula must work even if some files don't have all keys
3. List all value columns to keep
4. ALWAYS state SINGLE FILE output
5. State data handling strategy: {"KEEP_ALL (preserve all detail rows)" if strategy == "KEEP_ALL" else "AGGREGATE (one row per master entity)"}

{feedback_context}

OUTPUT:
# GRANULARITY
<All same level OR mixed (master+detail)>

# DATA HANDLING STRATEGY
<{"KEEP_ALL - Preserve all detail rows, join master data to each detail row" if strategy == "KEEP_ALL" else "AGGREGATE - Single file with smart aggregation per data type, one row per master entity"}>

# KEYS AND MASTER_UID CREATION
Key columns from each file:
- File1: <column_name_1, column_name_2>
- File2: <column_name_3>

MASTER_UID Formula (step-by-step):
1. For File1: Combine <column_name_1> + '_' + <column_name_2>
2. For File2: Use <column_name_3> directly (it's equivalent to File1's combined key)
3. Fill missing with '0' or handle NULL values
4. Final formula: MASTER_UID = combined_key_1 OR combined_key_2 (coalesce/fallback logic)

Example: If File1 has PropertyID and File2 has BuildingID+UnitID:
- File1: MASTER_UID = str(PropertyID)
- File2: MASTER_UID = str(BuildingID) + '_' + str(UnitID)
- These should identify the SAME entities

# COLUMNS
<Value columns to retain with their source files>
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

CHECK (100 points):
1. MASTER_UID formula is explicit and implementable? (30 pts)
   - Shows which columns from which files
   - Handles cases where files have different key structures
   - Clear step-by-step formula for each file

2. Keys handle all unique combinations? (20 pts)
   - No duplicate MASTER_UIDs possible
   - Covers all entities from all files

3. All value columns included? (20 pts)
   - No data loss
   - Columns properly attributed to source files

4. Strategy matches granularity? (15 pts)
   - Single file output stated
   - Aggregation strategy clear

5. Implementation clarity? (15 pts)
   - A programmer can write code from this schema
   - No ambiguous instructions

CRITICAL FAILURES (auto-score < 70):
- MASTER_UID formula uses same logic for all files (when keys differ)
- No step-by-step key creation per file
- Unclear how to merge files with different key structures

SCORE: 0-100
- 90+: Approve
- <90: Reject with specific issues and what to fix

OUTPUT (JSON):
{{
  "confidence_score": <number>,
  "feedback_text": "<Specific issues if rejected: which files lack clear MASTER_UID logic, what's ambiguous, how to fix. Else 'Approved'>"
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
        
        # Include strategy section based on one-to-many handling
        aggregation_section = ""
        strategy = state.get("aggregation_strategy", "")
        
        if state.get("one_to_many_detected") and strategy == "KEEP_ALL":
            # KEEP_ALL strategy - preserve all detail rows
            aggregation_section = f"""
ONE-TO-MANY STRATEGY: KEEP_ALL (Preserve All Detail Records)

⚠️ CRITICAL RULES FOR KEEP_ALL:
1. ALWAYS create ONE file only to '{safe_output_path}'
2. DO NOT aggregate or groupby - preserve every detail row
3. Join master data to detail data using merge/join (left join on MASTER_UID)
4. Each detail record (repair, transaction, log entry) appears as its own row
5. Master data columns are REPEATED for each detail row
6. Output will have MULTIPLE rows per master entity - THIS IS CORRECT
7. DO NOT use groupby().agg() - that destroys the detail data!

Code structure for KEEP_ALL:
- Load all files, create MASTER_UID per file
- Identify which files are MASTER (one row per entity) and DETAIL (many rows per entity)
- Concatenate all data with MASTER_UID
- Use pd.merge() to join master columns to detail rows
- OR simply pd.concat() if all files share MASTER_UID column
- Save ALL rows to output - do NOT collapse/aggregate
"""
        elif state.get("one_to_many_detected"):
            # Aggregation strategy
            agg_type = strategy.replace('AGGREGATE_', '').lower() if strategy else 'sum'
            
            aggregation_section = f"""
ONE-TO-MANY AGGREGATION STRATEGY: {strategy}

⚠️ CRITICAL RULES FOR AGGREGATION:
1. ALWAYS create ONE file only to '{safe_output_path}'
2. For NUMERIC fields:
    - Apply {agg_type} using groupby aggregation
3. For TEXT/CATEGORICAL fields (e.g., names, statuses, categories, types):
    - Use MODE (most frequent value) via: lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) > 0 else None
    - This picks the most representative/common value from the group
4. For DATE/DATETIME fields:
    - If strategy is MAX: use the latest (max) date
    - If strategy is MIN: use the earliest (min) date
    - Otherwise: use the latest (max) date as default (most recent is usually most relevant)
5. For DESCRIPTIVE/LONG TEXT fields (e.g., notes, comments, descriptions):
    - Concatenate unique non-null values with ' | ' separator
    - Use: lambda x: ' | '.join(x.dropna().unique().astype(str)) if len(x.dropna()) > 0 else None
6. For fields that can't be aggregated properly:
    - Use first non-null value: lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None
7. DO NOT create multiple files
8. Result: ONE row per master record with intelligently aggregated values
"""
        
        if strategy == "KEEP_ALL":
            prompt = f"""Generate Python code to implement the schema with KEEP_ALL strategy.

SCHEMA:
{state['schema']}{aggregation_section}

FILES:
{state['file_paths']}

CRITICAL REQUIREMENTS:
1. Output EXACTLY ONE file to '{safe_output_path}'
2. PRESERVE ALL DETAIL ROWS - do NOT aggregate/collapse
3. Join master data to each detail row using MASTER_UID
4. Output has MULTIPLE rows per master entity (this is the correct behavior)

Code structure (FOLLOW EXACTLY):

1. **Load all dataframes** (handle CSV/Excel with multiple sheets):
   ```python
   import pandas as pd
   import os
   
   dfs = {{}}  # Dict to store dataframes by source name
   ```

2. **Normalize keys for EACH dataframe INDIVIDUALLY**:
   - Convert key columns to string, strip whitespace
   - Handle NaN/None (replace with '0' or 'UNKNOWN')

3. **Create MASTER_UID for EACH dataframe based on schema**:
   IMPORTANT: Different files may have different key columns!
   Follow the schema's MASTER_UID formula EXACTLY for each file.

4. **Add source tracking**:
   ```python
   df['_source_file'] = 'filename_sheet'
   ```

5. **Merge strategy for KEEP_ALL**:
   - Identify MASTER files (one row per MASTER_UID) and DETAIL files (many rows per MASTER_UID)
   - Start with the DETAIL file(s) as the base (to preserve all rows)
   - Left-join MASTER file columns onto DETAIL rows using MASTER_UID
   - If multiple DETAIL files: concatenate them first, then join master data
   
   ```python
   # Example:
   # detail_df has many rows per MASTER_UID (e.g., repairs, transactions)
   # master_df has one row per MASTER_UID (e.g., property info)
   # final_df = detail_df.merge(master_df, on='MASTER_UID', how='left', suffixes=('', '_master'))
   ```

6. **Handle column conflicts** (same column name in multiple files):
   - Use suffixes in merge OR rename before merge

7. **DO NOT groupby or aggregate** - keep all rows as-is

8. **Save to Excel**:
   ```python
   final_df.to_excel('{safe_output_path}', index=False, engine='openpyxl')
   print(f"SUCCESS: Unified data saved with {{len(final_df)}} total records")
   print(f"MASTER_UIDs: {{final_df['MASTER_UID'].nunique()}} unique entities")
   ```

COMMON PITFALLS TO AVOID:
- ❌ Using groupby().agg() - this DESTROYS detail rows!
- ❌ Creating MASTER_UID with same formula for all files
- ❌ Not normalizing keys before merge
- ❌ Losing detail rows during merge (use left join from detail side)

OUTPUT: Complete, executable Python code only. No markdown, no explanations."""

        else:
            prompt = f"""Generate Python code to implement the schema.

SCHEMA:
{state['schema']}{aggregation_section}

FILES:
{state['file_paths']}

CRITICAL REQUIREMENTS:
1. ALWAYS output EXACTLY ONE file to '{safe_output_path}'
2. NEVER create multiple files
3. FOCUS: Create one unified master file with aggregated/merged data

For one-to-many data:
- Group by master record (MASTER_UID)
- Apply aggregation strategy to numeric columns
- For text/categorical fields: use MODE (most frequent value) for short text, concatenate unique values for long text/descriptions
- For date fields: use latest (max) date by default
- For any other non-numeric: use first non-null value
- Result: ONE row per unique MASTER_UID

Code structure (FOLLOW EXACTLY):

1. **Load all dataframes** (handle CSV/Excel with multiple sheets):
   ```python
   import pandas as pd
   import os
   
   dfs = []  # List to store (df, source_name) tuples
   ```

2. **Normalize keys for EACH dataframe INDIVIDUALLY**:
   - Convert key columns to string
   - Strip whitespace
   - Handle NaN/None (replace with '0' or 'UNKNOWN')
   - DO THIS BEFORE creating MASTER_UID

3. **Create MASTER_UID for EACH dataframe based on schema**:
   IMPORTANT: Different files may have different key columns!
   
   Example logic:
   ```python
   # For File1 with columns: PropertyID
   df1['MASTER_UID'] = df1['PropertyID'].astype(str).str.strip()
   
   # For File2 with columns: BuildingID, UnitID
   df2['MASTER_UID'] = (df2['BuildingID'].astype(str).str.strip() + '_' + 
                        df2['UnitID'].astype(str).str.strip())
   ```
   
   KEY POINT: Follow the schema's MASTER_UID formula EXACTLY for each file!

4. **Add source tracking** (before concatenation):
   ```python
   df['_source_file'] = 'filename_sheet'
   ```

5. **Concatenate all dataframes**:
   ```python
   merged_df = pd.concat(dfs, ignore_index=True, sort=False)
   ```

6. **Handle column conflicts** (if same column name exists in multiple files):
   - Use first non-null value OR
   - Apply aggregation logic

7. **Identify column data types for smart aggregation**:
   ```python
   numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
   date_cols = merged_df.select_dtypes(include=['datetime64']).columns.tolist()
   text_cols = merged_df.select_dtypes(include=['object']).columns.tolist()
   # Also try to detect date columns stored as strings
   for col in text_cols[:]:
       try:
           pd.to_datetime(merged_df[col].dropna().head(20), errors='raise')
           date_cols.append(col)
           text_cols.remove(col)
           merged_df[col] = pd.to_datetime(merged_df[col], errors='coerce')
       except:
           pass
   # Remove MASTER_UID and _source_file from aggregation lists
   ```

8. **Build SMART aggregation dictionary**:
   ```python
   agg_dict = {{}}
   # Numeric columns: apply the chosen aggregation
   for col in numeric_cols:
       if col != 'MASTER_UID':
           agg_dict[col] = '{strategy.replace("AGGREGATE_", "").lower() if strategy else "sum"}'
   
   # Date columns: use max (latest date) by default
   for col in date_cols:
       if col != 'MASTER_UID':
           agg_dict[col] = 'max'
   
   # Text columns: intelligent handling per column type
   # Short categorical fields (names, statuses, types) -> MODE (most frequent)
   # Long descriptive fields (notes, comments) -> concatenate unique values
   for col in text_cols:
       if col not in ('MASTER_UID', '_source_file'):
           avg_len = merged_df[col].dropna().astype(str).str.len().mean()
           if avg_len > 50:
               # Long text -> concatenate unique values
               agg_dict[col] = lambda x: ' | '.join(x.dropna().unique().astype(str)) if len(x.dropna()) > 0 else None
           else:
               # Short text -> mode (most frequent value)
               agg_dict[col] = lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) > 0 else (x.dropna().iloc[0] if len(x.dropna()) > 0 else None)
   ```

9. **Group by MASTER_UID and aggregate**:
   ```python
   final_df = merged_df.groupby('MASTER_UID', as_index=False).agg(agg_dict)
   ```

10. **Verify output**:
    ```python
    assert final_df['MASTER_UID'].nunique() == len(final_df), "ERROR: Duplicate MASTER_UIDs found!"
    print(f"✅ Unified data created: {{len(final_df)}} unique records")
    ```

11. **Preserve integer data types** (CRITICAL - prevents int→float conversion):
    ```python
    # After all processing, fix columns that were originally integers but became floats due to NaN
    for col in final_df.columns:
        if final_df[col].dtype == 'float64':
            # Check if all non-null values are whole numbers
            non_null = final_df[col].dropna()
            if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                final_df[col] = final_df[col].astype('Int64')  # Nullable integer type (handles NaN)
    ```

12. **Save to Excel**:
    ```python
    final_df.to_excel('{safe_output_path}', index=False, engine='openpyxl')
    print("SUCCESS: Unified data saved to master_unified_data.xlsx")
    ```

COMMON PITFALLS TO AVOID:
- ❌ Creating MASTER_UID with same formula for all files (keys differ by file!)
- ❌ Not normalizing keys before concatenation (leads to mismatches)
- ❌ Forgetting to handle NaN in key columns (breaks groupby)
- ❌ Using wrong aggregation type (text columns don't sum!)
- ❌ Not verifying unique MASTER_UIDs after aggregation
- ❌ Integer columns becoming float due to NaN (use Int64 nullable type!)

OUTPUT: Complete, executable Python code only. No markdown, no explanations."""
        
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
        
        if user_choice == "keep_all":
            # User explicitly wants ALL detail rows preserved
            logger.info("👤 USER CHOICE: KEEP ALL RECORDS")
            logger.info("   All detail/transaction rows will be preserved in output")
            logger.info("   Output will have multiple rows per master entity")
            logger.info("✅ Proceeding with KEEP_ALL strategy")
            
            return {
                "one_to_many_detected": True,
                "one_to_many_resolution": "keep_all",
                "aggregation_strategy": "KEEP_ALL",
                "user_intent": "keep_all"
            }
        
        elif user_choice == "auto_solve":
            logger.info("🤖 AUTO-SOLVE MODE: LLM will intelligently determine data handling...")
            
            # Check if target mode is active - if so, use target text to inform strategy
            target_text = state.get("target_schema_input", "")
            target_context = ""
            if state.get("target_mode_enabled", False) and target_text:
                target_context = f"""
USER'S TARGET SPECIFICATION (CRITICAL - this tells you what the user actually wants):
{target_text}

IMPORTANT: Analyze the user's target text carefully:
- If user mentions "complete history", "all records", "full log", "detailed", "every transaction", 
  "repair history", "all repairs", etc. → recommend KEEP_ALL
- If user mentions "summary", "total", "aggregate", "per property", "one row per" → recommend aggregation
- The user's intent in target text OVERRIDES default aggregation behavior
"""
            
            prompt = f"""Analyze this data structure and recommend the BEST strategy for handling one-to-many data.

IDENTIFIERS:
{state['identifiers']}

DATA:
{state['dfs_sample_str']}
{target_context}

Your data has one-to-many relationships. Choose ONE strategy:
1. KEEP_ALL: Preserve ALL detail rows (multiple rows per master entity in output)
   - Best when: user wants complete history/logs/transaction details
   - Output: Each detail record appears as its own row, joined with master data
2. AGGREGATE_MAX: Keep maximum values from detail records
3. AGGREGATE_MIN: Keep minimum values from detail records  
4. AGGREGATE_SUM: Sum all detail record values
5. AGGREGATE_AVG: Average all detail record values
6. AGGREGATE_COUNT: Count detail records per master

Considering the data nature, business logic, and especially the USER'S TARGET TEXT (if provided), 
choose the BEST ONE ONLY.

REQUIREMENT: You MUST recommend ONE and ONLY ONE strategy.

OUTPUT (JSON):
{{
  "recommended_strategy": "<KEEP_ALL, AGGREGATE_MAX, AGGREGATE_MIN, AGGREGATE_SUM, AGGREGATE_AVG, or AGGREGATE_COUNT>",
  "reasoning": "<Why this is best for this data and user intent>"
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
            
            logger.info(f"✅ Auto-solve complete - proceeding with strategy: {strategy}")
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
        logger.info("   - Text fields intelligently aggregated (mode/concatenation)")
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
            "detail_files_info": "",
            
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
                if agg_strategy == 'KEEP_ALL':
                    logger.info(f"   - One-to-many data: ALL detail rows preserved")
                    logger.info("   - Master data joined to each detail row")
                else:
                    logger.info(f"   - One-to-many data aggregated using: {agg_strategy}")
                    logger.info("   - Numeric fields: aggregated per strategy")
                    logger.info("   - Text fields: intelligently aggregated (mode/concatenation)")
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