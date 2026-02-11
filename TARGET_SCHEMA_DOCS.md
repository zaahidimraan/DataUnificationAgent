# Target Schema Feature - Technical Documentation

## Overview

The Target Schema feature adds user-defined schema capability to the DataUnificationAgent. Users can now specify their desired output format, and the system will attempt to map source data to match that target.

## Architecture

### Dual-Mode Design

The system operates in one of two modes:

1. **Auto Mode (Default)**
   - System analyzes data and generates optimal schema automatically
   - Uses existing `node_schema_maker` → `node_schema_evaluator` flow
   - No changes to existing behavior

2. **Target Mode (Optional)**
   - User provides target schema (file template or text description)
   - System maps source data to target using `node_target_schema_mapper` → `node_target_schema_validator`
   - Includes retry logic (up to 3 attempts)
   - Automatic fallback to Auto Mode if mapping fails

### State Machine Flow

```
┌─────────────────────────────────────────────────────────┐
│              Phase 1: Identification                     │
│         (Detect keys & relationships)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
         Has one-to-many? ──Yes→ One-to-Many Resolver
                   │
                   No
                   ↓
          Is target_mode_enabled?
                   │
        ┌──────────┴──────────┐
        │                     │
       YES                   NO
        │                     │
        ↓                     ↓
┌────────────────┐    ┌────────────────┐
│ Target Schema  │    │ Auto Schema    │
│    Mapper      │    │    Maker       │
└───────┬────────┘    └───────┬────────┘
        │                     │
        ↓                     ↓
┌────────────────┐    ┌────────────────┐
│ Target Schema  │    │ Auto Schema    │
│   Validator    │    │  Evaluator     │
└───────┬────────┘    └───────┬────────┘
        │                     │
  Score ≥90?              Score ≥90?
        │                     │
  Yes ──┴→ Code Generation ←─┴── Yes
        │
  No, retries < 3
        │
        ↓
    Retry with feedback
        │
  No, retries ≥ 3
        │
        ↓
┌────────────────┐
│ Target Fallback│
└───────┬────────┘
        │
        ↓
  Switch to Auto Mode
```

## Components

### 1. State Extensions

**New State Fields:**
```python
target_mode_enabled: bool          # Is target mode active?
target_schema_input: str           # Parsed target specification
target_schema_type: str            # 'file' or 'text'
target_validation_retries: int     # Current retry count (0-3)
target_validation_feedback: str    # Feedback from last validation
target_fallback_triggered: bool    # Did we fall back to auto?
```

### 2. Parser Function

**`_parse_target_schema(target_input, input_type)`**

Purpose: Convert user input into structured format for LLM

**File Input:**
- Reads CSV/Excel headers only (no data)
- Extracts column names
- Returns formatted list

**Text Input:**
- Takes raw text description
- Minimal formatting (preserves user intent)
- Returns structured text

**Output Format:**
```
TARGET COLUMNS (5 total):
  - customer_id
  - customer_name
  - total_revenue
  - order_count
  - average_order_value
```

### 3. Target Schema Mapper Node

**`node_target_schema_mapper(state)`**

Purpose: Map source data columns to user's target schema

**Inputs:**
- Target specification (from state)
- Source data samples
- Identifiers (from Phase 1)
- Aggregation strategy (if one-to-many)
- Previous feedback (if retry)

**Process:**
1. Analyze target columns vs source columns
2. Identify direct mappings
3. Determine required transformations/derivations
4. Mark unmappable columns with reason
5. Generate comprehensive mapping plan

**Output Schema Structure:**
```
# GRANULARITY
<One row per what entity?>

# TARGET MODE
ENABLED - Mapping to user-defined target schema

# MAPPING STRATEGY
<Transformation approach>

# TARGET COLUMNS
For each target column:
- Column Name: <exact name>
- Source Mapping: <source column(s)>
- Transformation: <calculation if needed>
- Status: MAPPED or UNMAPPABLE

# KEYS
<Identifier columns + Master_UID formula>

# ADDITIONAL COLUMNS
<Source columns not in target but preserved>
```

### 4. Target Schema Validator Node

**`node_target_schema_validator(state)`**

Purpose: Score mapping quality and provide feedback

**Validation Criteria (100 points):**

1. **Target Column Coverage (40 pts)**
   - All requested columns mapped: 40 pts
   - Some mapped with valid UNMAPPABLE reasons: 20-35 pts
   - Missing columns without explanation: 0 pts

2. **Mapping Correctness (30 pts)**
   - Semantically correct mappings: 30 pts
   - Data type compatibility: ±10 pts
   - Appropriate transformations: ±10 pts

3. **Data Preservation (15 pts)**
   - No important source data lost: 15 pts
   - Proper aggregation applied: ±5 pts

4. **Implementation Clarity (15 pts)**
   - Clear implementation instructions: 15 pts
   - Unambiguous mapping logic: ±5 pts

**Output:**
```json
{
  "confidence_score": 85,
  "target_columns_mapped": 4,
  "target_columns_total": 5,
  "unmappable_columns": ["impossible_metric_xyz"],
  "feedback_text": "Missing proper derivation for 'average_order_value'. Should use: SUM(order_total) / COUNT(orders)"
}
```

**Decision Logic:**
- Score ≥90: Approve → Code Generation
- Score <90 AND retries <3: Retry with feedback
- Score <90 AND retries ≥3: Trigger fallback

### 5. Target Fallback Node

**`node_target_fallback(state)`**

Purpose: Handle graceful degradation when target mapping fails

**Actions:**
1. Log clear warning message
2. Set `target_mode_enabled = False`
3. Set `target_fallback_triggered = True`
4. Reset retry counters for auto mode
5. Provide feedback explaining fallback

**Output:**
Routes to `schema_maker` (Auto Mode) to complete processing successfully

## User Interface

### Input Options

**Advanced Section (Collapsible):**
```html
🎯 Advanced: Define Target Schema (Optional)

Option 1: Upload Template File
  [File input: CSV/Excel with column headers]

OR

Option 2: Describe Target Schema
  [Text area: Natural language description]
```

### Success Messages

**Auto Mode:**
```
✅ Data unified successfully into single file.
```

**Target Mode - Success:**
```
✅ Data unified successfully using your target schema!
```

**Target Mode - Fallback:**
```
✅ Data unified successfully. 
⚠️ Note: Target schema mapping failed after 3 attempts - used auto-generated schema instead.
```

## Logging

### Target Mode Activation
```
📋 MODE: TARGET-DRIVEN SCHEMA
   Input: Template file (target_template.csv)
   Fallback: Auto mode (if target mapping fails after 3 attempts)
```

### Target Mapping Attempts
```
==========================================================
🎯 PHASE 2: TARGET-DRIVEN SCHEMA MAPPING
Attempt #1 / 3
==========================================================
🤖 Calling LLM to map source data to target schema...
✅ Target schema mapping proposal generated
📋 Mapping length: 1523 characters
```

### Validation Results
```
⚖️  Validating target schema mapping...
📊 Target Validation Score: 75/100
📊 Target Columns: 4/5 mapped
⚠️  Unmappable columns: impossible_metric_xyz
⚠️  Target mapping needs improvement (Score: 75)
💬 Feedback: Column 'impossible_metric_xyz' cannot be derived from source data...
🔄 Retry count: 1/3
```

### Fallback Trigger
```
⚠️  =======================================================
⚠️  TARGET MODE FALLBACK TRIGGERED
⚠️  =======================================================

📊 Unable to map source data to target schema after 3 attempts

🔄 FALLING BACK TO AUTO MODE:
   - Disabling target mode
   - Using automatic schema generation
   - This ensures your data is still unified successfully

💡 Recommendation: Review your target schema and source data compatibility
```

## Testing Strategy

### Test Cases

**1. Auto Mode (Baseline)**
- Verify existing functionality unchanged
- No target schema provided
- Standard output

**2. Target Mode - Achievable Requirements**
- Provide reasonable target schema
- Columns exist in source or can be derived
- Should succeed without fallback

**3. Target Mode - Challenging Requirements**
- Provide complex target schema
- May require multiple retry attempts
- Should either succeed or fall back gracefully

**4. Target Mode - Impossible Requirements**
- Request columns that don't exist
- Cannot be derived from source
- Must trigger fallback after 3 attempts

**5. Target Mode + One-to-Many**
- Combine target schema with one-to-many data
- Verify aggregation strategy applied correctly
- Target columns properly aggregated

### Expected Outcomes

All tests should:
- Never fail completely (fallback ensures success)
- Produce valid unified output
- Log clear status messages
- Preserve data integrity

## Edge Cases Handled

1. **No Target Input**
   - System defaults to Auto Mode
   - Zero impact on existing functionality

2. **Both File and Text Provided**
   - File takes precedence
   - Text is ignored

3. **Invalid Target File**
   - Parser returns None
   - Error message shown to user
   - Process stops gracefully

4. **Empty Target Description**
   - Treated as no target input
   - Falls through to Auto Mode

5. **Target File with No Headers**
   - Parser fails
   - Error logged
   - User notified

6. **One-to-Many Modal + Target Schema**
   - Target schema info stored in session
   - Restored after modal submission
   - Both features work together

7. **Duplicate Column Names in Target**
   - Mapper notes the issue
   - Validator flags it in feedback
   - May trigger retry or fallback

8. **Case Sensitivity in Column Names**
   - Mapper attempts case-insensitive matching
   - Documents exact names used

## Performance Considerations

### Additional LLM Calls

**Auto Mode:**
- Phase 2: 2 calls per attempt (maker + evaluator)
- Max: 6 calls (3 attempts × 2)

**Target Mode (Success):**
- Phase 2: 2 calls per attempt (mapper + validator)
- Max: 6 calls (3 attempts × 2)

**Target Mode (Fallback):**
- Phase 2: 6 calls (target) + up to 6 calls (auto)
- Max: 12 calls total

**Mitigation:**
- Fallback is rare (only on truly incompatible targets)
- Most requests succeed in 1-2 attempts
- Validation prevents unnecessary retries

## Security Considerations

1. **File Upload Validation**
   - Only CSV/Excel formats allowed
   - File size limits enforced
   - Secure filename handling

2. **Text Input Sanitization**
   - No execution of user text
   - Only used as LLM prompt context
   - Length limits could be added

3. **Session Storage**
   - Target schema info stored temporarily
   - Cleaned up after processing
   - No persistent storage of user data

## Future Enhancements

1. **Target Schema Templates**
   - Pre-built templates for common use cases
   - One-click selection

2. **Interactive Column Mapper**
   - Visual drag-and-drop interface
   - Real-time validation

3. **Confidence Threshold Adjustment**
   - User can set required validation score
   - Trade-off between accuracy and retries

4. **Partial Success Mode**
   - Accept 80%+ of target columns
   - Mark missing columns clearly
   - No fallback to auto

5. **Learning from History**
   - Remember successful mappings
   - Suggest similar mappings for repeat users

6. **Multi-File Target Support**
   - User specifies multiple target schemas
   - Different outputs for different granularities

## Maintenance Notes

### Key Files Modified

1. **app/services.py**
   - Extended `AgentState` with 6 new fields
   - Added `_parse_target_schema()` method
   - Added 3 new nodes (mapper, validator, fallback)
   - Updated graph routing logic
   - Updated `run()` method signature

2. **app/routes.py**
   - Added target schema input handling
   - Session storage for modal flow
   - Success message customization

3. **app/templates/index.html**
   - Added collapsible target schema section
   - Two input methods (file + text)
   - Status display in results

4. **README.md**
   - Comprehensive documentation
   - Usage examples
   - Architecture diagrams

### Breaking Changes

**None** - Feature is fully backward compatible

Existing functionality:
- Unchanged when target schema not provided
- Can be used exactly as before
- No migration needed

## Support & Troubleshooting

### Common Issues

**Issue:** Target mapping always fails
- **Cause:** Target columns don't match source semantics
- **Solution:** Review target vs source data structure
- **Workaround:** Let fallback work, then manually transform

**Issue:** Fallback takes too long
- **Cause:** 3 target attempts + 3 auto attempts
- **Solution:** Expected behavior for complex cases
- **Future:** Could add early-exit option

**Issue:** Aggregation not matching target
- **Cause:** One-to-many strategy conflicts with target
- **Solution:** Target mapper respects aggregation strategy
- **Check:** Logs show which strategy was applied

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key log markers:
- `🎯 TARGET MODE ENABLED` - Target mode activated
- `📊 Target Validation Score` - Validation results
- `⚠️  TARGET MODE FALLBACK` - Fallback triggered

## Conclusion

The Target Schema feature provides users with control over output format while maintaining system reliability through intelligent fallback mechanisms. The dual-mode design ensures existing functionality is preserved while adding powerful new capabilities for users who need specific output formats.
