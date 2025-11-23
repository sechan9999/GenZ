# Python vs SAS: Side-by-Side Code Comparison

## Quick Reference for Duplicate Detection QA

---

## STEP 1: Identify All Duplicates

### Python
```python
# Define key columns
key_columns = ['facility_id', 'resident_id', 'reporting_week']

# Find all duplicates (keep=False marks ALL duplicates)
duplicates = df[df.duplicated(subset=key_columns, keep=False)]
```

### SAS
```sas
/* Sort by key columns */
PROC SORT DATA=input_data OUT=sorted_data;
    BY facility_id resident_id reporting_week;
RUN;

/* Flag all duplicates using FIRST./LAST. */
DATA all_duplicates;
    SET sorted_data;
    BY facility_id resident_id reporting_week;

    /* If FIRST=1 AND LAST=1, record is unique (not duplicate) */
    /* If NOT (FIRST AND LAST), record is part of duplicate group */
    IF NOT (FIRST.reporting_week AND LAST.reporting_week) THEN OUTPUT;
RUN;
```

**Key Insight:**
- Python: Boolean masking with `.duplicated(keep=False)`
- SAS: BY-group processing with `FIRST./LAST.` automatic variables

---

## STEP 2: Count Total Duplicates

### Python
```python
if not duplicates.empty:
    print(f"CRITICAL: Found {len(duplicates)} duplicate records.")
```

### SAS
```sas
/* Count duplicates and store in macro variable */
PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :dup_total
    FROM all_duplicates;
QUIT;

/* Print to log */
%PUT CRITICAL: Found &dup_total duplicate records.;
```

**Key Insight:**
- Python: `len(df)` returns row count
- SAS: `PROC SQL COUNT(*)` with `INTO :macrovar` stores result

---

## STEP 3: Detect Conflicting Status Values

### Python
```python
# Group by key columns and count unique values in covid_status
conflicts = duplicates.groupby(key_columns)['covid_status'].nunique()

# Keep only groups with more than 1 unique value (conflicts)
true_conflicts = conflicts[conflicts > 1]
```

### SAS
```sas
/* Group by key columns and count distinct status values */
PROC SQL;
    CREATE TABLE status_conflicts AS
    SELECT
        facility_id,
        resident_id,
        reporting_week,
        COUNT(DISTINCT covid_status) AS status_count,
        COUNT(*) AS record_count
    FROM all_duplicates
    GROUP BY facility_id, resident_id, reporting_week
    HAVING status_count > 1;  /* Only keep groups with conflicts */
QUIT;
```

**Key Insight:**
- Python: `.groupby().nunique()` returns Series
- SAS: `GROUP BY` with `COUNT(DISTINCT)` and `HAVING` clause

---

## STEP 4: Report Results

### Python
```python
if not duplicates.empty:
    # ... process conflicts ...
    return true_conflicts
else:
    return "QA PASSED: No duplicates found."
```

### SAS
```sas
/* Count conflicts */
PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :conflict_count
    FROM status_conflicts;
QUIT;

/* Conditional reporting using macro logic */
%MACRO report_results;
    %IF &conflict_count > 0 %THEN %DO;
        %PUT ERROR: Found &conflict_count groups with conflicting COVID status!;

        TITLE "DUPLICATE RECORDS WITH CONFLICTING COVID STATUS";
        PROC PRINT DATA=status_conflicts;
            VAR facility_id resident_id reporting_week status_count record_count;
        RUN;
        TITLE;
    %END;
    %ELSE %IF &dup_total = 0 %THEN %DO;
        %PUT NOTE: QA PASSED: No duplicates found.;
    %END;
    %ELSE %DO;
        %PUT NOTE: Found &dup_total duplicates but no conflicting status values.;
    %END;
%MEND report_results;

%report_results;
```

**Key Insight:**
- Python: Function returns different types (DataFrame or string)
- SAS: Macro logic with log messages and PROC PRINT reports

---

## Common Operations: Quick Translation Table

| Task | Python Pandas | SAS |
|------|---------------|-----|
| **Find duplicates** | `df.duplicated(subset=cols, keep=False)` | `PROC SORT; DATA step with FIRST./LAST.` |
| **Remove duplicates** | `df.drop_duplicates(subset=cols)` | `PROC SORT NODUPKEY` |
| **Count rows** | `len(df)` | `PROC SQL COUNT(*)` |
| **Group by** | `df.groupby(['col1', 'col2'])` | `PROC SQL GROUP BY col1, col2` |
| **Count unique** | `.nunique()` | `COUNT(DISTINCT col)` |
| **Filter** | `df[df['col'] > 1]` | `HAVING col > 1` or `WHERE col > 1` |
| **Print** | `print(f"Value: {x}")` | `%PUT Value: &x;` |
| **Conditional** | `if condition:` | `%IF condition %THEN %DO;` |
| **Create dataset** | `new_df = df[...]` | `DATA new; SET df; ... RUN;` |

---

## Complete Working Example

### Python (Original)
```python
import pandas as pd

def quality_check_duplicates(df):
    """Check for duplicate records and conflicting status."""

    key_columns = ['facility_id', 'resident_id', 'reporting_week']
    duplicates = df[df.duplicated(subset=key_columns, keep=False)]

    if not duplicates.empty:
        print(f"CRITICAL: Found {len(duplicates)} duplicate records.")
        conflicts = duplicates.groupby(key_columns)['covid_status'].nunique()
        true_conflicts = conflicts[conflicts > 1]
        return true_conflicts
    else:
        return "QA PASSED: No duplicates found."

# Usage
result = quality_check_duplicates(my_dataframe)
print(result)
```

### SAS (Converted)
```sas
/* Sort by key columns */
PROC SORT DATA=input_data OUT=sorted_data;
    BY facility_id resident_id reporting_week;
RUN;

/* Identify all duplicates */
DATA all_duplicates;
    SET sorted_data;
    BY facility_id resident_id reporting_week;
    IF NOT (FIRST.reporting_week AND LAST.reporting_week) THEN OUTPUT;
RUN;

/* Count total duplicates */
PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :dup_total FROM all_duplicates;
QUIT;
%PUT CRITICAL: Found &dup_total duplicate records.;

/* Find conflicting status values */
PROC SQL;
    CREATE TABLE status_conflicts AS
    SELECT facility_id, resident_id, reporting_week,
           COUNT(DISTINCT covid_status) AS status_count,
           COUNT(*) AS record_count
    FROM all_duplicates
    GROUP BY facility_id, resident_id, reporting_week
    HAVING status_count > 1;
QUIT;

/* Report results */
PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :conflict_count FROM status_conflicts;
QUIT;

%MACRO report_results;
    %IF &conflict_count > 0 %THEN %DO;
        %PUT ERROR: Found &conflict_count groups with conflicting COVID status!;
        PROC PRINT DATA=status_conflicts; RUN;
    %END;
    %ELSE %IF &dup_total = 0 %THEN %DO;
        %PUT NOTE: QA PASSED: No duplicates found.;
    %END;
%MEND report_results;

%report_results;
```

---

## Key Takeaways

### When to Use Python
✅ Interactive data exploration
✅ Rapid prototyping
✅ Small to medium datasets (< 5GB)
✅ Integration with web apps/APIs
✅ Modern visualization (seaborn, plotly)

### When to Use SAS
✅ Large datasets (> 10GB)
✅ Production ETL pipelines
✅ Regulatory/validated environments (FDA, HIPAA)
✅ Legacy system integration
✅ Enterprise scheduling (SAS Grid)

### Best Practice: Bilingual Approach
Many data professionals use **both**:
1. **Develop** in Python (jupyter notebooks, pandas)
2. **Validate** in SAS (production-grade, auditable)
3. **Deploy** to appropriate platform based on requirements

---

## Testing Both Versions

### Sample Test Data

**Python:**
```python
import pandas as pd

test_data = pd.DataFrame({
    'facility_id': [101, 101, 102, 102, 102],
    'resident_id': [1001, 1001, 1002, 1003, 1003],
    'reporting_week': [202401, 202401, 202401, 202401, 202401],
    'covid_status': ['Positive', 'Negative', 'Positive', 'Positive', 'Positive']
})

result = quality_check_duplicates(test_data)
```

**SAS:**
```sas
DATA input_data;
    INPUT facility_id resident_id reporting_week covid_status $ @@;
    DATALINES;
101 1001 202401 Positive
101 1001 202401 Negative
102 1002 202401 Positive
102 1003 202401 Positive
102 1003 202401 Positive
;
RUN;

/* Run the quality check code here */
```

### Expected Output (Both Languages)

**Console/Log:**
```
CRITICAL: Found 3 duplicate records.
ERROR: Found 1 groups with conflicting COVID status!
```

**Conflict Details:**
- Facility 101, Resident 1001, Week 202401: **CONFLICT** (Positive vs Negative)
- Facility 102, Resident 1003, Week 202401: Duplicate but same status (OK)

---

**Document Version**: 1.0
**Created**: 2025-11-23
**Purpose**: Quick reference for Python-to-SAS code translation
