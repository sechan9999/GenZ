# SAS Conversions

This directory contains Python-to-SAS code conversions with comprehensive documentation.

## Overview

The code examples demonstrate how to convert common data quality checks from Python pandas to SAS, specifically focusing on duplicate detection and validation for healthcare data (NHSN LTCF COVID-19 surveillance).

## Files

### 1. `quality_check_duplicates.sas`
**Full SAS implementation** of the duplicate detection quality check with:
- Comprehensive header comments
- Step-by-step explanations (in English)
- Interview talking points
- Production-ready code structure

### 2. `PYTHON_TO_SAS_CONVERSION_GUIDE.md`
**Detailed conversion guide** covering:
- Line-by-line code mapping
- Conceptual differences between Python and SAS
- Performance considerations
- When to use each language
- Testing instructions
- Interview preparation tips

### 3. `SIDE_BY_SIDE_COMPARISON.md`
**Quick reference** with:
- Side-by-side code comparisons
- Common operations translation table
- Complete working examples
- Test data and expected outputs

## Use Cases

### Healthcare Data Quality Assurance
- **NHSN LTCF**: Long-Term Care Facility COVID-19 surveillance
- **Duplicate Detection**: Identify duplicate patient/resident records
- **Conflict Detection**: Find records with contradictory status values
- **Production ETL**: Integrate into batch processing pipelines

### Interview Preparation
These materials are ideal for demonstrating:
- Bilingual programming skills (Python + SAS)
- Understanding of data quality concepts
- Ability to translate between languages
- Healthcare data experience

## Quick Start

### Python Version
```python
import pandas as pd

# Load your data
df = pd.read_csv('ltcf_data.csv')

# Run quality check
from quality_check_duplicates import quality_check_duplicates
result = quality_check_duplicates(df)
print(result)
```

### SAS Version
```sas
/* Load your data */
PROC IMPORT DATAFILE="ltcf_data.csv" OUT=input_data DBMS=CSV REPLACE;
    GETNAMES=YES;
RUN;

/* Run quality check */
%INCLUDE 'quality_check_duplicates.sas';
```

## Key Concepts Covered

### Python Pandas
- `.duplicated(keep=False)` - Find all duplicates
- `.groupby().nunique()` - Count distinct values per group
- Boolean masking and filtering
- Conditional returns

### SAS
- `PROC SORT` with BY-group processing
- `FIRST.` and `LAST.` automatic variables
- `PROC SQL` with `GROUP BY` and `HAVING`
- Macro programming (`%MACRO`, `%IF/%THEN`)
- Macro variables (`INTO :var`)

## Translation Cheat Sheet

| Python | SAS |
|--------|-----|
| `df.duplicated()` | `FIRST./LAST.` logic |
| `df.drop_duplicates()` | `PROC SORT NODUPKEY` |
| `len(df)` | `PROC SQL COUNT(*)` |
| `df.groupby()` | `PROC SQL GROUP BY` |
| `.nunique()` | `COUNT(DISTINCT)` |
| `print()` | `%PUT` |
| `if/else` | `%IF/%THEN/%DO` |

## Performance Notes

- **Python**: Best for < 5GB data, interactive analysis
- **SAS**: Optimized for > 10GB data, production ETL

## Compliance

These examples follow best practices for:
- **FDA 21 CFR Part 11**: Validated code in regulated environments
- **HIPAA**: Healthcare data privacy and security
- **CDC NHSN**: Public health surveillance reporting

## References

- **SAS Documentation**: https://documentation.sas.com/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **CDC NHSN**: https://www.cdc.gov/nhsn/ltc/

## Author

Generated as part of the GenZ Agent project
Date: 2025-11-23

## License

Educational and demonstration purposes
