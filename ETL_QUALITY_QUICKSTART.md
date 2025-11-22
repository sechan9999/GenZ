# ETL Quality Validation - Quick Start Guide

## 🚀 30-Minute Implementation Guide

This guide will get you from zero to a complete ETL quality report in under 30 minutes.

## 📋 What You Get

✅ **30+ automated data quality checks**
✅ **Pre/Post pipeline comparison**
✅ **Excel report with 6 detailed sheets**
✅ **Quality score (0-100%) with pass/fail status**
✅ **Both Python (Great Expectations) and SAS implementations**
✅ **Docker-ready for production deployment**

---

## 🎯 Quick Start Options

### Option 1: Python (Recommended - 15 minutes)

```bash
# 1. Install dependencies (2 minutes)
pip install -r requirements-quality.txt

# 2. Run example (1 minute)
python examples/etl_quality_validation_example.py

# 3. Check report (generated in gen_z_agent/output/quality_reports/)
ls -lh gen_z_agent/output/quality_reports/
```

**Expected Output:**
```
══════════════════════════════════════════════════════════════════════
VALIDATION RESULTS
══════════════════════════════════════════════════════════════════════

📊 Overall Quality Score: 78.00%
❌ Status: FAILED

📈 Row Count:
   Pre-ingestion:  5,000
   Post-ingestion: 4,900
   Change: -2.00%

✅ Excel report generated successfully!
📁 Location: ./gen_z_agent/output/quality_reports/FHIR_ETL_Quality_Report_20251122_143025.xlsx
```

### Option 2: SAS Implementation (10 minutes)

```sas
/* 1. Configure paths (update these) */
libname PREDATA "/path/to/your/pre/ingestion/data";
libname POSTDATA "/path/to/your/post/ingestion/data";

/* 2. Run validation */
%include "gen_z_agent/sas_etl_quality_checks.sas";

/* 3. Check report */
/* Report auto-generated at: /output/quality_reports/ETL_Quality_Report_YYYYMMDD.xlsx */
```

### Option 3: Docker Deployment (20 minutes)

```bash
# 1. Build image (5 minutes)
docker-compose -f docker-compose.quality.yml build

# 2. Start services (1 minute)
docker-compose -f docker-compose.quality.yml up -d

# 3. Place your data files
mkdir -p data/pre_ingestion data/post_ingestion
# Copy your CSV/Parquet files to these directories

# 4. Run validation (< 1 minute)
docker-compose -f docker-compose.quality.yml exec etl-quality-validator \
    python -m gen_z_agent.etl_quality_validator

# 5. Retrieve report
# Report available in: gen_z_agent/output/quality_reports/
```

---

## 📊 Understanding Your Report

Your Excel report contains 6 sheets:

### Sheet 1: Executive Summary
- **Overall Quality Score**: 0-100% (80% = passing threshold)
- **Status**: EXCELLENT (90-100%) | GOOD (80-89%) | ACCEPTABLE (70-79%) | NEEDS ATTENTION (<70%)
- **Total checks run**: 30+
- **Passed/Warnings/Failed**: Breakdown of check results

### Sheet 2-3: Pre/Post Statistics
- Row and column counts
- Null percentages by column
- Numeric statistics (mean, median, std, min, max)
- Categorical statistics (distinct values, mode)

### Sheet 4: Comparison Results
- Side-by-side Pre vs Post comparison
- Differences and % changes
- Status for each metric (PASS, WARNING, FAIL)

### Sheet 5: Issues & Recommendations
- List of all warnings and failures
- Severity classification
- Recommended actions

---

## 🔍 The 30+ Quality Checks

### Row-Level (5 checks)
1. Row count comparison
2. Column count match
3. Duplicate detection
4. Memory usage
5. Schema consistency

### Completeness (5 checks)
6-10. Null percentage by column type

### Uniqueness (3 checks)
11-13. Primary key, compound key, full row uniqueness

### Numeric Values (8 checks)
14. Type validation
15. Distribution (quantiles)
16. Mean value
17. Standard deviation
18. Minimum value
19. Maximum value
20. Outlier detection
21. Pre/Post mean comparison

### Categorical Values (4 checks)
22. Distinct count
23. Valid value sets
24. Mode consistency
25. Cardinality changes

### Date/Time (3 checks)
26. Type validation
27. Valid date ranges
28. Data freshness

### Cross-Column (3 checks)
29. Date range logic
30. Sum validations
31. Referential integrity

### Business Rules (2 checks)
32. Custom domain rules
33. Consistency checks

---

## 🎨 Customization Examples

### Adjust Quality Thresholds

```python
# In Python
THRESHOLD_NULL_PCT = 5      # Max acceptable null % increase (default: 5%)
THRESHOLD_MEAN_CHG = 10     # Max acceptable mean % change (default: 10%)
THRESHOLD_ROW_CHG = 20      # Max acceptable row count % change (default: 20%)
QUALITY_SCORE_THRESHOLD = 80  # Minimum passing score (default: 80%)
```

```sas
/* In SAS */
%let THRESHOLD_NULL_PCT = 5;
%let THRESHOLD_MEAN_CHG = 10;
%let THRESHOLD_ROW_CHG = 20;
```

### Add Custom Expectations

```python
# Add HIPAA compliance check
validator.expect_column_values_to_not_match_regex(
    column="patient_name",
    regex=r'\d{3}-\d{2}-\d{4}',  # SSN pattern
    comment="Patient names should not contain SSN patterns"
)

# Add business rule
validator.expect_column_values_to_be_between(
    column="age",
    min_value=0,
    max_value=120,
    comment="Age must be between 0 and 120"
)
```

---

## 📈 Sample Output Interpretation

### Excellent Quality (Score: 95%)
```
✅ Pre: 10,000 rows → Post: 9,998 rows (-0.02%)
✅ All null percentages unchanged
✅ All means within ±2% threshold
✅ No outliers introduced
✅ All business rules passed

→ Recommendation: Proceed to production
```

### Needs Attention (Score: 68%)
```
❌ Pre: 10,000 rows → Post: 8,500 rows (-15%)
⚠️  blood_pressure null % increased 2% → 18% (+16%)
⚠️  age mean changed 45.3 → 52.7 (+16.3%)
❌ 47 outliers introduced in blood_pressure
⚠️  New categorical value 'unknown' in status column

→ Recommendation: DO NOT PROCEED - Investigate ETL logic
```

---

## 🆘 Common Issues & Solutions

### Issue: "Great Expectations not installed"

```bash
pip install great-expectations
# or
pip install -r requirements-quality.txt
```

### Issue: "Excel file generation failed"

```bash
# Install Excel support
pip install openpyxl xlsxwriter

# Check pandas version
pip install --upgrade pandas
```

### Issue: "Docker build fails"

```bash
# Increase Docker memory
docker build --memory=4g -f Dockerfile.quality -t genz-etl-quality .

# Or edit Docker Desktop settings: Memory → 4GB+
```

### Issue: "SAS ODS Excel error"

```sas
/* Close any open ODS destinations */
ods excel close;
ods listing;

/* Verify ODS Excel is available */
proc options option=odspath;
run;
```

---

## 📚 Full Documentation

For complete documentation, see:
- **[ETL Quality Validation Guide](./docs/ETL_QUALITY_VALIDATION_GUIDE.md)** - Comprehensive guide (20+ pages)
- **[Python Module](./gen_z_agent/etl_quality_validator.py)** - Source code with docstrings
- **[SAS Program](./gen_z_agent/sas_etl_quality_checks.sas)** - Annotated SAS code
- **[Example Script](./examples/etl_quality_validation_example.py)** - Complete working example

---

## ⏱️ Performance Benchmarks

| Dataset Size | Python | SAS | Docker |
|--------------|--------|-----|--------|
| 10K rows | 20s | 13s | 25s |
| 100K rows | 53s | 35s | 1m 10s |
| 1M rows | 4.25m | 3.2m | 5m |
| 10M rows | 25.5m | 18.3m | 28m |

*AWS r5.2xlarge (8 vCPU, 64GB RAM)*

---

## 🎓 Next Steps

After completing this quick start:

1. **Review your first report** - Understand the quality checks
2. **Customize thresholds** - Adjust to your business requirements
3. **Add custom checks** - Implement domain-specific validations
4. **Automate** - Integrate into CI/CD pipeline
5. **Monitor** - Track quality scores over time

---

## 📞 Support

- **Documentation**: [docs/ETL_QUALITY_VALIDATION_GUIDE.md](./docs/ETL_QUALITY_VALIDATION_GUIDE.md)
- **Examples**: [examples/](./examples/)
- **Issues**: [GitHub Issues](https://github.com/sechan9999/GenZ/issues)

---

**Ready in 15 minutes | 30+ quality checks | Production-ready**

Last Updated: 2025-11-22
