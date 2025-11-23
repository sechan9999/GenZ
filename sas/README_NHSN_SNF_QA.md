# NHSN SNF Quality Assurance Analysis

Comprehensive SAS-based quality assurance system for National Healthcare Safety Network (NHSN) Skilled Nursing Facility surveillance data.

## 📋 Overview

This QA system performs rigorous quality checks on NHSN SNF data, identifying:
- **Data quality issues** (missing values, invalid ranges, impossible values)
- **Statistical anomalies** (outliers, spikes, unusual patterns)
- **Geographic patterns** (facility, state, and HHS region-level analysis)
- **Temporal spikes** (sudden week-over-week changes)

## 🗂️ Files in This Directory

| File | Description | Purpose |
|------|-------------|---------|
| `nhsn_snf_qa_analysis.sas` | **Main QA program** (1,070 lines) | Complete QA analysis with 10 components |
| `nhsn_snf_sample_data_generator.sas` | Sample data generator | Creates test data with injected anomalies |
| `README_NHSN_SNF_QA.md` | This file | Quick start guide |
| `../docs/nhsn_snf_qa_methodology.md` | **Detailed methodology** | Statistical methods and interpretation |

## 🚀 Quick Start

### Step 1: Generate Sample Data (Optional - for testing)

```sas
/* Generate 500 facilities, 16 weeks of data with 5% anomaly rate */
%include '/home/user/GenZ/sas/nhsn_snf_sample_data_generator.sas';

/* This creates WORK.NHSN_SNF_DATA */
```

### Step 2: Run QA Analysis

```sas
/* Your data should be in WORK.NHSN_SNF_DATA */
/* Or load your own data: */
data work.nhsn_snf_data;
    set mylib.my_nhsn_data;
run;

/* Run comprehensive QA */
%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';
```

### Step 3: Review Results

The QA program generates multiple output datasets and CSV exports:

**Key Output Datasets:**
- `WORK.QA_SUMMARY` - Overall anomaly counts by category
- `WORK.FACILITY_ANOMALIES` - Facility-level spikes and anomalies
- `WORK.STATE_ANOMALIES` - State-level aggregated anomalies
- `WORK.REGION_ANOMALIES` - HHS region-level patterns
- `WORK.TEMPORAL_SPIKES` - Week-over-week sudden changes
- `WORK.STATISTICAL_OUTLIERS` - IQR-based outliers

**Exported CSV Files** (in `/home/user/GenZ/output/`):
- `nhsn_facility_anomalies_YYYYMMDD.csv`
- `nhsn_state_anomalies_YYYYMMDD.csv`
- `nhsn_qa_comprehensive_report_YYYYMMDD.csv`
- `nhsn_qa_summary_YYYYMMDD.csv`
- `nhsn_qa_event_log_YYYYMMDD.csv`

## 📊 Sample Output Review

### Review Top Anomalies

```sas
/* View most critical facility anomalies */
proc print data=work.facility_anomalies(obs=20);
    where severity = 'CRITICAL';
    var facility_name state metric_type week_start_date
        current_value baseline_mean z_score;
run;

/* View state-level spikes */
proc print data=work.state_anomalies;
    var state metric_type week_start_date
        current_value baseline_mean z_score anomaly_flag;
run;

/* View QA summary */
proc print data=work.qa_summary;
run;
```

### Example Output

```
Facility Anomalies Summary:
--------------------------------------------------
Facility: Oakwood SNF
State: FL
Metric Type: COVID_CASES
Week: 2025-11-15
Current Value: 45
Baseline Mean: 8.5
Z-Score: 6.2
Severity: CRITICAL
--------------------------------------------------
Interpretation: Facility experiencing outbreak
6.2 standard deviations above normal baseline.
Immediate investigation required.
```

## ⚙️ Configuration Options

You can customize the QA analysis by modifying macro variables:

```sas
/* At the top of nhsn_snf_qa_analysis.sas, modify: */

%let lookback_weeks = 12;    /* Baseline calculation period (default: 12) */
%let spike_threshold = 3;     /* Z-score threshold for spikes (default: 3) */
%let outlier_threshold = 3;   /* IQR multiplier for outliers (default: 3) */
```

**Recommendations:**
- **More sensitive detection**: Set `spike_threshold = 2` (more false positives)
- **More conservative**: Set `spike_threshold = 4` (fewer alerts, may miss outbreaks)
- **Longer baseline**: Set `lookback_weeks = 26` (6 months, better for seasonal data)

## 📋 Data Requirements

Your input dataset `WORK.NHSN_SNF_DATA` must contain:

### Required Variables

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `facility_id` | Char | Unique facility ID | "SNF00123" |
| `facility_name` | Char | Facility name | "Springfield SNF" |
| `state` | Char | State (2-letter) | "MA" |
| `hhs_region` | Num | HHS region (1-10) | 1 |
| `report_date` | Date | Report date | '01NOV2025'd |
| `metric_type` | Char | Metric type | "COVID_CASES" |
| `metric_value` | Num | Metric value | 12 |
| `population_at_risk` | Num | Denominator | 120 |

### Sample Data Creation

```sas
data work.nhsn_snf_data;
    length facility_id $10 facility_name $100 state $2 metric_type $30;
    input facility_id $ facility_name $ 1-40 state $ hhs_region
          report_date :date9. metric_type $ metric_value population_at_risk;
    format report_date yymmdd10.;
    datalines;
SNF00001 Springfield Skilled Nursing MA 1 01NOV2025 COVID_CASES 5 120
SNF00001 Springfield Skilled Nursing MA 1 08NOV2025 COVID_CASES 12 120
SNF00001 Springfield Skilled Nursing MA 1 15NOV2025 COVID_CASES 8 120
SNF00002 Portland Care Center ME 1 01NOV2025 COVID_CASES 3 85
;
run;
```

## 📈 Understanding the Results

### Severity Levels

| Severity | Z-Score | Interpretation | Action |
|----------|---------|----------------|--------|
| **CRITICAL** | > 5.0 | Extremely rare event | Investigate within 24 hours |
| **WARNING** | 3.0 - 5.0 | Significant anomaly | Investigate within 48 hours |
| **UNUSUAL_LOW** | < -3.0 | Unusually low values | Review for under-reporting |
| **NORMAL** | -3.0 to 3.0 | Within expected range | Routine monitoring |

### Anomaly Flags

- **HIGH_SPIKE**: Value >3 SD above baseline (potential outbreak)
- **EXTREME_CHANGE**: >200% increase from baseline (data error or major event)
- **LOW_SPIKE**: Value <-3 SD below baseline (reporting issue)
- **LOW_REPORTING**: <50% of facilities reporting (system issue)
- **HIGH_OUTLIER**: Above upper IQR fence (extreme compared to peers)
- **LOW_OUTLIER**: Below lower IQR fence (unusually low)

### Detection Methods

| Method | Best For | Strengths | Limitations |
|--------|----------|-----------|-------------|
| **Z-Score** | Normally distributed data | Statistically rigorous | Sensitive to outliers |
| **IQR** | Skewed distributions | Robust to outliers | Less specific |
| **Week-over-Week** | Sudden changes | Detects rapid spikes | Misses gradual trends |

## 🔍 Common Use Cases

### Use Case 1: Weekly Outbreak Surveillance

**Goal**: Identify facility outbreaks in near real-time

```sas
/* Focus on recent COVID cases with critical severity */
proc sql;
    create table outbreaks_to_investigate as
    select facility_name, state, week_start_date,
           current_value, baseline_mean, z_score, severity
    from work.facility_anomalies
    where metric_type = 'COVID_CASES'
        and severity in ('CRITICAL', 'WARNING')
        and week_start_date >= intnx('week', today(), -2, 'beginning')
    order by z_score desc;
quit;

proc print data=outbreaks_to_investigate;
    title "Facilities Requiring Immediate Investigation";
run;
```

### Use Case 2: State-Level Resource Allocation

**Goal**: Identify states with elevated COVID burden

```sas
proc sql;
    create table state_covid_burden as
    select state, hhs_region, current_value,
           baseline_mean, pct_change, anomaly_flag
    from work.state_anomalies
    where metric_type = 'COVID_CASES'
        and anomaly_flag = 'HIGH_SPIKE'
    order by current_value desc;
quit;
```

### Use Case 3: Data Quality Monitoring

**Goal**: Identify facilities with reporting issues

```sas
/* Facilities with frequent data quality violations */
proc freq data=work.qa_range_violations;
    tables facility_id * violation_type / out=violation_counts;
run;

proc print data=violation_counts(where=(count > 5));
    title "Facilities with Persistent Data Quality Issues";
run;
```

### Use Case 4: Performance Benchmarking

**Goal**: Compare facility to state peers

```sas
proc sql;
    create table facility_benchmarks as
    select facility_name, state, metric_type,
           total_value as facility_value,
           state_avg, state_z_score, value_rank,
           comparison_flag
    from work.state_facility_outliers
    where facility_id = 'SNF00123'  /* Your facility */
    order by metric_type, week_start_date desc;
quit;
```

## 📖 Detailed Documentation

For comprehensive methodology, statistical formulas, and interpretation guidelines:

**Read**: [`/home/user/GenZ/docs/nhsn_snf_qa_methodology.md`](../docs/nhsn_snf_qa_methodology.md)

This 2,000+ line document includes:
- Statistical method details (Z-score, IQR, control charts)
- Anomaly detection algorithms
- HHS region definitions
- Interpretation guidelines by severity
- Limitations and assumptions
- Recommended workflows

## 🧪 Testing the QA System

### Test with Sample Data

```sas
/* 1. Generate sample data with known anomalies */
%include '/home/user/GenZ/sas/nhsn_snf_sample_data_generator.sas';

/* 2. Run QA analysis */
%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';

/* 3. Verify anomalies were detected */
proc sql;
    select count(*) as detected_anomalies
    from work.nhsn_snf_data
    where anomaly_label ne 'NORMAL';

    select count(*) as qa_flagged_anomalies
    from work.facility_anomalies;
quit;

/* Expected: ~5% of records flagged (with default 5% anomaly rate) */
```

### Validate Detection Accuracy

```sas
/* Join injected anomalies with QA detections */
proc sql;
    create table validation as
    select
        a.facility_id,
        a.metric_type,
        a.week_start_date,
        b.anomaly_label as true_label,
        a.anomaly_flag as detected_flag,
        case
            when b.anomaly_label ne 'NORMAL' and a.anomaly_flag ne 'NORMAL'
                then 'TRUE_POSITIVE'
            when b.anomaly_label = 'NORMAL' and a.anomaly_flag ne 'NORMAL'
                then 'FALSE_POSITIVE'
            when b.anomaly_label ne 'NORMAL' and missing(a.anomaly_flag)
                then 'FALSE_NEGATIVE'
            else 'TRUE_NEGATIVE'
        end as classification
    from work.facility_anomalies as a
    full join (
        select facility_id, metric_type,
               intnx('week', report_date, 0, 'beginning') as week_start_date,
               anomaly_label
        from work.nhsn_snf_data
    ) as b
        on a.facility_id = b.facility_id
        and a.metric_type = b.metric_type
        and a.week_start_date = b.week_start_date;
quit;

proc freq data=validation;
    tables classification;
    title "QA Detection Performance";
run;
```

## 🔧 Troubleshooting

### Issue: No anomalies detected

**Possible Causes:**
1. Insufficient data (need ≥4 weeks per facility)
2. Thresholds too conservative (`spike_threshold` too high)
3. Data too uniform (low variance)

**Solutions:**
```sas
/* Lower detection threshold */
%let spike_threshold = 2;

/* Check data distribution */
proc means data=nhsn_snf_data n mean std min max;
    class metric_type;
    var metric_value;
run;
```

### Issue: Too many false positives

**Possible Causes:**
1. Thresholds too sensitive
2. High natural variance in data
3. Seasonal patterns not accounted for

**Solutions:**
```sas
/* Increase threshold */
%let spike_threshold = 4;

/* Use longer baseline for seasonal data */
%let lookback_weeks = 26;
```

### Issue: Missing output files

**Possible Causes:**
1. Output directory doesn't exist
2. Insufficient permissions

**Solutions:**
```bash
# Create output directory
mkdir -p /home/user/GenZ/output

# Check permissions
ls -la /home/user/GenZ/output
```

### Issue: SAS errors during execution

**Common Errors:**

**"Variable not found"**
```sas
/* Ensure all required variables exist */
proc contents data=work.nhsn_snf_data; run;
```

**"Division by zero"**
```sas
/* Check for facilities with zero variance */
proc means data=nhsn_snf_data;
    class facility_id metric_type;
    var metric_value;
    output out=check_variance std=std_val;
run;

proc print data=check_variance(where=(std_val = 0));
run;
```

## 📊 Output Interpretation Examples

### Example 1: Critical Facility Outbreak

```
facility_name: Riverside SNF
state: TX
hhs_region: 6
metric_type: COVID_CASES
week_start_date: 2025-11-15
current_value: 52
baseline_mean: 6.3
baseline_std: 2.1
z_score: 21.8
pct_change_from_baseline: 725%
anomaly_flag: HIGH_SPIKE
severity: CRITICAL
```

**Interpretation:**
- Facility has 52 cases vs. baseline of 6.3 (725% increase)
- Z-score of 21.8 = extremely rare event (p < 0.000001)
- **Action**: Immediate outbreak investigation, infection control assessment

---

### Example 2: State-Wide Reporting Issue

```
state: NY
hhs_region: 2
metric_type: COVID_CASES
week_start_date: 2025-11-15
current_value: 1,234
baseline_mean: 2,156
z_score: -1.8
pct_change: -43%
reporting_facilities: 45
pct_facilities_reporting: 38%
anomaly_flag: LOW_REPORTING
```

**Interpretation:**
- State total is lower than expected
- Only 38% of facilities reporting (vs. normal ~85%)
- **Action**: Technical support for reporting systems, contact state coordinator

---

## 📅 Recommended Schedule

### Daily (for high-priority surveillance)
- Quick review of CRITICAL severity alerts
- Contact flagged facilities for immediate verification

### Weekly (standard surveillance)
- Full QA analysis run
- Review WARNING and CRITICAL anomalies
- Generate weekly surveillance report
- Public health investigations as needed

### Monthly (data quality review)
- Comprehensive QA summary
- Identify persistent data quality issues
- Facility outreach and training
- Update baseline parameters if needed

## 🆘 Support

For issues or questions:

1. **Review SAS log** for specific error messages
2. **Check documentation**: `nhsn_snf_qa_methodology.md`
3. **Verify data requirements**: Ensure all required variables present
4. **Test with sample data**: Use data generator to isolate issues

## 📝 Version Information

- **QA Program Version**: 1.0
- **Last Updated**: 2025-11-23
- **SAS Version Required**: SAS 9.4 or later
- **Dependencies**: Base SAS, SAS/STAT (for PROC MEANS)

## 🔗 Related Files

- Main QA Program: `nhsn_snf_qa_analysis.sas`
- Sample Data Generator: `nhsn_snf_sample_data_generator.sas`
- Methodology Guide: `../docs/nhsn_snf_qa_methodology.md`
- Gen Z Agent README: `../README.md`

---

**Ready to start?** Run the sample data generator, execute the QA analysis, and review the outputs!

```sas
/* Complete workflow */
%include '/home/user/GenZ/sas/nhsn_snf_sample_data_generator.sas';
%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';

/* Review results */
proc print data=work.qa_summary; run;
proc print data=work.facility_anomalies(obs=50); run;
```
