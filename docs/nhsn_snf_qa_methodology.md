# NHSN SNF Data Quality Assurance Methodology

**Document Version**: 1.0
**Date**: 2025-11-23
**Author**: Gen Z Agent Project

---

## Table of Contents

1. [Overview](#overview)
2. [Data Requirements](#data-requirements)
3. [QA Components](#qa-components)
4. [Statistical Methods](#statistical-methods)
5. [Anomaly Detection Algorithms](#anomaly-detection-algorithms)
6. [Interpretation Guidelines](#interpretation-guidelines)
7. [Output Files](#output-files)
8. [Usage Examples](#usage-examples)
9. [Limitations and Considerations](#limitations-and-considerations)

---

## Overview

This QA system performs comprehensive quality assurance on **NHSN (National Healthcare Safety Network) Skilled Nursing Facility (SNF)** surveillance data. The system is designed to identify data quality issues, statistical anomalies, and unusual patterns at multiple geographic levels.

### Key Features

- **Multi-level Analysis**: Facility, State, and HHS Region aggregations
- **Temporal Detection**: Week-over-week spike detection
- **Statistical Rigor**: Multiple statistical methods (Z-score, IQR, control charts)
- **Comprehensive Coverage**: Data completeness, range validation, outlier detection
- **Actionable Output**: CSV exports with severity classifications

### Use Cases

1. **Public Health Surveillance**: Detect facility outbreaks early
2. **Data Quality Monitoring**: Identify reporting issues and data errors
3. **Performance Benchmarking**: Compare facilities within states/regions
4. **Regulatory Compliance**: Ensure data meets NHSN reporting standards
5. **Research Validation**: Screen data before analysis

---

## Data Requirements

### Required Input Dataset: `WORK.NHSN_SNF_DATA`

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `facility_id` | Character | Unique facility identifier | Yes |
| `facility_name` | Character | Facility name | Yes |
| `state` | Character | Two-letter state abbreviation | Yes |
| `hhs_region` | Numeric | HHS region number (1-10) | Yes |
| `report_date` | Date | Date of report (SAS date value) | Yes |
| `metric_type` | Character | Type of measure (e.g., COVID_CASES) | Yes |
| `metric_value` | Numeric | Value of the metric | Yes |
| `population_at_risk` | Numeric | Denominator (total beds, residents) | Recommended |

### Sample Metric Types

- `COVID_CASES` - Confirmed COVID-19 cases
- `COVID_DEATHS` - COVID-19 deaths
- `INFLUENZA_CASES` - Influenza cases
- `BEDS_OCCUPIED` - Number of occupied beds
- `STAFF_SHORTAGE` - Staff shortage count

### Data Format Example

```sas
data nhsn_snf_data;
    input facility_id $ facility_name $ 1-40 state $ hhs_region
          report_date :yymmdd10. metric_type $ metric_value population_at_risk;
    format report_date yymmdd10.;
    datalines;
SNF00001 Springfield SNF MA 1 2025-11-01 COVID_CASES 5 120
SNF00001 Springfield SNF MA 1 2025-11-08 COVID_CASES 12 120
SNF00002 Portland Care Center ME 1 2025-11-01 COVID_CASES 3 85
;
run;
```

---

## QA Components

The QA analysis consists of **10 major sections**:

### 1. Data Validation and Completeness

**Purpose**: Identify missing or invalid data elements

**Checks Performed**:
- Missing values in critical fields (facility_id, state, metric_value, etc.)
- Percentage of missingness by field
- Thresholds: Warning if >5% missing

**Output**: `WORK.QA_COMPLETENESS`

**Interpretation**:
- **<5% missing**: Acceptable
- **5-15% missing**: Warning - investigate reporting issues
- **>15% missing**: Critical - data may be unreliable

---

### 2. Range Validation

**Purpose**: Detect out-of-range or impossible values

**Validations**:
1. **Negative values**: `metric_value < 0`
2. **Exceeds population**: `metric_value > population_at_risk`
3. **Invalid HHS region**: Not in {1,2,3,4,5,6,7,8,9,10}
4. **Future dates**: `report_date > today()`

**Output**: `WORK.QA_RANGE_VIOLATIONS`

**Interpretation**:
- Any violations indicate data entry errors or system issues
- Negative values = data corruption or calculation errors
- Exceeds population = reporting error (e.g., duplicate counting)

---

### 3. Facility-Level Spike Detection

**Purpose**: Identify individual facilities with abnormal spikes

**Method**: Z-score analysis
- Calculate baseline mean and standard deviation for each facility/metric
- Baseline period: 12 weeks (configurable via `&lookback_weeks`)
- Spike threshold: 3 standard deviations (configurable via `&spike_threshold`)

**Formula**:
```
z_score = (current_value - baseline_mean) / baseline_std
```

**Classification**:
- `z_score > 3`: **HIGH_SPIKE** (Warning)
- `z_score > 5`: **CRITICAL** spike
- `z_score < -3`: **LOW_SPIKE** (Unusual decrease)
- `pct_change > 200%`: **EXTREME_CHANGE**

**Output**: `WORK.FACILITY_ANOMALIES`

**Example Interpretation**:
```
Facility: Oakwood SNF, State: FL, Metric: COVID_CASES
Current Value: 45, Baseline Mean: 8.5, Z-Score: 6.2
Classification: CRITICAL
Interpretation: Potential outbreak - 6.2 standard deviations above baseline
Action: Immediate investigation required
```

---

### 4. State-Level Anomaly Detection

**Purpose**: Detect state-wide patterns and aggregated anomalies

**Method**: State-level Z-score + reporting completeness

**Additional Checks**:
- Percentage of facilities reporting
- Comparison to state historical baseline

**Classification**:
- `z_score > 3`: State-wide spike
- `reporting < 50%`: **LOW_REPORTING** flag

**Output**: `WORK.STATE_ANOMALIES`

**Use Case**: Identify regional outbreaks or state-wide reporting issues

---

### 5. HHS Region-Level Analysis

**Purpose**: Detect regional trends across multiple states

**Method**: Regional aggregation with Z-score detection

**HHS Regions**:
- **Region 1**: CT, ME, MA, NH, RI, VT
- **Region 2**: NJ, NY, PR, VI
- **Region 3**: DE, DC, MD, PA, VA, WV
- **Region 4**: AL, FL, GA, KY, MS, NC, SC, TN
- **Region 5**: IL, IN, MI, MN, OH, WI
- **Region 6**: AR, LA, NM, OK, TX
- **Region 7**: IA, KS, MO, NE
- **Region 8**: CO, MT, ND, SD, UT, WY
- **Region 9**: AZ, CA, HI, NV, AS, GU, MP
- **Region 10**: AK, ID, OR, WA

**Output**: `WORK.REGION_ANOMALIES`

---

### 6. Statistical Outlier Detection (IQR Method)

**Purpose**: Identify extreme outliers using robust statistics

**Method**: Interquartile Range (IQR) with Tukey fences

**Formula**:
```
IQR = Q3 - Q1
Lower Fence = Q1 - (3 × IQR)
Upper Fence = Q3 + (3 × IQR)

Outlier if: value < Lower Fence OR value > Upper Fence
```

**Advantages**:
- Robust to skewed distributions
- Less sensitive to extreme outliers than Z-score
- Works well with non-normal data

**Output**: `WORK.STATISTICAL_OUTLIERS`

**Interpretation**:
- **Fence Distance**: How many IQRs beyond the fence
  - 1-2 IQRs: Mild outlier
  - 2-4 IQRs: Moderate outlier
  - >4 IQRs: Extreme outlier

---

### 7. Temporal Spike Detection (Week-over-Week)

**Purpose**: Detect sudden changes between consecutive weeks

**Method**: Week-over-week percent change

**Formula**:
```
WoW_change = current_week - previous_week
WoW_pct_change = (WoW_change / previous_week) × 100
```

**Classification**:
- `WoW_pct_change > 200%`: **EXTREME_INCREASE**
- `WoW_pct_change > 100%`: **LARGE_INCREASE**
- `WoW_pct_change < -75%`: **LARGE_DECREASE**

**Output**: `WORK.TEMPORAL_SPIKES`

**Use Case**: Detect sudden facility outbreaks or reporting errors

---

### 8. Cross-Facility Comparison Within States

**Purpose**: Benchmark facilities against state peers

**Method**:
1. Rank facilities within each state by metric value
2. Calculate state mean and standard deviation
3. Identify top outliers (top 5-10 facilities per state)

**Metrics**:
- **Value Rank**: Facility ranking within state (1 = highest)
- **State Z-Score**: Standard deviations from state mean

**Output**: `WORK.STATE_FACILITY_OUTLIERS`

**Use Case**: Performance comparison and outlier identification

---

### 9. Comprehensive QA Summary

**Purpose**: High-level overview of all QA findings

**Contents**:
- Total anomaly counts by category
- Distinct facilities/states affected
- Summary statistics

**Output**: `WORK.QA_SUMMARY`

---

### 10. Export and Reporting

**Purpose**: Generate exportable CSV files for downstream analysis

**Exported Files**:
1. `nhsn_facility_anomalies_YYYYMMDD.csv` - Facility-level anomalies
2. `nhsn_state_anomalies_YYYYMMDD.csv` - State-level anomalies
3. `nhsn_qa_comprehensive_report_YYYYMMDD.csv` - All anomalies combined
4. `nhsn_qa_summary_YYYYMMDD.csv` - Summary statistics
5. `nhsn_qa_event_log_YYYYMMDD.csv` - QA execution log

---

## Statistical Methods

### Z-Score (Standard Score)

**Formula**:
```
z = (x - μ) / σ

where:
  x = observed value
  μ = population mean
  σ = population standard deviation
```

**Interpretation**:
- `|z| < 2`: Within normal range (95% confidence)
- `|z| = 2-3`: Unusual but possible
- `|z| > 3`: Highly unlikely (<0.3% probability)
- `|z| > 5`: Extremely rare (<0.00006% probability)

**Assumptions**:
- Data is approximately normally distributed
- Sufficient sample size (n ≥ 30 recommended)
- Independent observations

**Limitations**:
- Sensitive to outliers (outliers inflate σ)
- Not robust for highly skewed data

---

### Interquartile Range (IQR) Method

**Formula**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1

Lower Fence = Q1 - k × IQR
Upper Fence = Q3 + k × IQR

where k = 1.5 (mild outliers) or 3.0 (extreme outliers)
```

**Advantages**:
- Robust to extreme outliers
- Works with skewed distributions
- No normality assumption required

**Our Implementation**: k = 3.0 (conservative approach)

---

### Baseline Calculation

**Rolling Baseline Window**: 12 weeks (configurable)

**Why 12 weeks?**
- Balances seasonal variation with recent trends
- Sufficient data for stable statistics (n ≥ 12 weekly points)
- Recent enough to detect emerging patterns

**Minimum Requirements**:
- At least 4 weeks of data to establish baseline
- Facilities with <4 weeks excluded from spike detection

---

## Anomaly Detection Algorithms

### Algorithm 1: Facility Z-Score Detection

```
For each facility F, metric M:
  1. Collect baseline: last 12 weeks of metric M for facility F
  2. Calculate: baseline_mean, baseline_std
  3. For current week W:
     z_score = (current_value - baseline_mean) / baseline_std
     IF z_score > threshold THEN flag = SPIKE
```

**Sensitivity**: Adjustable via `&spike_threshold` (default: 3)

---

### Algorithm 2: IQR Outlier Detection

```
For each metric M across all facilities:
  1. Calculate Q1, Q3, IQR for metric M
  2. Set fences: lower = Q1 - 3×IQR, upper = Q3 + 3×IQR
  3. For each observation O:
     IF O < lower OR O > upper THEN flag = OUTLIER
     fence_distance = |O - nearest_fence| / IQR
```

**Advantage**: Identifies facilities that are extreme compared to all peers, not just their own history

---

### Algorithm 3: Week-over-Week Change Detection

```
For each facility F, metric M:
  1. Order data by week ascending
  2. For each week W:
     prev_week_value = value at week W-1
     current_value = value at week W
     pct_change = ((current - prev) / prev) × 100
     IF |pct_change| > threshold THEN flag = SPIKE
```

**Thresholds**:
- Extreme: >200% increase
- Large: >100% increase
- Large decrease: <-75% decrease

---

## Interpretation Guidelines

### Severity Levels

| Severity | Z-Score Range | Action Required | Response Time |
|----------|---------------|-----------------|---------------|
| **CRITICAL** | > 5 | Immediate investigation | Within 24 hours |
| **WARNING** | 3 - 5 | Investigation required | Within 48 hours |
| **UNUSUAL_LOW** | < -3 | Review for reporting issues | Within 1 week |
| **NORMAL** | -3 to 3 | Routine monitoring | N/A |

---

### Anomaly Flag Interpretation

#### Facility-Level Flags

- **HIGH_SPIKE**: Value is >3 SD above baseline
  - **Possible Causes**: Outbreak, cluster event, data entry error
  - **Action**: Verify with facility, investigate outbreak protocols

- **EXTREME_CHANGE**: >200% increase from baseline
  - **Possible Causes**: Major outbreak, reporting backlog, duplicate entry
  - **Action**: Contact facility immediately, verify data

- **LOW_SPIKE**: Value is <-3 SD below baseline
  - **Possible Causes**: Under-reporting, system outage, improved conditions
  - **Action**: Verify reporting completeness

#### State-Level Flags

- **HIGH_SPIKE**: State aggregate >3 SD above baseline
  - **Possible Causes**: Regional outbreak, seasonal surge
  - **Action**: Public health alert, resource allocation

- **LOW_REPORTING**: <50% of facilities reporting
  - **Possible Causes**: System outage, holiday period, policy change
  - **Action**: Technical support, reminder communications

---

### Context-Specific Interpretation

#### COVID_CASES Spikes

**Normal Seasonal Pattern**:
- Winter surge (Nov-Mar): Expect 20-50% increase
- Summer trough (Jun-Aug): Expect 30-50% decrease

**Outbreak vs. Data Error**:
- **Outbreak**: Gradual increase over 2-3 weeks, affects multiple facilities in region
- **Data Error**: Sudden spike in single facility, returns to baseline next week

#### BEDS_OCCUPIED Anomalies

**Expected Range**: 70-95% occupancy for most SNFs

**Interpretation**:
- <50% occupancy: Potential closure, financial issues, or data error
- >100% occupancy: **Data error** (impossible value)

---

## Output Files

### 1. Facility Anomalies CSV

**File**: `nhsn_facility_anomalies_YYYYMMDD.csv`

**Columns**:
- `facility_id`, `facility_name`, `state`, `hhs_region`
- `metric_type`, `week_start_date`
- `current_value`, `baseline_mean`, `baseline_std`
- `z_score`, `pct_change_from_baseline`
- `anomaly_flag`, `severity`

**Use**: Investigate specific facility outbreaks

---

### 2. State Anomalies CSV

**File**: `nhsn_state_anomalies_YYYYMMDD.csv`

**Columns**:
- `state`, `hhs_region`, `metric_type`, `week_start_date`
- `current_value`, `baseline_mean`, `z_score`, `pct_change`
- `reporting_facilities`, `pct_facilities_reporting`
- `anomaly_flag`

**Use**: State-level surveillance and resource allocation

---

### 3. Comprehensive QA Report CSV

**File**: `nhsn_qa_comprehensive_report_YYYYMMDD.csv`

**Columns**:
- `level` (FACILITY, STATE, REGION)
- `entity_id`, `entity_name`, `state`, `hhs_region`
- `metric_type`, `analysis_date`
- `metric_value`, `baseline_mean`, `z_score`, `pct_change`
- `anomaly_flag`, `severity`, `detection_method`

**Use**: Comprehensive analysis and reporting dashboards

---

## Usage Examples

### Example 1: Basic QA Run

```sas
/* Load your data */
data work.nhsn_snf_data;
    set mylib.nhsn_weekly_data;
run;

/* Run QA analysis */
%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';

/* Review summary */
proc print data=work.qa_summary;
run;
```

---

### Example 2: Investigate Specific State

```sas
/* After running QA, filter to California facilities */
proc sql;
    select *
    from work.facility_anomalies
    where state = 'CA'
        and severity in ('CRITICAL', 'WARNING')
    order by z_score desc;
quit;
```

---

### Example 3: Customized Thresholds

```sas
/* More sensitive detection (2 SD instead of 3) */
%let spike_threshold = 2;
%let outlier_threshold = 2;

%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';
```

---

### Example 4: Time Period Restriction

```sas
/* Only analyze data from last 6 months */
data work.nhsn_snf_data;
    set mylib.nhsn_data;
    where report_date >= intnx('month', today(), -6, 'beginning');
run;

%include '/home/user/GenZ/sas/nhsn_snf_qa_analysis.sas';
```

---

## Limitations and Considerations

### 1. Statistical Assumptions

**Normal Distribution Assumption**:
- Z-score method assumes normal distribution
- Healthcare surveillance data is often right-skewed
- **Mitigation**: IQR method provided as robust alternative

**Small Sample Sizes**:
- Facilities with <4 weeks of data are excluded
- New facilities may not be adequately monitored
- **Mitigation**: Lower threshold for new facilities (manual review)

---

### 2. Temporal Considerations

**Seasonal Variation**:
- Respiratory illnesses show strong seasonality
- 12-week baseline may not capture year-over-year patterns
- **Mitigation**: Consider separate baselines for each season

**Reporting Delays**:
- Data may have 1-2 week lag
- Recent weeks may be incomplete
- **Mitigation**: Focus on data ≥2 weeks old for completeness

---

### 3. Data Quality Dependencies

**Garbage In, Garbage Out**:
- QA can only detect anomalies in submitted data
- Cannot detect facilities that don't report
- **Mitigation**: Track reporting completeness separately

**Metric Definition Changes**:
- Changes in case definitions affect baseline calculations
- **Example**: COVID case definition changes in 2023
- **Mitigation**: Reset baselines after definition changes

---

### 4. False Positives vs. False Negatives

**Current Settings (Conservative)**:
- 3 SD threshold = ~0.3% false positive rate (assuming normality)
- Lower sensitivity = may miss smaller outbreaks

**Adjusting Sensitivity**:
- Stricter (2 SD): Catches more anomalies, more false positives
- Looser (4 SD): Fewer false alarms, may miss real events

**Recommendation**: Start with 3 SD, adjust based on operational needs

---

### 5. Regional and Demographic Differences

**Facility Heterogeneity**:
- Large facilities (200 beds) vs. small (50 beds)
- Urban vs. rural settings
- Different patient populations

**Current Approach**: Within-facility baselines
**Future Enhancement**: Risk-adjusted comparisons

---

## Recommended Workflows

### Weekly Surveillance Workflow

```
1. Monday: Extract NHSN data from previous week
2. Tuesday: Run QA analysis
3. Tuesday: Review CRITICAL and WARNING anomalies
4. Wednesday: Contact flagged facilities for verification
5. Thursday: Public health investigation for confirmed outbreaks
6. Friday: Update surveillance dashboard and reports
```

---

### Monthly Data Quality Review

```
1. Run comprehensive QA on full month
2. Review completeness trends by state/region
3. Identify facilities with persistent quality issues
4. Generate data quality scorecards
5. Outreach to low-quality reporters
```

---

## Contact and Support

For questions about this QA methodology:

- **Technical Issues**: Review SAS log files, check data requirements
- **Interpretation**: Consult epidemiologist or biostatistician
- **Enhancements**: Submit feature requests via project repository

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-23 | Initial release with 10 QA components |

---

**Document End**
