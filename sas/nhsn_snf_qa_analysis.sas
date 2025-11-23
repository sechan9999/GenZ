/*****************************************************************************
* Program: nhsn_snf_qa_analysis.sas
* Purpose: Rigorous Quality Assurance for NHSN SNF (Skilled Nursing Facility) Data
* Author: Generated for Gen Z Agent Project
* Date: 2025-11-23
*
* Description:
*   Performs comprehensive QA on NHSN SNF inferred data including:
*   - Data completeness and validity checks
*   - Temporal spike detection at facility, state, and HHS region levels
*   - Statistical outlier identification using multiple methods
*   - Geographic aggregation and comparison
*   - Facility-level anomaly flagging
*   - Trend analysis and control charts
*
* Input Dataset: WORK.NHSN_SNF_DATA
*   Required variables:
*     - facility_id: Unique facility identifier
*     - facility_name: Facility name
*     - state: State abbreviation
*     - hhs_region: HHS region number (1-10)
*     - report_date: Date of report (YYMMDD10. format)
*     - metric_type: Type of measure (e.g., 'COVID_CASES', 'INFLUENZA', 'BEDS_OCCUPIED')
*     - metric_value: Numeric value of the metric
*     - population_at_risk: Denominator (e.g., total beds, residents)
*
* Output:
*   - WORK.QA_SUMMARY: Overall QA summary statistics
*   - WORK.FACILITY_ANOMALIES: Flagged facility-level anomalies
*   - WORK.STATE_ANOMALIES: State-level anomalies
*   - WORK.REGION_ANOMALIES: HHS region-level anomalies
*   - WORK.TEMPORAL_SPIKES: Detected temporal spikes
*   - WORK.QA_REPORT: Comprehensive QA report for export
*****************************************************************************/

%let qa_date = %sysfunc(today(), yymmddn8.);
%let lookback_weeks = 12;  /* Number of weeks for baseline calculation */
%let spike_threshold = 3;   /* Standard deviations for spike detection */
%let outlier_threshold = 3; /* IQR multiplier for outlier detection */

/* Macro for logging QA events */
%macro log_qa_event(severity=, category=, message=);
    data _qa_log_temp;
        length severity $10 category $50 message $500;
        severity = "&severity";
        category = "&category";
        message = "&message";
        qa_timestamp = datetime();
        format qa_timestamp datetime20.;
    run;

    proc append base=work.qa_event_log data=_qa_log_temp force; run;
%mend log_qa_event;

/* Initialize QA log */
data work.qa_event_log;
    length severity $10 category $50 message $500;
    stop;
    format qa_timestamp datetime20.;
run;

%log_qa_event(severity=INFO, category=INITIALIZATION,
              message=Starting NHSN SNF QA Analysis);


/*****************************************************************************
* SECTION 1: DATA VALIDATION AND COMPLETENESS CHECKS
*****************************************************************************/

title "NHSN SNF QA Analysis - Data Validation";

/* 1.1 Check data availability and basic structure */
proc sql noprint;
    select count(*) into :total_records from nhsn_snf_data;
    select count(distinct facility_id) into :total_facilities from nhsn_snf_data;
    select count(distinct state) into :total_states from nhsn_snf_data;
    select count(distinct hhs_region) into :total_regions from nhsn_snf_data;
    select min(report_date) format=yymmdd10., max(report_date) format=yymmdd10.
        into :min_date, :max_date
        from nhsn_snf_data;
quit;

%put NOTE: Total Records: &total_records;
%put NOTE: Total Facilities: &total_facilities;
%put NOTE: Total States: &total_states;
%put NOTE: Date Range: &min_date to &max_date;

%log_qa_event(severity=INFO, category=DATA_SUMMARY,
              message=Total records: &total_records | Facilities: &total_facilities);

/* 1.2 Completeness checks - identify missing critical fields */
proc sql;
    create table work.qa_completeness as
    select
        'facility_id' as field_name,
        count(*) as total_records,
        sum(case when missing(facility_id) then 1 else 0 end) as missing_count,
        calculated missing_count / calculated total_records * 100 as pct_missing format=8.2
    from nhsn_snf_data

    union all

    select
        'state' as field_name,
        count(*) as total_records,
        sum(case when missing(state) then 1 else 0 end) as missing_count,
        calculated missing_count / calculated total_records * 100 as pct_missing
    from nhsn_snf_data

    union all

    select
        'hhs_region' as field_name,
        count(*) as total_records,
        sum(case when missing(hhs_region) then 1 else 0 end) as missing_count,
        calculated missing_count / calculated total_records * 100 as pct_missing
    from nhsn_snf_data

    union all

    select
        'report_date' as field_name,
        count(*) as total_records,
        sum(case when missing(report_date) then 1 else 0 end) as missing_count,
        calculated missing_count / calculated total_records * 100 as pct_missing
    from nhsn_snf_data

    union all

    select
        'metric_value' as field_name,
        count(*) as total_records,
        sum(case when missing(metric_value) then 1 else 0 end) as missing_count,
        calculated missing_count / calculated total_records * 100 as pct_missing
    from nhsn_snf_data

    order by pct_missing desc;
quit;

title2 "Data Completeness Check";
proc print data=work.qa_completeness noobs;
    var field_name total_records missing_count pct_missing;
run;

/* Flag high missingness */
data _null_;
    set work.qa_completeness;
    if pct_missing > 5 then do;
        call execute('%log_qa_event(severity=WARNING, category=COMPLETENESS, ' ||
                    'message=High missingness in ' || strip(field_name) ||
                    ': ' || put(pct_missing, 8.2) || '%%)');
    end;
run;

/* 1.3 Range validation - identify out-of-range values */
proc sql;
    create table work.qa_range_violations as
    select
        'Negative metric_value' as violation_type,
        count(*) as violation_count,
        min(metric_value) as min_value,
        max(metric_value) as max_value
    from nhsn_snf_data
    where metric_value < 0

    union all

    select
        'Metric > Population at Risk' as violation_type,
        count(*) as violation_count,
        min(metric_value - population_at_risk) as min_value,
        max(metric_value - population_at_risk) as max_value
    from nhsn_snf_data
    where metric_value > population_at_risk and not missing(population_at_risk)

    union all

    select
        'Invalid HHS Region' as violation_type,
        count(*) as violation_count,
        min(hhs_region) as min_value,
        max(hhs_region) as max_value
    from nhsn_snf_data
    where hhs_region not in (1,2,3,4,5,6,7,8,9,10)

    union all

    select
        'Future report_date' as violation_type,
        count(*) as violation_count,
        min(report_date - today()) as min_value,
        max(report_date - today()) as max_value
    from nhsn_snf_data
    where report_date > today();
quit;

title2 "Range Validation Violations";
proc print data=work.qa_range_violations noobs;
run;


/*****************************************************************************
* SECTION 2: TEMPORAL PREPARATION - CREATE TIME SERIES
*****************************************************************************/

/* 2.1 Create weekly aggregations for temporal analysis */
proc sql;
    create table work.weekly_facility_data as
    select
        facility_id,
        facility_name,
        state,
        hhs_region,
        metric_type,
        intnx('week', report_date, 0, 'beginning') as week_start_date format=yymmdd10.,
        count(*) as report_count,
        sum(metric_value) as total_value,
        mean(metric_value) as avg_value,
        std(metric_value) as std_value,
        min(metric_value) as min_value,
        max(metric_value) as max_value,
        sum(population_at_risk) as total_population
    from nhsn_snf_data
    where not missing(metric_value)
    group by facility_id, facility_name, state, hhs_region, metric_type,
             calculated week_start_date
    order by facility_id, metric_type, week_start_date;
quit;

/* 2.2 Create state-level weekly aggregations */
proc sql;
    create table work.weekly_state_data as
    select
        state,
        hhs_region,
        metric_type,
        intnx('week', report_date, 0, 'beginning') as week_start_date format=yymmdd10.,
        count(distinct facility_id) as facility_count,
        count(*) as report_count,
        sum(metric_value) as total_value,
        mean(metric_value) as avg_value,
        std(metric_value) as std_value,
        median(metric_value) as median_value,
        sum(population_at_risk) as total_population
    from nhsn_snf_data
    where not missing(metric_value)
    group by state, hhs_region, metric_type, calculated week_start_date
    order by state, metric_type, week_start_date;
quit;

/* 2.3 Create HHS region-level weekly aggregations */
proc sql;
    create table work.weekly_region_data as
    select
        hhs_region,
        metric_type,
        intnx('week', report_date, 0, 'beginning') as week_start_date format=yymmdd10.,
        count(distinct state) as state_count,
        count(distinct facility_id) as facility_count,
        count(*) as report_count,
        sum(metric_value) as total_value,
        mean(metric_value) as avg_value,
        std(metric_value) as std_value,
        sum(population_at_risk) as total_population
    from nhsn_snf_data
    where not missing(metric_value)
    group by hhs_region, metric_type, calculated week_start_date
    order by hhs_region, metric_type, week_start_date;
quit;


/*****************************************************************************
* SECTION 3: FACILITY-LEVEL SPIKE DETECTION
*****************************************************************************/

title "NHSN SNF QA Analysis - Facility-Level Spike Detection";

/* 3.1 Calculate baseline statistics for each facility/metric combination */
proc sql;
    create table work.facility_baseline as
    select
        facility_id,
        facility_name,
        state,
        hhs_region,
        metric_type,
        count(*) as baseline_weeks,
        mean(total_value) as baseline_mean,
        std(total_value) as baseline_std,
        median(total_value) as baseline_median,
        min(total_value) as baseline_min,
        max(total_value) as baseline_max
    from work.weekly_facility_data
    where week_start_date >= intnx('week', today(), -&lookback_weeks, 'beginning')
    group by facility_id, facility_name, state, hhs_region, metric_type
    having calculated baseline_weeks >= 4;  /* Require at least 4 weeks of data */
quit;

/* 3.2 Identify facility-level spikes using z-score method */
proc sql;
    create table work.facility_anomalies as
    select
        w.facility_id,
        w.facility_name,
        w.state,
        w.hhs_region,
        w.metric_type,
        w.week_start_date,
        w.total_value as current_value,
        b.baseline_mean,
        b.baseline_std,
        b.baseline_median,
        /* Calculate z-score */
        case
            when b.baseline_std > 0 then (w.total_value - b.baseline_mean) / b.baseline_std
            else .
        end as z_score,
        /* Calculate percent change from baseline */
        case
            when b.baseline_mean > 0 then ((w.total_value - b.baseline_mean) / b.baseline_mean) * 100
            else .
        end as pct_change_from_baseline,
        /* Spike flag */
        case
            when calculated z_score > &spike_threshold then 'HIGH_SPIKE'
            when calculated z_score < -&spike_threshold then 'LOW_SPIKE'
            when abs(calculated pct_change_from_baseline) > 200 then 'EXTREME_CHANGE'
            else 'NORMAL'
        end as anomaly_flag,
        /* Severity score */
        case
            when calculated z_score > 5 then 'CRITICAL'
            when calculated z_score > &spike_threshold then 'WARNING'
            when calculated z_score < -&spike_threshold then 'UNUSUAL_LOW'
            else 'NORMAL'
        end as severity
    from work.weekly_facility_data as w
    inner join work.facility_baseline as b
        on w.facility_id = b.facility_id
        and w.metric_type = b.metric_type
    where w.week_start_date >= intnx('week', today(), -8, 'beginning')  /* Recent 8 weeks */
    order by calculated z_score desc;
quit;

/* 3.3 Filter to only anomalies */
data work.facility_anomalies;
    set work.facility_anomalies;
    where anomaly_flag ne 'NORMAL';
run;

title2 "Facility-Level Anomalies (Sorted by Z-Score)";
proc print data=work.facility_anomalies(obs=50) noobs;
    var facility_name state hhs_region metric_type week_start_date
        current_value baseline_mean z_score pct_change_from_baseline
        anomaly_flag severity;
    format current_value baseline_mean comma12.2
           z_score pct_change_from_baseline 8.2;
run;

/* 3.4 Facility anomaly summary by severity */
proc freq data=work.facility_anomalies;
    tables severity * metric_type / nocum nopercent;
    title2 "Facility Anomalies by Severity and Metric Type";
run;


/*****************************************************************************
* SECTION 4: STATE-LEVEL ANOMALY DETECTION
*****************************************************************************/

title "NHSN SNF QA Analysis - State-Level Anomaly Detection";

/* 4.1 Calculate state baseline statistics */
proc sql;
    create table work.state_baseline as
    select
        state,
        hhs_region,
        metric_type,
        count(*) as baseline_weeks,
        mean(total_value) as baseline_mean,
        std(total_value) as baseline_std,
        median(total_value) as baseline_median,
        mean(facility_count) as avg_facilities_reporting
    from work.weekly_state_data
    where week_start_date >= intnx('week', today(), -&lookback_weeks, 'beginning')
    group by state, hhs_region, metric_type
    having calculated baseline_weeks >= 4;
quit;

/* 4.2 Identify state-level anomalies using IQR and z-score methods */
proc sql;
    create table work.state_anomalies as
    select
        w.state,
        w.hhs_region,
        w.metric_type,
        w.week_start_date,
        w.total_value as current_value,
        w.facility_count as reporting_facilities,
        b.baseline_mean,
        b.baseline_std,
        b.baseline_median,
        b.avg_facilities_reporting,
        /* Z-score */
        case
            when b.baseline_std > 0 then (w.total_value - b.baseline_mean) / b.baseline_std
            else .
        end as z_score,
        /* Percent change */
        case
            when b.baseline_mean > 0 then ((w.total_value - b.baseline_mean) / b.baseline_mean) * 100
            else .
        end as pct_change,
        /* Reporting completeness */
        case
            when b.avg_facilities_reporting > 0
                then (w.facility_count / b.avg_facilities_reporting) * 100
            else .
        end as pct_facilities_reporting,
        /* Anomaly classification */
        case
            when calculated z_score > &spike_threshold then 'HIGH_SPIKE'
            when calculated z_score < -&spike_threshold then 'LOW_SPIKE'
            when calculated pct_facilities_reporting < 50 then 'LOW_REPORTING'
            else 'NORMAL'
        end as anomaly_flag
    from work.weekly_state_data as w
    inner join work.state_baseline as b
        on w.state = b.state
        and w.metric_type = b.metric_type
    where w.week_start_date >= intnx('week', today(), -8, 'beginning')
    having calculated anomaly_flag ne 'NORMAL'
    order by calculated z_score desc;
quit;

title2 "State-Level Anomalies";
proc print data=work.state_anomalies(obs=30) noobs;
    var state hhs_region metric_type week_start_date current_value
        baseline_mean z_score pct_change pct_facilities_reporting anomaly_flag;
    format current_value baseline_mean comma12.2
           z_score pct_change pct_facilities_reporting 8.2;
run;


/*****************************************************************************
* SECTION 5: HHS REGION-LEVEL ANALYSIS
*****************************************************************************/

title "NHSN SNF QA Analysis - HHS Region-Level Analysis";

/* 5.1 Calculate region baseline */
proc sql;
    create table work.region_baseline as
    select
        hhs_region,
        metric_type,
        count(*) as baseline_weeks,
        mean(total_value) as baseline_mean,
        std(total_value) as baseline_std,
        median(total_value) as baseline_median
    from work.weekly_region_data
    where week_start_date >= intnx('week', today(), -&lookback_weeks, 'beginning')
    group by hhs_region, metric_type
    having calculated baseline_weeks >= 4;
quit;

/* 5.2 Identify region-level anomalies */
proc sql;
    create table work.region_anomalies as
    select
        w.hhs_region,
        w.metric_type,
        w.week_start_date,
        w.total_value as current_value,
        w.state_count,
        w.facility_count,
        b.baseline_mean,
        b.baseline_std,
        case
            when b.baseline_std > 0 then (w.total_value - b.baseline_mean) / b.baseline_std
            else .
        end as z_score,
        case
            when b.baseline_mean > 0 then ((w.total_value - b.baseline_mean) / b.baseline_mean) * 100
            else .
        end as pct_change,
        case
            when calculated z_score > &spike_threshold then 'HIGH_SPIKE'
            when calculated z_score < -&spike_threshold then 'LOW_SPIKE'
            else 'NORMAL'
        end as anomaly_flag
    from work.weekly_region_data as w
    inner join work.region_baseline as b
        on w.hhs_region = b.hhs_region
        and w.metric_type = b.metric_type
    where w.week_start_date >= intnx('week', today(), -8, 'beginning')
    having calculated anomaly_flag ne 'NORMAL'
    order by calculated z_score desc;
quit;

title2 "HHS Region-Level Anomalies";
proc print data=work.region_anomalies noobs;
    var hhs_region metric_type week_start_date current_value baseline_mean
        z_score pct_change state_count facility_count anomaly_flag;
    format current_value baseline_mean comma12.2 z_score pct_change 8.2;
run;


/*****************************************************************************
* SECTION 6: ADVANCED STATISTICAL OUTLIER DETECTION
*****************************************************************************/

title "NHSN SNF QA Analysis - Statistical Outlier Detection";

/* 6.1 IQR-based outlier detection across all facilities */
proc means data=nhsn_snf_data noprint;
    class metric_type;
    var metric_value;
    output out=work.iqr_stats
        n=n
        mean=mean
        std=std
        median=median
        q1=q1
        q3=q3
        qrange=iqr;
run;

/* 6.2 Flag outliers using IQR method */
proc sql;
    create table work.statistical_outliers as
    select
        n.facility_id,
        n.facility_name,
        n.state,
        n.hhs_region,
        n.metric_type,
        n.report_date,
        n.metric_value,
        i.median,
        i.q1,
        i.q3,
        i.iqr,
        i.q1 - (&outlier_threshold * i.iqr) as lower_fence,
        i.q3 + (&outlier_threshold * i.iqr) as upper_fence,
        case
            when n.metric_value < calculated lower_fence then 'LOW_OUTLIER'
            when n.metric_value > calculated upper_fence then 'HIGH_OUTLIER'
            else 'NORMAL'
        end as outlier_flag,
        /* Distance from fence */
        case
            when n.metric_value < calculated lower_fence
                then (calculated lower_fence - n.metric_value) / i.iqr
            when n.metric_value > calculated upper_fence
                then (n.metric_value - calculated upper_fence) / i.iqr
            else 0
        end as fence_distance
    from nhsn_snf_data as n
    inner join work.iqr_stats as i
        on n.metric_type = i.metric_type
    where i._type_ = 1  /* Class statistics only */
        and calculated outlier_flag ne 'NORMAL'
        and n.report_date >= intnx('month', today(), -3, 'beginning')  /* Recent 3 months */
    order by calculated fence_distance desc;
quit;

title2 "Statistical Outliers (IQR Method, 3x IQR Fences)";
proc print data=work.statistical_outliers(obs=50) noobs;
    var facility_name state metric_type report_date metric_value
        median lower_fence upper_fence outlier_flag fence_distance;
    format metric_value median lower_fence upper_fence comma12.2
           fence_distance 8.2;
run;


/*****************************************************************************
* SECTION 7: TEMPORAL SPIKE DETECTION - WEEK-OVER-WEEK CHANGES
*****************************************************************************/

title "NHSN SNF QA Analysis - Week-over-Week Spike Detection";

/* 7.1 Calculate week-over-week changes at facility level */
data work.facility_wow_changes;
    set work.weekly_facility_data;
    by facility_id metric_type week_start_date;

    retain prev_week_value prev_week_date;

    if first.metric_type then do;
        prev_week_value = .;
        prev_week_date = .;
        wow_change = .;
        wow_pct_change = .;
    end;
    else do;
        wow_change = total_value - prev_week_value;
        if prev_week_value > 0 then
            wow_pct_change = (wow_change / prev_week_value) * 100;
        else
            wow_pct_change = .;
    end;

    /* Flag significant spikes */
    if not missing(wow_pct_change) then do;
        if wow_pct_change > 200 then spike_flag = 'EXTREME_INCREASE';
        else if wow_pct_change > 100 then spike_flag = 'LARGE_INCREASE';
        else if wow_pct_change < -75 then spike_flag = 'LARGE_DECREASE';
        else spike_flag = 'NORMAL';
    end;
    else spike_flag = 'NO_COMPARISON';

    prev_week_value = total_value;
    prev_week_date = week_start_date;

    format prev_week_date yymmdd10.;
run;

/* 7.2 Extract only flagged spikes */
proc sql;
    create table work.temporal_spikes as
    select
        facility_id,
        facility_name,
        state,
        hhs_region,
        metric_type,
        week_start_date,
        prev_week_date,
        prev_week_value,
        total_value as current_week_value,
        wow_change,
        wow_pct_change,
        spike_flag
    from work.facility_wow_changes
    where spike_flag not in ('NORMAL', 'NO_COMPARISON')
        and week_start_date >= intnx('week', today(), -8, 'beginning')
    order by wow_pct_change desc;
quit;

title2 "Temporal Spikes - Week-over-Week Changes";
proc print data=work.temporal_spikes(obs=50) noobs;
    var facility_name state metric_type week_start_date
        prev_week_value current_week_value wow_change wow_pct_change spike_flag;
    format prev_week_value current_week_value wow_change comma12.2
           wow_pct_change 8.2;
run;


/*****************************************************************************
* SECTION 8: CROSS-FACILITY COMPARISON WITHIN STATES
*****************************************************************************/

title "NHSN SNF QA Analysis - Cross-Facility Comparison";

/* 8.1 Rank facilities within each state by recent metric values */
proc rank data=work.weekly_facility_data(where=(week_start_date >= intnx('week', today(), -4, 'beginning')))
           out=work.facility_ranks descending;
    by state metric_type;
    var total_value;
    ranks value_rank;
run;

/* 8.2 Identify top outlier facilities per state */
proc sql;
    create table work.state_facility_outliers as
    select
        a.state,
        a.hhs_region,
        a.metric_type,
        a.facility_id,
        a.facility_name,
        a.week_start_date,
        a.total_value,
        a.value_rank,
        b.avg_value as state_avg,
        b.std_value as state_std,
        case
            when b.std_value > 0 then (a.total_value - b.avg_value) / b.std_value
            else .
        end as state_z_score,
        case
            when a.value_rank <= 5 then 'TOP_5_IN_STATE'
            when calculated state_z_score > 2 then 'ABOVE_STATE_AVG'
            else 'NORMAL'
        end as comparison_flag
    from work.facility_ranks as a
    left join (
        select
            state,
            metric_type,
            week_start_date,
            mean(total_value) as avg_value,
            std(total_value) as std_value
        from work.weekly_facility_data
        group by state, metric_type, week_start_date
    ) as b
        on a.state = b.state
        and a.metric_type = b.metric_type
        and a.week_start_date = b.week_start_date
    where a.value_rank <= 10  /* Top 10 facilities in each state */
    order by a.state, a.metric_type, a.value_rank;
quit;

title2 "Top Facilities by State (Ranked by Metric Value)";
proc print data=work.state_facility_outliers(obs=50) noobs;
    var state facility_name metric_type week_start_date
        total_value state_avg state_z_score value_rank comparison_flag;
    format total_value state_avg comma12.2 state_z_score 8.2;
run;


/*****************************************************************************
* SECTION 9: COMPREHENSIVE QA SUMMARY REPORT
*****************************************************************************/

title "NHSN SNF QA Analysis - Comprehensive Summary Report";

/* 9.1 Count anomalies by category */
proc sql;
    create table work.qa_summary as
    select
        'Facility-Level Anomalies' as qa_category,
        count(*) as anomaly_count,
        count(distinct facility_id) as distinct_facilities,
        count(distinct state) as distinct_states
    from work.facility_anomalies

    union all

    select
        'State-Level Anomalies' as qa_category,
        count(*) as anomaly_count,
        . as distinct_facilities,
        count(distinct state) as distinct_states
    from work.state_anomalies

    union all

    select
        'Region-Level Anomalies' as qa_category,
        count(*) as anomaly_count,
        . as distinct_facilities,
        . as distinct_states
    from work.region_anomalies

    union all

    select
        'Statistical Outliers (IQR)' as qa_category,
        count(*) as anomaly_count,
        count(distinct facility_id) as distinct_facilities,
        count(distinct state) as distinct_states
    from work.statistical_outliers

    union all

    select
        'Temporal Spikes (WoW)' as qa_category,
        count(*) as anomaly_count,
        count(distinct facility_id) as distinct_facilities,
        count(distinct state) as distinct_states
    from work.temporal_spikes;
quit;

title2 "QA Summary - Anomaly Counts by Category";
proc print data=work.qa_summary noobs;
run;

/* 9.2 Create comprehensive QA report for export */
proc sql;
    create table work.qa_report as
    select
        'FACILITY' as level,
        facility_id as entity_id,
        facility_name as entity_name,
        state,
        hhs_region,
        metric_type,
        week_start_date as analysis_date,
        current_value as metric_value,
        baseline_mean,
        z_score,
        pct_change_from_baseline as pct_change,
        anomaly_flag,
        severity,
        'Z-Score Method' as detection_method
    from work.facility_anomalies

    union all

    select
        'STATE' as level,
        state as entity_id,
        state as entity_name,
        state,
        hhs_region,
        metric_type,
        week_start_date as analysis_date,
        current_value as metric_value,
        baseline_mean,
        z_score,
        pct_change,
        anomaly_flag,
        'N/A' as severity,
        'Z-Score Method' as detection_method
    from work.state_anomalies

    union all

    select
        'REGION' as level,
        put(hhs_region, 2.) as entity_id,
        catx(' ', 'HHS Region', put(hhs_region, 2.)) as entity_name,
        'N/A' as state,
        hhs_region,
        metric_type,
        week_start_date as analysis_date,
        current_value as metric_value,
        baseline_mean,
        z_score,
        pct_change,
        anomaly_flag,
        'N/A' as severity,
        'Z-Score Method' as detection_method
    from work.region_anomalies

    union all

    select
        'FACILITY' as level,
        facility_id as entity_id,
        facility_name as entity_name,
        state,
        hhs_region,
        metric_type,
        report_date as analysis_date,
        metric_value,
        median as baseline_mean,
        . as z_score,
        . as pct_change,
        outlier_flag as anomaly_flag,
        'N/A' as severity,
        'IQR Method' as detection_method
    from work.statistical_outliers

    order by level, state, hhs_region, metric_type, analysis_date;
quit;

title2 "Comprehensive QA Report (All Anomalies)";
proc print data=work.qa_report(obs=100) noobs;
    var level entity_name state hhs_region metric_type analysis_date
        metric_value baseline_mean z_score pct_change anomaly_flag
        severity detection_method;
    format metric_value baseline_mean comma12.2 z_score pct_change 8.2;
run;


/*****************************************************************************
* SECTION 10: EXPORT RESULTS TO CSV FOR FURTHER ANALYSIS
*****************************************************************************/

/* 10.1 Export facility anomalies */
proc export data=work.facility_anomalies
    outfile="/home/user/GenZ/output/nhsn_facility_anomalies_&qa_date..csv"
    dbms=csv replace;
run;

/* 10.2 Export state anomalies */
proc export data=work.state_anomalies
    outfile="/home/user/GenZ/output/nhsn_state_anomalies_&qa_date..csv"
    dbms=csv replace;
run;

/* 10.3 Export comprehensive QA report */
proc export data=work.qa_report
    outfile="/home/user/GenZ/output/nhsn_qa_comprehensive_report_&qa_date..csv"
    dbms=csv replace;
run;

/* 10.4 Export QA summary */
proc export data=work.qa_summary
    outfile="/home/user/GenZ/output/nhsn_qa_summary_&qa_date..csv"
    dbms=csv replace;
run;

/* 10.5 Export QA event log */
proc export data=work.qa_event_log
    outfile="/home/user/GenZ/output/nhsn_qa_event_log_&qa_date..csv"
    dbms=csv replace;
run;

%log_qa_event(severity=INFO, category=COMPLETION,
              message=NHSN SNF QA Analysis completed successfully);

title;
footnote "NHSN SNF QA Analysis - Generated on &sysdate9 at &systime";

/* Print final summary to log */
proc print data=work.qa_summary noobs label;
    title "Final QA Summary Statistics";
    label
        qa_category = "QA Category"
        anomaly_count = "Total Anomalies"
        distinct_facilities = "Distinct Facilities"
        distinct_states = "Distinct States";
run;

/* End of Program */
