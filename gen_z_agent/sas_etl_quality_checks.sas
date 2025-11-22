/*****************************************************************************
ETL QUALITY VALIDATION - SAS CODE EXAMPLES
A/B Testing Framework for Pre/Post Pipeline Data Quality

This SAS program implements comprehensive data quality checks for ETL pipelines,
comparing pre-ingestion (source) and post-ingestion (target) data.

Features:
- 30+ data quality checks
- Pre/Post pipeline comparison
- Excel report generation via ODS
- Statistical validation
- Anomaly detection

Author: Gen Z Agent Team
Created: 2025-11-22
*****************************************************************************/

/*****************************************************************************
SECTION 1: ENVIRONMENT SETUP AND CONFIGURATION
*****************************************************************************/

/* Set up library paths */
libname PREDATA "/path/to/pre/ingestion/data";
libname POSTDATA "/path/to/post/ingestion/data";
libname REPORTS "/path/to/quality/reports";

/* Global macro variables */
%let PROJECT_NAME = ETL_Quality_Validation;
%let RUN_DATE = %sysfunc(today(), yymmddn8.);
%let RUN_TIME = %sysfunc(time(), time8.);
%let THRESHOLD_NULL_PCT = 5;      /* Max acceptable null % increase */
%let THRESHOLD_MEAN_CHG = 10;     /* Max acceptable mean % change */
%let THRESHOLD_ROW_CHG = 20;      /* Max acceptable row count % change */

/* Create output directory for reports */
%let OUTPUT_DIR = /home/user/GenZ/gen_z_agent/output/quality_reports;

/* Initialize log */
data _null_;
    put "=" *80;
    put "ETL QUALITY VALIDATION STARTED";
    put "Project: &PROJECT_NAME";
    put "Run Date: &RUN_DATE";
    put "Run Time: &RUN_TIME";
    put "=" *80;
run;

/*****************************************************************************
SECTION 2: DATA LOADING AND PREPARATION
*****************************************************************************/

/* Load Pre-Ingestion (Source) Data */
data work.pre_ingestion;
    set PREDATA.source_table;
    /* Add processing timestamp */
    load_timestamp = datetime();
    format load_timestamp datetime20.;
run;

/* Load Post-Ingestion (Target) Data */
data work.post_ingestion;
    set POSTDATA.target_table;
    /* Add processing timestamp */
    load_timestamp = datetime();
    format load_timestamp datetime20.;
run;

/* Create metadata about datasets */
proc contents data=work.pre_ingestion out=work.pre_metadata noprint;
run;

proc contents data=work.post_ingestion out=work.post_metadata noprint;
run;

/*****************************************************************************
SECTION 3: QUALITY CHECK #1-5 - ROW LEVEL STATISTICS
*****************************************************************************/

/* QC #1: Row Count Comparison */
proc sql noprint;
    /* Pre-ingestion row count */
    select count(*) into :pre_row_count trimmed
    from work.pre_ingestion;

    /* Post-ingestion row count */
    select count(*) into :post_row_count trimmed
    from work.post_ingestion;

    /* Calculate difference */
    %let row_count_diff = %sysevalf(&post_row_count - &pre_row_count);
    %let row_count_pct = %sysevalf((&row_count_diff / &pre_row_count) * 100);
quit;

data work.qc_01_row_count;
    length check_name $50 status $10;
    check_name = "Row Count Comparison";
    pre_value = &pre_row_count;
    post_value = &post_row_count;
    difference = &row_count_diff;
    pct_change = &row_count_pct;

    /* Determine status */
    if abs(pct_change) <= &THRESHOLD_ROW_CHG then status = "PASS";
    else if abs(pct_change) <= &THRESHOLD_ROW_CHG * 2 then status = "WARNING";
    else status = "FAIL";

    severity = case
        when status = "FAIL" then 3
        when status = "WARNING" then 2
        else 1
    end;
run;

/* QC #2: Column Count and Schema Comparison */
proc sql;
    create table work.qc_02_column_check as
    select
        'Column Count' as check_name,
        count(*) as pre_value,
        0 as post_value,
        0 as difference,
        0 as pct_change,
        'PENDING' as status,
        1 as severity
    from work.pre_metadata;

    update work.qc_02_column_check
    set post_value = (select count(*) from work.post_metadata);

    update work.qc_02_column_check
    set difference = post_value - pre_value,
        pct_change = ((post_value - pre_value) / pre_value) * 100;

    update work.qc_02_column_check
    set status = case
        when difference = 0 then 'PASS'
        when abs(difference) <= 2 then 'WARNING'
        else 'FAIL'
    end,
    severity = case
        when difference = 0 then 1
        when abs(difference) <= 2 then 2
        else 3
    end;
quit;

/* QC #3: Duplicate Row Detection */
proc sort data=work.pre_ingestion out=work.pre_sorted nodupkey dupout=work.pre_dups;
    by _all_;
run;

proc sort data=work.post_ingestion out=work.post_sorted nodupkey dupout=work.post_dups;
    by _all_;
run;

proc sql;
    create table work.qc_03_duplicates as
    select
        'Duplicate Rows' as check_name,
        (select count(*) from work.pre_dups) as pre_value,
        (select count(*) from work.post_dups) as post_value,
        calculated post_value - calculated pre_value as difference,
        case
            when calculated pre_value > 0
            then ((calculated post_value - calculated pre_value) / calculated pre_value) * 100
            else 0
        end as pct_change,
        case
            when calculated post_value = 0 then 'PASS'
            when calculated post_value <= 10 then 'WARNING'
            else 'FAIL'
        end as status,
        case
            when calculated post_value = 0 then 1
            when calculated post_value <= 10 then 2
            else 3
        end as severity;
quit;

/* QC #4: Memory Usage Comparison */
proc sql;
    create table work.qc_04_memory as
    select
        'Memory Usage (KB)' as check_name,
        sum(memsize)/1024 as pre_value format=comma12.2,
        0 as post_value,
        0 as difference,
        0 as pct_change,
        'PENDING' as status,
        1 as severity
    from work.pre_metadata;

    update work.qc_04_memory
    set post_value = (select sum(memsize)/1024 from work.post_metadata);

    update work.qc_04_memory
    set difference = post_value - pre_value,
        pct_change = ((post_value - pre_value) / pre_value) * 100,
        status = case
            when abs(calculated pct_change) <= 20 then 'PASS'
            when abs(calculated pct_change) <= 50 then 'WARNING'
            else 'FAIL'
        end;
quit;

/* QC #5: Data Type Consistency */
proc sql;
    create table work.qc_05_dtypes as
    select
        coalesce(a.name, b.name) as column_name,
        a.type as pre_type,
        b.type as post_type,
        case
            when a.type = b.type then 'PASS'
            when a.type is null or b.type is null then 'MISSING'
            else 'FAIL'
        end as status
    from work.pre_metadata a
    full join work.post_metadata b
        on a.name = b.name;
quit;

/*****************************************************************************
SECTION 4: QUALITY CHECK #6-15 - NULL VALUE ANALYSIS
*****************************************************************************/

/* QC #6-15: Null Count and Percentage by Column */
%macro check_nulls(dataset=, prefix=);
    proc sql;
        /* Get all column names */
        select distinct name into :col_list separated by ' '
        from dictionary.columns
        where libname = 'WORK' and memname = upcase("&dataset");
    quit;

    /* Calculate null statistics for each column */
    proc sql;
        create table work.&prefix._null_stats (
            column_name char(32),
            null_count num,
            total_count num,
            null_pct num format=5.2
        );
    quit;

    %let i = 1;
    %let col = %scan(&col_list, &i);

    %do %while(&col ne );
        proc sql;
            insert into work.&prefix._null_stats
            select
                "&col" as column_name,
                sum(case when &col is null then 1 else 0 end) as null_count,
                count(*) as total_count,
                calculated null_count / calculated total_count * 100 as null_pct
            from work.&dataset;
        quit;

        %let i = %eval(&i + 1);
        %let col = %scan(&col_list, &i);
    %end;
%mend;

/* Run null checks for both datasets */
%check_nulls(dataset=pre_ingestion, prefix=pre);
%check_nulls(dataset=post_ingestion, prefix=post);

/* Compare null percentages */
proc sql;
    create table work.qc_06_15_null_comparison as
    select
        'Null % - ' || a.column_name as check_name,
        a.null_pct as pre_value,
        b.null_pct as post_value,
        b.null_pct - a.null_pct as difference,
        case
            when a.null_pct > 0
            then ((b.null_pct - a.null_pct) / a.null_pct) * 100
            else b.null_pct
        end as pct_change,
        case
            when abs(calculated difference) <= &THRESHOLD_NULL_PCT then 'PASS'
            when abs(calculated difference) <= &THRESHOLD_NULL_PCT * 2 then 'WARNING'
            else 'FAIL'
        end as status,
        case
            when abs(calculated difference) <= &THRESHOLD_NULL_PCT then 1
            when abs(calculated difference) <= &THRESHOLD_NULL_PCT * 2 then 2
            else 3
        end as severity
    from work.pre_null_stats a
    inner join work.post_null_stats b
        on a.column_name = b.column_name
    order by abs(calculated difference) desc;
quit;

/*****************************************************************************
SECTION 5: QUALITY CHECK #16-23 - NUMERIC COLUMN STATISTICS
*****************************************************************************/

/* QC #16-23: Numeric Column Statistical Comparisons */
%macro compare_numeric_stats(dataset1=, dataset2=, prefix1=, prefix2=);
    /* Get numeric columns */
    proc sql noprint;
        select name into :num_cols separated by ' '
        from dictionary.columns
        where libname = 'WORK'
            and memname = upcase("&dataset1")
            and type = 1;  /* Numeric type */
    quit;

    /* Calculate statistics for each numeric column */
    %let i = 1;
    %let col = %scan(&num_cols, &i);

    %do %while(&col ne );
        /* Pre-ingestion stats */
        proc means data=work.&dataset1 noprint;
            var &col;
            output out=work.temp_pre_&i
                n=n
                mean=mean
                median=median
                std=std
                min=min
                max=max
                p25=p25
                p75=p75;
        run;

        data work.temp_pre_&i;
            set work.temp_pre_&i;
            column_name = "&col";
        run;

        /* Post-ingestion stats */
        proc means data=work.&dataset2 noprint;
            var &col;
            output out=work.temp_post_&i
                n=n
                mean=mean
                median=median
                std=std
                min=min
                max=max
                p25=p25
                p75=p75;
        run;

        data work.temp_post_&i;
            set work.temp_post_&i;
            column_name = "&col";
        run;

        %let i = %eval(&i + 1);
        %let col = %scan(&num_cols, &i);
    %end;

    /* Combine all pre stats */
    data work.&prefix1._numeric_stats;
        set work.temp_pre_:;
    run;

    /* Combine all post stats */
    data work.&prefix2._numeric_stats;
        set work.temp_post_:;
    run;

    /* Clean up temp datasets */
    proc datasets library=work nolist;
        delete temp_pre_: temp_post_:;
    quit;
%mend;

%compare_numeric_stats(
    dataset1=pre_ingestion,
    dataset2=post_ingestion,
    prefix1=pre,
    prefix2=post
);

/* Create comparison table */
proc sql;
    create table work.qc_16_23_numeric_stats as
    select
        a.column_name,

        /* Mean comparison */
        a.mean as pre_mean,
        b.mean as post_mean,
        b.mean - a.mean as mean_diff,
        case
            when a.mean ne 0
            then ((b.mean - a.mean) / a.mean) * 100
            else 0
        end as mean_pct_chg,

        /* Median comparison */
        a.median as pre_median,
        b.median as post_median,

        /* Std dev comparison */
        a.std as pre_std,
        b.std as post_std,

        /* Min/Max comparison */
        a.min as pre_min,
        b.min as post_min,
        a.max as pre_max,
        b.max as post_max,

        /* IQR comparison */
        a.p75 - a.p25 as pre_iqr,
        b.p75 - b.p25 as post_iqr,

        /* Status determination */
        case
            when abs(calculated mean_pct_chg) <= &THRESHOLD_MEAN_CHG then 'PASS'
            when abs(calculated mean_pct_chg) <= &THRESHOLD_MEAN_CHG * 2 then 'WARNING'
            else 'FAIL'
        end as status

    from work.pre_numeric_stats a
    inner join work.post_numeric_stats b
        on a.column_name = b.column_name;
quit;

/*****************************************************************************
SECTION 6: QUALITY CHECK #24-28 - CATEGORICAL COLUMN ANALYSIS
*****************************************************************************/

/* QC #24-28: Categorical Column Comparisons */
%macro compare_categorical(dataset1=, dataset2=, prefix1=, prefix2=);
    /* Get categorical (character) columns */
    proc sql noprint;
        select name into :cat_cols separated by ' '
        from dictionary.columns
        where libname = 'WORK'
            and memname = upcase("&dataset1")
            and type = 2;  /* Character type */
    quit;

    /* Calculate statistics for each categorical column */
    %let i = 1;
    %let col = %scan(&cat_cols, &i);

    %do %while(&col ne );
        /* Pre-ingestion categorical stats */
        proc freq data=work.&dataset1 noprint;
            tables &col / out=work.cat_pre_&i;
        run;

        proc sql;
            create table work.cat_sum_pre_&i as
            select
                "&col" as column_name,
                count(distinct &col) as distinct_count,
                (select &col from work.cat_pre_&i
                 where count = max(count)) as mode_value,
                max(count) as mode_count
            from work.&dataset1;
        quit;

        /* Post-ingestion categorical stats */
        proc freq data=work.&dataset2 noprint;
            tables &col / out=work.cat_post_&i;
        run;

        proc sql;
            create table work.cat_sum_post_&i as
            select
                "&col" as column_name,
                count(distinct &col) as distinct_count,
                (select &col from work.cat_post_&i
                 where count = max(count)) as mode_value,
                max(count) as mode_count
            from work.&dataset2;
        quit;

        %let i = %eval(&i + 1);
        %let col = %scan(&cat_cols, &i);
    %end;

    /* Combine results */
    data work.&prefix1._cat_stats;
        set work.cat_sum_pre_:;
    run;

    data work.&prefix2._cat_stats;
        set work.cat_sum_post_:;
    run;

    /* Clean up */
    proc datasets library=work nolist;
        delete cat_pre_: cat_post_: cat_sum_pre_: cat_sum_post_:;
    quit;
%mend;

%compare_categorical(
    dataset1=pre_ingestion,
    dataset2=post_ingestion,
    prefix1=pre,
    prefix2=post
);

/* Create comparison table */
proc sql;
    create table work.qc_24_28_categorical as
    select
        a.column_name,
        a.distinct_count as pre_distinct,
        b.distinct_count as post_distinct,
        b.distinct_count - a.distinct_count as distinct_diff,
        case
            when a.distinct_count = b.distinct_count then 'PASS'
            when abs(b.distinct_count - a.distinct_count) <= 5 then 'WARNING'
            else 'FAIL'
        end as status
    from work.pre_cat_stats a
    inner join work.post_cat_stats b
        on a.column_name = b.column_name;
quit;

/*****************************************************************************
SECTION 7: QUALITY CHECK #29-33 - ADVANCED VALIDATIONS
*****************************************************************************/

/* QC #29: Referential Integrity Check (if foreign keys exist) */
/* Example: Check if all patient_ids in post exist in pre */
proc sql;
    create table work.qc_29_referential as
    select
        'Referential Integrity - patient_id' as check_name,
        count(*) as violations,
        case
            when calculated violations = 0 then 'PASS'
            else 'FAIL'
        end as status
    from work.post_ingestion a
    where not exists (
        select 1 from work.pre_ingestion b
        where a.patient_id = b.patient_id
    );
quit;

/* QC #30: Business Rule Validation */
/* Example: Age should be between 0 and 120 */
proc sql;
    create table work.qc_30_business_rules as
    select
        'Business Rule - Valid Age Range' as check_name,
        sum(case when age < 0 or age > 120 then 1 else 0 end) as pre_violations,
        0 as post_violations,
        'PENDING' as status
    from work.pre_ingestion;

    update work.qc_30_business_rules
    set post_violations = (
        select sum(case when age < 0 or age > 120 then 1 else 0 end)
        from work.post_ingestion
    );

    update work.qc_30_business_rules
    set status = case
        when post_violations = 0 then 'PASS'
        when post_violations <= pre_violations then 'WARNING'
        else 'FAIL'
    end;
quit;

/* QC #31: Date Range Validation */
proc sql;
    create table work.qc_31_date_range as
    select
        'Date Range - Visit Dates' as check_name,
        min(visit_date) as pre_min_date format=date9.,
        max(visit_date) as pre_max_date format=date9.,
        0 as post_min_date,
        0 as post_max_date,
        'PENDING' as status
    from work.pre_ingestion;

    update work.qc_31_date_range
    set post_min_date = (select min(visit_date) from work.post_ingestion),
        post_max_date = (select max(visit_date) from work.post_ingestion);

    update work.qc_31_date_range
    set status = case
        when post_min_date >= pre_min_date
             and post_max_date <= pre_max_date then 'PASS'
        else 'WARNING'
    end;
quit;

/* QC #32: Outlier Detection using IQR method */
%macro detect_outliers(dataset=, var=, prefix=);
    proc means data=work.&dataset noprint;
        var &var;
        output out=work.temp_iqr
            p25=q1
            p75=q3;
    run;

    data _null_;
        set work.temp_iqr;
        iqr = q3 - q1;
        lower_bound = q1 - 1.5 * iqr;
        upper_bound = q3 + 1.5 * iqr;
        call symputx('lower', lower_bound);
        call symputx('upper', upper_bound);
    run;

    proc sql;
        create table work.&prefix._outliers_&var as
        select count(*) as outlier_count
        from work.&dataset
        where &var < &lower or &var > &upper;
    quit;
%mend;

%detect_outliers(dataset=pre_ingestion, var=age, prefix=pre);
%detect_outliers(dataset=post_ingestion, var=age, prefix=post);

proc sql;
    create table work.qc_32_outliers as
    select
        'Outliers - age' as check_name,
        (select outlier_count from work.pre_outliers_age) as pre_outliers,
        (select outlier_count from work.post_outliers_age) as post_outliers,
        case
            when calculated post_outliers <= calculated pre_outliers then 'PASS'
            else 'WARNING'
        end as status;
quit;

/* QC #33: Data Freshness Check */
proc sql;
    create table work.qc_33_freshness as
    select
        'Data Freshness - Days Since Last Update' as check_name,
        intck('day', max(visit_date), today()) as days_since_update,
        case
            when calculated days_since_update <= 7 then 'PASS'
            when calculated days_since_update <= 30 then 'WARNING'
            else 'FAIL'
        end as status
    from work.post_ingestion;
quit;

/*****************************************************************************
SECTION 8: AGGREGATE RESULTS AND CALCULATE QUALITY SCORE
*****************************************************************************/

/* Combine all quality check results */
data work.all_quality_checks;
    set
        work.qc_01_row_count
        work.qc_02_column_check
        work.qc_03_duplicates
        work.qc_04_memory
        work.qc_06_15_null_comparison
        work.qc_29_referential
        work.qc_30_business_rules
        work.qc_32_outliers
        work.qc_33_freshness;
run;

/* Calculate overall quality score */
proc sql;
    create table work.quality_score_summary as
    select
        count(*) as total_checks,
        sum(case when status = 'PASS' then 1 else 0 end) as passed,
        sum(case when status = 'WARNING' then 1 else 0 end) as warnings,
        sum(case when status = 'FAIL' then 1 else 0 end) as failed,

        /* Calculate quality score (0-100) */
        (calculated passed / calculated total_checks) * 100 as quality_score,

        /* Overall status */
        case
            when calculated quality_score >= 90 then 'EXCELLENT'
            when calculated quality_score >= 80 then 'GOOD'
            when calculated quality_score >= 70 then 'ACCEPTABLE'
            else 'NEEDS ATTENTION'
        end as overall_status

    from work.all_quality_checks;
quit;

/*****************************************************************************
SECTION 9: GENERATE EXCEL REPORT
*****************************************************************************/

/* Set ODS options for Excel output */
ods excel file="&OUTPUT_DIR/ETL_Quality_Report_&RUN_DATE..xlsx"
    options(
        sheet_name="Executive Summary"
        embedded_titles="yes"
        embedded_footnotes="yes"
    );

/* Sheet 1: Executive Summary */
title "ETL Quality Validation Report";
title2 "Project: &PROJECT_NAME";
title3 "Run Date: &RUN_DATE &RUN_TIME";

proc print data=work.quality_score_summary noobs label;
    var total_checks passed warnings failed quality_score overall_status;
    label
        total_checks = "Total Checks"
        passed = "Passed"
        warnings = "Warnings"
        failed = "Failed"
        quality_score = "Quality Score (%)"
        overall_status = "Overall Status";
run;

/* Sheet 2: All Quality Checks */
ods excel options(sheet_name="All Quality Checks");

proc print data=work.all_quality_checks noobs label;
    var check_name pre_value post_value difference pct_change status severity;
    label
        check_name = "Quality Check"
        pre_value = "Pre Value"
        post_value = "Post Value"
        difference = "Difference"
        pct_change = "% Change"
        status = "Status"
        severity = "Severity";
run;

/* Sheet 3: Numeric Statistics */
ods excel options(sheet_name="Numeric Statistics");

proc print data=work.qc_16_23_numeric_stats noobs label;
    var column_name pre_mean post_mean mean_pct_chg status;
    label
        column_name = "Column"
        pre_mean = "Pre Mean"
        post_mean = "Post Mean"
        mean_pct_chg = "Mean % Change"
        status = "Status";
run;

/* Sheet 4: Null Analysis */
ods excel options(sheet_name="Null Analysis");

proc print data=work.qc_06_15_null_comparison noobs label;
    var check_name pre_value post_value difference status;
    label
        check_name = "Column"
        pre_value = "Pre Null %"
        post_value = "Post Null %"
        difference = "Difference"
        status = "Status";
run;

/* Sheet 5: Failed Checks */
ods excel options(sheet_name="Failed Checks");

title "Critical Issues Requiring Attention";
proc print data=work.all_quality_checks noobs label;
    where status in ('FAIL', 'WARNING');
    var check_name status difference pct_change;
    label
        check_name = "Quality Check"
        status = "Status"
        difference = "Difference"
        pct_change = "% Change";
run;

/* Close ODS Excel */
ods excel close;
title;

/*****************************************************************************
SECTION 10: LOGGING AND CLEANUP
*****************************************************************************/

/* Log final results */
data _null_;
    set work.quality_score_summary;

    put "=" *80;
    put "ETL QUALITY VALIDATION COMPLETED";
    put "=" *80;
    put "Total Checks: " total_checks;
    put "Passed: " passed;
    put "Warnings: " warnings;
    put "Failed: " failed;
    put "Quality Score: " quality_score "%" ;
    put "Overall Status: " overall_status;
    put "=" *80;
    put "Report Location: &OUTPUT_DIR/ETL_Quality_Report_&RUN_DATE..xlsx";
    put "=" *80;
run;

/* Optional: Export results to permanent library */
data REPORTS.quality_check_results_&RUN_DATE;
    set work.all_quality_checks;
run;

data REPORTS.quality_score_&RUN_DATE;
    set work.quality_score_summary;
run;

/* Clean up work library (optional) */
/*
proc datasets library=work kill nolist;
quit;
*/

/*****************************************************************************
END OF SAS PROGRAM
*****************************************************************************/
