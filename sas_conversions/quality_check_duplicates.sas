/*******************************************************************************
 * PROGRAM: quality_check_duplicates.sas
 * PURPOSE: Replicates Python duplicate detection logic for NHSN LTCF Data QA
 * AUTHOR: Converted from Python to SAS
 * DATE: 2025-11-23
 *
 * STRATEGY: Moving from Python Pandas to SAS PROC SORT/SQL for duplicate QA
 * CONTEXT: NHSN Long-Term Care Facility (LTCF) Data Quality Assurance
 *
 * DESCRIPTION:
 * This program replicates the 'Weekly Duplicates Report' logic to ensure
 * Governor's Report accuracy by identifying duplicate records and checking
 * for conflicting COVID status values.
 ******************************************************************************/

/*-----------------------------------------------------------------------------
 * STEP 1: IDENTIFY ALL DUPLICATES
 *
 * EXPLANATION:
 * - Sort data by key columns (facility_id, resident_id, reporting_week)
 * - Use FIRST. and LAST. processing to flag duplicate groups
 * - Unlike NODUPKEY (which keeps only first), this marks ALL duplicates
 * - Equivalent to Python's df.duplicated(keep=False)
 *---------------------------------------------------------------------------*/

PROC SORT DATA=input_data OUT=sorted_data;
    BY facility_id resident_id reporting_week;
RUN;

DATA all_duplicates;
    SET sorted_data;
    BY facility_id resident_id reporting_week;

    /* Flag if record is part of a duplicate group */
    /* If FIRST=1 and LAST=1, it's unique (not a duplicate) */
    IF NOT (FIRST.reporting_week AND LAST.reporting_week) THEN OUTPUT;

    /* Alternative method: Count occurrences */
    RETAIN dup_count;
    IF FIRST.reporting_week THEN dup_count = 0;
    dup_count + 1;
RUN;

/*-----------------------------------------------------------------------------
 * STEP 2: COUNT TOTAL DUPLICATES
 *
 * EXPLANATION:
 * - Calculate total number of duplicate records found
 * - This replicates: len(duplicates) in Python
 *---------------------------------------------------------------------------*/

PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :dup_total
    FROM all_duplicates;
QUIT;

%PUT CRITICAL: Found &dup_total duplicate records.;

/*-----------------------------------------------------------------------------
 * STEP 3: IDENTIFY CONFLICTING STATUS VALUES
 *
 * EXPLANATION:
 * - Group duplicates by key columns
 * - Count distinct covid_status values per group
 * - If count > 1, there are conflicting statuses (e.g., Positive vs Negative)
 * - This replicates: duplicates.groupby(key_columns)['covid_status'].nunique()
 *---------------------------------------------------------------------------*/

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
    HAVING status_count > 1;  /* Only show true conflicts */
QUIT;

/*-----------------------------------------------------------------------------
 * STEP 4: REPORT RESULTS
 *
 * EXPLANATION:
 * - Print conflict summary to log
 * - Generate detailed conflict report if conflicts exist
 * - This replicates the return logic in Python function
 *---------------------------------------------------------------------------*/

PROC SQL NOPRINT;
    SELECT COUNT(*) INTO :conflict_count
    FROM status_conflicts;
QUIT;

%MACRO report_results;
    %IF &conflict_count > 0 %THEN %DO;
        %PUT ERROR: Found &conflict_count groups with conflicting COVID status!;

        /* Print detailed conflict report */
        TITLE "DUPLICATE RECORDS WITH CONFLICTING COVID STATUS";
        PROC PRINT DATA=status_conflicts;
            VAR facility_id resident_id reporting_week status_count record_count;
        RUN;
        TITLE;

        /* Optional: Show all records in conflict groups for investigation */
        TITLE "DETAILED VIEW: ALL RECORDS IN CONFLICT GROUPS";
        PROC SQL;
            CREATE TABLE conflict_details AS
            SELECT a.*
            FROM all_duplicates AS a
            INNER JOIN status_conflicts AS b
                ON a.facility_id = b.facility_id
                AND a.resident_id = b.resident_id
                AND a.reporting_week = b.reporting_week
            ORDER BY a.facility_id, a.resident_id, a.reporting_week, a.covid_status;
        QUIT;

        PROC PRINT DATA=conflict_details;
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

/*-----------------------------------------------------------------------------
 * OPTIONAL: CREATE CLEAN DATASET (REMOVE DUPLICATES)
 *
 * EXPLANATION:
 * - If you need to create a deduplicated dataset (like Python's drop_duplicates)
 * - NODUPKEY keeps only the FIRST occurrence of each duplicate group
 *---------------------------------------------------------------------------*/

PROC SORT DATA=input_data OUT=clean_data NODUPKEY;
    BY facility_id resident_id reporting_week;
RUN;

/*-----------------------------------------------------------------------------
 * INTERVIEW TALKING POINT:
 *
 * "In Python, this was an inline function that runs BEFORE data hits the
 * dashboard, blocking bad data immediately. In SAS, this is traditionally
 * a batch process that outputs to the log and generates reports. However,
 * I can integrate this into ETL pipelines using macro logic to halt
 * processing if critical errors are found (&SYSERR, PROC SQL return codes)."
 *
 * KEY SAS vs PYTHON DIFFERENCES:
 *
 * 1. DUPLICATE DETECTION:
 *    - Python: df.duplicated(subset=cols, keep=False)
 *    - SAS: FIRST./LAST. processing in DATA step
 *
 * 2. GROUPBY + AGGREGATION:
 *    - Python: groupby().nunique()
 *    - SAS: PROC SQL GROUP BY with COUNT(DISTINCT)
 *
 * 3. RETURN VALUES:
 *    - Python: Function returns DataFrame or string
 *    - SAS: Outputs datasets, log messages, and reports
 *
 * 4. ERROR HANDLING:
 *    - Python: Conditional logic with if/else
 *    - SAS: Macro logic (%IF/%THEN) with log messages
 *
 * 5. PERFORMANCE:
 *    - Python: In-memory processing (good for interactive analysis)
 *    - SAS: Disk-based with excellent optimization for large datasets
 ******************************************************************************/
