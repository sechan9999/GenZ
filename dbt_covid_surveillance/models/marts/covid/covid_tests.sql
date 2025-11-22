-- models/marts/covid/covid_tests.sql
-- Final COVID test results with interpretation, demographics, and analytics fields
-- This is the primary fact table for COVID testing surveillance

{{ config(
    materialized='incremental',
    unique_key='accession_number',
    on_schema_change='sync_all_columns',
    tags=['covid', 'daily', 'fact_table'],
    partition_by={
        "field": "test_event_date",
        "data_type": "date",
        "granularity": "day"
    }
) }}

WITH raw_lims AS (
    SELECT * FROM {{ ref('stg_lims_results') }}
    {% if is_incremental() %}
    WHERE result_released_dttm >= (SELECT MAX(result_released_dttm) FROM {{ this }})
    {% endif %}
),

assay_rules AS (
    SELECT * FROM {{ ref('dim_assay_interpretation_rules') }}
    WHERE is_current = TRUE
),

-- Group results by accession (handle multi-target assays)
results_grouped AS (
    SELECT
        accession_number,
        pat_id,
        specimen_collected_dttm,
        specimen_received_dttm,
        result_released_dttm,
        assay_name,
        assay_manufacturer,
        reagent_lot,
        specimen_type,
        specimen_source,
        ordering_provider_npi,

        -- Aggregate CT values and targets
        MIN(ct_value) AS ct_min,
        MAX(ct_value) AS ct_max,
        AVG(ct_value) AS ct_mean,
        COUNT(DISTINCT target) AS targets_tested,
        COUNTIF(ct_value < {{ var('default_ct_cutoff', 40) }}) AS targets_detected,

        -- Collect target-specific results
        MAX(CASE WHEN target = 'N_GENE' THEN ct_value END) AS ct_n_gene,
        MAX(CASE WHEN target = 'ORF1AB' THEN ct_value END) AS ct_orf1ab,
        MAX(CASE WHEN target = 'E_GENE' THEN ct_value END) AS ct_e_gene,
        MAX(CASE WHEN target = 'S_GENE' THEN ct_value END) AS ct_s_gene,

        -- Antigen test results (no CT values)
        MAX(qualitative_result) AS qualitative_result,

        -- Metadata
        source_system,
        created_dttm,
        updated_dttm

    FROM raw_lims
    GROUP BY ALL
),

-- Apply interpretation logic using assay-specific rules
interpreted AS (
    SELECT
        r.*,

        -- Apply custom interpretation logic
        CASE
            -- PCR tests: use CT value cutoffs
            WHEN r.assay_name LIKE '%PCR%' OR r.assay_name LIKE '%RT-PCR%' THEN
                CASE
                    -- Positive: Multiple targets below cutoff
                    WHEN COALESCE(
                        (CASE WHEN r.ct_n_gene <= COALESCE(a.ct_cutoff_n, 40) THEN 1 ELSE 0 END) +
                        (CASE WHEN r.ct_orf1ab <= COALESCE(a.ct_cutoff_orf, 40) THEN 1 ELSE 0 END) +
                        (CASE WHEN r.ct_e_gene <= COALESCE(a.ct_cutoff_e, 40) THEN 1 ELSE 0 END) +
                        (CASE WHEN r.ct_s_gene <= COALESCE(a.ct_cutoff_s, 40) THEN 1 ELSE 0 END),
                        0
                    ) >= COALESCE(a.min_targets_for_positive, 2) THEN 'Detected'

                    -- Inconclusive: Single weak positive
                    WHEN r.targets_detected = 1 AND r.ct_min > COALESCE(a.inconclusive_threshold, 35) THEN 'Inconclusive'

                    -- Negative: No targets detected or all above cutoff
                    WHEN r.targets_detected = 0 OR r.ct_min > COALESCE(a.ct_cutoff_n, 40) THEN 'Not Detected'

                    ELSE 'Inconclusive'
                END

            -- Antigen tests: use qualitative result directly
            WHEN r.assay_name LIKE '%Antigen%' THEN
                CASE
                    WHEN r.qualitative_result = 'POSITIVE' THEN 'Detected'
                    WHEN r.qualitative_result = 'NEGATIVE' THEN 'Not Detected'
                    ELSE 'Inconclusive'
                END

            -- Default: use targets_detected
            WHEN r.targets_detected >= 2 THEN 'Detected'
            WHEN r.targets_detected = 0 THEN 'Not Detected'
            ELSE 'Inconclusive'
        END AS final_result,

        -- Flag for repeat testing recommendation
        CASE
            WHEN r.targets_detected = 1 AND r.ct_min > COALESCE(a.inconclusive_threshold, 35)
            THEN TRUE
            ELSE FALSE
        END AS recommend_repeat_testing,

        -- Assay metadata
        a.fda_eua_number,
        a.clia_complexity,
        a.is_point_of_care

    FROM results_grouped r
    LEFT JOIN assay_rules a
        ON r.assay_name = a.assay_name
       AND COALESCE(r.reagent_lot, 'UNKNOWN') = COALESCE(a.reagent_lot, 'UNKNOWN')
),

-- Join with patient demographics
with_demographics AS (
    SELECT
        i.*,

        -- Patient demographics
        d.patient_sk,
        d.date_of_birth,
        d.sex,
        d.race,
        d.ethnicity,
        d.preferred_language,
        d.zip5,
        d.zip3,
        d.city,
        d.state,
        d.county_name,
        d.county_fips,

        -- Calculate age at test
        {{ dbt_utils.dateadd(
            'year',
            dbt_utils.datediff(
                'i.specimen_collected_dttm',
                'd.date_of_birth',
                'year'
            ),
            'd.date_of_birth'
        ) }} AS age_at_test_approx,

        DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) AS age_at_test,

        -- Age groups for reporting
        CASE
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) < 18 THEN '0-17'
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) BETWEEN 18 AND 29 THEN '18-29'
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) BETWEEN 30 AND 39 THEN '30-39'
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) BETWEEN 40 AND 49 THEN '40-49'
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) BETWEEN 50 AND 64 THEN '50-64'
            WHEN DATEDIFF('year', d.date_of_birth, COALESCE(i.specimen_collected_dttm, i.result_released_dttm)) >= 65 THEN '65+'
            ELSE 'Unknown'
        END AS age_group_cdc

    FROM interpreted i
    LEFT JOIN {{ ref('dim_patient') }} d
        ON i.pat_id = d.pat_id
),

-- Final table with all fields
final AS (
    SELECT
        -- Primary key
        accession_number,

        -- Surrogate key for joins
        {{ dbt_utils.generate_surrogate_key(['accession_number']) }} AS test_sk,

        -- Test event date (primary partition key)
        COALESCE(specimen_collected_dttm::DATE, result_released_dttm::DATE) AS test_event_date,

        -- Timestamps
        specimen_collected_dttm,
        specimen_received_dttm,
        result_released_dttm,

        -- Patient
        pat_id,
        patient_sk,
        age_at_test,
        age_group_cdc,
        sex,
        race,
        ethnicity,
        preferred_language,

        -- Geography
        zip5,
        zip3,
        city,
        state,
        county_name,
        county_fips,

        -- Specimen
        specimen_type,
        specimen_source,

        -- Test and result
        assay_name,
        assay_manufacturer,
        reagent_lot,
        fda_eua_number,
        clia_complexity,
        is_point_of_care,
        final_result,
        recommend_repeat_testing,

        -- CT values
        ct_min,
        ct_max,
        ct_mean,
        ct_n_gene,
        ct_orf1ab,
        ct_e_gene,
        ct_s_gene,

        -- Target counts
        targets_tested,
        targets_detected,

        -- Ordering provider
        ordering_provider_npi,

        -- Metadata
        source_system,
        created_dttm,
        updated_dttm

    FROM with_demographics
)

SELECT *
FROM final
