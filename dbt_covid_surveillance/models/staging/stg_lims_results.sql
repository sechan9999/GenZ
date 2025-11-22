-- models/staging/stg_lims_results.sql
-- Staging model for laboratory information management system (LIMS) results
-- Cleans and standardizes raw LIMS data for downstream processing

{{ config(
    materialized='view',
    tags=['staging', 'lims']
) }}

WITH source AS (
    SELECT * FROM {{ source('raw_lims', 'laboratory_results') }}
),

cleaned AS (
    SELECT
        -- Identifiers
        TRIM(UPPER(accession_number)) AS accession_number,
        TRIM(pat_id) AS pat_id,
        TRIM(ordering_provider_npi) AS ordering_provider_npi,

        -- Specimen information
        TRIM(specimen_type) AS specimen_type,
        TRIM(UPPER(specimen_source)) AS specimen_source,
        TRY_CAST(specimen_collected_dttm AS TIMESTAMP) AS specimen_collected_dttm,
        TRY_CAST(specimen_received_dttm AS TIMESTAMP) AS specimen_received_dttm,

        -- Test information
        TRIM(order_code) AS order_code,
        TRIM(order_name) AS order_name,
        TRIM(assay_name) AS assay_name,
        TRIM(assay_manufacturer) AS assay_manufacturer,
        TRIM(reagent_lot) AS reagent_lot,
        TRIM(instrument_id) AS instrument_id,

        -- Result information
        TRIM(test_code) AS test_code,
        TRIM(test_name) AS test_name,
        TRIM(UPPER(target)) AS target,  -- N_GENE, ORF1AB, etc.

        -- CT value parsing (handle various formats)
        CASE
            WHEN REGEXP_LIKE(ct_value_raw, '^[0-9]+\.?[0-9]*$')
                THEN TRY_CAST(ct_value_raw AS DECIMAL(10,2))
            WHEN LOWER(ct_value_raw) IN ('not detected', 'nd', 'undetermined', 'null')
                THEN NULL
            ELSE NULL
        END AS ct_value,

        TRIM(UPPER(qualitative_result)) AS qualitative_result,  -- POSITIVE, NEGATIVE, INDETERMINATE
        TRIM(result_comment) AS result_comment,
        TRIM(result_status) AS result_status,  -- FINAL, PRELIMINARY, CORRECTED

        -- Timestamps
        TRY_CAST(result_dttm AS TIMESTAMP) AS result_dttm,
        TRY_CAST(result_released_dttm AS TIMESTAMP) AS result_released_dttm,
        TRY_CAST(result_verified_dttm AS TIMESTAMP) AS result_verified_dttm,

        -- Verification and quality flags
        TRIM(verified_by) AS verified_by,
        CASE
            WHEN LOWER(qc_flag) = 'pass' OR qc_flag IS NULL THEN TRUE
            ELSE FALSE
        END AS qc_passed,

        -- Metadata
        TRY_CAST(created_dttm AS TIMESTAMP) AS created_dttm,
        TRY_CAST(updated_dttm AS TIMESTAMP) AS updated_dttm,
        TRIM(source_system) AS source_system

    FROM source
),

-- Filter out invalid records
validated AS (
    SELECT *
    FROM cleaned
    WHERE TRUE
        AND accession_number IS NOT NULL
        AND pat_id IS NOT NULL
        AND assay_name IS NOT NULL
        -- Must have either CT value or qualitative result
        AND (ct_value IS NOT NULL OR qualitative_result IS NOT NULL)
        -- Must have collection or result date
        AND (specimen_collected_dttm IS NOT NULL OR result_released_dttm IS NOT NULL)
        -- Only final or corrected results
        AND result_status IN ('FINAL', 'CORRECTED')
        -- QC must have passed
        AND qc_passed = TRUE
)

SELECT *
FROM validated
