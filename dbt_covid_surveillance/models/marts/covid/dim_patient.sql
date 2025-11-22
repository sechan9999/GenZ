-- models/marts/covid/dim_patient.sql
-- Patient dimension with demographics for COVID testing analysis
-- Includes age calculation at time of test

{{ config(
    materialized='table',
    tags=['dimension', 'patient']
) }}

WITH patient_base AS (
    SELECT * FROM {{ ref('stg_patient') }}
),

patient_with_age AS (
    SELECT
        pat_id,
        patient_sk,
        mrn,

        -- Demographics
        date_of_birth,
        sex_standardized AS sex,
        gender_identity,
        race_cdc_category AS race,
        ethnicity_cdc_category AS ethnicity,
        preferred_language,

        -- Age groups (calculated dynamically at query time against test date)
        -- These are helper columns for common age groupings

        -- Geographic
        zip5,
        zip3,
        city,
        state,
        county_name,
        county_fips,

        -- Clinical
        is_deceased,
        date_of_death,

        -- Metadata
        created_dttm,
        updated_dttm,
        source_system

    FROM patient_base
)

SELECT *
FROM patient_with_age
