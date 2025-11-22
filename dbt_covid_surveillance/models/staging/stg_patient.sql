-- models/staging/stg_patient.sql
-- Staging model for patient demographics
-- Handles PHI with proper de-identification for public health reporting

{{ config(
    materialized='view',
    tags=['staging', 'patient', 'phi']
) }}

WITH source AS (
    SELECT * FROM {{ source('raw_ehr', 'patient_demographics') }}
),

cleaned AS (
    SELECT
        -- Patient identifiers (PHI - handle with care)
        TRIM(pat_id) AS pat_id,
        TRIM(mrn) AS mrn,
        {{ dbt_utils.generate_surrogate_key(['pat_id']) }} AS patient_sk,

        -- Demographics
        TRY_CAST(date_of_birth AS DATE) AS date_of_birth,
        TRIM(UPPER(sex)) AS sex,  -- M, F, U, O
        TRIM(UPPER(gender_identity)) AS gender_identity,
        TRIM(race) AS race,
        TRIM(ethnicity) AS ethnicity,
        TRIM(UPPER(preferred_language)) AS preferred_language,

        -- Geographic information
        TRIM(address_line1) AS address_line1,
        TRIM(address_line2) AS address_line2,
        TRIM(city) AS city,
        TRIM(UPPER(state)) AS state,
        TRIM(zip_code) AS zip_code,
        LEFT(TRIM(zip_code), 5) AS zip5,  -- 5-digit ZIP for privacy
        LEFT(TRIM(zip_code), 3) AS zip3,  -- 3-digit ZIP for further de-identification
        TRIM(county_name) AS county_name,
        TRIM(county_fips) AS county_fips,

        -- Contact information (PHI)
        TRIM(phone_number) AS phone_number,
        TRIM(LOWER(email)) AS email,

        -- Clinical flags
        CASE WHEN LOWER(is_deceased) = 'true' THEN TRUE ELSE FALSE END AS is_deceased,
        TRY_CAST(date_of_death AS DATE) AS date_of_death,

        -- Metadata
        TRY_CAST(created_dttm AS TIMESTAMP) AS created_dttm,
        TRY_CAST(updated_dttm AS TIMESTAMP) AS updated_dttm,
        TRIM(source_system) AS source_system

    FROM source
),

-- Standardize race and ethnicity categories (CDC format)
standardized AS (
    SELECT
        *,

        -- Standardize race to CDC categories
        CASE
            WHEN race LIKE '%White%' THEN 'White'
            WHEN race LIKE '%Black%' OR race LIKE '%African%' THEN 'Black or African American'
            WHEN race LIKE '%Asian%' THEN 'Asian'
            WHEN race LIKE '%Native Hawaiian%' OR race LIKE '%Pacific Islander%' THEN 'Native Hawaiian or Other Pacific Islander'
            WHEN race LIKE '%American Indian%' OR race LIKE '%Alaska Native%' THEN 'American Indian or Alaska Native'
            WHEN race LIKE '%Multiple%' OR race LIKE '%More than one%' THEN 'Multiple Races'
            WHEN race IS NULL OR race LIKE '%Unknown%' OR race LIKE '%Declined%' THEN 'Unknown'
            ELSE 'Other'
        END AS race_cdc_category,

        -- Standardize ethnicity to CDC categories
        CASE
            WHEN ethnicity LIKE '%Hispanic%' OR ethnicity LIKE '%Latino%' THEN 'Hispanic or Latino'
            WHEN ethnicity LIKE '%Not Hispanic%' OR ethnicity LIKE '%Not Latino%' THEN 'Not Hispanic or Latino'
            WHEN ethnicity IS NULL OR ethnicity LIKE '%Unknown%' OR ethnicity LIKE '%Declined%' THEN 'Unknown'
            ELSE 'Unknown'
        END AS ethnicity_cdc_category,

        -- Standardize sex
        CASE
            WHEN sex IN ('M', 'MALE') THEN 'Male'
            WHEN sex IN ('F', 'FEMALE') THEN 'Female'
            WHEN sex IN ('U', 'UNKNOWN', 'UNK') THEN 'Unknown'
            WHEN sex IN ('O', 'OTHER') THEN 'Other'
            ELSE 'Unknown'
        END AS sex_standardized

    FROM cleaned
)

SELECT *
FROM standardized
WHERE pat_id IS NOT NULL
