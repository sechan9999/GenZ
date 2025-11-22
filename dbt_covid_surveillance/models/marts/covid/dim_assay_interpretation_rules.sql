-- models/marts/covid/dim_assay_interpretation_rules.sql
-- Business rules for interpreting COVID test results
-- Uses SCD Type 2 pattern to track rule changes over time

{{ config(
    materialized='table',
    tags=['dimension', 'assay_rules']
) }}

WITH assay_rules_seed AS (
    SELECT * FROM {{ ref('assay_rules') }}
),

current_rules AS (
    SELECT
        {{ dbt_utils.generate_surrogate_key(['assay_name', 'reagent_lot', 'effective_start_date']) }} AS rule_sk,

        assay_name,
        assay_manufacturer,
        reagent_lot,

        -- CT value cutoffs for interpretation
        ct_cutoff_n,
        ct_cutoff_orf,
        ct_cutoff_e,
        ct_cutoff_s,

        -- Interpretation logic
        min_targets_for_positive,  -- How many targets must be positive
        inconclusive_threshold,    -- CT value range that triggers inconclusive
        repeat_testing_threshold,  -- CT value that triggers repeat testing

        -- EUA and regulatory information
        fda_eua_number,
        clia_complexity,  -- waived, moderate, high
        is_point_of_care,

        -- Validity period (SCD Type 2)
        effective_start_date,
        effective_end_date,
        CASE
            WHEN effective_end_date IS NULL OR effective_end_date >= CURRENT_DATE
            THEN TRUE
            ELSE FALSE
        END AS is_current,

        -- Metadata
        created_by,
        created_dttm,
        notes

    FROM assay_rules_seed
)

SELECT *
FROM current_rules
