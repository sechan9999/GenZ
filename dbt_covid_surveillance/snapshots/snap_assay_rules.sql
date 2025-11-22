-- snapshots/snap_assay_rules.sql
-- Slowly Changing Dimension (SCD) Type 2 snapshot for assay interpretation rules
-- Tracks historical changes to CT cutoffs, EUA approvals, and interpretation logic

{% snapshot snap_assay_rules %}

{{
    config(
      target_schema='snapshots',
      unique_key='rule_id',
      strategy='timestamp',
      updated_at='updated_dttm',
      invalidate_hard_deletes=True
    )
}}

SELECT
    {{ dbt_utils.generate_surrogate_key([
        'assay_name',
        'reagent_lot',
        'effective_start_date'
    ]) }} AS rule_id,

    assay_name,
    assay_manufacturer,
    reagent_lot,

    -- CT cutoffs (these change as new data emerges)
    ct_cutoff_n,
    ct_cutoff_orf,
    ct_cutoff_e,
    ct_cutoff_s,

    -- Interpretation parameters
    min_targets_for_positive,
    inconclusive_threshold,
    repeat_testing_threshold,

    -- Regulatory
    fda_eua_number,
    clia_complexity,
    is_point_of_care,

    -- Validity period
    effective_start_date,
    effective_end_date,

    -- Metadata
    created_by,
    created_dttm,
    CURRENT_TIMESTAMP AS updated_dttm,
    notes

FROM {{ ref('assay_rules') }}

{% endsnapshot %}
