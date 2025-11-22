-- snapshots/snap_lineage_classifications.sql
-- Snapshot of WHO variant classifications over time
-- Lineage designations can change (VOC → VOI → delisted) so we need history

{% snapshot snap_lineage_classifications %}

{{
    config(
      target_schema='snapshots',
      unique_key='variant_id',
      strategy='timestamp',
      updated_at='updated_dttm',
      invalidate_hard_deletes=True
    )
}}

SELECT
    {{ dbt_utils.generate_surrogate_key(['who_label']) }} AS variant_id,

    who_label,
    pango_lineages,
    classification,  -- VOC, VOI, VUM, or delisted

    -- Important dates
    emergence_date,
    detection_date,
    designation_date,

    -- Defining characteristics
    key_mutations,
    notes,

    -- Metadata
    CURRENT_TIMESTAMP AS updated_dttm

FROM {{ ref('who_variants') }}

{% endsnapshot %}
