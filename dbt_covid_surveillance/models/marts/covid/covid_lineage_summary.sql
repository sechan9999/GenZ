-- models/marts/covid/covid_lineage_summary.sql
-- Weekly lineage proportions for CDC submission and public health surveillance
-- Aggregates by epi week, geography, and lineage

{{ config(
    materialized='incremental',
    unique_key=['epi_week', 'geography_level', 'geography_value', 'pangolin_lineage'],
    tags=['cdc_submission', 'weekly'],
    on_schema_change='sync_all_columns'
) }}

WITH variants AS (
    SELECT * FROM {{ ref('covid_variants') }}
    WHERE test_event_date >= '{{ var('start_date', '2021-01-01') }}'
    {% if is_incremental() %}
        AND test_event_date >= (SELECT MAX(test_event_date) FROM {{ this }})
    {% endif %}
),

-- Calculate epi weeks using CDC MMWR week standard
with_epi_week AS (
    SELECT
        *,
        {{ epi_week('test_event_date') }} AS epi_week,
        EXTRACT(YEAR FROM test_event_date) AS epi_year
    FROM variants
),

-- Aggregate by multiple geographic levels
county_level AS (
    SELECT
        epi_week,
        epi_year,
        'county' AS geography_level,
        county_fips AS geography_value,
        county_name AS geography_name,
        state,

        pangolin_lineage,
        who_variant,

        COUNT(*) AS sequences_count,
        AVG(genome_coverage) AS avg_genome_coverage,
        AVG(ct_min) AS avg_ct_value,

        -- Demographics
        COUNTIF(age_group_cdc IN ('0-17', '18-29')) AS sequences_under_30,
        COUNTIF(age_group_cdc IN ('65+')) AS sequences_65_plus

    FROM with_epi_week
    WHERE county_fips IS NOT NULL
    GROUP BY ALL
),

state_level AS (
    SELECT
        epi_week,
        epi_year,
        'state' AS geography_level,
        state AS geography_value,
        state AS geography_name,
        NULL AS state,  -- state field is redundant at state level

        pangolin_lineage,
        who_variant,

        COUNT(*) AS sequences_count,
        AVG(genome_coverage) AS avg_genome_coverage,
        AVG(ct_min) AS avg_ct_value,

        COUNTIF(age_group_cdc IN ('0-17', '18-29')) AS sequences_under_30,
        COUNTIF(age_group_cdc IN ('65+')) AS sequences_65_plus

    FROM with_epi_week
    WHERE state IS NOT NULL
    GROUP BY ALL
),

national_level AS (
    SELECT
        epi_week,
        epi_year,
        'national' AS geography_level,
        'USA' AS geography_value,
        'United States' AS geography_name,
        NULL AS state,

        pangolin_lineage,
        who_variant,

        COUNT(*) AS sequences_count,
        AVG(genome_coverage) AS avg_genome_coverage,
        AVG(ct_min) AS avg_ct_value,

        COUNTIF(age_group_cdc IN ('0-17', '18-29')) AS sequences_under_30,
        COUNTIF(age_group_cdc IN ('65+')) AS sequences_65_plus

    FROM with_epi_week
    GROUP BY ALL
),

-- Union all geographic levels
all_geographies AS (
    SELECT * FROM county_level
    UNION ALL
    SELECT * FROM state_level
    UNION ALL
    SELECT * FROM national_level
),

-- Calculate proportions within each geography
with_proportions AS (
    SELECT
        *,

        -- Total sequences for this epi week and geography
        SUM(sequences_count) OVER (
            PARTITION BY epi_week, geography_level, geography_value
        ) AS total_sequences_geo_week,

        -- Calculate percentage
        ROUND(
            100.0 * sequences_count / NULLIF(
                SUM(sequences_count) OVER (
                    PARTITION BY epi_week, geography_level, geography_value
                ),
                0
            ),
            2
        ) AS percent_lineage,

        -- Rank lineages by prevalence within geography/week
        ROW_NUMBER() OVER (
            PARTITION BY epi_week, geography_level, geography_value
            ORDER BY sequences_count DESC
        ) AS lineage_rank

    FROM all_geographies
),

-- Filter to CDC reporting thresholds and add metadata
final AS (
    SELECT
        -- Time dimension
        epi_week,
        epi_year,
        {{ dbt_utils.date_trunc('week', 'epi_week') }} AS week_start_date,
        {{ dbt_utils.dateadd('day', 6, dbt_utils.date_trunc('week', 'epi_week')) }} AS week_end_date,

        -- Geography
        geography_level,
        geography_value,
        geography_name,
        state,

        -- Lineage
        pangolin_lineage,
        who_variant,

        -- Counts and proportions
        sequences_count,
        total_sequences_geo_week,
        percent_lineage,
        lineage_rank,

        -- Quality metrics
        avg_genome_coverage,
        avg_ct_value,

        -- Demographics
        sequences_under_30,
        sequences_65_plus,
        ROUND(100.0 * sequences_under_30 / NULLIF(sequences_count, 0), 1) AS pct_under_30,
        ROUND(100.0 * sequences_65_plus / NULLIF(sequences_count, 0), 1) AS pct_65_plus,

        -- Flags
        CASE
            WHEN sequences_count >= {{ var('cdc_minimum_sequences', 5) }} THEN TRUE
            ELSE FALSE
        END AS meets_cdc_threshold,

        CASE
            WHEN lineage_rank = 1 THEN TRUE
            ELSE FALSE
        END AS is_dominant_lineage,

        -- Metadata
        CURRENT_TIMESTAMP AS report_generated_at

    FROM with_proportions
    WHERE TRUE
        -- Apply CDC minimum threshold
        AND sequences_count >= {{ var('cdc_minimum_sequences', 5) }}
        -- Only include known lineages
        AND pangolin_lineage IS NOT NULL
        AND pangolin_lineage != 'Unknown'
)

SELECT *
FROM final
ORDER BY
    epi_week DESC,
    geography_level,
    geography_value,
    percent_lineage DESC
