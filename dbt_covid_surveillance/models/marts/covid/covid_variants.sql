-- models/marts/covid/covid_variants.sql
-- SARS-CoV-2 variant surveillance with lineage tracking and mutation profiles
-- Links sequencing results to test results for comprehensive epidemiological analysis

{{ config(
    materialized='table',
    tags=['variants', 'genomics', 'surveillance']
) }}

WITH sequencing_runs AS (
    SELECT * FROM {{ ref('stg_sequencing_results') }}
),

test_results AS (
    SELECT * FROM {{ ref('covid_tests') }}
),

-- Join sequencing to test results
sequences_with_tests AS (
    SELECT
        s.accession_number,
        s.sequencing_run_id,
        s.sample_id,

        -- Test information
        t.test_event_date,
        t.specimen_collected_dttm,
        t.final_result,
        t.ct_min,
        t.age_at_test,
        t.age_group_cdc,
        t.sex,
        t.race,
        t.ethnicity,
        t.county_name,
        t.county_fips,
        t.state,
        t.zip3,

        -- Sequencing metadata
        s.sequenced_date,
        s.sequencing_platform,
        s.library_prep_kit,

        -- Genome quality
        s.genome_coverage,
        s.mean_depth,
        s.n_content_percent,
        s.passes_genome_qc,

        -- Lineage assignments
        s.pangolin_lineage,
        s.pangolin_conflict,
        s.pangolin_probability,
        s.pangolin_version,
        s.pango_version,
        s.nextclade_clade,
        s.nextclade_lineage,
        s.nextclade_qc_score,
        s.nextclade_qc_status,

        -- Mutation counts
        s.nextclade_total_substitutions,
        s.nextclade_total_deletions,
        s.nextclade_total_insertions,
        s.nextclade_total_missing,

        -- WHO classification
        s.who_variant,
        s.who_variant_status,

        -- Mutation list (JSON)
        s.mutation_list,

        -- Pre-flagged key mutations
        s.has_e484k,
        s.has_n501y,
        s.has_l452r,
        s.has_del69_70,

        -- Analysis metadata
        s.pipeline_version,
        s.analysis_lab,
        s.created_dttm,
        s.updated_dttm

    FROM sequencing_runs s
    LEFT JOIN test_results t
        ON s.accession_number = t.accession_number
),

-- Parse mutation list JSON to extract all mutations
mutations_parsed AS (
    SELECT
        accession_number,
        TRIM(mutation.value::STRING) AS mutation
    FROM sequences_with_tests,
    LATERAL FLATTEN(input => PARSE_JSON(mutation_list)) mutation
    WHERE mutation_list IS NOT NULL
),

-- Aggregate mutation flags
mutation_flags AS (
    SELECT
        accession_number,

        -- Spike protein mutations of interest (partial list)
        MAX(CASE WHEN mutation LIKE 'S:E484%' THEN 1 ELSE 0 END) AS has_spike_e484_any,
        MAX(CASE WHEN mutation = 'S:E484K' THEN 1 ELSE 0 END) AS has_spike_e484k,
        MAX(CASE WHEN mutation = 'S:E484A' THEN 1 ELSE 0 END) AS has_spike_e484a,
        MAX(CASE WHEN mutation = 'S:N501Y' THEN 1 ELSE 0 END) AS has_spike_n501y,
        MAX(CASE WHEN mutation = 'S:L452R' THEN 1 ELSE 0 END) AS has_spike_l452r,
        MAX(CASE WHEN mutation = 'S:K417N' THEN 1 ELSE 0 END) AS has_spike_k417n,
        MAX(CASE WHEN mutation = 'S:T478K' THEN 1 ELSE 0 END) AS has_spike_t478k,
        MAX(CASE WHEN mutation = 'S:P681R' THEN 1 ELSE 0 END) AS has_spike_p681r,
        MAX(CASE WHEN mutation = 'S:P681H' THEN 1 ELSE 0 END) AS has_spike_p681h,

        -- Deletions
        MAX(CASE WHEN mutation LIKE '%del69_70%' OR mutation LIKE '%del69-70%' THEN 1 ELSE 0 END) AS has_del69_70,
        MAX(CASE WHEN mutation LIKE '%del144%' THEN 1 ELSE 0 END) AS has_del144,

        -- Nucleocapsid mutations
        MAX(CASE WHEN mutation LIKE 'N:%' THEN 1 ELSE 0 END) AS has_n_gene_mutations,

        -- Count total mutations
        COUNT(DISTINCT mutation) AS total_unique_mutations

    FROM mutations_parsed
    GROUP BY accession_number
),

-- Classify into WHO variant categories based on defining mutations
who_classification AS (
    SELECT
        s.*,
        COALESCE(m.has_spike_e484_any, 0) AS has_spike_e484_any,
        COALESCE(m.has_spike_e484k, 0) AS has_spike_e484k,
        COALESCE(m.has_spike_e484a, 0) AS has_spike_e484a,
        COALESCE(m.has_spike_n501y, 0) AS has_spike_n501y,
        COALESCE(m.has_spike_l452r, 0) AS has_spike_l452r,
        COALESCE(m.has_spike_k417n, 0) AS has_spike_k417n,
        COALESCE(m.has_spike_t478k, 0) AS has_spike_t478k,
        COALESCE(m.has_spike_p681r, 0) AS has_spike_p681r,
        COALESCE(m.has_spike_p681h, 0) AS has_spike_p681h,
        COALESCE(m.has_del69_70, 0) AS has_del69_70,
        COALESCE(m.has_del144, 0) AS has_del144,
        COALESCE(m.has_n_gene_mutations, 0) AS has_n_gene_mutations,
        COALESCE(m.total_unique_mutations, 0) AS total_unique_mutations,

        -- Classify WHO variant if not already provided
        COALESCE(
            s.who_variant,
            CASE
                -- Alpha (B.1.1.7)
                WHEN s.pangolin_lineage LIKE 'B.1.1.7%' OR s.pangolin_lineage LIKE 'Q.%' THEN 'Alpha'

                -- Beta (B.1.351)
                WHEN s.pangolin_lineage LIKE 'B.1.351%' THEN 'Beta'

                -- Gamma (P.1)
                WHEN s.pangolin_lineage LIKE 'P.1%' THEN 'Gamma'

                -- Delta (B.1.617.2)
                WHEN s.pangolin_lineage LIKE 'B.1.617.2%' OR s.pangolin_lineage LIKE 'AY.%' THEN 'Delta'

                -- Omicron (B.1.1.529)
                WHEN s.pangolin_lineage LIKE 'B.1.1.529%'
                  OR s.pangolin_lineage LIKE 'BA.%'
                  OR s.pangolin_lineage LIKE 'BE.%'
                  OR s.pangolin_lineage LIKE 'BF.%'
                  OR s.pangolin_lineage LIKE 'BQ.%'
                  OR s.pangolin_lineage LIKE 'XBB%'
                  OR s.pangolin_lineage LIKE 'JN.%'
                  THEN 'Omicron'

                ELSE 'Other'
            END
        ) AS who_variant_classified

    FROM sequences_with_tests s
    LEFT JOIN mutation_flags m
        ON s.accession_number = m.accession_number
),

-- Final output with all fields
final AS (
    SELECT
        -- Identifiers
        accession_number,
        sequencing_run_id,
        sample_id,

        -- Test information
        test_event_date,
        specimen_collected_dttm,
        sequenced_date,
        final_result,
        ct_min,

        -- Demographics
        age_at_test,
        age_group_cdc,
        sex,
        race,
        ethnicity,

        -- Geography
        county_name,
        county_fips,
        state,
        zip3,

        -- Sequencing platform
        sequencing_platform,
        library_prep_kit,

        -- Quality metrics
        genome_coverage,
        mean_depth,
        n_content_percent,
        passes_genome_qc,

        -- Lineage classification
        pangolin_lineage,
        pangolin_conflict,
        pangolin_probability,
        pangolin_version,
        pango_version,
        nextclade_clade,
        nextclade_lineage,
        nextclade_qc_score,
        nextclade_qc_status,

        -- WHO classification
        who_variant_classified AS who_variant,
        who_variant_status,

        -- Mutation profile
        nextclade_total_substitutions,
        nextclade_total_deletions,
        nextclade_total_insertions,
        nextclade_total_missing,
        total_unique_mutations,

        -- Key mutations (binary flags)
        has_spike_e484_any,
        has_spike_e484k,
        has_spike_e484a,
        has_spike_n501y,
        has_spike_l452r,
        has_spike_k417n,
        has_spike_t478k,
        has_spike_p681r,
        has_spike_p681h,
        has_del69_70,
        has_del144,
        has_n_gene_mutations,

        -- Full mutation list (JSON)
        mutation_list,

        -- Metadata
        pipeline_version,
        analysis_lab,
        created_dttm,
        updated_dttm

    FROM who_classification
    WHERE passes_genome_qc = TRUE
)

SELECT *
FROM final
