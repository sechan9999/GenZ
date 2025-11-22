-- models/staging/stg_sequencing_results.sql
-- Staging model for SARS-CoV-2 genomic sequencing results
-- Cleans and standardizes data from Pangolin, Nextclade, and variant calling pipelines

{{ config(
    materialized='view',
    tags=['staging', 'genomics', 'sequencing']
) }}

WITH source AS (
    SELECT * FROM {{ source('raw_sequencing', 'sequence_analysis_results') }}
),

cleaned AS (
    SELECT
        -- Identifiers
        TRIM(UPPER(accession_number)) AS accession_number,
        TRIM(sequencing_run_id) AS sequencing_run_id,
        TRIM(sample_id) AS sample_id,

        -- Sequencing metadata
        TRY_CAST(sequenced_date AS DATE) AS sequenced_date,
        TRIM(sequencing_platform) AS sequencing_platform,  -- Illumina, ONT, etc.
        TRIM(library_prep_kit) AS library_prep_kit,
        TRY_CAST(cycle_threshold AS DECIMAL(10,2)) AS cycle_threshold,

        -- Consensus genome
        consensus_genome_fasta AS consensus_genome_fasta,
        TRY_CAST(genome_coverage AS DECIMAL(5,2)) AS genome_coverage,
        TRY_CAST(mean_depth AS DECIMAL(10,2)) AS mean_depth,
        TRY_CAST(n_content_percent AS DECIMAL(5,2)) AS n_content_percent,

        -- Pangolin lineage assignment
        TRIM(pangolin_lineage) AS pangolin_lineage,
        TRIM(pangolin_conflict) AS pangolin_conflict,
        TRY_CAST(pangolin_probability AS DECIMAL(5,4)) AS pangolin_probability,
        TRIM(pangolin_version) AS pangolin_version,
        TRIM(pango_version) AS pango_version,
        TRY_CAST(pangolin_analysis_date AS DATE) AS pangolin_analysis_date,

        -- Nextclade classification
        TRIM(nextclade_clade) AS nextclade_clade,
        TRIM(nextclade_lineage) AS nextclade_lineage,
        TRY_CAST(nextclade_qc_score AS DECIMAL(10,2)) AS nextclade_qc_score,
        TRIM(nextclade_qc_status) AS nextclade_qc_status,  -- good, mediocre, bad
        TRY_CAST(nextclade_total_substitutions AS INTEGER) AS nextclade_total_substitutions,
        TRY_CAST(nextclade_total_deletions AS INTEGER) AS nextclade_total_deletions,
        TRY_CAST(nextclade_total_insertions AS INTEGER) AS nextclade_total_insertions,
        TRY_CAST(nextclade_total_missing AS INTEGER) AS nextclade_total_missing,

        -- WHO variant classification
        TRIM(who_variant) AS who_variant,  -- Alpha, Beta, Gamma, Delta, Omicron
        TRIM(who_variant_status) AS who_variant_status,  -- VOC, VOI, VUM

        -- Mutation list (JSON array from Freyja, V-pipe, or custom caller)
        TRY_PARSE_JSON(mutation_list) AS mutation_list,

        -- Key mutations of concern (pre-parsed flags)
        CASE WHEN LOWER(has_e484k) = 'true' THEN TRUE ELSE FALSE END AS has_e484k,
        CASE WHEN LOWER(has_n501y) = 'true' THEN TRUE ELSE FALSE END AS has_n501y,
        CASE WHEN LOWER(has_l452r) = 'true' THEN TRUE ELSE FALSE END AS has_l452r,
        CASE WHEN LOWER(has_del69_70) = 'true' THEN TRUE ELSE FALSE END AS has_del69_70,

        -- Quality control
        CASE
            WHEN genome_coverage >= 95 AND n_content_percent <= 5 THEN TRUE
            ELSE FALSE
        END AS passes_genome_qc,

        -- Metadata
        TRY_CAST(created_dttm AS TIMESTAMP) AS created_dttm,
        TRY_CAST(updated_dttm AS TIMESTAMP) AS updated_dttm,
        TRIM(pipeline_version) AS pipeline_version,
        TRIM(analysis_lab) AS analysis_lab

    FROM source
),

-- Filter to high-quality sequences only
validated AS (
    SELECT *
    FROM cleaned
    WHERE TRUE
        AND accession_number IS NOT NULL
        AND sequenced_date IS NOT NULL
        -- Must have lineage assignment from at least one tool
        AND (pangolin_lineage IS NOT NULL OR nextclade_clade IS NOT NULL)
        -- Quality thresholds
        AND genome_coverage >= {{ var('genome_coverage_threshold', 80) }}
        AND n_content_percent <= {{ var('max_n_content_percent', 10) }}
        AND passes_genome_qc = TRUE
)

SELECT *
FROM validated
