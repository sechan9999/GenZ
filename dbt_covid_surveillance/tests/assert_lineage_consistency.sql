-- tests/assert_lineage_consistency.sql
-- Check that WHO variant classification is consistent with Pango lineage

SELECT
    accession_number,
    pangolin_lineage,
    who_variant
FROM {{ ref('covid_variants') }}
WHERE TRUE
    -- Alpha should be B.1.1.7 or Q.*
    AND (
        (who_variant = 'Alpha' AND NOT (pangolin_lineage LIKE 'B.1.1.7%' OR pangolin_lineage LIKE 'Q.%'))
        -- Delta should be B.1.617.2 or AY.*
        OR (who_variant = 'Delta' AND NOT (pangolin_lineage LIKE 'B.1.617.2%' OR pangolin_lineage LIKE 'AY.%'))
        -- Omicron should be BA.* family
        OR (who_variant = 'Omicron' AND NOT (
            pangolin_lineage LIKE 'B.1.1.529%'
            OR pangolin_lineage LIKE 'BA.%'
            OR pangolin_lineage LIKE 'BE.%'
            OR pangolin_lineage LIKE 'BF.%'
            OR pangolin_lineage LIKE 'BQ.%'
            OR pangolin_lineage LIKE 'XBB%'
            OR pangolin_lineage LIKE 'JN.%'
        ))
    )
