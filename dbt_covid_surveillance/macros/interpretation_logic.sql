-- macros/interpretation_logic.sql
-- Reusable macros for COVID test result interpretation

-- Interpret PCR result based on CT values and targets
{% macro interpret_pcr_result(
    ct_n_gene,
    ct_orf1ab,
    ct_e_gene=None,
    ct_s_gene=None,
    ct_cutoff_n=40,
    ct_cutoff_orf=40,
    ct_cutoff_e=40,
    ct_cutoff_s=40,
    min_targets_positive=2,
    inconclusive_threshold=35
) %}

    CASE
        -- Count positive targets (below cutoff)
        WHEN (
            (CASE WHEN {{ ct_n_gene }} <= {{ ct_cutoff_n }} THEN 1 ELSE 0 END) +
            (CASE WHEN {{ ct_orf1ab }} <= {{ ct_cutoff_orf }} THEN 1 ELSE 0 END)
            {% if ct_e_gene %}
            + (CASE WHEN {{ ct_e_gene }} <= {{ ct_cutoff_e }} THEN 1 ELSE 0 END)
            {% endif %}
            {% if ct_s_gene %}
            + (CASE WHEN {{ ct_s_gene }} <= {{ ct_cutoff_s }} THEN 1 ELSE 0 END)
            {% endif %}
        ) >= {{ min_targets_positive }} THEN 'Detected'

        -- Single weak positive (above inconclusive threshold)
        WHEN (
            (CASE WHEN {{ ct_n_gene }} <= {{ ct_cutoff_n }} THEN 1 ELSE 0 END) +
            (CASE WHEN {{ ct_orf1ab }} <= {{ ct_cutoff_orf }} THEN 1 ELSE 0 END)
            {% if ct_e_gene %}
            + (CASE WHEN {{ ct_e_gene }} <= {{ ct_cutoff_e }} THEN 1 ELSE 0 END)
            {% endif %}
            {% if ct_s_gene %}
            + (CASE WHEN {{ ct_s_gene }} <= {{ ct_cutoff_s }} THEN 1 ELSE 0 END)
            {% endif %}
        ) = 1
        AND LEAST(
            COALESCE({{ ct_n_gene }}, 999),
            COALESCE({{ ct_orf1ab }}, 999)
            {% if ct_e_gene %}, COALESCE({{ ct_e_gene }}, 999) {% endif %}
            {% if ct_s_gene %}, COALESCE({{ ct_s_gene }}, 999) {% endif %}
        ) > {{ inconclusive_threshold }} THEN 'Inconclusive'

        -- No targets detected
        ELSE 'Not Detected'
    END

{% endmacro %}


-- Flag for S-gene target failure (SGTF) - marker for certain variants
{% macro s_gene_target_failure(ct_n_gene, ct_orf1ab, ct_s_gene, sgtf_threshold=30) %}

    CASE
        WHEN {{ ct_s_gene }} IS NULL
          OR {{ ct_s_gene }} > {{ sgtf_threshold }}
        THEN
            CASE
                WHEN {{ ct_n_gene }} < {{ sgtf_threshold }}
                 AND {{ ct_orf1ab }} < {{ sgtf_threshold }}
                THEN TRUE
                ELSE FALSE
            END
        ELSE FALSE
    END

{% endmacro %}


-- Classify test turnaround time
{% macro test_turnaround_category(collected_dttm, released_dttm) %}

    CASE
        WHEN DATEDIFF('hour', {{ collected_dttm }}, {{ released_dttm }}) <= 24
            THEN 'Same Day (<24h)'
        WHEN DATEDIFF('hour', {{ collected_dttm }}, {{ released_dttm }}) <= 48
            THEN '1-2 Days'
        WHEN DATEDIFF('hour', {{ collected_dttm }}, {{ released_dttm }}) <= 72
            THEN '2-3 Days'
        WHEN DATEDIFF('hour', {{ collected_dttm }}, {{ released_dttm }}) <= 120
            THEN '3-5 Days'
        ELSE '>5 Days'
    END

{% endmacro %}


-- Calculate test positivity rate for a given group
{% macro test_positivity_rate(result_column, detected_value='Detected') %}

    ROUND(
        100.0 * COUNTIF({{ result_column }} = '{{ detected_value }}') / NULLIF(COUNT(*), 0),
        2
    )

{% endmacro %}


-- Detect if specimen is likely a retest (same patient, recent test)
{% macro is_likely_retest(pat_id, test_date, lookback_days=14) %}

    CASE
        WHEN COUNT(*) OVER (
            PARTITION BY {{ pat_id }}
            ORDER BY {{ test_date }}
            RANGE BETWEEN INTERVAL '{{ lookback_days }}' DAY PRECEDING AND INTERVAL '1' DAY PRECEDING
        ) > 0
        THEN TRUE
        ELSE FALSE
    END

{% endmacro %}
