-- macros/data_quality_checks.sql
-- Reusable data quality check macros for COVID surveillance

-- Check for duplicate accession numbers
{% test no_duplicate_accessions(model, column_name) %}

    WITH duplicates AS (
        SELECT
            {{ column_name }},
            COUNT(*) AS duplicate_count
        FROM {{ model }}
        GROUP BY {{ column_name }}
        HAVING COUNT(*) > 1
    )

    SELECT *
    FROM duplicates
    WHERE duplicate_count > 1

{% endtest %}


-- Check that CT values are within valid range
{% test valid_ct_value_range(model, column_name, min_value=0, max_value=50) %}

    SELECT *
    FROM {{ model }}
    WHERE {{ column_name }} IS NOT NULL
      AND ({{ column_name }} < {{ min_value }} OR {{ column_name }} > {{ max_value }})

{% endtest %}


-- Check that dates are in chronological order
{% test date_sequence(model, earlier_date, later_date) %}

    SELECT *
    FROM {{ model }}
    WHERE {{ earlier_date }} IS NOT NULL
      AND {{ later_date }} IS NOT NULL
      AND {{ earlier_date }} > {{ later_date }}

{% endtest %}


-- Check for unusual surge in test volumes (>3 std devs from mean)
{% test anomalous_daily_volume(model, date_column) %}

    WITH daily_counts AS (
        SELECT
            {{ date_column }}::DATE AS test_date,
            COUNT(*) AS daily_count
        FROM {{ model }}
        GROUP BY {{ date_column }}::DATE
    ),

    stats AS (
        SELECT
            AVG(daily_count) AS mean_count,
            STDDEV(daily_count) AS stddev_count
        FROM daily_counts
    )

    SELECT
        dc.test_date,
        dc.daily_count,
        s.mean_count,
        s.stddev_count,
        (dc.daily_count - s.mean_count) / NULLIF(s.stddev_count, 0) AS z_score
    FROM daily_counts dc
    CROSS JOIN stats s
    WHERE ABS((dc.daily_count - s.mean_count) / NULLIF(s.stddev_count, 0)) > 3

{% endtest %}


-- Check for suspiciously high positivity rate (>50% in a day)
{% test suspicious_positivity_rate(model, date_column, result_column, threshold=50) %}

    WITH daily_positivity AS (
        SELECT
            {{ date_column }}::DATE AS test_date,
            COUNT(*) AS total_tests,
            COUNTIF({{ result_column }} = 'Detected') AS positive_tests,
            ROUND(100.0 * COUNTIF({{ result_column }} = 'Detected') / NULLIF(COUNT(*), 0), 2) AS positivity_rate
        FROM {{ model }}
        GROUP BY {{ date_column }}::DATE
    )

    SELECT *
    FROM daily_positivity
    WHERE positivity_rate > {{ threshold }}
      AND total_tests >= 10  -- Only flag if sufficient volume

{% endtest %}


-- Check for missing critical demographics
{% test demographic_completeness(model, demographic_columns) %}

    {% set columns = demographic_columns.split(',') %}

    WITH missing_counts AS (
        SELECT
            {% for col in columns %}
            COUNTIF({{ col.strip() }} IS NULL) AS {{ col.strip() }}_missing,
            {% endfor %}
            COUNT(*) AS total_records
        FROM {{ model }}
    )

    SELECT
        {% for col in columns %}
        '{{ col.strip() }}' AS column_name,
        {{ col.strip() }}_missing AS missing_count,
        ROUND(100.0 * {{ col.strip() }}_missing / NULLIF(total_records, 0), 2) AS missing_percent
        {% if not loop.last %}
        UNION ALL
        {% endif %}
        {% endfor %}
    FROM missing_counts

{% endtest %}


-- Check for genome coverage below threshold for sequences
{% test genome_quality_threshold(model, coverage_column, n_content_column, min_coverage=90, max_n_content=5) %}

    SELECT *
    FROM {{ model }}
    WHERE {{ coverage_column }} < {{ min_coverage }}
       OR {{ n_content_column }} > {{ max_n_content }}

{% endtest %}
