-- macros/epi_week.sql
-- Calculate CDC MMWR epidemiological week from a date
-- Returns the Saturday of the epi week (standard CDC format)
-- Epi weeks start on Sunday and end on Saturday

{% macro epi_week(date_column) %}
    {% set epi_week_start_day = var('epi_week_start_day', 0) %}  -- 0 = Sunday (CDC standard)

    {% if target.type == 'snowflake' %}
        -- Snowflake implementation
        DATEADD(
            'day',
            (6 - DAYOFWEEK({{ date_column }})) % 7,
            {{ date_column }}
        )::DATE

    {% elif target.type == 'bigquery' %}
        -- BigQuery implementation
        DATE_ADD(
            {{ date_column }},
            INTERVAL ((7 - EXTRACT(DAYOFWEEK FROM {{ date_column }})) % 7) DAY
        )

    {% elif target.type == 'postgres' %}
        -- PostgreSQL implementation
        ({{ date_column }} + ((6 - EXTRACT(DOW FROM {{ date_column }})) % 7) * INTERVAL '1 day')::DATE

    {% elif target.type == 'redshift' %}
        -- Redshift implementation
        DATEADD(
            day,
            (6 - EXTRACT(DOW FROM {{ date_column }})) % 7,
            {{ date_column }}
        )

    {% else %}
        -- Generic SQL implementation
        DATE({{ date_column }})
    {% endif %}

{% endmacro %}


-- Alternative: Calculate MMWR week number (1-53)
{% macro mmwr_week_number(date_column) %}

    {% if target.type == 'snowflake' %}
        WEEKOFYEAR({{ date_column }})

    {% elif target.type == 'bigquery' %}
        EXTRACT(ISOWEEK FROM {{ date_column }})

    {% elif target.type == 'postgres' or target.type == 'redshift' %}
        EXTRACT(WEEK FROM {{ date_column }})

    {% else %}
        WEEK({{ date_column }})
    {% endif %}

{% endmacro %}


-- Calculate MMWR year (handles weeks spanning calendar years)
{% macro mmwr_year(date_column) %}

    CASE
        WHEN EXTRACT(MONTH FROM {{ date_column }}) = 1
         AND EXTRACT(WEEK FROM {{ date_column }}) >= 52
            THEN EXTRACT(YEAR FROM {{ date_column }}) - 1
        WHEN EXTRACT(MONTH FROM {{ date_column }}) = 12
         AND EXTRACT(WEEK FROM {{ date_column }}) = 1
            THEN EXTRACT(YEAR FROM {{ date_column }}) + 1
        ELSE EXTRACT(YEAR FROM {{ date_column }})
    END

{% endmacro %}
