{% test is_non_negative(model, column_name) %}
    -- Generic test that fails (returns rows) if any value in column_name is
    -- strictly negative. A test passes when it returns 0 rows.
    --
    -- Referenced in schema.yml as:
    --   tests:
    --     - is_non_negative
    --
    -- Named with the test_ prefix so dbt's macro resolver recognises it as a
    -- generic test and not a utility macro. The column_name argument is injected
    -- by dbt from the schema.yml column definition.
    SELECT {{ column_name }}
    FROM {{ model }}
    WHERE {{ column_name }} < 0
{% endtest %}
