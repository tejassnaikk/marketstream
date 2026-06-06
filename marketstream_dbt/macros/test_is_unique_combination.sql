{% test is_unique_combination(model, combination_of) %}
    -- Generic test that fails (returns rows) when any combination of the specified
    -- columns appears more than once in the model.
    --
    -- Referenced in schema.yml at the model level (not column level) as:
    --   tests:
    --     - is_unique_combination:
    --         combination_of: ["window_start", "symbol"]
    --
    -- The combination_of argument is a Jinja list; the | join filter renders it
    -- as a comma-separated string for the GROUP BY clause.
    WITH duplicates AS (
        SELECT
            {{ combination_of | join(', ') }},
            COUNT(*) AS n
        FROM {{ model }}
        GROUP BY {{ combination_of | join(', ') }}
    )
    SELECT *
    FROM duplicates
    WHERE n > 1
{% endtest %}
