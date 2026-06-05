/*
  stg_order_book — Bronze → Staging

  Responsibilities:
    1. Read the Delta Bronze table via delta_scan()
    2. Deduplicate on last_update_id (Spark micro-batch retries can produce
       duplicate offsets on failure/restart, even with Delta's ACID guarantees)
    3. Cast price/qty strings to DOUBLE for arithmetic in downstream models
    4. Drop rows with null or zero best prices (updates with no top-of-book change)

  Materialized as: table (inherited from silver/ config in dbt_project.yml)
  Downstream: silver_order_book
*/

WITH bronze AS (

    -- delta_scan() is a DuckDB function provided by the delta extension. It reads
    -- the _delta_log/ transaction log at the given path to determine which Parquet
    -- files belong to the current table version, then reads only those files.
    -- We cannot use a plain FROM clause or dbt source() here because the Bronze
    -- table is not registered in DuckDB's catalog — it lives as raw files managed
    -- by Spark's Delta writer. delta_scan() bridges that gap without requiring
    -- any Spark session or catalog registration.
    SELECT *
    FROM delta_scan('/Volumes/Tejas SSD/marketstream/delta/order_book')

),

deduped AS (

    -- Deduplicate on last_update_id rather than on a composite key because
    -- last_update_id is the Binance sequence cursor and is the natural unique
    -- identifier for a processed update batch. We dedup here in staging (not
    -- in silver_order_book) so that every downstream model inherits a clean,
    -- unique-keyed foundation without each having to implement its own dedup.
    --
    -- ROW_NUMBER() ordered by event_time DESC means: if two rows share the same
    -- last_update_id (a Spark retry duplicate), keep the one with the later
    -- event_time. In practice they will be identical, so the ordering is a
    -- tiebreaker rather than a business decision.
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY last_update_id
            ORDER BY event_time DESC
        ) AS _rn

    FROM bronze

),

cast_prices AS (

    SELECT
        event_time,
        date,
        symbol,
        first_update_id,
        last_update_id,
        bid_count,
        ask_count,

        -- Prices and quantities are stored as strings in Bronze to preserve the
        -- exact 8-decimal-place representation Binance sends (e.g. "65521.37000000").
        -- String storage avoids IEEE 754 rounding at write time. The cast to DOUBLE
        -- is acceptable here in the analytical layer: the rounding error (~1e-11 USD)
        -- is far below market microstructure noise and we need arithmetic for spreads.
        --
        -- TRY_CAST returns NULL on failure instead of raising an error, so a
        -- malformed price string (e.g. from a Kafka parse failure) silently becomes
        -- NULL and is filtered in the next CTE rather than aborting the entire run.
        TRY_CAST(best_bid_price AS DOUBLE) AS best_bid_price,
        TRY_CAST(best_bid_qty   AS DOUBLE) AS best_bid_qty,
        TRY_CAST(best_ask_price AS DOUBLE) AS best_ask_price,
        TRY_CAST(best_ask_qty   AS DOUBLE) AS best_ask_qty,

        -- Keep raw arrays for the depth aggregation in silver_order_book.
        -- We do not cast or transform them here — that work happens downstream
        -- where it is closer to the models that use it.
        raw_bids,
        raw_asks

    FROM deduped
    WHERE _rn = 1  -- keep only the deduplicated winner from the window above

),

filtered AS (

    SELECT *
    FROM cast_prices

    -- Drop rows where either price is null or zero. A null price means TRY_CAST
    -- failed (bad data) or best_bid_price was null in Bronze (the depth update
    -- contained no bid-side changes, so b[0][0] was null in Spark).
    -- A zero price should never occur in a live BTC market but guards against
    -- data corruption. These rows would produce zero mid_price in silver,
    -- making spread_bps divide by zero and order_imbalance meaningless.
    WHERE best_bid_price IS NOT NULL
      AND best_ask_price IS NOT NULL
      AND best_bid_price > 0
      AND best_ask_price > 0

)

SELECT * FROM filtered
