/*
  silver_order_book — Staging → Silver

  Adds six derived market microstructure columns on top of stg_order_book:
    mid_price        — reference price between best bid and ask
    spread           — raw USD cost of crossing the spread
    spread_bps       — spread normalised by mid_price (dimensionless, comparable)
    order_imbalance  — directional pressure indicator (-1 to +1)
    bid_depth_total  — total size of all non-deletion bid changes in this update
    ask_depth_total  — total size of all non-deletion ask changes in this update

  Materialized as: table (inherited from silver/ config in dbt_project.yml)
  Upstream: stg_order_book
*/

WITH base AS (

    -- ref() generates the correct qualified table name for the current target
    -- (e.g. main.stg_order_book in DuckDB). It also records the dependency edge
    -- in dbt's DAG so that dbt always runs stg_order_book before this model.
    SELECT * FROM {{ ref('stg_order_book') }}

),

with_fundamentals AS (

    -- Compute mid_price and spread in a separate CTE so they can be referenced
    -- by name in the outer SELECT. SQL does not allow referencing a column alias
    -- defined in the same SELECT list (e.g. using "spread" to compute "spread_bps"
    -- in the same SELECT clause), so we stage them one level up.
    SELECT
        *,

        -- Mid-price is the reference price used by market makers and the basis for
        -- all percentage-based metrics. Using /2.0 (float literal) ensures DuckDB
        -- returns DOUBLE rather than truncated integer arithmetic.
        (best_bid_price + best_ask_price) / 2.0 AS mid_price,

        -- Raw spread in USD. Always non-negative in a valid order book
        -- (ask >= bid by definition). The not_null and is_non_negative schema
        -- tests will catch violations.
        best_ask_price - best_bid_price AS spread

    FROM base

)

SELECT
    -- Pass through all staging columns unchanged so the Silver table is a superset
    -- of Staging — downstream consumers never need to join back to Staging.
    event_time,
    date,
    symbol,
    first_update_id,
    last_update_id,
    bid_count,
    ask_count,
    best_bid_price,
    best_bid_qty,
    best_ask_price,
    best_ask_qty,
    raw_bids,
    raw_asks,

    -- Fundamental derived columns
    mid_price,
    spread,

    -- Spread in basis points (1 bps = 0.01%). Raw USD spread is not comparable
    -- across time: a $10 spread is tight when BTC trades at $100k but wide at $20k.
    -- Dividing by mid_price normalises for the price level, making spread_bps
    -- a stable indicator of market liquidity that is comparable across all BTC
    -- price regimes and against other assets (ETH spread_bps vs BTC spread_bps).
    -- NULLIF(mid_price, 0) guards against division by zero — stg_order_book
    -- already filters mid_price = 0, but defensive SQL costs nothing.
    (spread / NULLIF(mid_price, 0)) * 10000 AS spread_bps,

    -- Order imbalance: (bids_changed - asks_changed) / total_changes.
    -- Ranges from -1.0 (only ask-side changes) to +1.0 (only bid-side changes).
    -- A persistently positive imbalance signals more aggressive buying pressure;
    -- negative signals selling pressure. Useful as a feature for short-horizon
    -- price prediction models.
    --
    -- NULLIF(bid_count + ask_count, 0) prevents division by zero for the rare
    -- (but valid) case where a depth update carries no bid or ask changes at all.
    -- Without NULLIF, SQL raises a runtime error and the entire dbt run fails.
    -- With it, those rows produce NULL imbalance, which is the correct answer:
    -- an update with no changes carries no directional information.
    CAST(bid_count - ask_count AS DOUBLE)
        / NULLIF(bid_count + ask_count, 0) AS order_imbalance,

    -- Total resting bid size across all levels in this update, excluding
    -- deletion entries (qty = 0.00000000).
    --
    -- Why not use bid_count? bid_count is the count of all bid changes including
    -- deletions. A message with 50 bid changes where 48 are deletions and 2 are
    -- new quotes has bid_count=50 but bid_depth_total reflects only the 2 genuine
    -- additions. bid_count overstates liquidity; bid_depth_total measures it.
    --
    -- list_transform/list_filter/list_aggregate are DuckDB's native higher-order
    -- list functions — they execute inside the SQL engine with no Python or UDFs:
    --   list_transform(arr, x -> expr)    — map: apply expr to every element
    --   list_filter(arr, x -> cond)       — filter: keep elements where cond is TRUE
    --   list_aggregate(arr, 'sum')        — reduce: aggregate the resulting list
    --
    -- raw_bids is VARCHAR[][] (list of [price_string, qty_string] pairs).
    -- DuckDB uses 1-based list indexing: x[1] = price, x[2] = qty.
    -- TRY_CAST returns NULL on parse failure; the filter then excludes NULLs
    -- because NULL > 0 evaluates to NULL (not TRUE), which list_filter drops.
    list_aggregate(
        list_filter(
            list_transform(raw_bids, x -> TRY_CAST(x[2] AS DOUBLE)),
            x -> x > 0
        ),
        'sum'
    ) AS bid_depth_total,

    list_aggregate(
        list_filter(
            list_transform(raw_asks, x -> TRY_CAST(x[2] AS DOUBLE)),
            x -> x > 0
        ),
        'sum'
    ) AS ask_depth_total

FROM with_fundamentals
