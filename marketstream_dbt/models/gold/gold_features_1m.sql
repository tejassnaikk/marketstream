/*
  gold_features_1m — Silver → Gold (1-minute tumbling windows)

  Aggregates every Silver order book update into non-overlapping 1-minute
  buckets aligned to wall-clock minute boundaries. One output row per
  (window_start, symbol) pair.

  Produces 16 columns used directly as ML features or as BI dashboard metrics:
    - Timing   : window_start, window_end
    - Activity : update_count
    - Price    : mid_price_open, mid_price_close, mid_price_change,
                 mid_price_avg, mid_price_stddev, vwap
    - Liquidity: spread_bps_avg, spread_bps_min
    - Flow     : imbalance_avg, bid_depth_avg, ask_depth_avg, depth_ratio

  Materialized as: table (inherited from gold/ config in dbt_project.yml)
  Upstream: silver_order_book
*/

WITH source AS (

    SELECT
        *,
        -- event_time is VARCHAR in Silver because Spark's from_unixtime() returns
        -- StringType, and DuckDB reads that from Delta as VARCHAR. time_bucket()
        -- requires a TIMESTAMP argument, so we cast here once rather than
        -- repeating the cast in every expression below.
        CAST(event_time AS TIMESTAMP) AS event_ts
    FROM {{ ref('silver_order_book') }}

),

bucketed AS (

    SELECT
        *,
        -- time_bucket() divides continuous time into fixed-width, non-overlapping
        -- intervals and returns the start of each interval.
        --
        -- Why time_bucket() and not DATE_TRUNC('minute', ...)?
        -- DATE_TRUNC only accepts calendar-aligned units: second, minute, hour,
        -- day, month, year. It cannot express "every 5 minutes" or "every 15
        -- minutes" — those are not calendar units. time_bucket() accepts any
        -- INTERVAL, so the same pattern works for 1m, 5m, 15m, or 37m buckets
        -- with no structural change to the query. Using it here (even for 1m
        -- where DATE_TRUNC would also work) keeps the two Gold models consistent
        -- and makes the interval the only difference between them.
        --
        -- Bucket alignment: DuckDB aligns time_bucket() to origin 2000-01-03
        -- for calendar intervals, which for minute-sized buckets means buckets
        -- always start on exact clock minutes (:00, :01, ...) — the expected
        -- behavior for market data.
        time_bucket(INTERVAL '1 MINUTE', event_ts) AS window_start

    FROM source

),

agg AS (

    SELECT
        window_start,
        symbol,

        -- Row count per bucket. Acts as a data density signal: low update_count
        -- in a 1-minute window means either the market was quiet or the producer
        -- was lagging. High count during normal trading hours is expected (~10-30
        -- updates/minute for BTC/USDT).
        COUNT(*) AS update_count,

        -- arg_min(value, key) returns `value` at the row where `key` is minimum.
        -- This is DuckDB's efficient aggregate equivalent of:
        --   FIRST_VALUE(mid_price) OVER (PARTITION BY window_start ORDER BY event_ts ASC)
        -- The window function approach would require a second GROUP BY pass to
        -- collapse the windowed rows; arg_min/arg_max do it in a single scan.
        -- Explicit ordering by event_ts (not row order) is critical: relying on
        -- arbitrary row order from a GROUP BY would give non-deterministic results
        -- as DuckDB may process rows in any order depending on partition layout.
        arg_min(mid_price, event_ts) AS mid_price_open,
        arg_max(mid_price, event_ts) AS mid_price_close,

        -- Simple mean — treats all updates within the window equally.
        AVG(mid_price)              AS mid_price_avg,

        -- Sample standard deviation of mid_price within the window. Acts as a
        -- within-window volatility proxy: high stddev means the price moved a
        -- lot during this minute; low stddev means a stable, range-bound minute.
        -- STDDEV_SAMP (not STDDEV_POP) because the Silver rows are a sample of
        -- all possible mid-price observations, not the complete population.
        STDDEV_SAMP(mid_price)      AS mid_price_stddev,

        -- Volume-Weighted Average Price, weighted by total resting depth rather
        -- than by trade volume. Trade volume is not available in the depth stream;
        -- resting depth (bid + ask) is a proxy for how much liquidity backed each
        -- price observation. A mid_price observed when the book was deep (high
        -- total depth) is more reliable than one observed during a thin book, so
        -- weighting by depth gives a more representative central price than a
        -- simple average. NULLIF on the denominator prevents division-by-zero when
        -- all depth values in the window are NULL (e.g. first batch before any
        -- depth data populates the arrays).
        SUM(mid_price * (bid_depth_total + ask_depth_total))
            / NULLIF(SUM(bid_depth_total + ask_depth_total), 0) AS vwap,

        AVG(spread_bps)             AS spread_bps_avg,

        -- Minimum spread_bps in the window = the tightest the bid-ask spread got
        -- during this minute. A low spread_bps_min indicates the market briefly
        -- offered very tight liquidity, even if the average was wider.
        MIN(spread_bps)             AS spread_bps_min,

        -- AVG naturally ignores NULLs in SQL, so rows where order_imbalance is
        -- NULL (updates with zero total changes) are excluded without any special
        -- handling. This is the correct behavior: a zero-change update carries no
        -- directional information and should not dilute the average.
        AVG(order_imbalance)        AS imbalance_avg,

        AVG(bid_depth_total)        AS bid_depth_avg,
        AVG(ask_depth_total)        AS ask_depth_avg

    FROM bucketed
    GROUP BY window_start, symbol

)

SELECT
    window_start,

    -- window_end makes the interval explicit so consumers don't need to know
    -- the bucket width from context. Storing both endpoints also makes joins
    -- between the 1m and 5m tables unambiguous.
    window_start + INTERVAL '1 MINUTE' AS window_end,

    symbol,
    update_count,
    mid_price_open,
    mid_price_close,

    -- Signed price change across the window: positive = price rose,
    -- negative = price fell. Simple and fast to compute from the already-
    -- aggregated open/close without re-scanning the source rows.
    mid_price_close - mid_price_open AS mid_price_change,

    mid_price_avg,
    mid_price_stddev,
    vwap,
    spread_bps_avg,
    spread_bps_min,
    imbalance_avg,
    bid_depth_avg,
    ask_depth_avg,

    -- Bid-to-ask depth ratio: values above 1.0 mean more bid liquidity than
    -- ask liquidity in this window — potential buy-side pressure or an ask-side
    -- supply shortage. Values below 1.0 signal more ask liquidity — potential
    -- selling pressure. A ratio near 1.0 indicates a balanced market. NULLIF
    -- prevents division by zero when ask_depth_avg is zero (extremely rare for
    -- BTC/USDT but possible in very short windows with only deletion updates).
    bid_depth_avg / NULLIF(ask_depth_avg, 0) AS depth_ratio

FROM agg
