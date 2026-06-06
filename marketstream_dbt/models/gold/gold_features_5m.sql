/*
  gold_features_5m — Silver → Gold (5-minute tumbling windows)

  Aggregates every Silver order book update into non-overlapping 5-minute
  buckets aligned to wall-clock 5-minute boundaries (e.g. :00, :05, :10...).
  One output row per (window_start, symbol) pair.

  Produces 16 columns used directly as ML features or as BI dashboard metrics:
    - Timing   : window_start, window_end
    - Activity : update_count
    - Price    : mid_price_open, mid_price_close, mid_price_change,
                 mid_price_avg, mid_price_stddev, vwap
    - Liquidity: spread_bps_avg, spread_bps_min
    - Flow     : imbalance_avg, bid_depth_avg, ask_depth_avg, depth_ratio

  Why build from silver_order_book directly (not from gold_features_1m)?
  Aggregating five 1-minute rows into one 5-minute row would give each 1-minute
  bucket equal weight regardless of how many raw events fell inside it — a window
  with 3 events would count the same as a window with 30. Aggregating the raw
  Silver rows ensures the 5-minute statistics (mean, stddev, VWAP) reflect the
  true distribution of observations, not a double-summarised approximation.

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
        -- day, month, year. It cannot express "every 5 minutes" — that is not a
        -- calendar unit. time_bucket() accepts any INTERVAL, which is why this 5m
        -- model only differs from gold_features_1m in the INTERVAL literal.
        --
        -- Bucket alignment: DuckDB aligns time_bucket() to origin 2000-01-03
        -- for calendar intervals. For a 5-minute INTERVAL, this produces buckets
        -- starting on clock multiples of 5 minutes (:00, :05, :10, ...) — standard
        -- for market data candles and compatible with exchange OHLCV bar timestamps.
        time_bucket(INTERVAL '5 MINUTES', event_ts) AS window_start

    FROM source

),

agg AS (

    SELECT
        window_start,
        symbol,

        -- Row count per bucket. For a 5-minute window ~50-150 updates is typical
        -- for BTC/USDT during active trading hours. Very low counts (< 10) in a
        -- 5-minute window may indicate a gap in the Kafka stream or a period of
        -- extreme illiquidity. Downstream ML pipelines can use update_count as
        -- a confidence weight for the other aggregated features.
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

        -- Simple mean — treats all updates within the 5-minute window equally.
        -- Over a 5-minute span, this is a smoother estimate of the "fair price"
        -- than the 1-minute average, less sensitive to individual outlier updates.
        AVG(mid_price)              AS mid_price_avg,

        -- Sample standard deviation of mid_price within the 5-minute window.
        -- Over a longer horizon than the 1m model, this captures more of the
        -- intra-period volatility and is a better feature for regime detection
        -- (trending vs mean-reverting) than the 1-minute counterpart.
        -- STDDEV_SAMP (not STDDEV_POP) because the Silver rows are a sample of
        -- all possible mid-price observations, not the complete population.
        STDDEV_SAMP(mid_price)      AS mid_price_stddev,

        -- Volume-Weighted Average Price, weighted by total resting depth rather
        -- than by trade volume. Trade volume is not available in the depth stream;
        -- resting depth (bid + ask) is a proxy for how much liquidity backed each
        -- price observation. Over 5 minutes, this VWAP estimate is more stable
        -- than the 1-minute version because depth spikes from a single update
        -- are diluted across more rows. NULLIF on the denominator prevents
        -- division-by-zero when all depth values in the window are NULL.
        SUM(mid_price * (bid_depth_total + ask_depth_total))
            / NULLIF(SUM(bid_depth_total + ask_depth_total), 0) AS vwap,

        AVG(spread_bps)             AS spread_bps_avg,

        -- Minimum spread_bps in the window = the tightest the bid-ask spread got
        -- during any single update in this 5-minute period. Over a longer window,
        -- spread_bps_min is useful for detecting temporary liquidity injections
        -- (market makers briefly tightening spreads before pulling back).
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
    -- between the 1m and 5m tables unambiguous without inspecting column names.
    window_start + INTERVAL '5 MINUTES' AS window_end,

    symbol,
    update_count,
    mid_price_open,
    mid_price_close,

    -- Signed price change across the 5-minute window: positive = price rose,
    -- negative = price fell. Over a 5-minute horizon, mid_price_change is a
    -- simple momentum signal: consistent positive values across consecutive
    -- windows suggest an uptrend; alternating signs suggest mean reversion.
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
    -- selling pressure. Over a 5-minute window, this ratio averages out short-lived
    -- depth fluctuations and better captures sustained directional order flow.
    -- NULLIF prevents division by zero when ask_depth_avg is zero.
    bid_depth_avg / NULLIF(ask_depth_avg, 0) AS depth_ratio

FROM agg
