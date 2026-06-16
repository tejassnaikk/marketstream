"""
scripts/refresh_gold.py

Rebuilds the Silver and Gold DuckDB tables from the Bronze layer written
by kafka_to_duckdb.py.

Pipeline:
  bronze_order_book (raw Kafka rows)
    → silver_order_book (1-minute OHLC + metrics aggregation)
    → gold_features_1m  (rolling features + direction label)

Run:
    python scripts/refresh_gold.py

Environment variables:
  DUCKDB_PATH  — default ~/marketstream/duckdb_ec2/marketstream.duckdb
"""

import logging
import os
import time
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DUCKDB_PATH = os.environ.get(
    "DUCKDB_PATH",
    str(Path.home() / "marketstream/duckdb_ec2/marketstream.duckdb")
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("refresh_gold")

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

SILVER_SQL = """
CREATE OR REPLACE TABLE silver_order_book AS
SELECT
    time_bucket(INTERVAL '1 minute', CAST(event_time AS TIMESTAMPTZ))           AS window_start,
    symbol,
    AVG((best_bid_price + best_ask_price) / 2.0)           AS mid_price_avg,
    AVG(
        (best_ask_price - best_bid_price)
        / NULLIF((best_bid_price + best_ask_price) / 2.0, 0)
        * 10000
    )                                                       AS spread_bps_avg,
    AVG(
        (best_bid_qty - best_ask_qty)
        / NULLIF(best_bid_qty + best_ask_qty, 0)
    )                                                       AS imbalance_avg,
    AVG(bid_count::DOUBLE / NULLIF(ask_count, 0))          AS depth_ratio,
    SUM((best_bid_price + best_ask_price) / 2.0
        * (best_bid_qty + best_ask_qty))
    / NULLIF(SUM(best_bid_qty + best_ask_qty), 0)          AS vwap,
    AVG(best_ask_price - best_bid_price)                   AS spread_avg,
    COUNT(*)                                               AS tick_count
FROM stg_order_book
WHERE CAST(event_time AS TIMESTAMPTZ) IS NOT NULL
  AND best_bid_price IS NOT NULL
  AND best_ask_price IS NOT NULL
GROUP BY 1, 2
ORDER BY 1 ASC
"""

GOLD_SQL = """
CREATE OR REPLACE TABLE gold_features_1m AS
SELECT
    window_start,
    symbol,
    mid_price_avg,
    spread_bps_avg,
    imbalance_avg,
    depth_ratio,
    vwap,
    spread_avg,
    tick_count,
    spread_bps_avg
        - LAG(spread_bps_avg, 1) OVER w                   AS spread_change,
    imbalance_avg
        - LAG(imbalance_avg, 1) OVER w                    AS imbalance_change,
    (vwap - mid_price_avg)
        / NULLIF(mid_price_avg, 0) * 10000                AS vwap_vs_mid,
    AVG(spread_bps_avg) OVER (
        PARTITION BY symbol ORDER BY window_start
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    )                                                      AS rolling_spread_mean,
    STDDEV(imbalance_avg) OVER (
        PARTITION BY symbol ORDER BY window_start
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    )                                                      AS rolling_imbalance_std,
    (LEAD(mid_price_avg, 5) OVER w > mid_price_avg)::INTEGER AS label
FROM silver_order_book
WINDOW w AS (PARTITION BY symbol ORDER BY window_start)
ORDER BY window_start ASC
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(f"DuckDB path: {DUCKDB_PATH}")
    t0 = time.perf_counter()

    con = duckdb.connect(DUCKDB_PATH)
    try:
        log.info("Building silver_order_book...")
        con.execute(SILVER_SQL)
        silver_rows = con.execute("SELECT COUNT(*) FROM silver_order_book").fetchone()[0]
        log.info(f"silver_order_book: {silver_rows:,} rows")

        log.info("Building gold_features_1m...")
        con.execute(GOLD_SQL)
        gold_rows = con.execute("SELECT COUNT(*) FROM gold_features_1m").fetchone()[0]
        log.info(f"gold_features_1m: {gold_rows:,} rows")

    finally:
        con.close()

    elapsed = time.perf_counter() - t0
    log.info(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
