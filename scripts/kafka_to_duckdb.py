"""
scripts/kafka_to_duckdb.py

Lightweight Kafka consumer that reads btcusdt_depth messages and writes
raw order book data to DuckDB Bronze table.

Replaces stream_parser_s3.py for the DuckDB pipeline path.
Runs continuously — restart-safe via offset tracking in DuckDB.

Pipeline:
  Kafka btcusdt_depth topic
    → parse JSON
    → extract best bid/ask + counts
    → append to bronze_order_book in DuckDB

Run:
    python scripts/kafka_to_duckdb.py

Environment variables:
  KAFKA_BOOTSTRAP  — default localhost:9092
  DUCKDB_PATH      — default ~/marketstream/duckdb_ec2/marketstream.duckdb
  BATCH_SIZE       — rows to accumulate before writing, default 100
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from kafka import KafkaConsumer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
DUCKDB_PATH     = os.environ.get(
    "DUCKDB_PATH",
    str(Path.home() / "marketstream/duckdb_ec2/marketstream.duckdb")
)
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "100"))
TOPIC           = "btcusdt_depth"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("kafka_to_duckdb")

# ---------------------------------------------------------------------------
# DuckDB setup
# ---------------------------------------------------------------------------

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS bronze_order_book (
            event_time      TIMESTAMPTZ,
            symbol          VARCHAR,
            first_update_id BIGINT,
            last_update_id  BIGINT,
            best_bid_price  DOUBLE,
            best_bid_qty    DOUBLE,
            best_ask_price  DOUBLE,
            best_ask_qty    DOUBLE,
            bid_count       INTEGER,
            ask_count       INTEGER,
            ingested_at     TIMESTAMPTZ
        )
    """)
    log.info("bronze_order_book table ready")

# ---------------------------------------------------------------------------
# Parse Kafka message
# ---------------------------------------------------------------------------

def parse_message(msg_value: bytes) -> dict | None:
    try:
        d = json.loads(msg_value.decode("utf-8"))
        bids = d.get("b", [])
        asks = d.get("a", [])
        return {
            "event_time":      datetime.fromtimestamp(d["E"] / 1000, tz=timezone.utc),
            "symbol":          d.get("s", "BTCUSDT"),
            "first_update_id": d.get("U"),
            "last_update_id":  d.get("u"),
            "best_bid_price":  float(bids[0][0]) if bids else None,
            "best_bid_qty":    float(bids[0][1]) if bids else None,
            "best_ask_price":  float(asks[0][0]) if asks else None,
            "best_ask_qty":    float(asks[0][1]) if asks else None,
            "bid_count":       len(bids),
            "ask_count":       len(asks),
            "ingested_at":     datetime.now(timezone.utc),
        }
    except Exception as e:
        log.warning(f"Failed to parse message: {e}")
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(f"Connecting to Kafka at {KAFKA_BOOTSTRAP}")
    log.info(f"DuckDB path: {DUCKDB_PATH}")
    log.info(f"Batch size: {BATCH_SIZE}")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="kafka-to-duckdb",
        consumer_timeout_ms=-1,
    )

    con = duckdb.connect(DUCKDB_PATH)
    init_db(con)

    batch = []
    total = 0

    log.info("Consuming messages...")

    try:
        for msg in consumer:
            row = parse_message(msg.value)
            if row:
                batch.append(row)

            if len(batch) >= BATCH_SIZE:
                con.executemany("""
                    INSERT INTO bronze_order_book VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, [list(r.values()) for r in batch])
                total += len(batch)
                log.info(f"Wrote {len(batch)} rows (total: {total})")
                batch = []

    except KeyboardInterrupt:
        if batch:
            con.executemany("""
                INSERT INTO bronze_order_book VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, [list(r.values()) for r in batch])
            log.info(f"Flushed {len(batch)} remaining rows")
        log.info("Shutting down")
    finally:
        con.close()
        consumer.close()


if __name__ == "__main__":
    main()
