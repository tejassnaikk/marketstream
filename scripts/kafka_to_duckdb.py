"""
scripts/kafka_to_duckdb.py

Lightweight Kafka consumer — opens DuckDB only during batch writes,
releasing the lock between batches so refresh_gold.py can run concurrently.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
from kafka import KafkaConsumer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:29092")
DUCKDB_PATH     = os.environ.get(
    "DUCKDB_PATH",
    str(Path.home() / "marketstream/duckdb_ec2/marketstream.duckdb")
)
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", "100"))
TOPIC           = "btcusdt_depth"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("kafka_to_duckdb")


def write_batch(batch):
    con = duckdb.connect(DUCKDB_PATH)
    try:
        con.executemany(
            "INSERT INTO stg_order_book VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
    finally:
        con.close()


def parse_message(msg_value: bytes):
    try:
        d = json.loads(msg_value.decode("utf-8"))
        bids = d.get("b", [])
        asks = d.get("a", [])
        event_dt = datetime.fromtimestamp(d["E"] / 1000, tz=timezone.utc)
        return (
            event_dt.strftime("%Y-%m-%d %H:%M:%S"),
            event_dt.date(),
            d.get("s", "BTCUSDT"),
            d.get("U"),
            d.get("u"),
            len(bids),
            len(asks),
            float(bids[0][0]) if bids else None,
            float(bids[0][1]) if bids else None,
            float(asks[0][0]) if asks else None,
            float(asks[0][1]) if asks else None,
            None,
            None,
        )
    except Exception as e:
        log.warning(f"Failed to parse: {e}")
        return None


def main():
    log.info(f"Kafka: {KAFKA_BOOTSTRAP} | DuckDB: {DUCKDB_PATH} | batch: {BATCH_SIZE}")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="kafka-to-duckdb",
        consumer_timeout_ms=-1,
    )

    batch = []
    total = 0

    log.info("Consuming messages...")
    try:
        for msg in consumer:
            row = parse_message(msg.value)
            if row:
                batch.append(row)
            if len(batch) >= BATCH_SIZE:
                write_batch(batch)
                total += len(batch)
                log.info(f"Wrote {len(batch)} rows (total: {total})")
                batch = []
    except KeyboardInterrupt:
        if batch:
            write_batch(batch)
            log.info(f"Flushed {len(batch)} remaining rows")
        log.info("Shutting down")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
