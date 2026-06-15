"""
scripts/watchdog.py

Dead man's switch watchdog for MarketStream.
Runs in a loop every 60 seconds and checks:
  1. Kafka — is the producer publishing new messages to btcusdt_depth?
  2. API   — is /health returning status ok?

If either check fails CONSECUTIVE_FAILURES times in a row, writes an alert
to /app/logs/watchdog_alerts.log.

Environment variables:
  KAFKA_BOOTSTRAP   — default kafka:9092
  API_URL           — default http://api:8000
  CHECK_INTERVAL    — seconds between checks, default 60
  FAILURE_THRESHOLD — consecutive failures before alert, default 3
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import requests
from kafka import KafkaConsumer
from kafka.structs import TopicPartition

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP   = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
API_URL           = os.environ.get("API_URL", "http://api:8000")
CHECK_INTERVAL    = int(os.environ.get("CHECK_INTERVAL", "60"))
FAILURE_THRESHOLD = int(os.environ.get("FAILURE_THRESHOLD", "3"))
SNS_TOPIC_ARN     = os.environ.get("SNS_TOPIC_ARN", "")
AWS_REGION        = os.environ.get("AWS_REGION", "us-east-1")
TOPIC             = "btcusdt_depth"
LOG_PATH          = Path("/app/logs/watchdog_alerts.log")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH),
    ],
)
log = logging.getLogger("watchdog")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
kafka_failures = 0
api_failures   = 0
last_offset    = None


def check_kafka() -> bool:
    """Return True if new messages have arrived since last check."""
    global last_offset
    try:
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            consumer_timeout_ms=5000,
        )
        tp = TopicPartition(TOPIC, 0)
        consumer.assign([tp])
        end_offsets = consumer.end_offsets([tp])
        current_offset = end_offsets[tp]
        consumer.close()

        if last_offset is None:
            last_offset = current_offset
            log.info(f"Kafka initial offset: {current_offset}")
            return True

        new_messages = current_offset - last_offset
        log.info(f"Kafka offset: {current_offset} (+{new_messages} since last check)")
        last_offset = current_offset
        return new_messages > 0

    except Exception as e:
        log.error(f"Kafka check failed: {e}")
        return False


def check_api() -> bool:
    """Return True if /health returns status ok."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        ok = r.status_code == 200 and data.get("status") == "ok"
        log.info(f"API health: {data.get('status')} (HTTP {r.status_code})")
        return ok
    except Exception as e:
        log.error(f"API check failed: {e}")
        return False


def write_alert(component: str, consecutive: int) -> None:
    alert = {
        "ts":                   datetime.now(timezone.utc).isoformat(),
        "component":            component,
        "consecutive_failures": consecutive,
        "message":              f"ALERT: {component} has failed {consecutive} consecutive checks",
    }
    log.warning(json.dumps(alert))

    if SNS_TOPIC_ARN:
        try:
            sns = boto3.client("sns", region_name=AWS_REGION)
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=f"MarketStream Alert: {component} failure",
                Message=(
                    f"Component: {component}\n"
                    f"Consecutive failures: {consecutive}\n"
                    f"Time: {alert['ts']}\n\n"
                    f"Check EC2 instance i-09b47be8d76bca124 immediately."
                ),
            )
            log.info(f"SNS alert sent for {component}")
        except Exception as e:
            log.error(f"SNS publish failed: {e}")
    else:
        log.warning("SNS_TOPIC_ARN not set — skipping SNS alert")


def main():
    global kafka_failures, api_failures
    log.info("Watchdog started")
    log.info(f"Kafka: {KAFKA_BOOTSTRAP} | API: {API_URL} | interval: {CHECK_INTERVAL}s | threshold: {FAILURE_THRESHOLD}")

    while True:
        log.info("--- watchdog check ---")

        # Kafka check
        if check_kafka():
            kafka_failures = 0
        else:
            kafka_failures += 1
            log.warning(f"Kafka failure {kafka_failures}/{FAILURE_THRESHOLD}")
            if kafka_failures >= FAILURE_THRESHOLD:
                write_alert("kafka", kafka_failures)

        # API check
        if check_api():
            api_failures = 0
        else:
            api_failures += 1
            log.warning(f"API failure {api_failures}/{FAILURE_THRESHOLD}")
            if api_failures >= FAILURE_THRESHOLD:
                write_alert("api", api_failures)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
