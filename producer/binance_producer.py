import json
import os
import signal
import sys
import time

from kafka import KafkaProducer
from websocket import WebSocketApp


BINANCE_DEPTH_STREAM_URL = os.getenv(
    "BINANCE_DEPTH_STREAM_URL",
    "wss://stream.binance.com:9443/ws/btcusdt@depth",
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "btcusdt_depth")

running = True


def build_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: key.encode("utf-8"),
    )


def handle_shutdown(_signum, _frame) -> None:
    global running
    running = False
    print("shutting down")


def main() -> int:
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    producer = build_producer()

    def on_open(_ws: WebSocketApp) -> None:
        print("connected to Binance BTC-USDT depth stream")

    def on_message(_ws: WebSocketApp, message: str) -> None:
        payload = json.loads(message)
        producer.send(KAFKA_TOPIC, key="btcusdt", value=payload)
        print("depth update")

    def on_error(_ws: WebSocketApp, error: Exception) -> None:
        print(f"websocket error: {error}", file=sys.stderr)

    def on_close(_ws: WebSocketApp, status_code: int, message: str) -> None:
        print(f"websocket closed: {status_code} {message}")

    while running:
        websocket = WebSocketApp(
            BINANCE_DEPTH_STREAM_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        websocket.run_forever(ping_interval=20, ping_timeout=10)

        if running:
            print("reconnecting in 5 seconds")
            time.sleep(5)

    producer.flush()
    producer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
