"""
spark/read_kafka.py

Reads the btcusdt_depth Kafka topic as a Spark Structured Streaming source
and prints decoded rows to the console every 5 seconds.

This is Phase 1 of the Spark layer: pure ingestion with no transformation
or sink. It exists to verify that Spark can reach Kafka, that the Kafka
connector JAR resolves correctly, and that the raw payload bytes are intact
before we build the parsing and Delta Lake sink on top.

Run:
    python spark/read_kafka.py
Stop:
    Ctrl+C
"""

import os
import sys

# ---------------------------------------------------------------------------
# JAVA_HOME
#
# Homebrew's openjdk@11 keg is not symlinked into the system PATH by default
# on Apple Silicon Macs, so the shell `java` command is absent until the user
# adds it manually. PySpark locates the JVM via the JAVA_HOME environment
# variable before importing anything from pyspark, so we must set it here
# in the process environment before any pyspark import occurs.
# If JAVA_HOME is already set (e.g. in a CI environment), we leave it alone.
# ---------------------------------------------------------------------------
if "JAVA_HOME" not in os.environ:
    _brew_java = (
        "/opt/homebrew/Cellar/openjdk@11/11.0.31/libexec/openjdk.jdk/Contents/Home"
    )
    if os.path.isdir(_brew_java):
        os.environ["JAVA_HOME"] = _brew_java
    else:
        sys.exit(
            "JAVA_HOME is not set and the expected Homebrew Java 11 path does not exist.\n"
            "Run: brew install openjdk@11\n"
            "Then set JAVA_HOME or let this script find it automatically."
        )

from pyspark.sql import SparkSession  # noqa: E402  (import after env setup)
from pyspark.sql.functions import col  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME          = "MarketStream-KafkaReader"
KAFKA_BOOTSTRAP   = "localhost:9092"
KAFKA_TOPIC       = "btcusdt_depth"
TRIGGER_INTERVAL  = "5 seconds"

# Maven coordinates for the Kafka DataSource V2 connector.
# Version must match the Spark version exactly (3.5.1) and the Scala binary
# version must match the Spark build (2.12 for Spark 3.x on most distributions).
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName(APP_NAME)

    # local[*] runs the Spark driver and all executors as threads inside this
    # single Python process using every available CPU core. There is no cluster,
    # no YARN, no Kubernetes. This is correct for development on a laptop:
    # the Kafka connector, schema parsing, and Delta Lake all behave identically
    # to a real cluster — only the parallelism and fault-tolerance model differ.
    # Replace with a cluster URL (e.g. "spark://host:7077") when deploying.
    .master("local[*]")

    # Tells Spark's internal dependency resolver to download the Kafka connector
    # JAR and its transitive dependencies (Kafka client, Jackson, etc.) from
    # Maven Central on the first run, then cache them in ~/.ivy2/. Without this,
    # .format("kafka") raises an AnalysisException because Spark has no built-in
    # Kafka source — it is deliberately kept separate (see read_kafka.py docstring
    # for why the JAR is not bundled).
    .config("spark.jars.packages", KAFKA_PACKAGE)

    # Suppresses the Kafka connector's own verbose logging. The connector logs
    # every consumer poll, offset commit, and partition assignment at INFO level,
    # which produces hundreds of lines per micro-batch. WARN keeps the console
    # readable while still surfacing real problems.
    .config("spark.sql.streaming.kafka.consumer.poll.ms", "512")

    # Prevents Spark from inferring a streaming DataFrame's schema by sampling
    # records. For Kafka sources the value column is always binary — there is
    # nothing to infer — so this setting has no effect here, but setting it
    # explicitly avoids a schema inference scan when we later add JSON parsing.
    .config("spark.sql.streaming.schemaInference", "false")

    .getOrCreate()
)

# Suppress INFO log noise from the Spark and Kafka internals.
# Must be called after getOrCreate() because the logger is initialised as part
# of the SparkContext startup.
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Streaming source: Kafka
# ---------------------------------------------------------------------------

# readStream returns an unbounded streaming DataFrame. No data is read yet —
# Spark records the source configuration and builds the query plan lazily.
# The actual Kafka consumer is not created until writeStream.start() is called.
raw_stream = (
    spark.readStream
    .format("kafka")

    # The Kafka connector spawns one consumer per Spark task per partition.
    # "kafka." prefix is passed through directly to the Kafka ConsumerConfig,
    # so any standard Kafka consumer property can be set this way.
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)

    # subscribe to a single topic. Use "subscribePattern" for a regex if you
    # later multiplex multiple symbols (e.g. "btc.*_depth").
    .option("subscribe", KAFKA_TOPIC)

    # "latest" means this run starts from messages that arrive after the query
    # starts — it does not replay historical messages. Use "earliest" only when
    # you want to reprocess all retained messages from the beginning of the log.
    .option("startingOffsets", "latest")

    .load()
)

# ---------------------------------------------------------------------------
# Schema of raw_stream (from the Kafka connector, before any parsing):
#
#   key:            binary   — message key bytes (null for our producer)
#   value:          binary   — the raw JSON payload; all Binance data is here
#   topic:          string
#   partition:      integer
#   offset:         long     — Kafka offset, not Binance update ID
#   timestamp:      timestamp — Kafka broker timestamp, not Binance event time
#   timestampType:  integer
# ---------------------------------------------------------------------------

# Cast `value` from binary to string (UTF-8). This is the only transformation
# in this script — no JSON parsing, no schema enforcement. The goal is to
# confirm the bytes survive the Kafka → Spark journey intact before adding
# the parsing layer on top.
decoded = raw_stream.select(
    col("value").cast("string").alias("raw_json"),
    col("topic"),
    col("partition"),
    col("offset"),
    col("timestamp").alias("kafka_timestamp"),
)

# ---------------------------------------------------------------------------
# Streaming sink: console
# ---------------------------------------------------------------------------

query = (
    decoded.writeStream
    .format("console")

    # processingTime trigger: Spark waits 5 seconds wall-clock between batches.
    # If the previous batch took longer than 5 seconds, the next fires immediately
    # with no gap — Spark never queues multiple triggers concurrently.
    .trigger(processingTime=TRIGGER_INTERVAL)

    # append: each micro-batch outputs only the new rows that arrived since the
    # last batch. This is the only valid outputMode for a source with no
    # aggregation. "complete" and "update" require aggregation operators.
    .outputMode("append")

    # truncate=False prevents Spark from cutting off long strings in the console
    # output. Binance depth messages can be several hundred characters wide;
    # the default truncation at 20 characters would hide almost all content.
    .option("truncate", "false")

    .start()
)

print(f"Streaming from Kafka topic '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP}")
print(f"Trigger: every {TRIGGER_INTERVAL}. Press Ctrl+C to stop.\n")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nShutting down…")
finally:
    query.stop()
    spark.stop()
    print("Done.")
