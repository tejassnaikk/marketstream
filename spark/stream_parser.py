"""
spark/stream_parser.py

Extends raw Kafka ingestion (read_kafka.py) by parsing the binary Binance
depth update payload into typed columns. This is stage two of the Spark layer:

  Kafka binary value
    → UTF-8 string
    → from_json struct (schema-validated)
    → flattened typed columns
    → console sink

The schema declared here is the contract between the Binance stream and
everything downstream. Delta Lake, aggregations, and the order book
reconstructor all depend on the types declared in DEPTH_SCHEMA being correct.

Run:
    /opt/anaconda3/bin/python spark/stream_parser.py
Stop:
    Ctrl+C
"""

import os
import sys

# ---------------------------------------------------------------------------
# JAVA_HOME must be set before any pyspark import.
#
# PySpark locates the JVM shared library via JAVA_HOME at import time, not at
# SparkSession creation time. Homebrew's openjdk@11 keg is intentionally not
# symlinked onto the default PATH on Apple Silicon, so the variable is absent
# from a fresh shell. We inject it here in the process environment so the
# script is self-contained — the user does not need to modify their shell
# profile or run `export JAVA_HOME=...` before each invocation.
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
            "Or: export JAVA_HOME=<path> before running this script."
        )

# All pyspark imports must come after JAVA_HOME is written to os.environ.
from pyspark.sql import SparkSession                                    # noqa: E402
from pyspark.sql.functions import col, from_json, from_unixtime, size  # noqa: E402
from pyspark.sql.types import (                                         # noqa: E402
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME         = "MarketStream-StreamParser"
KAFKA_BOOTSTRAP  = "localhost:9092"
KAFKA_TOPIC      = "btcusdt_depth"
TRIGGER_INTERVAL = "5 seconds"

# Kafka DataSource V2 connector coordinates.
# _2.12 is Spark's Scala binary version — must match the Spark build exactly.
# The trailing 3.5.1 is the connector version — must match the Spark version
# exactly. Either mismatch produces a ClassNotFoundException at runtime, not
# a build error, so there is no compile-time safety net.
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# ---------------------------------------------------------------------------
# Binance depth update schema
#
# Explicit schema is required for three reasons:
#   1. Spark cannot sample an unbounded streaming source to infer types —
#      no rows exist at query-plan construction time.
#   2. from_json is a schema-guided single-pass tokenizer. It reads only the
#      fields listed here and discards the rest; without the schema it has no
#      instruction for what to materialise.
#   3. Catalyst validates every downstream column reference against this schema
#      at compile time, catching field-name typos before any data flows.
#
# Why each field has its type:
#
#   e  StringType — event type tag, always "depthUpdate" for this topic.
#                   Kept as string because we filter on it, not compute with it.
#
#   E  LongType   — event time in Unix milliseconds. LongType avoids overflow:
#                   the max safe Int32 epoch second is year 2038; a millisecond
#                   counter overflows Int32 in 1970+24 days. BTC will outlast both.
#
#   s  StringType — symbol e.g. "BTCUSDT". String because it is a label, not
#                   a quantity.
#
#   U  LongType   — first update ID in this batch. Used by the order book
#                   reconstructor to validate sequence continuity.
#
#   u  LongType   — last update ID in this batch. The sequence cursor: the next
#                   message's U must equal this value + 1.
#
#   b  ArrayType(ArrayType(StringType())) — bid changes. Each inner array is
#      [price_string, qty_string]. StringType is deliberate: casting to
#      DoubleType here would introduce IEEE 754 rounding error into values
#      Binance sends as exact 8-decimal-place strings (e.g. "65521.37000000").
#      The loss is small per value but accumulates across aggregations and
#      breaks exact equality comparisons used in order book level lookups.
#
#   a  ArrayType(ArrayType(StringType())) — ask changes; same structure as bids.
# ---------------------------------------------------------------------------
DEPTH_SCHEMA = StructType([
    StructField("e", StringType()),
    StructField("E", LongType()),
    StructField("s", StringType()),
    StructField("U", LongType()),
    StructField("u", LongType()),
    StructField("b", ArrayType(ArrayType(StringType()))),
    StructField("a", ArrayType(ArrayType(StringType()))),
])

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

spark = (
    SparkSession.builder
    .appName(APP_NAME)

    # local[*] runs the driver and all executor threads inside this single
    # Python process, using every available CPU core. No cluster is needed.
    # The Kafka connector, JSON parsing, and Delta Lake (when added) behave
    # identically to a real cluster — only fault-tolerance and parallelism
    # limits differ. Replace with a cluster URL when deploying.
    .master("local[*]")

    # Triggers Maven/Ivy resolution on first run to download the Kafka connector
    # JAR and its transitive dependencies (Kafka client, Jackson, Snappy) into
    # ~/.ivy2/. Subsequent runs use the local cache. Without this config,
    # .format("kafka") raises AnalysisException: "Failed to find data source: kafka".
    .config("spark.jars.packages", KAFKA_PACKAGE)

    # Disables schema inference for streaming sources. Has no effect on Kafka
    # (the value column is always BinaryType — nothing to infer), but prevents
    # an accidental inference scan if a file-based source is added alongside
    # Kafka later in this session.
    .config("spark.sql.streaming.schemaInference", "false")

    # Required because DEPTH_SCHEMA contains both "e" (event type, StringType)
    # and "E" (event time ms, LongType). Spark's default case-insensitive mode
    # treats these as the same field name, so col("d.E") matches twice and
    # Catalyst raises AMBIGUOUS_REFERENCE_TO_FIELDS. The same collision affects
    # "U" (first update ID) vs "u" (last update ID). Enabling case sensitivity
    # makes the resolver treat "e"/"E" and "U"/"u" as four distinct fields,
    # exactly matching the Binance wire format where the case is meaningful.
    .config("spark.sql.caseSensitive", "true")

    .getOrCreate()
)

# Must be called after getOrCreate(): the log4j logger hierarchy is initialised
# as a side effect of SparkContext construction, so any level set before that
# point is overwritten during startup.
spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Streaming source
# ---------------------------------------------------------------------------

# readStream is entirely lazy. No Kafka consumer is created here; Spark records
# the source parameters into the logical plan. The physical consumer is
# instantiated per-task when writeStream.start() submits the first micro-batch.
raw_stream = (
    spark.readStream
    .format("kafka")

    # "kafka." prefix is forwarded verbatim to the Kafka ConsumerConfig, so any
    # standard Kafka consumer property (e.g. fetch.max.bytes, session.timeout.ms)
    # can be injected the same way.
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)

    # subscribe to one topic by exact name. Use "subscribePattern" with a regex
    # to multiplex multiple symbols (e.g. ".*usdt_depth") into one stream later.
    .option("subscribe", KAFKA_TOPIC)

    # "latest" starts consumption from the moment the query launches, ignoring
    # all messages already in the Kafka log. Use "earliest" only when you want
    # to replay the full retention window — on a live BTC stream that can mean
    # millions of historical rows flooding the first micro-batch.
    .option("startingOffsets", "latest")

    .load()
    # Schema delivered by the Kafka connector at this point:
    #   key            binary    (null — our producer sets no key)
    #   value          binary    ← the entire Binance JSON payload as raw bytes
    #   topic          string
    #   partition      int
    #   offset         long      (Kafka log offset, NOT the Binance update ID)
    #   timestamp      timestamp (broker ingestion time, NOT Binance event time E)
    #   timestampType  int
)

# ---------------------------------------------------------------------------
# Parsing: binary → string → struct → flat columns
# ---------------------------------------------------------------------------

# Step 1 — decode bytes to string.
#
# The Kafka connector delivers `value` as BinaryType (raw bytes). from_json
# requires StringType input; if you pass BinaryType directly, from_json treats
# every row as null without raising an error — a silent data loss that is hard
# to debug. The cast is a UTF-8 decode only; no JSON structure is touched here.
value_str = col("value").cast("string")

# Step 2 — parse the JSON string into a typed struct.
#
# from_json is a Catalyst expression, not a Python call that executes now.
# Spark inserts it into the logical plan with output type StructType(DEPTH_SCHEMA).
# The Jackson tokenizer runs per-row on executor threads during micro-batch
# execution, not here on the driver.
#
# The result is aliased as "d" immediately so that every field reference below
# uses col("d.E"), col("d.b"), etc. Without the alias, referencing a field
# would require repeating the full from_json(...) expression, causing Spark to
# tokenize the same JSON string once per output column — N full parse passes
# instead of one.
parsed = raw_stream.select(
    from_json(value_str, DEPTH_SCHEMA).alias("d")
)

# Step 3 — flatten the struct into named top-level columns.
#
# Expanding to top-level columns (rather than leaving them as struct.field)
# means Delta Lake predicate pushdown, SQL queries over the table, and the
# order book reconstructor can all reference columns by simple name without
# knowing the nesting depth. It also makes the console output readable.
messages = parsed.select(

    # from_unixtime expects integer epoch seconds and returns a formatted string
    # ("yyyy-MM-dd HH:mm:ss" by default). E is epoch milliseconds, so we divide
    # by 1000. LongType integer division truncates sub-second precision — fine
    # for display and date-based partitioning. If you need millisecond-accurate
    # timestamps, use (col("d.E") / 1000).cast("timestamp") instead, which
    # preserves fractional seconds as a TimestampType.
    from_unixtime(col("d.E") / 1000).alias("event_time"),

    col("d.s").alias("symbol"),

    # Both update IDs are preserved because the order book reconstructor needs
    # to check: current.U == previous.u + 1. Keeping only one would force a
    # stateful join or re-parse of the raw message to recover the other.
    col("d.U").alias("first_update_id"),
    col("d.u").alias("last_update_id"),

    # size() is a Catalyst array function that executes on the executor.
    # Python's len() cannot be used here — col("d.b") is a Column expression
    # referencing a future value, not a Python list that exists on the driver.
    # bid_count and ask_count let us measure message density and deletion ratio
    # without unpacking the full arrays in every downstream query.
    size(col("d.b")).alias("bid_count"),
    size(col("d.a")).alias("ask_count"),

    # b[0] is the first (best) bid because Binance sends bids in descending
    # price order — highest price first. b[0][0] is the price string; b[0][1]
    # is the quantity string. Both are null when the bids array is empty, which
    # is valid: not every depth update touches the bid side of the book.
    col("d.b")[0][0].alias("best_bid_price"),
    col("d.b")[0][1].alias("best_bid_qty"),

    # a[0] is the best ask because Binance sends asks in ascending price order —
    # lowest price first.
    col("d.a")[0][0].alias("best_ask_price"),
    col("d.a")[0][1].alias("best_ask_qty"),

    # Preserve the full price-level arrays. The console output shows only the
    # best level, but the Delta Lake sink (next stage) and the order book
    # reconstructor need every changed level to maintain correct book state.
    # Dropping these here would make this stage a lossy transformation.
    col("d.b").alias("raw_bids"),
    col("d.a").alias("raw_asks"),
)

# ---------------------------------------------------------------------------
# Streaming sink: console
# ---------------------------------------------------------------------------

query = (
    messages.writeStream
    .format("console")
    .trigger(processingTime=TRIGGER_INTERVAL)

    # append is the only valid outputMode for a query with no aggregation
    # operators. "complete" requires a groupBy; "update" requires a stateful
    # operator (window, watermark, or deduplication). Catalyst rejects the
    # other modes with AnalysisException at query compile time if there is no
    # aggregation to justify them.
    .outputMode("append")

    # Without truncate=False, Spark clips any string longer than 20 characters
    # in the console output. The raw_bids and raw_asks arrays printed as strings
    # can exceed 500 characters on a busy update — the truncation would hide
    # almost all of the data we are trying to inspect.
    .option("truncate", "false")

    .start()
)

print(f"Parsing '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP}")
print(f"Schema: {len(DEPTH_SCHEMA.fields)} top-level fields → {len(messages.columns)} output columns")
print(f"Trigger: every {TRIGGER_INTERVAL}. Press Ctrl+C to stop.\n")

try:
    query.awaitTermination()
except KeyboardInterrupt:
    print("\nShutting down…")
finally:
    query.stop()
    spark.stop()
    print("Done.")
