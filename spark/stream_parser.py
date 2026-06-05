"""
spark/stream_parser.py

Extends raw Kafka ingestion (read_kafka.py) by parsing the binary Binance
depth update payload into typed columns and writing to a Delta Lake table.
This is stage three of the Spark layer:

  Kafka binary value
    → UTF-8 string
    → from_json struct (schema-validated)
    → flattened typed columns
    ├→ console sink      (visual confirmation, 5-second trigger)
    └→ Delta Lake sink   (durable storage, 10-second trigger)

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
from pyspark.sql import SparkSession                                              # noqa: E402
from pyspark.sql.functions import col, from_json, from_unixtime, lit, size, to_date  # noqa: E402
from pyspark.sql.types import (                                                   # noqa: E402
    ArrayType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# DeltaTable is the Python API for Delta Lake DDL (MERGE, OPTIMIZE, VACUUM).
# We don't use it in this file yet, but importing it here validates that the
# delta-spark Python package is installed alongside the JAR. If this import
# fails, run: pip install delta-spark==3.2.0
from delta.tables import DeltaTable  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME        = "MarketStream-StreamParser"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC     = "btcusdt_depth"

CONSOLE_TRIGGER = "5 seconds"   # fast refresh for visual confirmation
DELTA_TRIGGER   = "10 seconds"  # slightly slower; Delta commit overhead per batch

# Kafka DataSource V2 connector coordinates.
# _2.12 is the Scala binary version of the Spark build — must match exactly.
# 3.5.1 is the connector version — must match the Spark version exactly.
# A mismatch in either suffix produces ClassNotFoundException at runtime
# (not at build time), so there is no compile-time safety net.
KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"

# Delta Lake connector coordinates.
# delta-spark_2.12 must match Spark's Scala binary version (_2.12).
# 3.2.0 is the Delta Lake version compatible with Spark 3.5.x — the Delta
# version is NOT the same as the Spark version. Compatibility matrix:
#   Delta 3.2.x → Spark 3.5.x   (this project)
#   Delta 3.1.x → Spark 3.5.x
#   Delta 2.4.x → Spark 3.4.x
# Using a Delta version built for a different Spark version causes obscure
# Catalyst plan errors rather than a clean startup failure.
DELTA_PACKAGE = "io.delta:delta-spark_2.12:3.2.0"

# Both packages passed as a single comma-separated string. Spark's Ivy resolver
# handles transitive dependencies for each independently; the comma is the
# only delimiter — no spaces around it.
ALL_PACKAGES = f"{KAFKA_PACKAGE},{DELTA_PACKAGE}"

# Delta table written here. Spark creates the directory on first write.
# The path is on the external SSD — fast enough for local development.
DELTA_PATH = "/Volumes/Tejas SSD/marketstream/delta/order_book"

# Checkpoint location records which Kafka offsets have been committed to Delta
# so the query can resume exactly where it left off after a restart.
# MUST be a different path than DELTA_PATH — if they share a directory,
# Delta's transaction log and Spark's checkpoint files collide and both
# become unreadable.
CHECKPOINT_PATH = "/Volumes/Tejas SSD/marketstream/checkpoints/order_book"

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

    # Downloads both the Kafka and Delta Lake JARs (plus their transitive
    # dependencies) from Maven Central on first run; cached in ~/.ivy2/ after that.
    # The comma-separated string is the only way to specify multiple packages —
    # calling .config("spark.jars.packages", ...) twice silently discards the first.
    .config("spark.jars.packages", ALL_PACKAGES)

    # Registers Delta Lake's SQL parser extension so that Delta-specific SQL
    # syntax (DESCRIBE HISTORY, RESTORE, GENERATE) works inside spark.sql().
    # Without this, those statements raise ParseException even though the JARs
    # are present.
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")

    # Replaces Spark's default in-memory catalog with Delta's catalog
    # implementation. This is what makes .format("delta") work as a table
    # format — the catalog tells Spark how to resolve Delta table paths,
    # versions, and metadata. Without it, .format("delta") raises
    # AnalysisException: "delta is not a valid data source".
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )

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

    # Partition key for Delta Lake. to_date() extracts the calendar date from
    # the epoch-second value, producing a DateType column (e.g. 2026-06-04).
    # Partitioning by date means Delta writes each day's data into its own
    # directory (date=2026-06-04/), so a query with a WHERE date = '2026-06-04'
    # filter skips every other day's files entirely — O(1) file scan instead
    # of O(n_days). Without this, every query scans all files ever written.
    to_date(from_unixtime(col("d.E") / 1000)).alias("date"),

    # Hardcoded as a literal rather than read from d.s because this pipeline
    # processes only BTCUSDT. Using lit() makes the partition column value
    # constant and visible to the query planner, enabling it to prune the
    # symbol=BTCUSDT/ partition directory without reading any file metadata.
    # When we add ETH/SOL streams later, replace lit() with col("d.s").
    lit("BTCUSDT").alias("symbol"),

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
# Sink 1: Delta Lake (durable storage)
# ---------------------------------------------------------------------------

# Ensure the checkpoint directory exists before starting the query.
# Spark creates the Delta table directory on first write, but the checkpoint
# directory must exist (or be creatable) before the StreamingQuery starts.
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

delta_query = (
    messages.writeStream
    .format("delta")

    # append adds new rows; it never modifies or deletes existing rows.
    # "complete" would require a groupBy and rewrite the entire table each
    # batch — correct for aggregations, catastrophic for raw tick data.
    # "update" is for stateful operators (watermarks, deduplication) and
    # requires a key column to identify which rows changed.
    .outputMode("append")

    # Each micro-batch is one ACID transaction to the Delta table. A 10-second
    # trigger means: collect all Kafka messages that arrive in a 10-second
    # window, then commit them atomically. Shorter triggers increase commit
    # frequency and create more small Parquet files (the "small file problem");
    # longer triggers reduce commit overhead but increase end-to-end latency.
    # 10 seconds is a reasonable balance for a dev environment.
    .trigger(processingTime=DELTA_TRIGGER)

    # The checkpoint records the exact Kafka offsets that have been written to
    # Delta after each successful commit. On restart, Spark reads this to resume
    # from the last committed offset rather than re-processing from "latest".
    # Without a checkpoint, every restart either loses data (latest) or floods
    # the pipeline with duplicates (earliest).
    # MUST NOT be inside DELTA_PATH — Delta's _delta_log/ would collide with
    # Spark's checkpoint files and corrupt both.
    .option("checkpointLocation", CHECKPOINT_PATH)

    # Partitioning splits the Parquet files into subdirectories by column value:
    #   delta/order_book/date=2026-06-04/symbol=BTCUSDT/part-00000.parquet
    # Queries that filter on date or symbol skip entire directory trees without
    # opening any files — this is Spark's partition pruning optimisation.
    # "date" is chosen over "event_time" because hourly or per-second
    # partitions would create thousands of tiny directories; daily is the
    # standard granularity for market tick data at this volume.
    .partitionBy("date", "symbol")

    .start(DELTA_PATH)
)

# ---------------------------------------------------------------------------
# Sink 2: console (visual confirmation)
# ---------------------------------------------------------------------------

console_query = (
    messages.writeStream
    .format("console")
    # 5-second trigger gives faster visual feedback than the Delta sink's
    # 10-second trigger. Both sinks run independently — each has its own
    # trigger loop and its own Kafka offset tracking.
    .trigger(processingTime=CONSOLE_TRIGGER)
    .outputMode("append")
    # Without truncate=False, Spark clips strings at 20 characters.
    # raw_bids and raw_asks can exceed 500 characters per row.
    .option("truncate", "false")
    .start()
)

# ---------------------------------------------------------------------------
# Startup banner and shutdown
# ---------------------------------------------------------------------------

print(f"\nParsing '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP}")
print(f"  Console sink : every {CONSOLE_TRIGGER}")
print(f"  Delta sink   : every {DELTA_TRIGGER}  →  {DELTA_PATH}")
print(f"  Checkpoint   : {CHECKPOINT_PATH}")
print(f"  Partitioned  : date / symbol")
print(f"  Columns      : {len(messages.columns)}")
print("Press Ctrl+C to stop.\n")

try:
    # awaitAnyTermination() blocks until either query fails or is stopped.
    # Using awaitTermination() on just one query would leave the other running
    # as an orphan after Ctrl+C; this ensures both are covered.
    spark.streams.awaitAnyTermination()
except KeyboardInterrupt:
    print("\nShutting down…")
finally:
    delta_query.stop()
    console_query.stop()
    spark.stop()
    print("Done.")
