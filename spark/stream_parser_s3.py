"""
spark/stream_parser_s3.py

Production variant of stream_parser.py for EC2 deployment.
Reads from a local Kafka broker and writes Delta Lake to S3.

Pipeline:
  Kafka binary value
    → UTF-8 string
    → from_json struct (schema-validated)
    → flattened typed columns
    └→ Delta Lake sink on S3   (10-second trigger)

Key differences from stream_parser.py (local dev):
  - JAVA_HOME set for Amazon Corretto 11 (not Homebrew openjdk)
  - Delta output path and checkpoint path are s3a:// URIs
  - Two extra packages: hadoop-aws (S3A connector) and aws-java-sdk-bundle
  - Three extra SparkSession configs for S3A filesystem routing
  - Console sink removed (stdout on EC2 is ephemeral; use CloudWatch or journald)

Run on EC2:
    python spark/stream_parser_s3.py
Stop:
    Ctrl+C  (or kill the process — Delta's ACID log ensures no partial writes)
"""

import os
import sys

# ---------------------------------------------------------------------------
# JAVA_HOME — Amazon Corretto 11 (EC2 default for Amazon Linux 2)
#
# On EC2 with Amazon Linux 2, the Corretto 11 JDK is installed at the path
# below by the amazon-corretto-11 RPM package. Unlike macOS Homebrew's keg,
# Corretto's bin/ is on PATH, but PySpark also needs JAVA_HOME to locate
# libjvm.so when spawning the JVM. Setting it explicitly here makes the script
# self-contained regardless of whether the ec2-user shell profile exports it.
# ---------------------------------------------------------------------------
if "JAVA_HOME" not in os.environ:
    _corretto = "/usr/lib/jvm/java-11-amazon-corretto"
    if os.path.isdir(_corretto):
        os.environ["JAVA_HOME"] = _corretto
    else:
        sys.exit(
            "JAVA_HOME is not set and Amazon Corretto 11 was not found at "
            f"{_corretto}.\n"
            "Install with: sudo yum install -y java-11-amazon-corretto-devel\n"
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

from delta.tables import DeltaTable  # noqa: E402  — validates delta-spark install

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME        = "MarketStream-StreamParser-S3"
KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC     = "btcusdt_depth"
DELTA_TRIGGER   = "10 seconds"

# ---------------------------------------------------------------------------
# Package coordinates
#
# Why four packages instead of two?
#
# stream_parser.py (local) only needed:
#   spark-sql-kafka   — Kafka DataSource V2 connector
#   delta-spark       — Delta Lake table format
#
# This S3 variant adds:
#
#   hadoop-aws (org.apache.hadoop:hadoop-aws:3.3.4)
#     Implements the s3a:// filesystem scheme inside the Hadoop I/O layer
#     that Spark and Delta use for all storage operations. Without this JAR,
#     Spark has no driver for s3a:// URIs and raises
#     "No FileSystem for scheme: s3a" at query startup — not at SparkSession
#     creation, which makes the missing JAR hard to detect early.
#     Version 3.3.4 matches the hadoop-client bundled in Spark 3.5.1.
#     Mismatching major versions (e.g. hadoop-aws 3.2.x with Spark 3.5) causes
#     NoSuchMethodError at runtime because the S3A API surface changed.
#
#   aws-java-sdk-bundle (com.amazonaws:aws-java-sdk-bundle:1.12.262)
#     The AWS SDK v1 "bundle" JAR — a single fat JAR containing the S3, STS,
#     and IAM clients that hadoop-aws calls internally to authenticate and sign
#     requests. hadoop-aws declares aws-java-sdk-s3 as a provided dependency
#     (not bundled), so we must supply it explicitly. The "bundle" variant is
#     preferred over individual SDK module JARs because it avoids transitive
#     dependency conflicts between AWS modules. 1.12.262 is the version tested
#     with hadoop-aws 3.3.4; using a substantially different SDK version can
#     cause ClassNotFoundException for internal AWS SDK classes that hadoop-aws
#     references by name.
#
# Why NOT s3:// ?
#   s3:// was Hadoop's legacy S3 filesystem driver that stored file content
#   inside S3 object metadata. It was deprecated in 2010 and removed in
#   Hadoop 3.x. s3a:// is the current native-S3 driver: it uses the AWS SDK
#   directly, supports multipart upload, S3-compatible endpoints, and IAM
#   credential providers. All Spark documentation since 2018 uses s3a://.
# ---------------------------------------------------------------------------

KAFKA_PACKAGE   = "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
DELTA_PACKAGE   = "io.delta:delta-spark_2.12:3.2.0"
HADOOP_AWS      = "org.apache.hadoop:hadoop-aws:3.3.4"
AWS_SDK_BUNDLE  = "com.amazonaws:aws-java-sdk-bundle:1.12.262"

ALL_PACKAGES = ",".join([KAFKA_PACKAGE, DELTA_PACKAGE, HADOOP_AWS, AWS_SDK_BUNDLE])

# S3A paths for Delta table and checkpoint.
# Bucket: marketstream-delta-tejas (us-east-1, private, versioning on)
# The s3a:// scheme routes through hadoop-aws + aws-java-sdk-bundle.
DELTA_PATH      = "s3a://marketstream-delta-tejas/delta/order_book"
CHECKPOINT_PATH = "s3a://marketstream-delta-tejas/checkpoints/order_book"

# ---------------------------------------------------------------------------
# Binance depth update schema — identical to stream_parser.py
#
# Why caseSensitive = true is still required here:
#   DEPTH_SCHEMA has both "e" (event type, StringType) and "E" (event time ms,
#   LongType). Spark's default case-insensitive resolver treats them as the same
#   field, so col("d.E") matches twice and Catalyst raises
#   AMBIGUOUS_REFERENCE_TO_FIELDS. Same collision for "U" / "u". This is a
#   property of the schema, not of the storage backend, so it is required
#   regardless of whether the sink is local disk or S3.
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
    .master("local[*]")
    .config("spark.jars.packages", ALL_PACKAGES)

    # Delta Lake SQL extension — same as local version.
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")

    # Delta Lake catalog — same as local version.
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )

    # Case-sensitive schema resolution — required for e/E and U/u disambiguation.
    .config("spark.sql.caseSensitive", "true")

    # -------------------------------------------------------------------------
    # S3A filesystem configuration
    #
    # fs.s3a.impl
    #   Registers S3AFileSystem as the handler for the s3a:// scheme inside
    #   Hadoop's FileSystem registry. Without this, Spark would attempt to load
    #   a default handler (none) and fail with "No FileSystem for scheme: s3a".
    #   The FQN must match the class inside the hadoop-aws JAR exactly.
    #
    # fs.s3a.aws.credentials.provider
    #   Tells the S3A connector how to obtain AWS credentials. The value is a
    #   comma-separated list of credential provider classes tried in order.
    #   DefaultAWSCredentialsProviderChain is the standard AWS SDK chain:
    #     1. Environment variables:  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    #     2. Java system properties: aws.accessKeyId / aws.secretKey
    #     3. AWS credentials file:  ~/.aws/credentials  (ec2-user's profile)
    #     4. EC2 instance metadata: http://169.254.169.254/latest/meta-data/...
    #        (IAM role attached to the EC2 instance — the preferred production
    #         approach because no credentials are stored on disk at all)
    #   Using DefaultAWSCredentialsProviderChain means this script works without
    #   any hardcoded credentials: on EC2 with an IAM role it uses the metadata
    #   service; in local testing it falls back to ~/.aws/credentials.
    #   Never hardcode AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY directly in
    #   code — secrets in source files rotate badly and leak through git history.
    #
    # fs.s3a.endpoint
    #   Routes S3A requests to the correct AWS regional endpoint. Without this,
    #   the SDK defaults to the us-east-1 global endpoint (s3.amazonaws.com),
    #   which works for us-east-1 buckets but adds a redirect round-trip for
    #   buckets in other regions and causes signature validation failures if
    #   bucket-level redirect enforcement is on. Being explicit prevents the
    #   redirect and is the required config when using path-style access or
    #   VPC S3 endpoints.
    # -------------------------------------------------------------------------
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    .config("spark.hadoop.fs.s3a.endpoint",
            "s3.us-east-1.amazonaws.com")

    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ---------------------------------------------------------------------------
# Streaming source
# ---------------------------------------------------------------------------

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
    .option("subscribe", KAFKA_TOPIC)
    # "latest" — same reasoning as stream_parser.py. On EC2 the producer and
    # consumer start together; replaying from "earliest" on every restart would
    # rewrite data already committed to S3 Delta (duplicates that the Silver
    # dedup CTE handles, but wasteful at scale).
    .option("startingOffsets", "latest")
    .load()
)

# ---------------------------------------------------------------------------
# Parsing: binary → string → struct → flat columns
# ---------------------------------------------------------------------------

value_str = col("value").cast("string")

parsed = raw_stream.select(
    from_json(value_str, DEPTH_SCHEMA).alias("d")
)

messages = parsed.select(
    from_unixtime(col("d.E") / 1000).alias("event_time"),
    to_date(from_unixtime(col("d.E") / 1000)).alias("date"),
    lit("BTCUSDT").alias("symbol"),
    col("d.U").alias("first_update_id"),
    col("d.u").alias("last_update_id"),
    size(col("d.b")).alias("bid_count"),
    size(col("d.a")).alias("ask_count"),
    col("d.b")[0][0].alias("best_bid_price"),
    col("d.b")[0][1].alias("best_bid_qty"),
    col("d.a")[0][0].alias("best_ask_price"),
    col("d.a")[0][1].alias("best_ask_qty"),
    col("d.b").alias("raw_bids"),
    col("d.a").alias("raw_asks"),
)

# ---------------------------------------------------------------------------
# Sink: Delta Lake on S3
#
# Why partitionBy("date", "symbol") matters more on S3 than on local disk:
#
#   On local disk, a full directory scan of delta/order_book/ costs milliseconds
#   because the filesystem inode table is in memory.
#
#   On S3, listing objects is an HTTP API call (LIST with delimiter="/") billed
#   per 1000 objects returned and taking 50-200ms per page. A Delta table with
#   no partitioning accumulates all Parquet files in one flat "directory" (S3
#   prefix). Scanning a month of tick data with no partition pruning means
#   listing thousands of objects and then opening each file's footer to evaluate
#   the date predicate — O(n_files) LIST calls plus O(n_files) GET calls.
#
#   With date/symbol partitioning, the S3 prefix hierarchy is:
#     delta/order_book/date=2026-06-07/symbol=BTCUSDT/part-00000.parquet
#   A WHERE date = '2026-06-07' query issues exactly ONE LIST call for that
#   prefix and opens only those files — O(1) independent of total table size.
#
#   The checkpoint location on S3 works identically to local disk: Spark writes
#   JSON offset files per micro-batch. S3's eventual consistency is not a risk
#   here because the checkpoint path uses a unique prefix isolated from the Delta
#   table path, and S3 provides read-after-write consistency for new objects
#   since November 2020.
# ---------------------------------------------------------------------------

delta_query = (
    messages.writeStream
    .format("delta")
    .outputMode("append")
    .trigger(processingTime=DELTA_TRIGGER)
    .option("checkpointLocation", CHECKPOINT_PATH)
    .partitionBy("date", "symbol")
    .start(DELTA_PATH)
)

# ---------------------------------------------------------------------------
# Startup banner and shutdown
# ---------------------------------------------------------------------------

print(f"\nParsing '{KAFKA_TOPIC}' on {KAFKA_BOOTSTRAP}")
print(f"  Delta sink  : every {DELTA_TRIGGER}  →  {DELTA_PATH}")
print(f"  Checkpoint  : {CHECKPOINT_PATH}")
print(f"  Partitioned : date / symbol")
print(f"  Columns     : {len(messages.columns)}")
print("Press Ctrl+C to stop.\n")

try:
    delta_query.awaitTermination()
except KeyboardInterrupt:
    print("\nShutting down…")
finally:
    delta_query.stop()
    spark.stop()
    print("Done.")
