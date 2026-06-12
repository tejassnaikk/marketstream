"""
scripts/smoke_test.py

End-to-end health check for the full MarketStream pipeline.
Prints ✓ or ✗ for each check and exits 0 (all pass) or 1 (any fail).

Run from the project root:
    /opt/anaconda3/bin/python scripts/smoke_test.py

The script is deliberately self-contained (no ml.* imports) so it can be run
on any machine that has the model artifacts and credentials, without needing
the full Python package structure on sys.path.
"""

import json
import sys
from pathlib import Path

import boto3
import botocore
import duckdb
import joblib
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

import os
API_URL       = os.environ.get("API_URL", "http://localhost:8000")
MODEL_PATH    = Path(os.environ.get("MODEL_PATH", "/Volumes/Tejas SSD/marketstream/models/lgbm_direction.pkl"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", "/Volumes/Tejas SSD/marketstream/models/model_metadata.json"))
DUCKDB_PATH   = Path(os.environ.get("DUCKDB_PATH", "/Volumes/Tejas SSD/marketstream/duckdb/marketstream.duckdb"))
S3_BUCKET     = "marketstream-delta-tejas"
S3_PREFIX     = "delta/order_book/"
LOCAL_DELTA   = Path(os.environ.get("LOCAL_DELTA", "/Volumes/Tejas SSD/marketstream/delta/order_book"))

# ---------------------------------------------------------------------------
# Failure accumulator
# ---------------------------------------------------------------------------

FAILURES: list[str] = []

# ---------------------------------------------------------------------------
# check() helper
# ---------------------------------------------------------------------------

def check(label: str, fn) -> None:
    """
    Run fn() and print ✓ or ✗.

    Any exception (including AssertionError) is caught, its message printed,
    and the label appended to FAILURES. Using a single broad except means a
    check that raises an unexpected error (e.g. ImportError, OSError) is
    always recorded as a failure rather than crashing the whole script and
    skipping the remaining checks.
    """
    try:
        fn()
        print(f"  ✓  {label}")
    except Exception as e:
        print(f"  ✗  {label} — {e}")
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_model_artifact():
    """Verify the joblib model file exists and loads a predict_proba-capable object."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise AssertionError(
            f"Loaded object {type(model).__name__!r} has no predict_proba method"
        )


def check_metadata():
    """Verify the JSON metadata file exists and contains the expected keys."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Not found: {METADATA_PATH}")
    metadata = json.loads(METADATA_PATH.read_text())
    required = {"model_version", "optimal_threshold", "feature_cols", "test_metrics"}
    missing  = required - metadata.keys()
    if missing:
        raise AssertionError(f"Missing keys in metadata: {sorted(missing)}")


def check_duckdb():
    """Verify gold_features_1m is populated with at least one row."""
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        count = conn.execute("SELECT COUNT(*) FROM gold_features_1m").fetchone()[0]
    finally:
        conn.close()
    if count == 0:
        raise AssertionError("gold_features_1m is empty — run the Spark pipeline first")


def check_duckdb_schema():
    """Verify gold_features_1m contains every column the ML pipeline depends on."""
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        rows = conn.execute("DESCRIBE gold_features_1m").fetchall()
    finally:
        conn.close()
    # DESCRIBE returns (column_name, column_type, null, key, default, extra)
    present  = {row[0] for row in rows}
    required = {
        "window_start", "mid_price_avg", "spread_bps_avg",
        "imbalance_avg", "depth_ratio", "vwap",
    }
    missing = required - present
    if missing:
        raise AssertionError(f"Missing columns in gold_features_1m: {sorted(missing)}")


def check_local_delta():
    """Verify the local Bronze Delta table directory exists and contains Parquet files."""
    if not LOCAL_DELTA.exists():
        raise AssertionError(f"Delta directory not found: {LOCAL_DELTA}")
    parquet_files = list(LOCAL_DELTA.rglob("*.parquet"))
    if not parquet_files:
        raise AssertionError(
            f"No .parquet files found under {LOCAL_DELTA} — "
            "run the Spark pipeline to write Bronze data"
        )


def check_s3_delta():
    """
    Verify the S3 Bronze Delta table prefix contains at least one object.

    Uses MaxKeys=1 to minimise API cost — we only need to know whether any
    object exists under the prefix, not how many. botocore.exceptions is
    imported for explicit NoCredentialsError handling so the failure message
    is actionable ("configure AWS credentials") rather than a raw exception
    dump.
    """
    s3 = boto3.client("s3", region_name="us-east-1")
    try:
        response = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=S3_PREFIX,
            MaxKeys=1,
        )
    except botocore.exceptions.NoCredentialsError:
        raise AssertionError(
            "AWS credentials not found — run `aws configure` or attach an IAM role"
        )
    if "Contents" not in response:
        raise AssertionError(
            f"No objects found at s3://{S3_BUCKET}/{S3_PREFIX} — "
            "run stream_parser_s3.py on EC2 to write S3 Bronze data"
        )


def check_api_health():
    """Verify the FastAPI service is reachable and returns a valid health response."""
    response = requests.get(f"{API_URL}/health", timeout=5)
    if response.status_code != 200:
        raise AssertionError(
            f"Unexpected status {response.status_code}: {response.text[:120]}"
        )
    body = response.json()
    if "model_version" not in body:
        raise AssertionError(
            f"'model_version' missing from /health response: {body}"
        )


def check_api_predict():
    """Verify /predict returns a well-formed prediction response."""
    response = requests.get(f"{API_URL}/predict", timeout=10)
    if response.status_code != 200:
        raise AssertionError(
            f"Unexpected status {response.status_code}: {response.text[:120]}"
        )
    data = response.json()

    required_keys = {"symbol", "prediction", "confidence", "probability", "threshold", "timestamp"}
    missing = required_keys - data.keys()
    if missing:
        raise AssertionError(f"Missing keys in /predict response: {sorted(missing)}")

    if data["prediction"] not in ("up", "down"):
        raise AssertionError(
            f"Unexpected prediction value: {data['prediction']!r} (expected 'up' or 'down')"
        )

    confidence = data["confidence"]
    if not (0.0 <= confidence <= 1.0):
        raise AssertionError(
            f"confidence {confidence} is out of range [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MarketStream Smoke Test")
    print("─" * 40)

    print("\n── Model artifacts")
    check("Model artifact (lgbm_direction.pkl)", check_model_artifact)
    check("Model metadata (model_metadata.json)", check_metadata)

    print("\n── Data layers")
    check("DuckDB row count (gold_features_1m > 0)", check_duckdb)
    check("DuckDB schema (required columns present)", check_duckdb_schema)
    check("Local Delta (Bronze Parquet files exist)", check_local_delta)
    check("S3 Delta (Bronze objects in s3 bucket)", check_s3_delta)

    print("\n── API")
    check("API /health (status 200 + model_version)", check_api_health)
    check("API /predict (valid prediction response)", check_api_predict)

    print()
    if not FAILURES:
        print("All checks passed ✓")
        sys.exit(0)
    else:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
