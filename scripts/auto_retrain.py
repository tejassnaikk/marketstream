"""
scripts/auto_retrain.py

Automated retraining trigger for MarketStream.
Runs every 30 minutes via cron. Checks /drift endpoint and retrains
the LightGBM model if drift is detected.

Pipeline:
  GET /drift
    → if drift_detected: run ml.train + ml.evaluate
    → copy new model to models/
    → restart API container
    → log result

Environment variables:
  API_URL        — default http://localhost:8000
  DUCKDB_PATH    — default ~/marketstream/duckdb_ec2/marketstream.duckdb
  MODEL_DIR      — default ~/marketstream/models
  MLFLOW_TRACKING_URI — default sqlite:////home/ec2-user/marketstream/mlflow.db
"""

import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL     = os.environ.get("API_URL", "http://localhost:8000")
DUCKDB_PATH = os.environ.get(
    "DUCKDB_PATH",
    str(Path.home() / "marketstream/duckdb_ec2/marketstream.duckdb")
)
MODEL_DIR   = os.environ.get(
    "MODEL_DIR",
    str(Path.home() / "marketstream/models")
)
MLFLOW_URI  = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:////{Path.home()}/marketstream/mlflow.db"
)
LOG_PATH    = Path.home() / "marketstream/logs/auto_retrain.log"

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
log = logging.getLogger("auto_retrain")


def check_drift() -> dict:
    r = requests.get(f"{API_URL}/drift", timeout=10)
    r.raise_for_status()
    return r.json()


def run_retrain() -> bool:
    env = os.environ.copy()
    env["DUCKDB_PATH"]           = DUCKDB_PATH
    env["MODEL_DIR"]             = MODEL_DIR
    env["MLFLOW_TRACKING_URI"]   = MLFLOW_URI

    for module in ["ml.train", "ml.evaluate"]:
        log.info(f"Running {module}...")
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(Path.home() / "marketstream"),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.error(f"{module} failed:\n{result.stderr}")
            return False
        log.info(result.stdout.strip())

    return True


def restart_api():
    log.info("Restarting API container...")
    result = subprocess.run(
        ["docker-compose", "-f",
         str(Path.home() / "marketstream/docker-compose.prod.yml"),
         "up", "-d", "--build", "api"],
        cwd=str(Path.home() / "marketstream"),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"API restart failed:\n{result.stderr}")
    else:
        log.info("API restarted successfully")


def main():
    log.info("=== Auto-retrain check ===")

    try:
        drift = check_drift()
        log.info(f"Drift status: {drift['status']} | up_ratio: {drift.get('up_ratio')} | down_ratio: {drift.get('down_ratio')}")
    except Exception as e:
        log.error(f"Failed to check drift: {e}")
        return

    if drift.get("status") != "drift_detected":
        log.info("No drift detected — skipping retrain")
        return

    log.info("Drift detected — starting retrain")
    success = run_retrain()

    if success:
        restart_api()
        log.info("Retrain complete")
    else:
        log.error("Retrain failed — keeping existing model")


if __name__ == "__main__":
    main()
