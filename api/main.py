"""
api/main.py

FastAPI prediction service for MarketStream's LightGBM direction classifier.

Exposes three endpoints:
  GET /health   — liveness probe; confirms the model is loaded
  GET /metrics  — returns the full model metadata written by ml/evaluate.py
  GET /predict  — reads the latest Gold feature row from DuckDB and returns
                  a price-direction prediction with calibrated confidence

Run from the project root:
    /opt/anaconda3/bin/python -m api.main
Or directly:
    /opt/anaconda3/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

from ml.features import FEATURE_COLS, get_training_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH    = Path(os.environ.get("MODEL_PATH",    "/Volumes/Tejas SSD/marketstream/models/lgbm_direction.pkl"))
METADATA_PATH = Path(os.environ.get("METADATA_PATH", "/Volumes/Tejas SSD/marketstream/models/model_metadata.json"))

# ---------------------------------------------------------------------------
# Module-level state
#
# Both are populated once at startup by the lifespan handler and then treated
# as read-only for the lifetime of the process. Using module-level variables
# (rather than app.state) keeps the endpoint handlers free of any FastAPI-
# specific imports — they read plain Python objects that are already in scope.
# ---------------------------------------------------------------------------

_model    = None
_metadata = None


# ---------------------------------------------------------------------------
# Lifespan
#
# FastAPI's lifespan pattern replaces the deprecated @app.on_event("startup")
# decorator. The asynccontextmanager wraps the startup/shutdown boundary:
# everything before `yield` runs on startup, everything after `yield` runs on
# shutdown. Using lifespan also makes the startup logic testable — it can be
# invoked independently of the HTTP server.
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata

    # -- Startup -------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run /opt/anaconda3/bin/python -m ml.train first."
        )
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"Metadata file not found: {METADATA_PATH}\n"
            "Run /opt/anaconda3/bin/python -m ml.evaluate first."
        )

    _model = joblib.load(MODEL_PATH)
    print("Model loaded")

    with open(METADATA_PATH) as f:
        _metadata = json.load(f)
    print("Metadata loaded")

    yield

    # -- Shutdown ------------------------------------------------------------
    # Nothing to clean up for this service (no DB connections, no background
    # threads). The yield boundary is kept explicit so shutdown hooks can be
    # added here later without restructuring the lifespan function.


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MarketStream Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness probe. Returns 200 as long as the process is running and the
    model loaded successfully at startup. Orchestrators (ECS, Kubernetes,
    docker-compose healthcheck) poll this endpoint to decide whether to route
    traffic to this instance.
    """
    return {
        "status":        "ok",
        "model_version": _metadata["model_version"],
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@app.get("/metrics")
def metrics():
    """
    Return the full model metadata written by ml/evaluate.py.

    Includes the optimal threshold, test-set evaluation metrics, feature list,
    label horizon, and model version. Useful for dashboards and for confirming
    that the deployed model matches the trained artifact without opening a
    Python session.
    """
    return _metadata


# ---------------------------------------------------------------------------
# GET /predict
# ---------------------------------------------------------------------------

@app.get("/predict")
def predict():
    """
    Predict the 5-minute BTC/USDT price direction from the latest Gold features.

    Reads the most recent row from gold_features_1m via get_training_data()
    (which queries DuckDB), runs it through the loaded LightGBM classifier,
    and applies the optimal threshold stored in _metadata to produce a hard
    "up" / "down" label alongside a calibrated confidence score.

    Confidence interpretation:
      - For "up"   predictions: confidence = P(up)            — how sure we are it rises
      - For "down" predictions: confidence = 1 − P(up) = P(down) — how sure we are it falls
    This makes confidence always represent the model's certainty in the
    direction it predicted, rather than always reporting P(up) regardless of
    the label, which would be confusing for "down" predictions with high P(up).

    Returns HTTP 500 if DuckDB is unreachable, the feature row contains
    unexpected nulls, or the model fails to produce a prediction.
    """
    try:
        # -- Load latest Gold feature row ------------------------------------

        # get_training_data() opens a read-only DuckDB connection, runs the
        # full feature pipeline (engineer_features + create_labels), and
        # returns a labeled DataFrame ordered by window_start ascending.
        # We take the last row, which is the most recent completed 1-minute
        # window. The label column is present but ignored here — we use the
        # model's own prediction, not the retrospective label.
        df = get_training_data()
        latest = df.iloc[-1]

        # -- Build the feature vector ----------------------------------------

        # to_dict() produces a {column_name: value} mapping. Wrapping in a
        # list and constructing a DataFrame preserves column names so the
        # model's internal feature-name validation (LightGBM stores the
        # training feature names and warns on mismatch) passes cleanly.
        features = latest[FEATURE_COLS].to_dict()
        X = pd.DataFrame([features])

        # -- Score -----------------------------------------------------------

        # predict_proba returns shape (n_samples, 2); [0][1] is P(label=1)
        # for the single row we passed in. Cast to Python float explicitly —
        # LightGBM returns numpy.float64, which JSON-serialises correctly
        # in modern Python but fails isinstance checks and can cause subtle
        # type errors in downstream consumers that expect plain float.
        y_prob    = float(_model.predict_proba(X)[0][1])
        threshold = float(_metadata["optimal_threshold"])

        prediction = "up" if y_prob >= threshold else "down"

        # Confidence = the model's probability mass behind its own prediction.
        # A model predicting "up" with P(up)=0.9 has confidence 0.9.
        # A model predicting "down" with P(up)=0.1 has confidence 0.9 (= P(down)).
        confidence = y_prob if prediction == "up" else 1.0 - y_prob

        return {
            "symbol":        "BTCUSDT",
            "prediction":    prediction,
            "confidence":    round(confidence, 4),
            "probability":   round(y_prob, 4),
            "threshold":     round(threshold, 4),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            # Include the raw feature values so the caller can audit what data
            # the prediction was based on, without needing to query DuckDB
            # separately. Values rounded to 6 decimal places to keep the
            # response payload compact while retaining meaningful precision.
            "features_used": {k: round(float(v), 6) for k, v in features.items()},
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
