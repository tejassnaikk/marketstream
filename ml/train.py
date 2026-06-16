import os
"""
ml/train.py

LightGBM binary classifier for BTC/USDT 1-minute price direction.

Trains on Gold features produced by ml/features.py, evaluates on a held-out
time-ordered test set, logs parameters/metrics/artifacts to MLflow, and saves
the sklearn-compatible model to disk via joblib.

Run from the project root:
    /opt/anaconda3/bin/python -m ml.train

Why -m ml.train (module form) rather than python ml/train.py (script form)?
  `python ml/train.py` adds ml/ to sys.path, which breaks the relative import
  `from ml.features import ...` — Python would look for ml.features inside ml/,
  not at the top-level marketstream package. Running with `-m ml.train` adds the
  current directory (project root) to sys.path instead, so `ml` resolves to the
  package and the import works correctly.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.features import FEATURE_COLS, get_training_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR  = Path(os.environ.get("MODEL_DIR", "/Volumes/Tejas SSD/marketstream/models"))
MODEL_PATH = MODEL_DIR / "lgbm_direction.pkl"

# MLflow tracking URI uses a local file-based store so there is no server to
# run. The URI scheme "file://" tells MLflow to write run metadata, params,
# metrics, and artifacts directly to this directory instead of posting them
# to a remote tracking server. Every `mlflow.start_run()` call creates a
# subdirectory under MLFLOW_DIR / <experiment_id> / <run_id> /.
MLFLOW_DIR = Path.home() / "marketstream_mlflow"

# Fraction of rows (by time position, not random sample) reserved for testing.
# 0.2 = last 20% of 1-minute windows. Time-based splitting is mandatory for
# time-series data: random splitting allows information from the future to leak
# into training data (e.g. a row from minute 500 in the train set and its
# neighbour from minute 501 in the test set), producing optimistically biased
# metrics that vanish in production.
TEST_SIZE = 0.2

# LightGBM hyperparameters kept here as a dict so the same object is passed
# to both LGBMClassifier and mlflow.log_params() — single source of truth,
# no risk of them drifting out of sync.
LGBM_PARAMS = {
    "objective": "binary",
    # binary_logloss is the standard cross-entropy loss for binary classification.
    # LightGBM minimises it on the training set but we use it here as the
    # evaluation metric to monitor overfitting: if train loss decreases while
    # test loss plateaus or rises, the model has overfit.
    "metric": "binary_logloss",
    # 200 boosting rounds. At learning_rate=0.05 this is conservative —
    # it takes ~200 rounds to converge for most tabular datasets. Increasing
    # n_estimators with early stopping would be the next tuning step.
    "n_estimators": 200,
    # 0.05 is a deliberately slow learning rate. Smaller steps with more trees
    # typically generalises better than fewer large steps, at the cost of longer
    # training time. At this data scale (hundreds of rows) training takes < 1s
    # regardless of learning rate.
    "learning_rate": 0.05,
    # num_leaves controls tree complexity. 31 is LightGBM's default and is
    # appropriate for small datasets. Larger values increase capacity but raise
    # the risk of overfitting; min_child_samples provides a complementary
    # constraint at the leaf level.
    "num_leaves": 31,
    # Minimum number of training rows required to create a leaf node. This is
    # the primary regularisation knob for small datasets: setting it to 10
    # prevents the tree from creating leaves that represent fewer than 10 rows,
    # reducing variance at the cost of some bias. Without this, LightGBM can
    # memorise individual rows when the dataset is small.
    "min_child_samples": 10,
    "random_state": 42,
    # -1 suppresses LightGBM's per-iteration stdout output. Without this, every
    # boosting round prints a line to the console, overwhelming the run summary.
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# time_split
# ---------------------------------------------------------------------------

def time_split(df, test_size=TEST_SIZE):
    """
    Split a time-ordered DataFrame into train and test sets by position.

    Uses the last `test_size` fraction of rows as the test set, preserving
    chronological order. This is the correct split strategy for time-series:
    the model is trained on past windows and evaluated on future windows,
    exactly as it would be used in production.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered feature DataFrame from get_training_data().
    test_size : float
        Fraction of rows to hold out as test, applied to the tail.

    Returns
    -------
    train, test : tuple[pd.DataFrame, pd.DataFrame]
    """
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    return train, test


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def train_model(df):
    """
    Fit a LightGBM binary classifier and evaluate on the held-out test set.

    Parameters
    ----------
    df : pd.DataFrame
        Full labeled DataFrame from get_training_data().

    Returns
    -------
    model : lgb.LGBMClassifier
        Fitted classifier with sklearn-compatible API.
    metrics : dict[str, float]
        Evaluation metrics on the test set.
    X_test : pd.DataFrame
        Test features (retained so callers can run additional diagnostics).
    y_test : pd.Series
        True test labels.
    """
    train, test = time_split(df)

    X_train = train[FEATURE_COLS]
    y_train = train["label"]
    X_test  = test[FEATURE_COLS]
    y_test  = test["label"]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # predict_proba returns a (n_samples, 2) array: column 0 is P(label=0),
    # column 1 is P(label=1). We take column 1 for ROC-AUC, which requires
    # continuous probability scores rather than hard 0/1 predictions.
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        # Overall fraction of correct predictions. Can be misleading when the
        # positive class is rare (a model predicting all-0 gets high accuracy
        # but is useless). Included for completeness alongside precision/recall.
        "accuracy":  accuracy_score(y_test, y_pred),

        # Precision: of all windows predicted as "price up", how many actually
        # went up? High precision → few false buy signals.
        "precision": precision_score(y_test, y_pred, zero_division=0),

        # Recall: of all windows where price actually went up, how many did we
        # catch? High recall → few missed opportunities. Precision and recall
        # trade off against each other via the decision threshold.
        "recall":    recall_score(y_test, y_pred, zero_division=0),

        # F1: harmonic mean of precision and recall. Single number that balances
        # both; useful when comparing models where neither precision nor recall
        # should be sacrificed entirely.
        "f1":        f1_score(y_test, y_pred, zero_division=0),

        # ROC-AUC: probability that the model scores a random positive instance
        # higher than a random negative instance. Threshold-independent, so it
        # measures the model's raw discriminative ability without committing to
        # a specific 0.5 cut-off. Values above 0.55 suggest non-trivial signal
        # in a near-random financial time series.
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    return model, metrics, X_test, y_test


# ---------------------------------------------------------------------------
# run_training
# ---------------------------------------------------------------------------

def run_training():
    """
    End-to-end training run: load data → train → evaluate → log to MLflow
    → save model artifact.

    MLflow tracking structure written to MLFLOW_DIR:
      mlflow/
        <experiment_id>/        (created by set_experiment)
          <run_id>/
            params/             (LGBM_PARAMS keys/values)
            metrics/            (accuracy, precision, recall, f1, roc_auc)
            artifacts/
              lgbm_model/       (LightGBM booster in MLflow format)
    """
    # file:// URI tells MLflow to use the local filesystem as its tracking store.
    # No server process is needed. The experiment directory is created on first
    # use if it does not exist.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")

    # Creates the experiment if it doesn't exist; subsequent runs append to it.
    # All runs for this project share one experiment name so they appear together
    # in `mlflow ui` without filtering.
    mlflow.set_experiment("marketstream-direction")

    with mlflow.start_run():

        # -- Data ------------------------------------------------------------
        df = get_training_data()

        # -- Train + evaluate ------------------------------------------------
        model, metrics, X_test, y_test = train_model(df)

        # -- Log to MLflow ---------------------------------------------------

        # log_params records hyperparameters as key-value strings. Storing them
        # in MLflow (rather than just printing) means every future run is
        # self-documenting: `mlflow ui` shows which params produced which metrics
        # without needing to cross-reference the source file.
        mlflow.log_params(LGBM_PARAMS)

        # log_metrics records numeric evaluation results. MLflow stores one value
        # per metric per run; use log_metric with step= for per-epoch series.
        mlflow.log_metrics(metrics)

        # log_model saves the raw LightGBM booster (not the sklearn wrapper) in
        # MLflow's native format, which includes a conda environment spec and a
        # MLmodel descriptor. model.booster_ is the underlying lgb.Booster object
        # extracted from the LGBMClassifier wrapper. The sklearn wrapper is saved
        # separately via joblib below — one format for MLflow's model registry,
        # one for fast local inference without an MLflow dependency.
        mlflow.lightgbm.log_model(model.booster_, "lgbm_model")

        # -- Save sklearn-compatible artifact --------------------------------
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # joblib serialises the full LGBMClassifier (including threshold,
        # feature names, and sklearn interface). Downstream inference code loads
        # this with joblib.load() and calls model.predict() / predict_proba()
        # without needing to reconstruct the LightGBM booster manually.
        joblib.dump(model, MODEL_PATH)

        # -- Print summary ---------------------------------------------------
        print("\n── Evaluation on test set ──────────────────────────")
        for name, value in metrics.items():
            print(f"  {name:<12} {value:.4f}")
        print(f"\nModel saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_training()
