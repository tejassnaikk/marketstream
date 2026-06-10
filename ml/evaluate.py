"""
ml/evaluate.py

Post-training evaluation for the MarketStream LightGBM direction classifier.

Responsibilities:
  1. Reload the saved model and reconstruct the same test set used in training
  2. Find the decision threshold that maximises Youden's J statistic
  3. Plot and save an annotated ROC curve
  4. Print a full classification report at the optimal threshold
  5. Write model_metadata.json for the FastAPI inference layer

Run from the project root:
    /opt/anaconda3/bin/python -m ml.evaluate
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.features import FEATURE_COLS, get_training_data
from ml.train import TEST_SIZE, time_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATH    = Path("/Volumes/Tejas SSD/marketstream/models/lgbm_direction.pkl")
METADATA_PATH = Path("/Volumes/Tejas SSD/marketstream/models/model_metadata.json")
PLOTS_DIR     = Path("/Volumes/Tejas SSD/marketstream/models/plots")


# ---------------------------------------------------------------------------
# load_model_and_data
# ---------------------------------------------------------------------------

def load_model_and_data():
    """
    Reload the serialised model and reconstruct the identical test split used
    during training.

    The test set must be reconstructed with the same time_split() call that
    train.py used. If we called get_training_data() and time_split() with
    different parameters — or applied any transformation in a different order —
    we would evaluate on a different set of rows than the model was withheld
    from, producing metrics that are neither train-time nor genuinely held-out.
    Using the same function with the same TEST_SIZE constant imported from
    ml.train guarantees the split boundary falls on the exact same row index.

    Returns
    -------
    model     : lgb.LGBMClassifier  — loaded sklearn-compatible wrapper
    X_test    : pd.DataFrame        — test feature matrix
    y_test    : pd.Series           — true binary labels
    y_prob    : np.ndarray          — predicted positive-class probabilities
    """
    model = joblib.load(MODEL_PATH)

    df = get_training_data()

    # time_split returns (train, test); we only need test here.
    _, test = time_split(df, test_size=TEST_SIZE)

    X_test = test[FEATURE_COLS]
    y_test = test["label"]

    # Column 1 of predict_proba is P(label=1) = P(price goes up).
    # We need raw probabilities (not hard predictions) for threshold tuning
    # and ROC-AUC calculation.
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_test, y_test, y_prob


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_test, y_prob):
    """
    Find the classification threshold that maximises Youden's J statistic.

    Youden's J = Sensitivity + Specificity − 1 = TPR − FPR

    It is the vertical distance from each point on the ROC curve to the
    diagonal (the random-classifier line). Maximising J gives the threshold
    where the model simultaneously achieves the best balance of true positive
    rate and false positive rate, without placing a prior weight on either.

    Why not just use 0.5?
    The default 0.5 threshold is only optimal when the positive class is
    exactly 50% of the population and the costs of false positives and false
    negatives are equal. Financial time series rarely satisfy either condition:
    the positive class fraction varies with market regime, and acting on a
    false buy signal (false positive) may cost differently than missing a true
    move (false negative). Youden's J adapts the threshold to the actual
    score distribution observed on the test set.

    Parameters
    ----------
    y_test : pd.Series    — true binary labels
    y_prob : np.ndarray   — predicted probabilities for the positive class

    Returns
    -------
    optimal_threshold : float
    fpr               : np.ndarray  — false positive rates across thresholds
    tpr               : np.ndarray  — true positive rates across thresholds
    thresholds        : np.ndarray  — threshold values corresponding to fpr/tpr
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    # J is a vector, one value per threshold point on the ROC curve.
    youdens_j = tpr - fpr

    # argmax returns the index of the threshold with the highest J.
    optimal_idx       = np.argmax(youdens_j)
    optimal_threshold = thresholds[optimal_idx]

    print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")

    return optimal_threshold, fpr, tpr, thresholds


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------

def plot_roc_curve(fpr, tpr, optimal_threshold, fpr_opt, tpr_opt, roc_auc):
    """
    Save an annotated ROC curve PNG with the optimal threshold marked.

    Parameters
    ----------
    fpr               : np.ndarray  — false positive rates (x-axis)
    tpr               : np.ndarray  — true positive rates (y-axis)
    optimal_threshold : float       — threshold value, shown in the point label
    fpr_opt           : float       — FPR at the optimal threshold (red dot x)
    tpr_opt           : float       — TPR at the optimal threshold (red dot y)
    roc_auc           : float       — area under the curve, shown in the legend
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    # ROC curve
    ax.plot(
        fpr, tpr,
        color="steelblue",
        linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )

    # Diagonal — represents a random classifier (AUC = 0.5). Any point above
    # this line adds predictive value; the further above, the better.
    ax.plot(
        [0, 1], [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Random classifier (AUC = 0.500)",
    )

    # Optimal threshold point
    ax.scatter(
        fpr_opt, tpr_opt,
        color="red",
        zorder=5,
        s=80,
        label=f"Optimal threshold = {optimal_threshold:.4f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("MarketStream — ROC Curve with Optimal Threshold")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    out_path = PLOTS_DIR / "roc_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"ROC curve saved to {out_path}")


# ---------------------------------------------------------------------------
# evaluate_at_threshold
# ---------------------------------------------------------------------------

def evaluate_at_threshold(y_test, y_prob, threshold):
    """
    Apply a specific decision threshold and print a full evaluation report.

    Using a tuned threshold instead of the default 0.5 can substantially
    change the precision/recall trade-off. This function applies the threshold,
    then reports every standard metric so the impact is immediately visible.

    Parameters
    ----------
    y_test    : pd.Series   — true binary labels
    y_prob    : np.ndarray  — predicted positive-class probabilities
    threshold : float       — decision boundary; P(up) >= threshold → predict 1

    Returns
    -------
    metrics : dict[str, float]
    """
    y_pred_tuned = (y_prob >= threshold).astype(int)

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred_tuned)
    print(cm)

    print("\nClassification report:")
    # target_names maps label 0 → "down", label 1 → "up" for human-readable output.
    print(classification_report(y_test, y_pred_tuned, target_names=["down", "up"]))

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred_tuned),
        "precision": precision_score(y_test, y_pred_tuned, zero_division=0),
        "recall":    recall_score(y_test, y_pred_tuned, zero_division=0),
        "f1":        f1_score(y_test, y_pred_tuned, zero_division=0),
        # ROC-AUC is threshold-independent — it uses y_prob, not y_pred_tuned,
        # so its value does not change when the threshold changes.
        "roc_auc":   roc_auc_score(y_test, y_prob),
    }

    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# save_metadata
# ---------------------------------------------------------------------------

def save_metadata(threshold, metrics):
    """
    Write model_metadata.json consumed by the FastAPI inference layer.

    The metadata file decouples the inference API from this evaluation script:
    FastAPI loads the threshold and feature list from JSON at startup rather
    than importing ml.evaluate or ml.train. This means the API container does
    not need the training dependencies (LightGBM training, matplotlib, dbt)
    installed — only joblib and the metadata file.

    Parameters
    ----------
    threshold : float           — optimal decision threshold from Youden's J
    metrics   : dict[str,float] — evaluation metrics at the optimal threshold
    """
    metadata = {
        "model_version":      "v1",
        "prediction_target":  "price_direction_5min",
        # label_horizon must match LABEL_HORIZON in ml/features.py.
        # Hardcoded here (rather than imported) to keep the JSON self-contained
        # and readable without tracing Python imports.
        "label_horizon":      5,
        "feature_cols":       FEATURE_COLS,
        "optimal_threshold":  float(threshold),
        "test_metrics":       {k: round(float(v), 4) for k, v in metrics.items()},
        # Placeholder: the FastAPI inference layer does not use this field.
        # A future training script can populate it from len(get_training_data()).
        "trained_on_rows":    None,
    }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {METADATA_PATH}")


# ---------------------------------------------------------------------------
# run_evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    """
    Full evaluation pipeline: load → threshold tune → plot → report → save.
    """
    model, X_test, y_test, y_prob = load_model_and_data()

    optimal_threshold, fpr, tpr, thresholds = find_optimal_threshold(y_test, y_prob)

    # Recover the (fpr, tpr) point that corresponds to the optimal threshold so
    # the red dot lands on the correct position on the ROC curve plot.
    optimal_idx = np.argmax(tpr - fpr)
    fpr_opt     = fpr[optimal_idx]
    tpr_opt     = tpr[optimal_idx]

    auc = roc_auc_score(y_test, y_prob)

    plot_roc_curve(fpr, tpr, optimal_threshold, fpr_opt, tpr_opt, roc_auc=auc)

    print("\n" + "─" * 50)
    print("── Evaluation at optimal threshold ──")

    metrics = evaluate_at_threshold(y_test, y_prob, optimal_threshold)

    save_metadata(optimal_threshold, metrics)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
