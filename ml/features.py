"""
ml/features.py

Feature pipeline for MarketStream's ML layer. Reads pre-aggregated Gold
features from DuckDB, engineers lagged and rolling features, and attaches
forward-looking price-direction labels.

Output contract:
  - One row per 1-minute window (window_start, symbol columns retained)
  - All columns in FEATURE_COLS are present and non-null
  - label is binary int (1 = mid_price rose over the next LABEL_HORIZON rows,
    0 = flat or fell)
  - No rows with NaN anywhere in the returned DataFrame

No ML model code lives here. This module is imported by training scripts and
real-time inference pipelines alike — both get the same feature set.
"""

import os

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUCKDB_PATH   = Path(os.environ.get("DUCKDB_PATH", "/Volumes/Tejas SSD/marketstream/duckdb/marketstream.duckdb"))

# Number of 1-minute rows to look forward when constructing the price-direction
# label. LABEL_HORIZON = 5 means: "did mid_price rise 5 minutes from now?"
# 5 minutes balances predictability (short enough that signal exists) against
# actionability (long enough to place and fill an order). Callers can override
# per-experiment via the horizon argument.
LABEL_HORIZON = 5

# Minimum acceptable training set size. Fewer than 100 rows means either the
# pipeline has barely started producing data or a filter removed too much.
# In both cases, training on this data would produce an unreliable model.
MIN_ROWS      = 100


# ---------------------------------------------------------------------------
# load_features
# ---------------------------------------------------------------------------

def load_features(horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """
    Read all rows from gold_features_1m, ordered by window_start ascending.

    Opens a read-only DuckDB connection so this function is safe to call from
    concurrent processes (e.g. a training job and a live inference loop running
    simultaneously). The connection is closed immediately after the query so
    the DuckDB file lock is held for the minimum possible time.

    Parameters
    ----------
    horizon : int
        Accepted but unused here — present so the public signature of every
        pipeline step is consistent and callers can thread `horizon` through
        without inspecting which step uses it.

    Returns
    -------
    pd.DataFrame
        Raw Gold table, one row per (window_start, symbol) pair, no feature
        engineering applied yet.
    """
    # read_only=True prevents accidental writes and allows concurrent readers.
    # DuckDB enforces a single-writer / multiple-reader policy at the file level;
    # without read_only=True, a second process opening the same file would raise
    # an IOException even if it only intends to SELECT.
    conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = conn.execute(
            "SELECT * FROM gold_features_1m ORDER BY window_start ASC"
        ).df()
    finally:
        # Close regardless of whether the query succeeded or raised. Leaving the
        # connection open would hold the DuckDB file lock and block any concurrent
        # writer (e.g. `dbt run`) until the Python process exits.
        conn.close()

    return df


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns that capture rate-of-change and rolling statistics.

    All five new columns require multiple rows of context:
      - diff(1) needs at least 2 consecutive rows → 1 NaN at the start
      - rolling(5) needs at least 5 rows → 4 NaNs at the start
    The union of all NaN positions is dropped in a single dropna() call so
    the returned DataFrame has exactly `rolling(5) - 1 = 4` fewer rows than
    the input (assuming no pre-existing NaNs).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_features(). Must contain: spread_bps, order_imbalance,
        vwap, mid_price.

    Returns
    -------
    pd.DataFrame
        Input columns plus five derived columns, NaN rows removed, index reset
        to 0-based integers.
    """
    out = df.copy()

    # Rate of change in bid-ask spread across consecutive 1-minute windows.
    # A positive spike in spread_change means liquidity suddenly thinned out
    # (market makers widened quotes). This is often a leading signal before
    # a directional move because MMs widen defensively when they detect flow.
    # Gold column name is spread_bps_avg (average spread over the 1-minute window).
    out["spread_change"] = out["spread_bps_avg"].diff(1)

    # Rate of change in order flow imbalance. A swing from balanced (0) to
    # strongly positive (+1) in one minute indicates a rapid shift toward
    # buy-side aggression that mid_price may not yet have fully reflected.
    # Gold column name is imbalance_avg (average imbalance over the window).
    out["imbalance_change"] = out["imbalance_avg"].diff(1)

    # Percentage gap between depth-weighted VWAP and average mid_price.
    # When VWAP > mid (positive gap), the weighted liquidity centroid sits
    # above the arithmetic mean — suggesting heavier depth on the ask side
    # pulling the weighted average up, which can precede ask-side exhaustion.
    # Dividing by mid_price_avg makes this dimensionless and stable across
    # BTC price levels (e.g. $30k vs $100k).
    out["vwap_vs_mid"] = (out["vwap"] - out["mid_price_avg"]) / out["mid_price_avg"]

    # 5-minute rolling mean of spread_bps_avg. Smooths the single-window spread
    # reading into a short-term liquidity baseline. The current spread_bps_avg
    # relative to rolling_spread_mean tells whether liquidity is tighter or
    # wider than its recent norm — more informative for a model than the raw
    # level alone.
    out["rolling_spread_mean"] = out["spread_bps_avg"].rolling(5).mean()

    # 5-minute rolling standard deviation of imbalance_avg. High std means
    # the imbalance has been flipping rapidly — a choppy, indecisive order flow.
    # Low std means sustained one-sided flow. This dispersion feature provides
    # information about *consistency* of directional pressure that the level
    # (imbalance_avg) does not capture.
    out["rolling_imbalance_std"] = out["imbalance_avg"].rolling(5).std()

    # Drop all rows that have NaN in any column. The rolling(5).mean() produces
    # the most NaNs (first 4 rows), so that is the binding constraint — exactly
    # 4 rows are dropped from the top of the DataFrame on a clean series.
    out = out.dropna()

    # Reset to 0-based integer index so downstream iloc slicing and train/test
    # splits behave predictably. Without reset_index(), the original row numbers
    # survive the drop, leaving gaps (e.g. index jumps from 3 to 8).
    out = out.reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# create_labels
# ---------------------------------------------------------------------------

def create_labels(df: pd.DataFrame, horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """
    Attach a binary price-direction label to each row.

    label = 1 if mid_price at row (i + horizon) > mid_price at row i, else 0.

    The last `horizon` rows have no future mid_price and are dropped; they
    cannot be labeled and must not be included in training.

    Parameters
    ----------
    df : pd.DataFrame
        Output of engineer_features(). Must contain: mid_price.
    horizon : int
        How many 1-minute rows ahead to look for the target mid_price.

    Returns
    -------
    pd.DataFrame
        Input columns plus 'label' (int), last `horizon` rows removed,
        future_mid helper column removed, index reset.
    """
    out = df.copy()

    # shift(-horizon) moves the mid_price column up by `horizon` positions,
    # aligning each row with the price that will be observed `horizon` minutes
    # later. The last `horizon` rows get NaN because there is no future data
    # to shift into them.
    out["future_mid"] = out["mid_price_avg"].shift(-horizon)

    # Binary classification target: 1 = price went up, 0 = flat or down.
    # The comparison produces a boolean Series; astype(int) converts True→1,
    # False→0. NaN in future_mid propagates as NaN through the comparison,
    # so those rows are handled by the dropna() below rather than producing
    # a silent 0 label.
    out["label"] = (out["future_mid"] > out["mid_price_avg"]).astype("Int64")

    # Drop the last `horizon` rows where future_mid is NaN (no label available).
    out = out.dropna(subset=["future_mid"])

    # Remove the helper column — it is an intermediate construct, not a feature.
    # Keeping it would let a model trivially predict the label from future_mid,
    # which would be a data leakage bug.
    out = out.drop(columns=["future_mid"])

    out = out.reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# get_training_data
# ---------------------------------------------------------------------------

def get_training_data(horizon: int = LABEL_HORIZON) -> pd.DataFrame:
    """
    Full pipeline: load → engineer features → attach labels.

    Parameters
    ----------
    horizon : int
        Forwarded to create_labels(). Controls how far ahead the label looks.

    Returns
    -------
    pd.DataFrame
        Clean, labeled DataFrame ready for train/test split and model fitting.

    Raises
    ------
    ValueError
        If the final DataFrame has fewer than MIN_ROWS rows, indicating that
        not enough Gold data has been collected for a reliable model.
    """
    df = load_features(horizon=horizon)
    df = engineer_features(df)
    df = create_labels(df, horizon=horizon)

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Training data has only {len(df)} rows (minimum is {MIN_ROWS}). "
            "Run the Spark pipeline longer to collect more data before training."
        )

    # Cast label back to plain int now that NaN rows are gone (Int64 → int64).
    df["label"] = df["label"].astype(int)

    print(
        f"Training data ready: {len(df)} rows, "
        f"{df['label'].mean():.2%} positive class"
    )

    return df


# ---------------------------------------------------------------------------
# Feature column list
#
# Defined after the functions that produce these columns so that the names
# here and the names in engineer_features() are never in two separate places
# that could drift out of sync.
#
# FEATURE_COLS is the canonical list of columns a model should train on.
# It excludes metadata columns (window_start, window_end, symbol,
# update_count) and the label itself. Training scripts do:
#   X = df[FEATURE_COLS]
#   y = df["label"]
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "vwap",                   # depth-weighted average price in the window
    "spread_bps_avg",         # average bid-ask spread (bps) over the window
    "imbalance_avg",          # average (bid_count - ask_count) / total, range [-1, 1]
    "depth_ratio",            # bid_depth_avg / ask_depth_avg
    "mid_price_avg",          # average (best_bid + best_ask) / 2 over the window
    "spread_change",          # 1-row diff of spread_bps_avg
    "imbalance_change",       # 1-row diff of imbalance_avg
    "vwap_vs_mid",            # (vwap - mid_price_avg) / mid_price_avg
    "rolling_spread_mean",    # 5-row rolling mean of spread_bps_avg
    "rolling_imbalance_std",  # 5-row rolling std of imbalance_avg
]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = get_training_data()
    print()
    print(df.head())
    print()
    print(df[FEATURE_COLS + ["label"]].describe())
