"""
dashboard/app.py

Streamlit live dashboard for MarketStream BTC/USDT market features.

Displays:
  - Latest model prediction (direction + confidence) from the FastAPI service
  - Four time-series charts from the Gold DuckDB table
  - Raw data expanders for debugging

Run from the project root:
    /opt/anaconda3/bin/streamlit run dashboard/app.py

The page auto-refreshes every REFRESH_INTERVAL seconds via st.rerun().
The FastAPI service must be running on localhost:8000 for the prediction card
to populate; the charts load directly from DuckDB and work independently.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUCKDB_PATH      = Path("/Volumes/Tejas SSD/marketstream/duckdb/marketstream.duckdb")
API_URL          = "http://localhost:8000"
REFRESH_INTERVAL = 60  # seconds between full page reruns

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MarketStream",
    layout="wide",
    page_icon="📈",
)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_prediction():
    """
    Call GET /predict on the FastAPI service and return the parsed JSON body.

    Returns {"error": <message>} on any failure so the caller never needs to
    handle exceptions — it just checks for the "error" key in the result.
    The 5-second timeout prevents a stalled API from blocking the entire
    dashboard render loop.
    """
    try:
        response = requests.get(f"{API_URL}/predict", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def load_gold_data() -> pd.DataFrame:
    """
    Read all rows from gold_features_1m ordered by window_start ascending.

    Opens a read-only DuckDB connection (safe for concurrent readers alongside
    dbt runs or ML training jobs) and closes it immediately after the query.
    Returns an empty DataFrame if the table does not exist yet rather than
    raising — render_charts() handles the empty case gracefully.
    """
    try:
        conn = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            df = conn.execute(
                "SELECT * FROM gold_features_1m ORDER BY window_start ASC"
            ).df()
        finally:
            conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_prediction_card(pred: dict):
    """
    Render a styled prediction card from the /predict API response.

    Uses st.markdown with inline HTML so the card can be colour-coded
    (green for "up", red for "down") without importing a third-party
    component library. unsafe_allow_html is required for inline styles.
    """
    if "error" in pred:
        st.error(f"API unavailable: {pred['error']}")
        return

    direction   = pred["prediction"]
    confidence  = pred["confidence"]
    probability = pred["probability"]

    color = "#00C853" if direction == "up" else "#D50000"
    arrow = "▲" if direction == "up" else "▼"

    st.markdown(
        f"""
        <div style="
            background: #1E1E1E;
            border-radius: 12px;
            padding: 28px 20px;
            text-align: center;
        ">
            <div style="
                font-size: 3.2rem;
                font-weight: 700;
                color: {color};
                line-height: 1.1;
            ">
                {arrow} {direction.upper()}
            </div>
            <div style="
                font-size: 1.15rem;
                color: #E0E0E0;
                margin-top: 14px;
            ">
                Confidence: <strong>{confidence:.1%}</strong>
            </div>
            <div style="
                font-size: 0.9rem;
                color: #9E9E9E;
                margin-top: 8px;
            ">
                Raw probability: {probability:.4f}
            </div>
            <div style="
                font-size: 0.85rem;
                color: #757575;
                margin-top: 6px;
            ">
                Threshold: {pred['threshold']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_charts(df: pd.DataFrame):
    """
    Render four time-series line charts from the Gold feature table.

    Chart layout:
      Row 1 (2 columns): Mid Price | Spread bps
      Row 2 (full width): Order Imbalance
      Row 3 (full width): Depth Ratio
    """
    if df.empty:
        st.warning("No Gold data yet. Run the Spark pipeline to populate gold_features_1m.")
        return

    # Ensure window_start is a proper datetime so Streamlit renders the x-axis
    # as a time series rather than a generic object axis.
    if not pd.api.types.is_datetime64_any_dtype(df["window_start"]):
        df = df.copy()
        df["window_start"] = pd.to_datetime(df["window_start"])

    # Use window_start as the index so every chart shares the same time axis.
    df = df.set_index("window_start")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Mid Price (USD)**")
        st.line_chart(df[["mid_price_avg"]])

    with col_right:
        st.markdown("**Spread (bps)**")
        st.line_chart(df[["spread_bps_avg"]])

    st.markdown("**Order Imbalance**")
    st.line_chart(df[["imbalance_avg"]])

    st.markdown("**Depth Ratio**")
    st.line_chart(df[["depth_ratio"]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("📈 MarketStream — BTC/USDT Live Dashboard")
    st.caption(
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        f"  |  Auto-refresh every {REFRESH_INTERVAL}s"
    )
    st.divider()

    # -- Fetch data (both calls are independent; Streamlit reruns the whole
    #    script on each refresh, so there is no stale-cache risk here) -------
    pred = fetch_prediction()
    df   = load_gold_data()

    # -- Main layout: 1:3 column ratio --------------------------------------
    col_pred, col_charts = st.columns([1, 3])

    with col_pred:
        st.subheader("Model Prediction")
        render_prediction_card(pred)

    with col_charts:
        st.subheader("Market Features")
        render_charts(df)

    st.divider()

    # -- Debug expanders -----------------------------------------------------
    with st.expander("Raw prediction data"):
        st.json(pred)

    with st.expander("Raw feature data (last 10 rows)"):
        st.dataframe(df.tail(10))

    # -- Auto-refresh --------------------------------------------------------
    # sleep() blocks the script for REFRESH_INTERVAL seconds, then st.rerun()
    # triggers a full top-to-bottom re-execution — re-fetching the prediction
    # and reloading the Gold table. This is simpler and more reliable than
    # st.experimental_rerun() + st_autorefresh component; the only trade-off
    # is that the page appears frozen during the sleep window, which is
    # acceptable for a 60-second interval dashboard.
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


# ---------------------------------------------------------------------------
# Entry point
#
# Streamlit executes the script from top to bottom on every render — it does
# not call `if __name__ == "__main__"` like a normal Python process. The
# `or True` ensures main() is always called regardless of how the script is
# invoked, including `streamlit run` (where __name__ == "__main__" is False
# in some Streamlit versions) and direct `python` execution for debugging.
# ---------------------------------------------------------------------------

if __name__ == "__main__" or True:
    main()
