"""
dashboard/app.py

MarketStream live monitoring dashboard.
Calls the FastAPI backend and displays live predictions,
system health, and drift monitoring.

Run:
    streamlit run dashboard/app.py
"""

import time
from datetime import datetime, timezone
from collections import deque

import pandas as pd
import requests
import streamlit as st

API_URL = "http://api:8000"

st.set_page_config(
    page_title="MarketStream",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.big-pred { font-size: 64px; font-weight: 700; line-height: 1; }
.up-color { color: #00c853; }
.down-color { color: #ff1744; }
.metric-label { font-size: 13px; color: #888; margin-bottom: 4px; }
.metric-big { font-size: 28px; font-weight: 600; }
.status-ok { color: #00c853; font-weight: 600; }
.status-warn { color: #ff9800; font-weight: 600; }
.status-bad { color: #ff1744; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "price_history" not in st.session_state:
    st.session_state.price_history = deque(maxlen=30)
if "conf_history" not in st.session_state:
    st.session_state.conf_history = deque(maxlen=30)
if "latency_history" not in st.session_state:
    st.session_state.latency_history = deque(maxlen=30)
if "time_history" not in st.session_state:
    st.session_state.time_history = deque(maxlen=30)
if "direction_history" not in st.session_state:
    st.session_state.direction_history = deque(maxlen=30)

def fetch_all():
    try:
        pred = requests.get(f"{API_URL}/predict", timeout=5).json()
        lat  = requests.get(f"{API_URL}/latency", timeout=5).json()
        drift = requests.get(f"{API_URL}/drift", timeout=5).json()
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        return pred, lat, drift, health, None
    except Exception as e:
        return None, None, None, None, str(e)

pred, lat, drift, health, err = fetch_all()

if pred:
    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
    st.session_state.price_history.append(pred["features_used"]["mid_price_avg"])
    st.session_state.conf_history.append(pred["confidence"])
    st.session_state.latency_history.append(lat.get("total_lag_seconds", 0))
    st.session_state.time_history.append(now)
    st.session_state.direction_history.append(pred["prediction"])

st.markdown("## 📈 MarketStream — Live BTC/USDT Prediction")
st.caption("Real-time order book → LightGBM → 5-minute price direction prediction")
st.divider()

if err:
    st.error(f"Cannot reach API: {err}")
    st.stop()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("### Live prediction")
    direction = pred["prediction"]
    conf = pred["confidence"]
    price = pred["features_used"]["mid_price_avg"]

    color_class = "up-color" if direction == "up" else "down-color"
    arrow = "↑" if direction == "up" else "↓"
    label = "Price likely to rise" if direction == "up" else "Price likely to fall"

    st.markdown(f'<div class="big-pred {color_class}">{arrow} {direction.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f"**{label}** in the next 5 minutes")
    st.markdown(f"**Confidence:** {conf:.1%}")
    st.markdown(f"**BTC price:** ${price:,.0f}")
    st.markdown(f"**Threshold:** {pred['threshold']}")
    ts = pred["timestamp"][:19].replace("T", " ") + " UTC"
    st.caption(f"Prediction at {ts}")

with col2:
    st.markdown("### System health")
    tick = lat.get("tick_lag_seconds", 0)
    gold = lat.get("gold_lag_seconds", 0)
    total = lat.get("total_lag_seconds", 0)

    st.metric("Tick → DuckDB", f"{tick:.1f}s", help="Time from Binance tick to database write")
    st.metric("Feature refresh lag", f"{gold:.0f}s", help="Age of latest Gold feature window")
    st.metric("Total end-to-end", f"{total:.0f}s", help="Full pipeline latency")
    st.metric("Model version", health.get("model_version", "—"))

with col3:
    st.markdown("### Drift monitoring")
    status = drift.get("status", "—")
    up_ratio = drift.get("up_ratio", 0)
    dn_ratio = drift.get("down_ratio", 0)
    total_preds = drift.get("total_predictions", 0)

    if status == "ok":
        st.markdown('<span class="status-ok">✓ No drift detected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">⚠ Drift detected — retraining soon</span>', unsafe_allow_html=True)

    st.progress(up_ratio, text=f"UP: {up_ratio:.0%}")
    st.progress(dn_ratio, text=f"DOWN: {dn_ratio:.0%}")
    st.caption(f"Based on last {total_preds} predictions. Auto-retrains at >80% skew.")

st.divider()

col4, col5 = st.columns(2)

with col4:
    st.markdown("### BTC price — last 30 predictions")
    if st.session_state.price_history:
        df_price = pd.DataFrame({
            "time": list(st.session_state.time_history),
            "price": list(st.session_state.price_history),
        }).set_index("time")
        st.line_chart(df_price, height=200)
    else:
        st.caption("Collecting data...")

with col5:
    st.markdown("### Model confidence — last 30 predictions")
    if st.session_state.conf_history:
        df_conf = pd.DataFrame({
            "time": list(st.session_state.time_history),
            "confidence": list(st.session_state.conf_history),
        }).set_index("time")
        st.line_chart(df_conf, height=200, color="#2a78d6")
    else:
        st.caption("Collecting data...")

st.divider()
st.markdown("### Features used in last prediction")
feats = pred["features_used"]
df_feats = pd.DataFrame([
    {"feature": k, "value": round(v, 4)}
    for k, v in feats.items()
])
st.dataframe(df_feats, use_container_width=True, hide_index=True)

st.divider()

col6, col7, col8 = st.columns(3)
with col6:
    st.markdown("**Pipeline**")
    st.markdown("""
    Binance WS → Kafka → DuckDB Bronze
    → refresh_gold.py (1min cron)
    → gold_features_1m → LightGBM → /predict
    """)
with col7:
    st.markdown("**Infrastructure**")
    st.markdown("""
    8 Docker containers on AWS EC2
    Prometheus + Grafana monitoring
    SNS email alerts on failure
    """)
with col8:
    st.markdown("**Model**")
    st.markdown("""
    LightGBM · 10 features · 279 rows
    ROC-AUC 0.785 · Threshold 0.7013
    Auto-retrains on drift detection
    """)

st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} · Auto-refreshes every 10 seconds")

time.sleep(10)
st.rerun()
