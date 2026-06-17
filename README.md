# MarketStream

[![CI](https://github.com/tejassnaikk/marketstream/actions/workflows/ci.yml/badge.svg)](https://github.com/tejassnaikk/marketstream/actions/workflows/ci.yml)
[![CD](https://github.com/tejassnaikk/marketstream/actions/workflows/cd.yml/badge.svg)](https://github.com/tejassnaikk/marketstream/actions/workflows/cd.yml)

Real-time cryptocurrency order book pipeline with ML-powered price direction prediction, full observability stack, and automated model retraining.

## Live Endpoints (EC2)

| Endpoint | URL |
|----------|-----|
| Prediction API | http://100.48.101.97:8000/predict |
| API Health | http://100.48.101.97:8000/health |
| Drift Detection | http://100.48.101.97:8000/drift |
| Latency | http://100.48.101.97:8000/latency |
| Grafana Dashboard | http://100.48.101.97:3000 |
| Prometheus | http://100.48.101.97:9090 |
| Streamlit Dashboard | http://100.48.101.97:8501 |

## Architecture
Binance WebSocket (wss://stream.binance.us:9443)

→ Kafka (Docker, topic: btcusdt_depth, ~140 msg/s)

→ kafka_to_duckdb.py (batch_size=10, ~1.7s tick lag)

→ stg_order_book (DuckDB Bronze)

→ refresh_gold.py (cron every 1min)

→ silver_order_book → gold_features_1m (DuckDB Silver/Gold)

→ FastAPI /predict (LightGBM, threshold=0.7013)

→ Streamlit Dashboard

**Observability layer (parallel):**
FastAPI /metrics → Prometheus (15s scrape) → Grafana

Watchdog container → Kafka lag + API health (60s) → AWS SNS alerts

/drift endpoint → up/down ratio (last 100 preds) → auto_retrain.py (cron 30min)

## ML Model

- **Task:** Binary classification — will BTC/USDT mid_price be higher 5 minutes from now?
- **Algorithm:** LightGBM (200 estimators, learning_rate=0.05)
- **Threshold:** 0.7013 (Youden's J on time-based split)
- **Auto-retraining:** triggered when drift ratio exceeds 80% in either direction

| Metric | v1 (148 rows) | v2 (279 rows) |
|--------|--------------|--------------|
| ROC-AUC | 0.760 | **0.785** |
| Accuracy | 0.670 | **0.750** |
| F1 | 0.670 | **0.759** |
| Precision (up) | — | **1.000** |

**Features (10):** `vwap`, `spread_bps_avg`, `imbalance_avg`, `depth_ratio`, `mid_price_avg`, `spread_change`, `imbalance_change`, `vwap_vs_mid`, `rolling_spread_mean`, `rolling_imbalance_std`

## Latency Profile

Measured on t3.small (2GB RAM) with 8 Docker containers running:

| Segment | Latency |
|---------|---------|
| Kafka tick → DuckDB write | ~1.7s |
| Gold table refresh | ~68s |
| **Total end-to-end** | **~70s** |

## Stack

| Layer | Technology |
|-------|-----------|
| Data source | Binance WebSocket API |
| Message broker | Apache Kafka (Confluent 7.4.0) |
| Stream storage | Delta Lake (local + S3) |
| Feature store | DuckDB (Bronze/Silver/Gold medallion) |
| Transformation | dbt (Silver/Gold SQL models) |
| ML training | LightGBM + MLflow |
| Serving | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Monitoring | Prometheus + Grafana |
| Alerting | AWS SNS (email) |
| Infrastructure | AWS EC2 (t3.small) + S3 |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |

## API Reference

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe + model version |
| `GET /predict` | Live BTC/USDT direction prediction |
| `GET /drift` | Up/down ratio over last 100 predictions |
| `GET /latency` | End-to-end pipeline lag measurement |
| `GET /metrics` | Prometheus scrape endpoint |
| `GET /model-info` | Model metadata + feature list |

### Sample `/predict` response

```json
{
  "symbol": "BTCUSDT",
  "prediction": "down",
  "confidence": 0.9881,
  "probability": 0.0119,
  "threshold": 0.7013,
  "timestamp": "2026-06-17T07:06:42Z",
  "features_used": {
    "vwap": 67190.97,
    "spread_bps_avg": 378.76,
    "imbalance_avg": 0.34
  }
}
```

## Monitoring & Observability

### Prometheus Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `marketstream_predictions_total` | Counter | Predictions by direction (up/down) |
| `marketstream_prediction_confidence` | Histogram | Confidence score distribution |
| `marketstream_model_version_info` | Gauge | Currently loaded model version |

### Grafana Dashboard (port 3000)
- Predictions Over Time
- Prediction Rate (per minute)
- API Latency p95

### Watchdog
- Checks Kafka offset growth + API `/health` every 60s
- 3 consecutive failures → SNS email alert to `marketstream-alerts` topic

### Drift Detection + Auto-Retraining
- `/drift` monitors up/down ratio over last 100 predictions
- `auto_retrain.py` runs every 30 minutes via cron
- If drift detected → retrain LightGBM → update threshold → restart API

## Project Structure
marketstream/

├── producer/

│   └── binance_producer.py        # Binance WebSocket → Kafka

├── spark/

│   └── stream_parser_s3.py        # Kafka → Delta Lake S3 (EC2)

├── marketstream_dbt/

│   └── models/

│       ├── silver/                # stg_order_book, silver_order_book

│       └── gold/                  # gold_features_1m, gold_features_5m

├── ml/

│   ├── features.py                # DuckDB → engineered features + labels

│   ├── train.py                   # LightGBM + MLflow training

│   └── evaluate.py                # Threshold tuning, ROC curve, metadata

├── api/

│   └── main.py                    # FastAPI: /health /predict /drift /latency /metrics

├── dashboard/

│   └── app.py                     # Streamlit live dashboard

├── scripts/

│   ├── kafka_to_duckdb.py         # Kafka consumer → DuckDB Bronze

│   ├── refresh_gold.py            # Bronze → Silver → Gold (cron 1min)

│   ├── auto_retrain.py            # Drift-triggered model retraining (cron 30min)

│   ├── watchdog.py                # Kafka + API health monitor → SNS

│   └── smoke_test.py              # 10-check end-to-end health test

├── monitoring/

│   └── prometheus.yml             # Prometheus scrape config

├── notebooks/

│   └── 01_demo.ipynb              # Live prediction demo with visualization

├── models/

│   ├── lgbm_direction.pkl         # Trained LightGBM classifier

│   └── model_metadata.json        # Threshold + metrics + feature list

├── Dockerfile.api

├── Dockerfile.dashboard

├── Dockerfile.producer

├── Dockerfile.watchdog

├── docker-compose.yml             # Local dev

└── docker-compose.prod.yml        # EC2 production (8 containers)

## CI/CD

Every push to `main`:
1. **CI** — validates model artifact loads + ROC-AUC > 0.70 + feature columns match
2. **CD** — SSH to EC2 → `git pull` → rebuild API container → deploy

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Local pipeline (Kafka → Spark → Delta → dbt → DuckDB) | ✅ |
| 2 | AWS deployment (EC2 + S3) | ✅ |
| 3 | ML + API + Dashboard | ✅ |
| 4 | Docker + Monitoring (Prometheus, Grafana, Watchdog, SNS) | ✅ |
| 5 | Stream processing (DuckDB pipeline, latency optimization) | ✅ |
| 6 | Model retraining, demo notebook, CI/CD | ✅ |
