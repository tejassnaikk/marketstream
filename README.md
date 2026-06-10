# MarketStream

Real-time cryptocurrency order book pipeline with ML-powered price direction prediction.

## Architecture

```
Binance WS → Kafka → Spark → Delta Lake (Bronze) → dbt → DuckDB (Gold) → LightGBM → FastAPI → Streamlit
```

## Stack

| Layer | Technology | Purpose |
|---|---|---|
| Ingestion | Binance WebSocket + Kafka | Real-time BTC/USDT order book stream |
| Stream Processing | Apache Spark Structured Streaming | Parse + write to Delta Lake |
| Storage | Delta Lake (local + S3) | Bronze layer with date/symbol partitioning |
| Transformation | dbt + DuckDB | Silver dedup/filter, Gold 1m/5m feature windows |
| ML | LightGBM + MLflow | Binary price direction classifier, experiment tracking |
| Serving | FastAPI | REST prediction API with /health /metrics /predict |
| Dashboard | Streamlit | Live feature charts + model prediction card |
| Cloud | AWS EC2 + S3 | EC2 Kafka/Spark producer, S3 Delta sink |

## Quick Start

### Prerequisites

- Python 3.13 (Anaconda), Java 11
- Docker + Colima (Mac) or Docker Engine (Linux)
- AWS CLI configured with S3 + EC2 access

### Run locally

1. **Start Kafka**
   ```bash
   docker-compose up -d
   ```

2. **Start producer**
   ```bash
   source venv/bin/activate && python producer/binance_producer.py
   ```

3. **Start Spark**
   ```bash
   export SPARK_LOCAL_IP=127.0.0.1 && /opt/anaconda3/bin/python spark/stream_parser.py
   ```

4. **Run dbt**
   ```bash
   cd marketstream_dbt && /opt/anaconda3/bin/dbt run --profiles-dir . --project-dir .
   ```

5. **Train model**
   ```bash
   /opt/anaconda3/bin/python -m ml.train
   ```

6. **Start API**
   ```bash
   /opt/anaconda3/bin/python -m api.main
   ```

7. **Start dashboard**
   ```bash
   /opt/anaconda3/bin/streamlit run dashboard/app.py
   ```

8. **Smoke test**
   ```bash
   /opt/anaconda3/bin/python scripts/smoke_test.py
   ```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Liveness check + model version |
| /metrics | GET | Model evaluation metrics + feature list |
| /predict | GET | Latest Gold features → LightGBM → prediction JSON |

Sample `/predict` response:

```json
{
  "symbol": "BTCUSDT",
  "prediction": "up",
  "confidence": 0.9999,
  "probability": 0.9999,
  "threshold": 0.9936,
  "timestamp": "2026-06-10T18:11:40+00:00",
  "features_used": {
    "vwap": 62265.42,
    "spread_bps_avg": 2.06,
    "...": "..."
  }
}
```

## ML Model

- **Target:** Binary classification — will BTC/USDT `mid_price` be higher 5 minutes from now?
- **Features:** 10 features derived from Gold layer (VWAP, spread, imbalance, depth ratio + rolling stats)
- **Algorithm:** LightGBM (200 estimators, learning_rate=0.05)
- **Threshold tuning:** Youden's J statistic on time-based train/test split
- **Tracking:** MLflow with SQLite backend

| Metric | Value |
|---|---|
| ROC-AUC | 0.76 |
| Accuracy | 0.67 |
| F1 | 0.67 |
| Train rows | 118 |
| Test rows | 30 |

> Metrics will improve as more data accumulates. Model is retrained periodically.

## Project Structure

```
marketstream/
├── producer/
│   └── binance_producer.py       # Binance WS → Kafka producer (certifi SSL, US endpoint)
├── spark/
│   ├── stream_parser.py          # Kafka → Delta Lake (local), console sink
│   └── stream_parser_s3.py       # Kafka → Delta Lake (S3), EC2 deployment variant
├── marketstream_dbt/
│   └── models/
│       ├── silver/               # stg_order_book (dedup/cast), silver_order_book (metrics)
│       └── gold/                 # gold_features_1m, gold_features_5m (tumbling windows)
├── ml/
│   ├── features.py               # DuckDB → engineered features + labels
│   ├── train.py                  # LightGBM training + MLflow logging
│   └── evaluate.py               # Threshold tuning (Youden's J), ROC curve, metadata JSON
├── api/
│   └── main.py                   # FastAPI service: /health /metrics /predict
├── dashboard/
│   └── app.py                    # Streamlit live dashboard, 60s auto-refresh
├── scripts/
│   └── smoke_test.py             # 8-check end-to-end health test, exits 0/1
└── models/
    ├── lgbm_direction.pkl         # Serialised LGBMClassifier (joblib)
    └── model_metadata.json        # Optimal threshold + feature list for FastAPI
```

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Local pipeline (Kafka → Spark → Delta → dbt → DuckDB) | Complete |
| 2 | AWS deployment (EC2 producer, S3 Delta sink) | Complete |
| 3 | ML + API + Dashboard (LightGBM, FastAPI, Streamlit) | Complete |
| 4 | Docker + full EC2 deploy | In progress |
