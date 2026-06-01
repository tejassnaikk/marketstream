# MarketStream Engineering Design

## Problem Statement

Build a real-time market data platform that ingests live order book updates and reconstructs a correct Level-2 order book.

## Architecture

Binance WebSocket -> Kafka -> Spark Structured Streaming -> Delta Lake -> Analytics Layer

## Data Source

Binance BTC-USDT depth stream

## Challenges

- Order book reconstruction
- Out-of-order updates
- Snapshot synchronization
- Data persistence
- Scalability

## Future Work

- Feature engineering
- ML evaluation
- API layer
- Dashboard
