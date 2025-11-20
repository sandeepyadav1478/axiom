# Data Pipeline Architecture & Workflow

## 🏗️ Current Architecture

### Why Only 1 Container?

Currently, we have **1 unified data ingestion pipeline** that handles:
- Real-time market data ingestion
- Multi-database storage (PostgreSQL + Redis + Neo4j)
- Continuous operation (60-second cycles)

This is a **monolithic pipeline design** - one container does everything.

### Alternative: Multi-Container Pipeline Architecture

If you want **multiple specialized pipelines**, we can create:

```yaml
services:
  # 1. Real-time price ingestion (current)
  realtime-prices:
    container_name: axiom-pipeline-realtime
    environment:
      PIPELINE_TYPE: realtime_prices
      SYMBOLS: AAPL,MSFT,GOOGL,TSLA,NVDA
      INTERVAL: 60

  # 2. Historical data backfill
  historical-data:
    container_name: axiom-pipeline-historical
    environment:
      PIPELINE_TYPE: historical_data
      SYMBOLS: AAPL,MSFT,GOOGL
      LOOKBACK_DAYS: 365

  # 3. Company fundamentals
  fundamentals:
    container_name: axiom-pipeline-fundamentals
    environment:
      PIPELINE_TYPE: fundamentals
      SYMBOLS: AAPL,MSFT,GOOGL
      INTERVAL: 3600  # Daily

  # 4. News & sentiment
  news-sentiment:
    container_name: axiom-pipeline-news
    environment:
      PIPELINE_TYPE: news_sentiment
      SOURCES: alpha_vantage,finnhub
      INTERVAL: 300  # 5 minutes

  # 5. Options data
  options-chain:
    container_name: axiom-pipeline-options
    environment:
      PIPELINE_TYPE: options
      SYMBOLS: SPY,QQQ,AAPL,MSFT
      INTERVAL: 60
```

---

## 🔄 Current Pipeline Workflow

### Overview
```
┌─────────────────────────────────────────────────────────┐
│         LIGHTWEIGHT DATA INGESTION PIPELINE             │
│              (axiom-pipeline-ingestion)                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │   Every 60 Seconds:    │
            │   Run Ingestion Cycle   │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 1. Fetch Market Data   │
            │    (yfinance API)      │
            │    - AAPL, MSFT, etc.  │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 2. Store in PostgreSQL │
            │    Table: price_data   │
            │    Columns: symbol,    │
            │    timestamp, OHLCV    │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 3. Cache in Redis      │
            │    Key: price:{symbol} │
            │    TTL: 60 seconds     │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 4. Update Neo4j        │
            │    Node: Stock         │
            │    Properties: price,  │
            │    last_updated        │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 5. Log Metrics         │
            │    - Symbols processed │
            │    - Records stored    │
            │    - Errors            │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │ 6. Sleep 60 seconds    │
            │    (await next cycle)  │
            └────────────────────────┘
                         │
                         └──── Loop back to step 1
```

---

## 📋 Detailed Step-by-Step Workflow

### Step 1: Initialize (One-time, on container start)
```python
pipeline = LightweightPipeline()

# Connects to:
├─ PostgreSQL: postgresql://axiom:****@postgres:5432/axiom_db
├─ Redis:      redis://redis:6379 (with password)
└─ Neo4j:      bolt://neo4j:7687 (with auth)

# Creates tables if not exist:
└─ price_data (id, symbol, timestamp, open, high, low, close, volume, source)
```

### Step 2: Continuous Loop (Every 60 seconds)
```python
while True:
    # === INGESTION CYCLE START ===
    
    # For each symbol (AAPL, MSFT, GOOGL, TSLA, NVDA):
    for symbol in symbols:
        
        # A. Fetch from yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        latest = hist.iloc[-1]  # Most recent price
        
        # B. Store in PostgreSQL
        price_record = PriceData(
            symbol=symbol,
            timestamp=now(),
            open=latest['Open'],
            high=latest['High'],
            low=latest['Low'],
            close=latest['Close'],
            volume=latest['Volume'],
            source='yfinance'
        )
        session.add(price_record)
        session.commit()
        
        # C. Cache in Redis (fast access)
        redis.hset(
            f"price:{symbol}:latest",
            {'close': latest['Close'], 'timestamp': now()}
        )
        redis.expire(f"price:{symbol}:latest", 60)
        
        # D. Update Neo4j graph
        neo4j.run("""
            MERGE (s:Stock {symbol: $symbol})
            SET s.last_price = $price,
                s.last_updated = $timestamp
        """, symbol=symbol, price=latest['Close'])
    
    # === INGESTION CYCLE END ===
    
    # Log metrics
    logger.info(f"Cycle complete: {processed}/{total}")
    
    # Wait 60 seconds
    await asyncio.sleep(60)
```

---

## 🎯 Data Flow Architecture

### Input Sources
```
┌─────────────────┐
│  Data Sources   │
├─────────────────┤
│ 1. yfinance     │ ← Currently active (free)
│ 2. Polygon.io   │ ← Ready (API key in .env)
│ 3. Alpha Vantage│ ← Ready (6 API keys in .env)
│ 4. Finnhub      │ ← Ready (API key in .env)
│ 5. FMP          │ ← Ready (API key in .env)
└─────────────────┘
```

### Storage Targets (Multi-Database)
```
┌──────────────────────────────────────────────────┐
│              STORAGE ARCHITECTURE                │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────┐    ┌───────────────┐       │
│  │  PostgreSQL    │    │    Redis      │       │
│  ├────────────────┤    ├───────────────┤       │
│  │ • price_data   │    │ • Latest      │       │
│  │ • fundamentals │    │   prices      │       │
│  │ • time series  │    │ • <1ms access │       │
│  │ • ACID         │    │ • TTL: 60s    │       │
│  └────────────────┘    └───────────────┘       │
│                                                  │
│  ┌────────────────┐    ┌───────────────┐       │
│  │    Neo4j       │    │  ChromaDB     │       │
│  ├────────────────┤    ├───────────────┤       │
│  │ • Stock nodes  │    │ • Embeddings  │       │
│  │ • Sectors      │    │ • Semantic    │       │
│  │ • Relationships│    │   search      │       │
│  │ • Graph queries│    │ • Similarity  │       │
│  └────────────────┘    └───────────────┘       │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## ⚙️ Current Pipeline Configuration

### Environment Variables
```bash
# What symbols to track
SYMBOLS=AAPL,MSFT,GOOGL,TSLA,NVDA

# How often to run (seconds)
PIPELINE_INTERVAL=60

# Which data source
DATA_SOURCE=yfinance  # Free, unlimited

# Database connections (from .env)
POSTGRES_HOST=postgres
POSTGRES_USER=axiom
POSTGRES_PASSWORD=axiom_secure_2024
REDIS_HOST=redis
NEO4J_URI=bolt://neo4j:7687
```

### Execution Pattern
```
Container starts
    ↓
Initialize databases connections
    ↓
Enter infinite loop:
    ├─ Fetch data for 5 symbols
    ├─ Store in PostgreSQL
    ├─ Cache in Redis
    ├─ Update Neo4j
    ├─ Log metrics
    ├─ Sleep 60 seconds
    └─ Repeat
```

---

## 🚀 Scaling to Multiple Pipelines

### Option 1: Add More Services to docker-compose.yml

```yaml
services:
  # Current: Real-time prices
  realtime-prices:
    container_name: axiom-pipeline-realtime
    ...

  # NEW: Historical backfill
  historical-backfill:
    container_name: axiom-pipeline-historical
    build:
      dockerfile: axiom/pipelines/Dockerfile.historical
    environment:
      SYMBOLS: AAPL,MSFT,GOOGL,TSLA,NVDA,SPY,QQQ
      LOOKBACK_YEARS: 5
      INTERVAL: 3600  # Run once per hour
    ...

  # NEW: Fundamentals scraper
  fundamentals:
    container_name: axiom-pipeline-fundamentals
    build:
      dockerfile: axiom/pipelines/Dockerfile.fundamentals
    environment:
      SYMBOLS: AAPL,MSFT,GOOGL
      INTERVAL: 86400  # Daily
    ...

  # NEW: Options chain
  options-chain:
    container_name: axiom-pipeline-options
    build:
      dockerfile: axiom/pipelines/Dockerfile.options
    environment:
      SYMBOLS: SPY,QQQ,AAPL
      INTERVAL: 300  # 5 minutes
    ...
```

### Option 2: Horizontal Scaling (Multiple Instances)

```yaml
services:
  # Scale by symbol groups
  ingestion-tech:
    environment:
      SYMBOLS: AAPL,MSFT,GOOGL,NVDA,META
  
  ingestion-finance:
    environment:
      SYMBOLS: JPM,GS,MS,BAC,C
  
  ingestion-energy:
    environment:
      SYMBOLS: XOM,CVX,COP,SLB
```

### Option 3: Kubernetes Deployment (Future)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: axiom-pipeline-ingestion
spec:
  replicas: 3  # Multiple instances
  selector:
    matchLabels:
      app: axiom-pipeline
  template:
    spec:
      containers:
      - name: ingestion
        image: pipelines-data-ingestion:latest
        env:
        - name: SYMBOLS
          value: "AAPL,MSFT,GOOGL,TSLA,NVDA"
```

---

## 📊 Current Single-Container Justification

### Why 1 Container is Sufficient Now:

1. **Simplicity**: Easier to manage, debug, monitor
2. **Resource Efficient**: One container handles 5 symbols easily
3. **Low Volume**: 5 symbols × 60s interval = low load
4. **Unified Workflow**: Same pattern for all symbols
5. **Cost Effective**: Minimal compute resources needed

### When to Scale to Multiple Containers:

1. **High Volume**: Tracking 100+ symbols
2. **Different Frequencies**: Some 1-min, some 5-min, some hourly
3. **Different Sources**: Mixing free + paid APIs with rate limits
4. **Resource Isolation**: GPU pipelines vs CPU pipelines
5. **Fault Isolation**: Critical symbols in separate containers

---

## 🔮 Future Pipeline Architecture

### Proposed Multi-Pipeline Design:

```
┌─────────────────────────────────────────────────────────────┐
│                  PIPELINE ORCHESTRATOR                      │
│            (Manages all pipeline containers)                │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────────┐
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  REALTIME     │  │  HISTORICAL  │  │ FUNDAMENTALS │  │   OPTIONS    │
│   PRICES      │  │   BACKFILL   │  │   SCRAPER    │  │    CHAIN     │
├───────────────┤  ├──────────────┤  ├──────────────┤  ├──────────────┤
│ • 5 symbols   │  │ • 100 symbols│  │ • 50 companies│  │ • 20 symbols │
│ • 60s interval│  │ • 5 years    │  │ • Daily      │  │ • 5min       │
│ • yfinance    │  │ • Polygon    │  │ • Alpha Vant │  │ • Polygon    │
└───────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │              │
        └─────────────────┼─────────────────┴──────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │  STORAGE LAYER       │
              │  (4 Databases)       │
              ├──────────────────────┤
              │ • PostgreSQL         │
              │ • Redis              │
              │ • Neo4j              │
              │ • ChromaDB           │
              └──────────────────────┘
```

---

## 📖 Current Workflow Detailed Explanation

### Container: `axiom-pipeline-ingestion`

**Purpose**: Continuous real-time market data ingestion

**Technology Stack**:
- Python 3.13
- SQLAlchemy (PostgreSQL ORM)
- redis-py (Redis client)
- neo4j-python-driver (Neo4j client)
- yfinance (Yahoo Finance API)

**Execution Flow**:

#### Phase 1: Initialization (Once)
```python
1. Connect to PostgreSQL
   └─ Create price_data table if not exists
   
2. Connect to Redis
   └─ Test connection with ping
   
3. Connect to Neo4j
   └─ Verify authentication

4. Log initialization status
   └─ Report which databases connected
```

#### Phase 2: Continuous Loop (Forever)
```python
Loop every 60 seconds:
    
    For each symbol in [AAPL, MSFT, GOOGL, TSLA, NVDA]:
        
        Step A: Fetch Data
        ├─ Call yfinance API
        ├─ Request 1-day history
        └─ Extract latest OHLCV
        
        Step B: Validate
        ├─ Check data not empty
        ├─ Validate price format
        └─ Skip if invalid
        
        Step C: Transform
        ├─ Convert to Decimal (precision)
        ├─ Add timestamp
        └─ Add source tag
        
        Step D: Store in PostgreSQL
        ├─ Create PriceData record
        ├─ Add to session
        └─ Commit transaction
        
        Step E: Cache in Redis
        ├─ Set key: price:{symbol}:latest
        ├─ Store: {close, timestamp}
        └─ Expire: 60 seconds
        
        Step F: Update Neo4j
        ├─ MERGE Stock node
        ├─ SET last_price, last_updated
        └─ Build graph relationships
        
        Step G: Log Success
        └─ "✅ AAPL: $150.25"
    
    End loop
    
    Step H: Report Metrics
    ├─ Symbols processed: 5/5
    ├─ Records stored: 5
    ├─ Records cached: 5
    └─ Errors: []
    
    Step I: Sleep
    └─ await asyncio.sleep(60)
    
    Repeat from Step A
```

---

## 🔍 Data Examples

### PostgreSQL Storage
```sql
SELECT * FROM price_data ORDER BY timestamp DESC LIMIT 5;

| id | symbol | timestamp           | open   | high   | low    | close  | volume     | source   |
|----|--------|---------------------|--------|--------|--------|--------|------------|----------|
| 1  | AAPL   | 2025-11-15 03:00:00 | 150.20 | 151.50 | 149.80 | 150.25 | 52000000   | yfinance |
| 2  | MSFT   | 2025-11-15 03:00:00 | 380.50 | 382.00 | 379.00 | 381.75 | 25000000   | yfinance |
| 3  | GOOGL  | 2025-11-15 03:00:00 | 140.10 | 141.00 | 139.50 | 140.80 | 18000000   | yfinance |
```

### Redis Cache
```bash
redis-cli> HGETALL price:AAPL:latest
1) "close"
2) "150.25"
3) "timestamp"
4) "2025-11-15T03:00:00.123456"

redis-cli> TTL price:AAPL:latest
(integer) 45  # Seconds until expiration
```

### Neo4j Graph
```cypher
MATCH (s:Stock {symbol: 'AAPL'})
RETURN s.symbol, s.last_price, s.last_updated

┌─────────┬─────────────┬──────────────────────┐
│ symbol  │ last_price  │ last_updated         │
├─────────┼─────────────┼──────────────────────┤
│ 'AAPL'  │ 150.25      │ 2025-11-15T03:00:00  │
└─────────┴─────────────┴──────────────────────┘
```

---

## 🎓 Why This Design?

### Single Container Advantages:
1. **Atomic Operations**: All or nothing - data consistency
2. **Simplified Monitoring**: One container to watch
3. **Unified Logging**: All logs in one place
4. **Lower Overhead**: Minimal resource usage
5. **Easier Debugging**: Single failure point

### When to Use Multiple Containers:
1. **Different Data Sources**: Some need authentication, some free
2. **Different Intervals**: Real-time (1s) vs batch (1h)
3. **Resource Isolation**: CPU vs GPU pipelines
4. **Fault Tolerance**: Critical vs non-critical data
5. **Scaling**: 1000+ symbols need parallel processing

---

## 💡 Recommendations

### For Current Scale (5 symbols):
✅ **Keep 1 container** - perfectly adequate

### To Add More Pipelines:

1. **Create new pipeline scripts**:
```bash
axiom/pipelines/
├── lightweight_data_ingestion.py     # Current (real-time)
├── historical_backfill.py            # New (batch)
├── fundamentals_scraper.py           # New (daily)
└── options_chain_ingestion.py        # New (5-min)
```

2. **Add services to docker-compose.yml**:
```yaml
services:
  realtime-ingestion:    # Current
  historical-backfill:   # Add this
  fundamentals-scraper:  # Add this
  options-chain:         # Add this
```

3. **Deploy**:
```bash
docker compose -f axiom/pipelines/docker-compose.yml up -d
```

---

## 📞 Quick Reference

### Current Pipeline Status
```bash
docker ps --filter "name=pipeline"
# axiom-pipeline-ingestion   Up X minutes (healthy)
```

### View Workflow in Action
```bash
docker logs -f axiom-pipeline-ingestion
# Shows each step: Fetch → Store → Cache → Update → Log
```

### Container Configuration
- **File**: `axiom/pipelines/docker-compose.yml`
- **Script**: `axiom/pipelines/lightweight_data_ingestion.py`
- **Networks**: axiom_network + database_axiom_network
- **Restart Policy**: unless-stopped

---

**The pipeline IS running. It's a single unified container by design, not a limitation.**