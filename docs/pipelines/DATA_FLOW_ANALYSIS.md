# 📊 Data Flow & Storage Analysis

## How Data Flows Into Databases

### Real-Time Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                              │
│  yfinance (5 symbols × 60s) → ~5 records/minute              │
└──────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┬─────────────┐
            │             │             │             │
            ▼             ▼             ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │PostgreSQL│  │  Redis   │  │  Neo4j   │  │ChromaDB  │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
         │             │             │             │
    Time-series   Latest     Knowledge    Vector
    Storage       Cache      Graph        Search
```

---

## 📈 Current Data Ingestion Rates

### Pipeline 1: Data Ingestion (60-second cycles)

**Per Cycle**:
```
Symbols: 5 (AAPL, MSFT, GOOGL, TSLA, NVDA)
Frequency: Every 60 seconds
Records per cycle: 5 price records
```

**Per Day**:
```
Cycles per day: 24 hours × 60 min/hour = 1,440 cycles
Records per day: 1,440 × 5 = 7,200 price records
```

**Storage Impact** (PostgreSQL):
```
Per Record Size:
├─ symbol: VARCHAR(20) = 20 bytes
├─ timestamp: TIMESTAMP = 8 bytes
├─ open/high/low/close: NUMERIC(20,6) × 4 = 64 bytes
├─ volume: NUMERIC(20,2) = 12 bytes
├─ source: VARCHAR(50) = 50 bytes
├─ indexes + overhead: ~50 bytes
└─ TOTAL: ~200 bytes per record

Daily Storage:
7,200 records × 200 bytes = 1.44 MB/day = 43.2 MB/month = 518.4 MB/year
```

**Answer**: At current rate, PostgreSQL would take **~23 months to reach 1GB** (price data only)

---

### Pipeline 2: Company Graph (hourly cycles)

**Per Cycle**:
```
Symbols: 8 companies
Frequency: Every 3600 seconds (hourly)
Nodes created: 8 Company + ~3 Sector nodes per cycle
Relationships: ~15-30 (BELONGS_TO, COMPETES_WITH, etc.)
```

**Per Day**:
```
Cycles per day: 24
Company nodes: 8 (but MERGE prevents duplicates, so only created once)
Sector nodes: 3-5 total
Relationships updated: 24 cycles × 20 avg = 480 relationship updates/day
```

**Storage Impact** (Neo4j):
```
Node Size:
├─ Company node: ~500 bytes (all properties)
├─ Sector node: ~200 bytes
├─ Relationship: ~150 bytes (with properties)

Initial Setup (Day 1):
├─ 8 companies × 500 = 4 KB
├─ 5 sectors × 200 = 1 KB
├─ 40 relationships × 150 = 6 KB
└─ TOTAL: ~11 KB

Daily Growth (after initial):
├─ New relationships: ~50 new/day × 150 = 7.5 KB/day
├─ Property updates: Minimal (existing nodes)
└─ TOTAL: ~8 KB/day = 240 KB/month = 2.88 MB/year
```

**Answer**: Neo4j would take **~28 years to reach 1GB** (at current scale)

---

### Pipeline 3: Events Tracker (5-minute cycles)

**Per Cycle**:
```
Symbols: 5 companies
Frequency: Every 300 seconds (5 min)
News items per symbol: ~2-10 (varies)
Event nodes created: ~10-30 per cycle
```

**Per Day**:
```
Cycles per day: 24 hours × 12 cycles/hour = 288 cycles
Events per cycle: ~20 (average)
Events per day: 288 × 20 = 5,760 events
```

**Storage Impact** (PostgreSQL + Neo4j):
```
PostgreSQL (event_log table):
Per event: ~500 bytes (title, description, metadata)
Daily: 5,760 × 500 = 2.88 MB/day = 86.4 MB/month = ~1 GB/year

Neo4j (MarketEvent nodes):
Per event node: ~300 bytes
Daily: 5,760 × 300 = 1.73 MB/day = 51.9 MB/month = 623 MB/year

Combined: ~1.6 GB/year for events
```

**Answer**: Events would fill **1GB in ~7 months** (PostgreSQL + Neo4j combined)

---

### Pipeline 4: Correlation Analyzer (hourly cycles)

**Per Cycle**:
```
Symbols: 5 stocks
Correlations calculated: 5 choose 2 = 10 pairs
Frequency: Every 3600 seconds (hourly)
```

**Per Day**:
```
Cycles per day: 24
Correlation records: 24 × 10 = 240 records/day
```

**Storage Impact** (PostgreSQL + Neo4j):
```
PostgreSQL (correlations table):
Per record: ~100 bytes (symbol1, symbol2, coefficient, period, timestamp)
Daily: 240 × 100 = 24 KB/day = 720 KB/month = 8.64 MB/year

Neo4j (CORRELATED_WITH edges):
Per edge: ~200 bytes (coefficient, explanation from Claude, metadata)
Daily: 240 × 200 = 48 KB/day = 1.44 MB/month = 17.28 MB/year

Combined: ~26 MB/year
```

**Answer**: Correlations would take **~38 years to reach 1GB**

---

## 📊 Total Storage Projections

### Summary Table

| Database | Daily | Monthly | Yearly | Time to 1GB |
|----------|-------|---------|--------|-------------|
| **PostgreSQL** | | | | |
| • Price data | 1.44 MB | 43.2 MB | 518 MB | 23 months |
| • Events log | 2.88 MB | 86.4 MB | 1.04 GB | 12 months |
| • Correlations | 0.024 MB | 0.72 MB | 8.6 MB | 116 months |
| **Subtotal PG** | **4.34 MB** | **130.3 MB** | **1.57 GB** | **~8 months** |
| | | | | |
| **Neo4j** | | | | |
| • Companies/Sectors | 0.008 MB | 0.24 MB | 2.9 MB | 346 months |
| • Events nodes | 1.73 MB | 51.9 MB | 623 MB | 20 months |
| • Correlations edges | 0.048 MB | 1.44 MB | 17.3 MB | 58 months |
| **Subtotal Neo4j** | **1.79 MB** | **53.6 MB** | **643 MB** | **~19 months** |
| | | | | |
| **Redis** | | | | |
| • Cache (TTL 60s) | 0 MB | 0 MB | 0 MB | Never (ephemeral) |
| | | | | |
| **ChromaDB** | | | | |
| • Embeddings | TBD | TBD | TBD | Future feature |
| | | | | |
| **TOTAL** | **~6 MB/day** | **~184 MB/mo** | **~2.2 GB/yr** | **~6 months** |

---

## 🔢 Detailed Calculations

### PostgreSQL Growth Rate

**price_data table**:
```sql
-- Assumptions
5 symbols × 1,440 cycles/day = 7,200 inserts/day

-- Record structure
CREATE TABLE price_data (
    id SERIAL,                          -- 4 bytes
    symbol VARCHAR(20),                 -- 20 bytes
    timestamp TIMESTAMP,                -- 8 bytes
    open NUMERIC(20,6),                 -- 16 bytes
    high NUMERIC(20,6),                 -- 16 bytes
    low NUMERIC(20,6),                  -- 16 bytes
    close NUMERIC(20,6),                -- 16 bytes
    volume NUMERIC(20,2),               -- 12 bytes
    source VARCHAR(50)                  -- 50 bytes
);
-- Base: 158 bytes + indexes (~40 bytes) = ~200 bytes/record

-- Growth rate
Day 1: 7,200 × 200 = 1.44 MB
Month 1: 7,200 × 30 = 216,000 records = 43.2 MB
Year 1: 7,200 × 365 = 2,628,000 records = 525.6 MB

To 1GB: 1GB / 1.44MB = ~694 days ≈ 23 months
```

**market_events table**:
```sql
-- Assumptions
5 symbols × 288 cycles/day × 4 events/symbol-cycle = 5,760 events/day

-- Record structure (larger due to text fields)
CREATE TABLE market_events (
    id SERIAL,                          -- 4 bytes
    symbol VARCHAR(20),                 -- 20 bytes
    title TEXT,                         -- ~200 bytes avg
    publisher VARCHAR(100),             -- 100 bytes
    link TEXT,                          -- ~150 bytes avg
    timestamp TIMESTAMP,                -- 8 bytes
    event_type VARCHAR(50),             -- 50 bytes
    impact_score FLOAT,                 -- 8 bytes
    raw_data JSONB                      -- ~200 bytes
);
-- Base: ~740 bytes compressed to ~500 bytes

-- Growth rate
Day 1: 5,760 × 500 = 2.88 MB
Month 1: 5,760 × 30 = 172,800 events = 86.4 MB
Year 1: 5,760 × 365 = 2,102,400 events = 1.05 GB

To 1GB: 1GB / 2.88MB = ~347 days ≈ 11.5 months
```

---

### Neo4j Growth Rate

**Nodes**:
```
Company nodes:
- Initial creation: 8 nodes × 500 bytes = 4 KB
- Updates: Hourly property updates (no size increase)
- Growth: Only when new companies added (rare)

Sector nodes:
- Initial: 5 nodes × 200 bytes = 1 KB  
- Growth: Minimal (sectors don't change often)

MarketEvent nodes:
- Per day: 5,760 events × 300 bytes = 1.73 MB/day
- Per month: 51.9 MB
- To 1GB: ~577 days ≈ 19 months
```

**Relationships**:
```
BELONGS_TO:
- One-time: 8 companies × 150 bytes = 1.2 KB
- No growth (stable)

COMPETES_WITH:
- Initial: ~20 edges × 150 bytes = 3 KB
- Updates: Intensity scores updated hourly (no size increase)

AFFECTED_BY:
- Per day: 5,760 events × 5 affected companies avg × 150 bytes = 4.32 MB/day
- This is the main growth driver!
- To 1GB: ~231 days ≈ 7.7 months
```

---

## 💾 Storage Optimization Strategies

### 1. Time-Based Archival

**PostgreSQL**:
```sql
-- Archive old price data (keep last 90 days hot)
INSERT INTO price_data_archive 
SELECT * FROM price_data 
WHERE timestamp < NOW() - INTERVAL '90 days';

DELETE FROM price_data 
WHERE timestamp < NOW() - INTERVAL '90 days';

-- Result: Keeps active table at ~13 MB (90 days × 1.44 MB)
```

**Neo4j**:
```cypher
// Archive old events (keep last 30 days)
MATCH (e:MarketEvent)
WHERE e.date < date() - duration('P30D')
SET e:ArchivedEvent
REMOVE e:MarketEvent

// Or delete entirely
MATCH (e:MarketEvent)
WHERE e.date < date() - duration('P90D')
DETACH DELETE e

// Result: Keeps active events at ~52 MB (30 days)
```

### 2. Data Compression

**PostgreSQL**:
- Enable compression: ~50% size reduction
- Yearly storage: 2.2GB → 1.1GB compressed

**Neo4j**:
- Store full text in PostgreSQL
- Keep only references in Neo4j
- Size reduction: ~60%

### 3. Sampling & Aggregation

Instead of storing every 60-second price:
```sql
-- Store 1-minute bars during market hours only
-- Market hours: 6.5 hours × 5 days = 32.5 hours/week
-- Records: 5 symbols × 390 min/day × 5 days = 9,750/week vs 50,400/week
-- Storage savings: ~80%
```

---

## 📊 Realistic Growth Projections

### Conservative Estimate (Current Settings):

**Year 1**:
```
PostgreSQL:
├─ Price data: 525 MB
├─ Events: 1,050 MB
├─ Fundamentals: 50 MB
├─ Correlations: 9 MB
└─ TOTAL: ~1.6 GB

Neo4j:
├─ Companies/Sectors: 3 MB
├─ Events: 623 MB
├─ Relationships: 20 MB
└─ TOTAL: ~650 MB

Redis:
└─ 0 MB (all ephemeral with TTL)

ChromaDB:
└─ TBD (not yet implemented)

COMBINED: ~2.25 GB/year
```

### With Archival (90-day retention):

**Steady State** (after 90 days):
```
PostgreSQL:
├─ Price data: 130 MB (90 days rolling)
├─ Events: 260 MB (90 days rolling)
├─ Archive tables: Growing but compressed
└─ TOTAL active: ~400 MB

Neo4j:
├─ Companies/Sectors: 3 MB (stable)
├─ Events: 52 MB (30 days rolling)
├─ Relationships: 20 MB (pruned monthly)
└─ TOTAL active: ~75 MB

COMBINED active data: ~475 MB (steady state)
```

**Answer with archival**: Never reaches 1GB active data! Stays at ~500MB steady state.

---

## 🚀 Scaling Projections

### If We Scale to 100 Symbols:

**Daily Ingestion**:
```
Price data: 
100 symbols × 1,440 = 144,000 records/day = 28.8 MB/day
Monthly: 864 MB
To 1GB: ~35 days

Events (assuming 5x more news):
28,800 events/day = 14.4 MB/day  
Monthly: 432 MB
To 1GB: ~69 days

COMBINED: ~43 MB/day
To 1GB: ~23 days
```

### If We Add Options Data:

**Options Chain** (per symbol, per day):
```
Strikes per symbol: ~20 (10 calls + 10 puts)
Updates per day: 288 (every 5 minutes during market hours)
Records: 100 symbols × 20 strikes × 288 = 576,000 options/day

Size per option: ~200 bytes
Daily: 576,000 × 200 = 115 MB/day
To 1GB: ~9 days
```

---

## 💡 Real-World Recommendations

### For Production at Scale:

**1. Immediate Optimizations**:
```yaml
# Adjust update frequencies
data-ingestion:
  INTERVAL: 300  # 5 min instead of 60s (saves 80% storage)

events-tracker:
  INTERVAL: 600  # 10 min instead of 5 min (saves 50%)

correlations:
  INTERVAL: 86400  # Daily instead of hourly (saves 95%)
```

**Storage Impact**:
- Price data: 525 MB/year → 105 MB/year (5min bars)
- Events: 1,050 MB/year → 525 MB/year (10min cycles)
- **New total: ~700 MB/year** (fits in 1GB with room to spare)

**2. Enable Compression**:
```sql
-- PostgreSQL
ALTER TABLE price_data SET (
    toast_compression = lz4
);

-- Saves ~40-50% space
-- 700 MB → 350-420 MB/year
```

**3. Implement Partitioning**:
```sql
-- PostgreSQL table partitioning by month
CREATE TABLE price_data_2025_11 PARTITION OF price_data
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');

-- Easy to drop old partitions
DROP TABLE price_data_2024_11;  -- Instantly free space
```

---

## 📈 Current vs Optimized Growth

### Current Settings (No optimization):
```
├─ 1 month: 184 MB
├─ 6 months: 1.1 GB ← Crosses 1GB threshold
├─ 1 year: 2.2 GB
└─ 2 years: 4.4 GB
```

### Optimized (5-min bars + compression + 90-day retention):
```
├─ Steady state: 400-500 MB (never grows beyond this)
├─ Archive: Grows but compressed and can be moved to cold storage
└─ Never reaches 1GB active data
```

---

## 🎯 Storage Recommendations

### Option A: Keep Everything (No Archival)
```
Good for: Research, backtesting, historical analysis
Storage needed: 2-5 GB/year
Cost: Minimal (a few GB is cheap)
Timeline to 1GB: 6-8 months at current scale
```

### Option B: Smart Archival (90-day hot data)
```
Good for: Production trading (only need recent data)
Storage needed: 500 MB steady state
Cost: Minimal
Timeline to 1GB: Never (stays at 500MB)
Archives: Compress and move to S3 ($0.023/GB/month)
```

### Option C: Aggressive Compression
```
Good for: Cost optimization
Storage needed: 200-300 MB/year
Method: 
  - 5-minute bars instead of 1-minute
  - 30-day retention
  - Compression enabled
  - Sample events (not every news item)
```

---

## 🔍 Current Database Sizes

Let me check actual current usage:

```sql
-- PostgreSQL
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Expected now (after ~4 minutes):
price_data: ~2 KB (20 records × 200 bytes)
company_fundamentals: ~1.5 KB (3 companies)
market_events: ~10-50 KB (depends on news volume)
```

```cypher
// Neo4j
CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypes
RETURN nodeCount, relCount, labels, relTypes;

// Expected now:
Nodes: 3-8 (Company + Sector)
Relationships: 0-10 (BELONGS_TO + maybe COMPETES_WITH)
Total size: <100 KB
```

---

## 💰 Cost Implications

### Storage Costs (AWS/Cloud):

**PostgreSQL (RDS)**:
- General Purpose SSD: $0.115/GB/month
- 2GB/year = $0.23/month (negligible)
- With PIOPS: $0.25/GB/month = $0.50/month

**Neo4j (self-hosted)**:
- Storage: Same as PostgreSQL
- Memory: More important (need 2-4GB RAM)
- Cost: $20-40/month for adequate server

**Redis (ElastiCache)**:
- cache.t3.micro: $12/month (ephemeral, no storage cost)

**Total cloud cost**: ~$35-55/month for databases at current scale

---

## 🎓 Key Insights

### 1. Storage is Not a Concern
At current scale (5 symbols), you'll never hit storage limits. The bottleneck is:
- Network bandwidth (fetching data)
- API rate limits
- Compute for analysis

### 2. Data Velocity is Low
```
6 MB/day = 0.25 MB/hour = 4.2 KB/minute = 70 bytes/second
```
This is **extremely low** - a single photo is bigger than a day's worth of data!

### 3. Scaling Changes Everything
If you scale to:
- 100 symbols: 10x growth → 1GB in 3-4 weeks
- 1,000 symbols: 100x growth → 1GB in 3-4 days
- Add options: Another 100x → 1GB in hours

### 4. Optimization Matters at Scale
For 1,000+ symbols:
- **Must** implement archival
- **Must** use sampling (5-min bars minimum)
- **Must** enable compression
- **Should** use partitioning

---

## 📌 Answer to Your Question

**Q: How quickly will 1GB fill up?**

**A: Depends on scale**:

**Current (5 symbols)**:
- PostgreSQL: ~8 months to 1GB
- Neo4j: ~19 months to 1GB
- **Combined: ~6 months to 1GB total**

**With optimization (90-day retention)**:
- Never reaches 1GB (steady state at 400-500MB)

**At 100 symbols**:
- Without optimization: ~23 days
- With optimization: Steady state at 2-3GB

**At 1,000 symbols**:
- Without optimization: 2-3 days
- With optimization: Steady state at 10-15GB
- **Requires**: Archival + compression + partitioning

---

## ✅ Recommendations

For **current 5-symbol scale**:
1. **No action needed** - storage is not a concern
2. Let it run for months without worry
3. Monitor when you scale beyond 20-50 symbols

For **scaling to 100+ symbols**:
1. Implement 90-day archival immediately
2. Use 5-minute bars (not 1-minute)
3. Enable compression
4. Set up automated cleanup

For **production 1,000+ symbols**:
1. **Must have** archival policy
2. **Must have** partitioning
3. Consider time-series database (TimescaleDB instead of plain PostgreSQL)
4. Use object storage (S3) for archives

**Your current setup will comfortably run for 6+ months before hitting 1GB!**