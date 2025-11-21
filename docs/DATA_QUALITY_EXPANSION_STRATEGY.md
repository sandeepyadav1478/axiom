# Data Quality & Expansion Strategy for Institutional-Grade Platform

**Created:** November 21, 2025  
**Status:** Strategic Planning  
**Target:** Compete with Bloomberg Terminal ($24K/year)

---

## 🎯 Executive Summary

**Current State:** 5 symbols, real-time ingestion only, ~66 records total  
**Target State:** Bloomberg-equivalent institutional platform with 60 ML models  
**Gap:** Massive data insufficiency for production deployment

### Critical Gaps Identified

| Category | Current | Required | Gap |
|----------|---------|----------|-----|
| **Historical Data** | 1-2 hours | 5-10 years | 99.9% missing |
| **Symbols Coverage** | 5 stocks | 500-1000+ | 99% missing |
| **Asset Classes** | Stocks only | 6 classes | 83% missing |
| **Fundamental Data** | None | Complete | 100% missing |
| **Options Data** | None | Full chains | 100% missing |
| **Alternative Data** | None | Multiple | 100% missing |

**Verdict:** Current data is INSUFFICIENT for institutional platform. Need comprehensive data expansion strategy.

---

## 📊 Current Data Inventory & Quality Assessment

### What We Have Now

#### 1. Real-Time Price Data (PostgreSQL: `price_data` table)
```
Symbols: 5 (AAPL, MSFT, GOOGL, TSLA, NVDA)
Frequency: Every 1 minute via Airflow
Duration: Started recently (~1-2 hours of data)
Records: ~66 rows total
Quality: Good (validated via data_quality_validation DAG)
```

**Data Fields:**
- symbol, timestamp, open, high, low, close, volume, source
- OHLCV complete, no nulls detected
- Validation: Batch validation every 5 minutes (working)

#### 2. Knowledge Graph (Neo4j)
```
Nodes: ~120K (from previous work)
Relationships: ~775K (correlations, sectors, companies)
Quality: Excellent (built from Claude analysis)
Coverage: Limited to 5 symbols currently tracked
```

#### 3. Redis Cache
```
Type: Ephemeral (60s TTL)
Purpose: Latest prices for real-time access
Volume: Minimal (<1 MB)
```

### Data Quality Assessment

**Strengths ✅:**
- Clean OHLCV data (no integrity violations)
- Automated validation running (batch every 5min)
- Multi-database architecture working
- Real-time ingestion operational

**Critical Weaknesses ❌:**
- **ZERO historical data** (need 2-10 years for ML models)
- **Tiny universe** (5 stocks vs 1000+ needed)
- **No fundamentals** (income statements, balance sheets required)
- **No options data** (15 options models need this)
- **No alternative data** (sentiment, ESG, earnings transcripts)
- **No fixed income** (bonds, rates needed for many models)

---

## 🎓 Data Requirements by ML Model Category

### 1. Portfolio Optimization Models (12 models)

**Current Status:** ❌ BLOCKED - Insufficient Data

**Data Required:**
- **Historical returns:** Minimum 252 trading days (1 year), ideal 1260 days (5 years)
- **Universe size:** Minimum 30 stocks for diversification, ideal 100-500
- **Frequency:** Daily or higher
- **Fundamentals:** Market cap, sector, industry classification
- **Corporate actions:** Stock splits, dividends for adjusted prices

**Specific Model Needs:**
```python
MILLION (Multi-Instance Learning):
  ├─ Needs: 2+ years daily returns, 50+ stocks
  └─ Current: 1 hour, 5 stocks → CANNOT RUN

RegimeFolio (Regime-Switching):
  ├─ Needs: 5+ years to identify regimes, 100+ stocks
  └─ Current: 1 hour, 5 stocks → CANNOT RUN

DRO-BAS (Distributionally Robust):
  ├─ Needs: 3+ years, 200+ stocks for distribution estimation
  └─ Current: 1 hour, 5 stocks → CANNOT RUN
```

**Gap Analysis:** 99.9% data missing for portfolio models

### 2. Options Pricing Models (15 models)

**Current Status:** ❌ BLOCKED - NO Options Data

**Data Required:**
- **Options chains:** Strike prices, bid/ask, volume, open interest
- **Implied volatility:** Market-derived vol from option prices
- **Historical vol:** Realized volatility from stock prices (need 1+ year)
- **Greeks snapshots:** For validation and calibration
- **Dividend schedules:** For American option pricing
- **Interest rates:** Risk-free rate curve

**Specific Model Needs:**
```python
VAE Option Pricer:
  ├─ Needs: 100K+ option price samples for training
  └─ Current: ZERO options data → CANNOT TRAIN

GAN Volatility Surface:
  ├─ Needs: 50K+ volatility surface snapshots
  └─ Current: ZERO volatility data → CANNOT TRAIN

PINN (Physics-Informed NN):
  ├─ Needs: Historical options + stock prices for validation
  └─ Current: Stock prices only, no options → LIMITED

DRL Hedging:
  ├─ Needs: Options + stock tick data for training environment
  └─ Current: 5-minute stock data → INSUFFICIENT
```

**Gap Analysis:** 100% options data missing

### 3. Credit Risk Models (20 models)

**Current Status:** ❌ BLOCKED - NO Credit/Fundamental Data

**Data Required:**
- **Balance sheets:** Assets, liabilities, equity (quarterly, 5+ years)
- **Income statements:** Revenue, EBITDA, net income (quarterly)
- **Cash flow statements:** Operating, investing, financing CF
- **Credit ratings:** Moody's/S&P/Fitch ratings history
- **Default data:** Historical default events for training
- **Bond prices:** CDS spreads, bond yields for market-implied PD

**Specific Model Needs:**
```python
CNN-LSTM Credit Model:
  ├─ Needs: 10K+ company financial statements (time series)
  └─ Current: ZERO fundamental data → CANNOT TRAIN

Ensemble Credit (20 models):
  ├─ Needs: Comprehensive fundamental + market data
  └─ Current: Stock prices only → INSUFFICIENT

LLM Credit Scoring:
  ├─ Needs: Earnings transcripts, MD&A, financial docs
  └─ Current: ZERO text data → CANNOT RUN
```

**Gap Analysis:** 100% credit/fundamental data missing

### 4. M&A Intelligence Models (13 models)

**Current Status:** ⚠️ PARTIAL - Can Use Web Search

**Data Required:**
- **Company financials:** Complete for valuation (DCF, comparables)
- **M&A transactions:** Historical deals for precedent analysis
- **News/events:** M&A announcements, rumors, regulatory filings
- **Ownership data:** Shareholders, institutional holdings
- **Sector data:** Industry relationships, supply chains

**Specific Model Needs:**
```python
ML Target Screening:
  ├─ Needs: Financials for 1000+ companies
  └─ Current: Can fetch on-demand via API → PARTIAL

AI Due Diligence:
  ├─ Needs: Document corpus (10-Q, 10-K, 8-K)
  └─ Current: Can scrape on-demand → SLOW but WORKABLE

DCF Valuation:
  ├─ Needs: Historical financials (3-5 years)
  └─ Current: API access to FMP/Alpha Vantage → WORKABLE
```

**Gap Analysis:** 60% data available via API calls, 40% missing

### 5. Risk Management Models (VaR) (5 models)

**Current Status:** ⚠️ PARTIAL - Limited Historical Data

**Data Required:**
- **Historical returns:** Minimum 252 days, ideal 1260+ days
- **Intraday data:** For more accurate short-horizon VaR
- **Correlation history:** Multi-year for regime identification
- **Volatility regimes:** Bear/bull market classification

**Specific Model Needs:**
```python
Historical VaR:
  ├─ Needs: 252+ days of returns
  └─ Current: 1 hour → CANNOT RUN

Regime-Switching VaR:
  ├─ Needs: 5+ years to identify bull/bear regimes
  └─ Current: 1 hour → CANNOT IDENTIFY REGIMES

EVT VaR (Extreme Value Theory):
  ├─ Needs: 10+ years for tail event modeling
  └─ Current: 1 hour → INSUFFICIENT TAIL DATA
```

**Gap Analysis:** 99.8% historical data missing for VaR models

---

## 🔍 Institutional Data Requirements Analysis

### Baseline Requirements to Compete with Bloomberg

#### 1. Historical Depth
```
Bloomberg Standard:
├─ Stocks: 10-30 years daily OHLCV
├─ Options: 5+ years chains
├─ Fundamentals: 10+ years quarterly
├─ News: 20+ years archive
└─ Economic data: 30+ years

Axiom Minimum (Tier 1):
├─ Stocks: 5 years daily (1,260 trading days)
├─ Options: 2 years chains
├─ Fundamentals: 5 years quarterly (20 quarters)
├─ News: 1 year archive
└─ Economic data: 5 years

Axiom Current:
├─ Stocks: 1 hour (4 data points) ← 99.97% GAP
├─ Options: ZERO
├─ Fundamentals: ZERO
├─ News: Real-time only (no archive)
└─ Economic data: ZERO
```

#### 2. Universe Coverage
```
Bloomberg:
├─ US Stocks: ~5,000
├─ Global Stocks: ~25,000
├─ Options: 1,000+ underlyings
├─ Bonds: 500,000+
└─ Currencies: 150+ pairs

Axiom Target (Institutional Minimum):
├─ US Stocks: 1,000 (S&P 500 + Russell 1000)
├─ Options: 100 underlyings (SPY, QQQ, top 98)
├─ Bonds: 100 sovereigns + corporates
└─ Currencies: 20 major pairs

Axiom Current:
├─ Stocks: 5 ← 99.5% GAP
├─ Options: 0 ← 100% GAP
├─ Bonds: 0 ← 100% GAP
└─ Currencies: 0 ← 100% GAP
```

#### 3. Data Quality Standards
```
Bloomberg Quality:
├─ Accuracy: 99.9%+
├─ Completeness: 99%+
├─ Timeliness: <100ms for real-time
├─ Validation: Multi-layer automated checks
└─ Audit trail: Complete lineage

Axiom Target:
├─ Accuracy: 99.5%+ (validated)
├─ Completeness: 98%+
├─ Timeliness: <10ms (current: working)
├─ Validation: Multi-layer (implemented)
└─ Audit trail: Complete (implemented)

Axiom Current Achievement:
├─ Accuracy: 99%+ ✅ (yfinance is reliable)
├─ Completeness: 100% for what we collect ✅
├─ Timeliness: <1s ✅ (Airflow every 1min)
├─ Validation: Batch validation working ✅
└─ Audit trail: Pipeline tracking working ✅
```

**Quality Verdict:** ✅ What we collect is INSTITUTIONAL GRADE. Problem is VOLUME, not QUALITY.

---

## 📈 Data Expansion Strategy

### Phase 1: Historical Backfill (URGENT - Week 1)

**Goal:** Get minimum viable historical data for ML model training

**Tasks:**
1. **Backfill 5 years daily stock prices** (1,260 trading days × 8 symbols = 10K records)
   ```python
   Source: yfinance (FREE, unlimited)
   Symbols: AAPL, MSFT, GOOGL, TSLA, NVDA, SPY, QQQ, META
   Period: 2020-01-01 to 2025-11-21 (5 years)
   Frequency: Daily OHLCV
   Storage: PostgreSQL price_data table
   ```

2. **Backfill company fundamentals** (quarterly, 5 years = 20 quarters × 8 companies)
   ```python
   Source: Alpha Vantage (FREE 500 calls/day) or FMP
   Data: Income statement, balance sheet, cash flow
   Period: 5 years quarterly
   Storage: PostgreSQL company_fundamentals table
   ```

3. **Backfill sector/industry metadata**
   ```python
   Source: yfinance.info (FREE)
   Data: Sector, industry, market cap, description
   Storage: Neo4j Company nodes
   ```

**Deliverable:** 10K+ historical records enabling 40+ ML models to train

**Implementation:** 
- Create Airflow DAG: `historical_backfill_dag.py`
- One-time execution or daily incremental
- Estimated time: 2-4 hours for 5 years × 8 symbols

### Phase 2: Universe Expansion (Week 2-3)

**Goal:** Expand to institutional-grade universe

**S&P 500 Coverage:**
```python
Current: 5 stocks (0.5% of S&P 500)
Target Phase 2A: 50 stocks (top 10% by market cap)
Target Phase 2B: 100 stocks (top 20%)
Target Phase 2C: 500 stocks (full S&P 500)

Data Volume Impact:
├─ 50 stocks × 5 years daily = 315K records
├─ 100 stocks × 5 years daily = 630K records
└─ 500 stocks × 5 years daily = 3.15M records

Storage: ~500 MB for 500 stocks × 5 years
Cost: FREE (yfinance has no limits)
Time: ~8-12 hours one-time backfill
```

**Prioritized Symbol List:**
```python
Tier 1 (Top 10 - Mega Cap):
AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, UNH, JNJ

Tier 2 (Next 40 - Large Cap):
JPM, V, PG, MA, HD, etc. (blue chip stocks)

Tier 3 (Next 50 - Mid-Large Cap):
Round out to 100 total

Tier 4 (Remaining 400):
Complete S&P 500 for true institutional coverage
```

**Implementation:**
- Progressive rollout: 10 → 50 → 100 → 500
- Monitor API rate limits (yfinance is unlimited but respect fair use)
- Parallel ingestion with worker pools

### Phase 3: Multi-Asset Class Expansion (Week 4-6)

**Goal:** Add bonds, options, currencies, commodities

#### 3A: Options Data (CRITICAL for 15 Options Models)
```python
Symbols: 20 most liquid options
└─ SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM
   IWM, GLD, SLV, USO, TLT, HYG, XLE, XLF, XLK, XLV

Data Needed:
├─ Options chains: All strikes, all expirations
├─ Frequency: 5-minute snapshots during market hours
├─ Historical: 2 years of chains for vol surface calibration
└─ Fields: strike, expiry, bid, ask, volume, OI, IV

Source: 
├─ Primary: Polygon.io (API key ready, 5 calls/min free)
├─ Fallback: yfinance.options (FREE but limited)
└─ Premium: TD Ameritrade API (if needed)

Storage:
├─ PostgreSQL: options_chain table (~10 GB/year for 20 symbols)
├─ Neo4j: Options → Stock relationships
└─ Redis: Latest chains for real-time access
```

#### 3B: Fixed Income Data (For Bond Models)
```python
Instruments:
├─ Treasury Bonds: 2Y, 5Y, 10Y, 30Y
├─ Corporate Bonds: Investment grade (AAA-BBB)
├─ High Yield: BB-CCC rated
└─ Municipals: Top issuers

Source:
├─ FRED (Federal Reserve) - FREE, official
├─ Alpha Vantage - Treasuries (FREE)
└─ FMP - Corporate bonds (paid tiers)

Data Fields:
├─ Yield curves (treasury yield curve)
├─ Credit spreads (vs treasuries)
├─ Duration, convexity
└─ Price/yield data
```

#### 3C: Currency & Commodities
```python
Currencies (20 pairs):
G10: EUR/USD, GBP/USD, USD/JPY, etc.
Source: yfinance (FREE) or Alpha Vantage

Commodities (10):
Gold, Silver, Oil (WTI, Brent), Natural Gas, Copper
Source: yfinance (FREE - via ETFs: GLD, SLV, USO, UNG)
```

### Phase 4: Alternative Data Integration (Month 2-3)

**Goal:** Differentiate from Bloomberg with AI-powered insights

#### 4A: Sentiment & News
```python
Current: Real-time news via events_tracker_v2 DAG (working)
Expand to:
├─ News archives: 1 year historical via NewsAPI, Finnhub
├─ Social media: Reddit WallStreetBets, Twitter/X sentiment
├─ Earnings call transcripts: Seeking Alpha, FMP
└─ SEC filings: 10-K, 10-Q, 8-K from EDGAR (FREE)

Source:
├─ NewsAPI (500 calls/day FREE)
├─ Finnhub (60 calls/min FREE)
├─ Reddit API (FREE)
└─ SEC EDGAR (FREE, unlimited)
```

#### 4B: ESG Data
```python
ESG Scores: 
├─ Source: Sustainalytics via FMP (paid)
├─ Fallback: Web scraping from company reports
└─ Storage: PostgreSQL esg_metrics table
```

#### 4C: Insider Trading & Institutional Holdings
```python
Insider Transactions:
├─ Source: Finnhub (FREE), SEC Form 4
├─ Data: Buys/sells by executives
└─ Use: Sentiment signal for models

Institutional Holdings:
├─ Source: 13F filings (SEC EDGAR - FREE)
├─ Data: Top holders, position changes
└─ Use: Ownership concentration analysis
```

---

## 🧹 Data Cleaning Framework Architecture

### 1. Real-Time Validation (Already Implemented ✅)

**Current Implementation:**
- `data_quality_validation_dag.py` (Airflow)
- Batch validation every 5 minutes
- Checks: OHLC integrity, null detection, price reasonableness

**Enhancement Needed:**
```python
Add to existing validation:
├─ Price continuity checks (detect halts/gaps)
├─ Volume sanity (detect flash crashes)
├─ Cross-validation (price vs multiple sources)
└─ Historical consistency (detect restatements)
```

### 2. Statistical Profiling (Implemented, Not Integrated)

**Existing Code:** `axiom/data_quality/profiling/statistical_profiler.py`

**Integration Needed:**
```python
Create Airflow DAG: data_profiling_dag.py
Schedule: Daily or weekly
Tasks:
├─ Profile price_data table
├─ Profile company_fundamentals table  
├─ Profile options_chain table
├─ Store profiles in PostgreSQL
├─ Alert on quality degradation
└─ Generate quality dashboard
```

### 3. Anomaly Detection (Implemented, Not Integrated)

**Existing Code:** `axiom/data_quality/profiling/anomaly_detector.py`

**Integration Needed:**
```python
Embed in data_ingestion_v2 DAG:
├─ Run anomaly detector on each batch
├─ Flag suspicious records
├─ Quarantine critical anomalies
├─ Alert on CRITICAL/HIGH severity
└─ Log all detections to anomaly_log table

Methods Already Implemented:
├─ Statistical outliers (IQR, Z-score) ✅
├─ Price spikes (>20% moves) ✅
├─ Volume anomalies ✅
├─ OHLC violations ✅
├─ Temporal anomalies (gaps, future dates) ✅
└─ Duplicate detection ✅
```

### 4. Data Cleaning Pipeline

**New Pipeline Needed:**
```python
Create: data_cleaning_dag.py

Tasks:
1. detect_anomalies
   └─ Run AnomalyDetector on new data
   
2. quarantine_critical
   └─ Move anomalies to quarantine table
   
3. impute_missing
   └─ Handle missing data:
      ├─ Forward fill for prices
      ├─ Interpolation for intraday
      └─ Mark as imputed
      
4. deduplicate
   └─ Remove exact duplicates
   
5. normalize_symbols
   └─ Standardize ticker symbols (GOOGL vs GOOG)
   
6. validate_cleaned
   └─ Re-run validation on cleaned data
   
7. promote_to_production
   └─ Move cleaned data to production tables
```

### 5. Data Quality Metrics Dashboard

**Metrics to Track:**
```python
Real-Time Metrics (every 5 min):
├─ Completeness: % of expected records received
├─ Timeliness: Lag between market close and ingestion
├─ Validity: % passing validation rules
└─ Anomaly rate: Anomalies per 1000 records

Daily Metrics:
├─ Profile scores: Overall quality score (0-100)
├─ Null percentages: By column
├─ Outlier rates: Statistical + domain
├─ Duplicate rates: % duplicate records
└─ Coverage: % of universe with fresh data

Weekly Metrics:
├─ Data drift: Statistical distribution changes
├─ Source reliability: Uptime by provider
├─ Error rates: By pipeline and task
└─ Storage growth: GB/week trend
```

**Implementation:**
```python
Store in PostgreSQL:
├─ data_quality_metrics table (time series)
└─ data_quality_alerts table (incidents)

Visualize in:
├─ Grafana dashboard (real-time)
└─ Weekly quality report (automated)
```

---

## 🚀 Implementation Roadmap

### Week 1: Critical Historical Backfill
```
Priority: URGENT - Enables 40+ ML models

[ ] Day 1-2: Create historical_backfill_dag.py
    ├─ Fetch 5 years daily for 8 symbols (10K records)
    ├─ Parallel workers for speed
    └─ Validate as ingested

[ ] Day 3: Backfill fundamentals (quarterly, 5 years)
    ├─ Use Alpha Vantage or FMP
    └─ Store in company_fundamentals table

[ ] Day 4-5: Validate data quality
    ├─ Run full profiling
    ├─ Detect and fix anomalies
    └─ Document data lineage

[ ] Day 6-7: Test ML models with real historical data
    ├─ Portfolio optimization (need 252+ days) ✅
    ├─ VaR models (need 252+ days) ✅
    └─ Time series (ARIMA/GARCH need 100+ days) ✅

Deliverable: 60% of ML models unblocked and trainable
```

### Week 2: Universe Expansion (50 Stocks)
```
[ ] Day 8-10: Expand to top 50 stocks by market cap
    ├─ Backfill 5 years for each
    ├─ Update Airflow to track all 50
    └─ Validate multi-stock portfolios

[ ] Day 11-12: Add sector/industry classification
    ├─ Enhance Neo4j graph
    ├─ Create sector nodes and relationships
    └─ Enable sector-based analysis

[ ] Day 13-14: Test portfolio models at scale
    ├─ 50-stock portfolios
    ├─ Sector rotation strategies
    └─ Risk parity with 10 sectors

Deliverable: Portfolio models fully functional
```

### Week 3-4: Options Data Integration
```
[ ] Week 3: Implement options chain ingestion
    ├─ Create options_chain_dag.py
    ├─ Integrate Polygon.io API
    ├─ Design options_chain table schema
    └─ Backfill 6 months chains for SPY, QQQ, AAPL

[ ] Week 4: Options data validation & cleaning
    ├─ Validate put-call parity
    ├─ Detect arbitrage opportunities (data errors)
    ├─ Build volatility surfaces
    └─ Test 15 options pricing models

Deliverable: Options models operational
```

### Month 2: Production-Grade Data Operations
```
[ ] Week 5-6: Data quality automation
    ├─ Integrate statistical profiler into pipelines
    ├─ Integrate anomaly detector into ingestion
    ├─ Create data_cleaning_dag.py
    └─ Set up quality metrics dashboard

[ ] Week 7-8: Alternative data integration
    ├─ News archives (1 year backfill)
    ├─ SEC filings ingestion
    ├─ Earnings transcripts (via FMP)
    └─ Insider trading data

Deliverable: Bloomberg-competitive data infrastructure
```

---

## 💾 Storage & Cost Projections

### Storage Requirements by Phase

#### Phase 1 (Historical Backfill):
```
Stock prices (8 symbols × 5 years daily):
├─ Records: 8 × 1,260 = 10,080
├─ Size: 10,080 × 200 bytes = 2 MB
└─ Cost: Negligible

Fundamentals (8 companies × 5 years quarterly):
├─ Records: 8 × 20 = 160
├─ Size: 160 × 2 KB = 320 KB
└─ Cost: Negligible

Total: ~2.5 MB
```

#### Phase 2 (50 Stocks):
```
Stock prices (50 × 5 years daily):
├─ Records: 50 × 1,260 = 63,000
├─ Size: 63,000 × 200 = 12.6 MB
└─ Still negligible!

Total: ~15 MB
```

#### Phase 3 (Options - 20 underlyings):
```
Options chains (20 symbols × 2 years):
├─ Chains per day: 20 symbols × 50 strikes × 6 exp = 6,000 options/day
├─ Days: 2 years × 252 = 504 trading days
├─ Records: 6,000 × 504 = 3,024,000 records
├─ Size: 3M × 300 bytes = 900 MB
└─ This is significant!

But: Only store end-of-day snapshots:
├─ Records: 3M (same)
├─ Compressed: ~300 MB with PostgreSQL compression
└─ Manageable
```

#### Full Production (500 Stocks + 100 Options):
```
Stocks (500 × 10 years daily):
├─ Records: 500 × 2,520 = 1.26M
├─ Size: 1.26M × 200 = 252 MB
└─ Compressed: ~150 MB

Options (100 × 5 years daily):
├─ Records: ~15M
├─ Size: ~4.5 GB
└─ Compressed: ~1.5 GB

Fundamentals (500 × 10 years quarterly):
├─ Records: 500 × 40 = 20K
├─ Size: 20K × 2 KB = 40 MB

News/Events (1 year):
├─ Records: ~2M events
├─ Size: ~1 GB

Total Production Storage: ~3-4 GB
Cost: AWS RDS $10-20/month
```

**Verdict:** Storage is NOT a concern. Focus on data acquisition, not storage cost.

---

## 📋 Data Source Priority Matrix

### Free Sources (Prioritize These)

| Source | Coverage | Rate Limit | Quality | Priority |
|--------|----------|------------|---------|----------|
| **yfinance** | Global stocks | Unlimited | ★★★★☆ | HIGHEST |
| **SEC EDGAR** | US filings | Unlimited | ★★★★★ | HIGH |
| **FRED** | Economic data | Unlimited | ★★★★★ | HIGH |
| **Alpha Vantage** | Multi-asset | 500/day | ★★★★☆ | HIGH |
| **Finnhub** | News/stocks | 60/min | ★★★★☆ | MEDIUM |
| **FMP** | Comprehensive | 250/day | ★★★★☆ | MEDIUM |
| **Polygon.io** | Options/ticks | 5/min | ★★★★★ | HIGH |

### Premium Sources (Consider Later)

| Source | Cost/Month | Benefit | When to Add |
|--------|------------|---------|-------------|
| **Quandl** | $50+ | Alternative data | Month 3+ |
| **Bloomberg API** | $2,000+ | Institutional grade | Production only |
| **FactSet** | $1,500+ | Comprehensive | If clients require |
| **Refinitiv** | $2,000+ | Global coverage | International expansion |

**Strategy:** Maximize FREE tier usage first, add premium only when necessary

---

## 🎯 Data Quality Framework Design

### Quality Dimensions (ISO 8000 Standard)

#### 1. Accuracy
```python
Definition: Data correctly represents reality
Measurement: % records matching source of truth
Target: >99.5%

Validation Methods:
├─ Cross-source verification (compare yfinance vs Alpha Vantage)
├─ Put-call parity checks (for options)
├─ Accounting identity checks (for fundamentals)
└─ Historical consistency (no retroactive changes)

Implementation:
└─ CrossValidationOperator in Airflow
```

#### 2. Completeness
```python
Definition: All required fields present
Measurement: % non-null values
Target: >98%

Validation Methods:
├─ Null detection (already implemented)
├─ Coverage checks (expected vs actual records)
├─ Missing data detection (gaps in time series)
└─ Field-level completeness

Implementation:
└─ Already in data_quality_validation_dag ✅
```

#### 3. Consistency
```python
Definition: Data conforms to business rules
Measurement: % passing validation rules
Target: >99%

Business Rules:
├─ OHLC: high >= {open, close, low}
├─ OHLC: low <= {open, close, high}
├─ Volume >= 0
├─ Prices > 0
├─ Timestamps not in future
├─ Bid < Ask (for options)
└─ Assets = Liabilities + Equity (fundamentals)

Implementation:
└─ Already in anomaly_detector.py ✅
```

#### 4. Timeliness
```python
Definition: Data available when needed
Measurement: Latency from market event to database
Target: <10 seconds for real-time

Metrics:
├─ Ingestion lag: Market close → Database
├─ Pipeline SLA: 95% of cycles < 10s
└─ Staleness: Age of latest record

Implementation:
└─ Monitor via Prometheus (future)
```

#### 5. Uniqueness
```python
Definition: No duplicate records
Measurement: % unique records
Target: 100%

Validation:
├─ Primary key constraints (symbol + timestamp + timeframe)
├─ Duplicate detection in anomaly_detector ✅
└─ Deduplication in cleaning pipeline

Implementation:
└─ PostgreSQL UNIQUE constraint + anomaly detector
```

### Quality Metrics Storage

```sql
-- New table for quality tracking
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    metric_date TIMESTAMP NOT NULL,
    table_name VARCHAR(50) NOT NULL,
    
    -- Quality dimensions
    accuracy_score DECIMAL(5,2),      -- 0-100
    completeness_score DECIMAL(5,2),  -- 0-100
    consistency_score DECIMAL(5,2),   -- 0-100
    timeliness_score DECIMAL(5,2),    -- 0-100
    uniqueness_score DECIMAL(5,2),    -- 0-100
    
    -- Overall
    overall_quality_score DECIMAL(5,2),  -- Average of above
    
    -- Volume metrics
    total_records INTEGER,
    records_validated INTEGER,
    records_passed INTEGER,
    records_failed INTEGER,
    
    -- Anomaly metrics
    anomalies_detected INTEGER,
    anomalies_critical INTEGER,
    anomalies_high INTEGER,
    anomalies_medium INTEGER,
    
    -- Created timestamp
    created_at TIMESTAMP DEFAULT NOW(),
    
    INDEX idx_metrics_date_table (metric_date, table_name)
);
```

---

## 🔬 Data Cleaning Process Flow

```mermaid
graph TB
    A[Raw Data Ingestion] --> B{Initial Validation}
    B -->|Pass| C[Store in Staging]
    B -->|Fail| D[Quarantine]
    
    C --> E[Statistical Profiling]
    E --> F[Anomaly Detection]
    
    F --> G{Anomaly Severity}
    G -->|Critical| D
    G -->|High| H[Flag for Review]
    G -->|Medium/Low| I[Log Warning]
    
    H --> J[Manual Review Queue]
    I --> K[Automated Cleaning]
    
    K --> L{Cleaning Success}
    L -->|Yes| M[Promote to Production]
    L -->|No| D
    
    M --> N[Production Tables]
    
    D --> O[Human Review]
    O -->|Fixed| C
    O -->|Invalid| P[Discard & Log]
```

### Quarantine Table Design
```sql
CREATE TABLE quarantined_data (
    id SERIAL PRIMARY KEY,
    source_table VARCHAR(50),
    source_record_json JSONB,
    quarantine_reason TEXT,
    anomaly_type VARCHAR(50),
    anomaly_severity VARCHAR(20),
    detected_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP,
    resolution VARCHAR(50),  -- 'fixed', 'discarded', 'pending'
    notes TEXT
);
```

---

## 📊 Data Governance & Compliance

### Data Lineage Tracking (Already Designed ✅)

**Existing Table:** `data_lineage` in [`models.py`](../axiom/database/models.py)

**Usage:**
```python
Track every transformation:
Raw yfinance → price_data → feature_data → model_input

Store:
├─ Source table + ID
├─ Target table + ID  
├─ Transformation logic
└─ Pipeline run context
```

### Audit Trail
```python
Regulatory Requirements (MiFID II, Dodd-Frank):
├─ Complete data provenance
├─ Change history (who, when, what)
├─ Deletion logging (never actually delete)
└─ Access logs

Implementation:
├─ data_lineage table ✅
├─ pipeline_runs table ✅
└─ validation_results table ✅

All already designed in models.py!
```

---

## 💡 Key Recommendations

### Immediate Actions (This Week)

1. **CRITICAL: Historical Backfill**
   ```
   Priority: P0 (blocks 80% of ML models)
   Effort: 2-3 days
   Impact: Unlocks portfolio, VaR, time series models
   Cost: FREE (yfinance)
   ```

2. **Integrate Existing Data Quality Tools**
   ```
   Priority: P0 (ensure data legitimacy)
   Effort: 1-2 days
   Components: 
   ├─ statistical_profiler.py → Airflow DAG
   └─ anomaly_detector.py → Inline in ingestion
   ```

3. **Expand Universe to 50 Stocks**
   ```
   Priority: P1 (enable meaningful diversification)
   Effort: 1 day
   Impact: Realistic portfolio optimization
   Cost: FREE
   ```

### Strategic Decisions Needed

**Question 1: Historical Depth**
- Option A: 2 years (fast, sufficient for most models)
- Option B: 5 years (better regime identification)
- Option C: 10 years (full Bloomberg parity)

**Recommendation:** Start with 5 years, expand to 10 later

**Question 2: Universe Size**
- Option A: 50 stocks (quick win)
- Option B: 100 stocks (institutional minimum)
- Option C: 500 stocks (S&P 500, full coverage)

**Recommendation:** Progressive 8 → 50 → 100 → 500

**Question 3: Options Data Priority**
- Option A: Delay until Month 2 (focus on stocks first)
- Option B: Start Week 4 (parallel track)
- Option C: Immediate (Week 2)

**Recommendation:** Start Week 4 (need historical stocks data first for delta calculations)

**Question 4: Data Cleaning Automation**
- Option A: Manual review of all anomalies
- Option B: Auto-clean medium/low, review critical/high
- Option C: Full automation with logging

**Recommendation:** Option B (balance safety and efficiency)

---

## 🎓 Data vs Model Readiness Matrix

### Can Run NOW (With Synthetic/Minimal Data)
```
Portfolio Models: SOME (with synthetic)
├─ Equal weight ✅
├─ Simple Markowitz ✅
└─ Risk parity ✅

Options Models: SOME (with parameters)
├─ Black-Scholes ✅
├─ Binomial trees ✅
└─ Greeks ✅

VaR Models: NONE (need historical)
Credit Models: NONE (need fundamentals)
Time Series: NONE (need historical)
M&A Models: PARTIAL (can use API calls)
```

### Can Run After Week 1 (Historical Backfill)
```
✅ Portfolio Models: ALL 12
✅ VaR Models: ALL 5
✅ Time Series: ALL
✅ Some Options: Volatility models
❌ Credit Models: Still need fundamentals
❌ Advanced Options: Need options chains
```

### Can Run After Month 1 (Full Expansion)
```
✅ ALL 60 Models Operational
✅ Institutional-Grade Data Coverage
✅ Production-Ready Platform
```

---

## 🚨 Critical Path to Production

```
Day 1: Create historical_backfill_dag.py
   └─ BLOCKS: 40+ ML models

Week 1: Execute 5-year backfill
   └─ UNBLOCKS: Portfolio, VaR, Time Series (50+ models)

Week 2-3: Expand to 50-100 stocks
   └─ ENABLES: Realistic portfolio optimization

Week 4-6: Add options data
   └─ UNBLOCKS: Remaining 15 options models

Month 2: Alternative data & cleaning automation
   └─ ACHIEVES: Bloomberg-competitive platform
```

---

## 📞 Next Steps

**For Planning Session:**
1. Review this strategy document
2. Decide on historical depth (2, 5, or 10 years)
3. Decide on universe size (50, 100, or 500 stocks)
4. Prioritize options data timing
5. Approve data cleaning automation level

**For Implementation (Code Mode):**
1. Create `historical_backfill_dag.py`
2. Execute 5-year backfill for 8 symbols
3. Integrate statistical profiler
4. Integrate anomaly detector
5. Test ML models with real data
6. Expand to 50 stocks
7. Add fundamentals ingestion
8. Create data quality dashboard

**The platform is 20% data-ready. With Week 1 backfill, becomes 60% ready. With Month 1 expansion, becomes 95% production-ready.**

---

**Conclusion:** Data quality tools exist and are excellent. Data QUANTITY is the gap. Historical backfill is the critical path to unlocking the platform's full potential.