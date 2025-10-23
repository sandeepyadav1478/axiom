# Axiom Platform - Complete Strategic Context & Development Guide
## For Future AI Development Threads

**Critical:** Read this entire document before making ANY code changes.

---

## 🎯 CORE MISSION

Axiom is building an **institutional-grade quantitative finance and investment banking platform** to **directly compete with and surpass** market leaders:

### Primary Competitors
1. **Bloomberg Terminal** - $24,000/year/seat
2. **FactSet Workstation** - $15,000/year/seat  
3. **BlackRock Aladdin** - Enterprise pricing ($50K+/year)
4. **Refinitiv Eikon** - $22,000/year/seat
5. **S&P Capital IQ** - $12,000/year/seat

### Our Advantage
- **Cost:** $0-100/month (99% cost savings)
- **Performance:** 100-1000x faster calculations
- **AI:** DSPy + SGLang (they don't have this)
- **Customization:** 47+ options (expandable to 200+)
- **Modern:** Microservices, webhooks, containers

---

## 🏢 Target Customers (Institutional Only)

### Tier 1: Hedge Funds
- Quantitative trading strategies
- Risk management (VaR, stress testing)
- Portfolio optimization
- Performance attribution
- **Needs:** Speed (<10ms), customizability, AI insights

### Tier 2: Proprietary Trading Firms
- High-frequency trading
- Algorithmic strategies
- Real-time risk monitoring
- Custom model development
- **Needs:** Ultra-low latency, webhook integration, GPU support

### Tier 3: Investment Banks
- M&A deal pipeline
- Due diligence automation
- Valuation analysis
- Regulatory compliance
- **Needs:** Audit trails, workflow automation, AI analysis

### Tier 4: Asset Managers
- Portfolio construction
- Risk budgeting
- Rebalancing automation
- Client reporting
- **Needs:** Multi-portfolio support, compliance, customization

---

## 🏗️ CRITICAL ARCHITECTURAL PRINCIPLES

### 1. Database Architecture (AI-Powered Quality Stack)

**Core Stack: PostgreSQL + Redis + Vector DB**

**Why These Three:**
- **PostgreSQL:** Structured data (prices, trades, portfolios) - Performance
- **Redis:** Real-time caching (<1ms) - Speed
- **Vector DB:** AI semantic search, RAG, similarity - **Quality**

**Vector DB is ESSENTIAL for AI Quality:**
- Semantic M&A target discovery (better than keyword search)
- Similar company recommendations (AI-powered)
- Document embeddings for research (SEC filings, news)
- RAG for investment research (superior context)
- This is our competitive advantage over Bloomberg

**PostgreSQL (Structured Data - Performance):**
```sql
-- Prices, trades, fundamentals
CREATE TABLE prices (
    symbol VARCHAR(10),
    date TIMESTAMP,
    open DECIMAL,
    high DECIMAL,
    low DECIMAL,
    close DECIMAL,
    volume BIGINT,
    adjusted_close DECIMAL
);

-- Portfolio positions
CREATE TABLE positions (
    portfolio_id UUID,
    symbol VARCHAR(10),
    quantity DECIMAL,
    cost_basis DECIMAL,
    current_value DECIMAL,
    updated_at TIMESTAMP
);

-- VaR calculations (audit trail)
CREATE TABLE var_calculations (
    id UUID PRIMARY KEY,
    portfolio_id UUID,
    calculation_time TIMESTAMP,
    method VARCHAR(20),
    var_amount DECIMAL,
    confidence_level DECIMAL,
    time_horizon INT,
    parameters JSONB
);
```

**Redis (Real-Time Performance):**
```python
# <1ms reads for speed
redis.set("portfolio:123:var", var_result, ex=60)
redis.set("price:AAPL:latest", price_data, ex=1)
redis.lpush("trades:stream", trade_data)
redis.publish("risk_alerts", alert_message)
```

**Vector DB (AI Quality - ESSENTIAL):**
```python
# Pinecone, Weaviate, or Qdrant
# This is CRITICAL for AI-powered quality

# 1. M&A Target Discovery (Superior to Bloomberg)
similar_targets = vector_db.query(
    vector=target_company_embedding,
    top_k=20,
    filter={"sector": "AI", "revenue": {"$gt": 100e6}}
)
# Returns semantically similar companies Bloomberg would miss

# 2. Investment Research (RAG)
relevant_docs = vector_db.query(
    vector=query_embedding,
    top_k=10,
    namespace="sec_filings"
)
# Better context than keyword search

# 3. Portfolio Similarity
similar_portfolios = vector_db.query(
    vector=portfolio_embedding,
    metadata_filter={"sharpe_ratio": {"$gt": 1.5}}
)
# Find portfolios with similar characteristics
```

**Why Vector DB is Non-Negotiable:**
- **Quality:** Semantic search >>> keyword search
- **AI:** Required for DSPy, RAG, embeddings
- **Competitive Edge:** Bloomberg doesn't have this
- **M&A:** Better target discovery than any competitor
- **Research:** Superior document retrieval

**Graph DB (Optional - Only if Needed):**
- Add only if relationship queries prove critical
- PostgreSQL can handle basic relationships
- Evaluate need after Vector DB implementation

**Final Stack: PostgreSQL + Redis + Vector DB (3 databases)**
- PostgreSQL: Structure & performance
- Redis: Real-time speed
- Vector DB: AI quality (our secret weapon)

### 2. Real-Time Data Architecture (CRITICAL)

**Webhooks (Primary):**
```python
# Market data webhooks (<1ms processing)
@app.post("/webhooks/market-data")
async def handle_market_update(data: MarketDataUpdate):
    # Process in <1ms
    await update_cache(data)
    
    # Trigger calculations if needed
    if should_recalculate_var(data):
        var = await calculate_var_async()  # <10ms
        await check_risk_limits(var)
        await notify_if_breached()

# News/Events webhooks
@app.post("/webhooks/news")  
async def handle_news(event: NewsEvent):
    # AI analysis <100ms
    sentiment = await analyze_sentiment_ai(event)
    impact = await assess_portfolio_impact(sentiment)
    await update_risk_models(impact)
```

**Streaming Data Pipeline:**
```
Exchange API → WebSocket → Kafka Topic → Stream Processor → Database
                             ↓
                      Real-time VaR (<10ms)
                      Risk Monitoring (<1s)
                      Alert System (<100ms)
```

**Technologies:**
- **WebSockets:** Sub-millisecond price updates
- **Server-Sent Events (SSE):** Live dashboard updates
- **Kafka/RabbitMQ:** High-throughput message broker
- **Redis Streams:** Real-time caching
- **gRPC:** Low-latency inter-service (<1ms)

### 3. Microservices Architecture (REQUIRED Design)

**Every algorithm MUST be containerizable:**

```dockerfile
# Example: VaR Service
FROM python:3.13-slim
WORKDIR /app
COPY requirements-var.txt .
RUN pip install -r requirements-var.txt
COPY axiom/models/risk/ ./risk/
EXPOSE 8010
CMD ["uvicorn", "risk.var_api:app", "--host", "0.0.0.0", "--port", "8010"]
```

**Service Mesh:**
```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ VaR Service │   │Portfolio Svc │   │TimeSeries   │
│  (Port 8010)│◄─►│  (Port 8011) │◄─►│  (Port 8012)│
└─────────────┘   └──────────────┘   └─────────────┘
       ▲                  ▲                  ▲
       └──────────────────┴──────────────────┘
                    API Gateway
                    (Port 8000)
```

**Benefits:**
- Independent scaling (scale VaR service only if needed)
- Independent deployment (update GARCH without touching VaR)
- Fault isolation (if ARIMA fails, VaR still works)
- Technology flexibility (rewrite one service in Rust/C++ if needed)

---

## 💎 EXTREME CUSTOMIZABILITY REQUIREMENTS

### Current: 47+ Options
### Target: 200+ Options for Institutional Control

**Every aspect must be configurable:**

**VaR Configuration (13 current → 30 target):**
```bash
# Basic
VAR_DEFAULT_CONFIDENCE_LEVEL=0.95
VAR_DEFAULT_TIME_HORIZON_DAYS=1
VAR_MONTE_CARLO_SIMULATIONS=10000

# Advanced (to be added)
VAR_BOOTSTRAP_METHOD=percentile|bca|abc
VAR_KERNEL_DENSITY=gaussian|epanechnikov
VAR_TAIL_INDEX_ESTIMATION=hill|pickands
VAR_EXTREME_VALUE_METHOD=pot|bm
VAR_COPULA_TYPE=gaussian|student|clayton
VAR_COVARIANCE_ESTIMATOR=sample|shrinkage|robust
VAR_GPU_ACCELERATION=true
VAR_PRECISION=float64|float32
```

**Portfolio Configuration (34 current → 80 target):**
```bash
# Already implemented
PORTFOLIO_RISK_FREE_RATE=0.02
PORTFOLIO_MAX_CONCENTRATION=0.3
PORTFOLIO_TRANSACTION_COST=0.001

# To be added
PORTFOLIO_SLIPPAGE_MODEL=fixed|square_root|market_impact
PORTFOLIO_REBALANCING_ALGORITHM=threshold|calendar|tolerance_band
PORTFOLIO_TAX_LOT_METHOD=fifo|lifo|hifo|spec_id
PORTFOLIO_DIVIDEND_REINVESTMENT=true
PORTFOLIO_CURRENCY_HEDGING=true
PORTFOLIO_ESG_SCORE_MINIMUM=7.0
PORTFOLIO_CARBON_INTENSITY_MAX=100
```

**Time Series Configuration (to be added):**
```bash
# ARIMA
ARIMA_AUTO_SELECT_METHOD=aic|bic|hqic
ARIMA_SEASONAL_DETECTION=auto|manual
ARIMA_OUTLIER_TREATMENT=winsorize|trim|interpolate

# GARCH
GARCH_INNOVATION_DISTRIBUTION=normal|student|ged|skewed
GARCH_MEAN_MODEL=constant|ar|arx
GARCH_VARIANCE_TARGET=true

# EWMA
EWMA_DECAY_ADAPTIVE=true
EWMA_VOLATILITY_SCALING=true
```

### Customization Layers (All Required)

**Layer 1: Algorithm Selection**
- Choose which VaR method
- Choose which optimizer
- Choose which forecaster

**Layer 2: Parameter Tuning**
- Every parameter exposed
- Sensible defaults
- Expert mode available

**Layer 3: Computational Settings**
- CPU/GPU selection
- Parallel workers
- Memory limits
- Precision tradeoffs

**Layer 4: Business Rules**
- Risk limits
- Sector constraints
- ESG requirements
- Regulatory compliance

---

## ⚡ PERFORMANCE REQUIREMENTS

### Non-Negotiable Performance Targets

| Operation | Maximum Latency | Target | Why |
|-----------|----------------|--------|-----|
| VaR Calculation | 10ms | <5ms | HFT needs sub-10ms |
| Portfolio Optimization | 100ms | <50ms | Real-time rebalancing |
| Market Data Retrieval | 50ms | <20ms | Webhook processing |
| Database Query | 10ms | <5ms | Real-time dashboards |

### Scalability Requirements

**Current (Development):**
- 100 portfolios
- 50 assets/portfolio
- 10 concurrent users

**Near-Term (6 months):**
- 10,000 portfolios
- 1,000 assets/portfolio
- 100 concurrent users

**Enterprise (12 months):**
- 100,000 portfolios
- 10,000 assets/portfolio
- 1,000 concurrent users
- 100,000 API calls/second

---

## 🎓 ALGORITHM SELECTION PHILOSOPHY

### "Best-in-Class Only" Principle

**We DON'T implement:**
- Outdated algorithms
- Poor-performing methods
- Redundant variations
- Academic-only models

**We DO implement:**
- Industry standard (Markowitz, Black-Scholes)
- Best performance (GARCH for volatility)
- Most customizable (Hierarchical Risk Parity)
- AI-enhanced (DSPy optimized)
- Production-proven (RiskMetrics)

### Quality Over Quantity

**Bloomberg:** 100+ mediocre optimizers  
**Axiom:** 6-8 elite optimizers, each with:
- 20+ configuration options
- Multiple solver engines
- GPU acceleration
- Custom constraints
- Real-time capable

**Example: Portfolio Optimization**

✅ **Include:**
- Markowitz (1952, proven)
- Black-Litterman (1992, industry standard)
- Risk Parity (modern, used by Bridgewater)
- HRP (2016, cutting-edge)
- CVaR optimization (tail risk focus)
- VaR-constrained (regulatory)

❌ **Exclude:**
- Mean-variance without constraints (impractical)
- Equal weighting (trivial, already included as baseline)
- Momentum-only (too simple)
- Academic-only methods without production use

---

## 🔧 CODE DESIGN STANDARDS

### Mandatory Requirements for ALL Code

**1. Containerization-Ready**
```python
# ✅ Good: Minimal dependencies, clear interface
class VaRCalculator:
    """
    Standalone VaR calculator.
    Can run in separate container.
    Dependencies: numpy, scipy only.
    """
    def calculate_var(self, ...): # REST API ready
        pass

# ❌ Bad: Tight coupling, hard to containerize
class System:
    def __init__(self):
        self.var = ...
        self.portfolio = ...
        self.data = ...
    # Everything entangled
```

**2. Configuration-Driven**
```python
# ✅ Good: Every parameter configurable
from axiom.config.settings import settings

class Model:
    def __init__(self):
        self.param1 = settings.model_param1
        self.param2 = settings.model_param2
        # All from environment

# ❌ Bad: Hardcoded values
class Model:
    def __init__(self):
        self.param = 0.95  # Hardcoded!
```

**3. Performance-Optimized**
```python
# ✅ Good: Vectorized, <10ms
import numpy as np
var = portfolio_value * np.percentile(returns, 5)  # <1ms

# ❌ Bad: Loop, slow
var = 0
for r in returns:
    # Slow iteration
```

**4. API-First**
```python
# ✅ Good: Can expose as REST/gRPC
def calculate_var(portfolio_value: float, 
                  returns: np.ndarray,
                  confidence: float) -> VaRResult:
    # Serializable input/output
    return VaRResult(var_amount=..., ...)

# ❌ Bad: Complex objects, not serializable
def calculate_var(portfolio_obj, config_obj):
    # Can't easily expose as API
```

---

## 🗄️ DATABASE STRATEGY (DETAILED)

### Why Multi-Database?

**PostgreSQL:** Best for structured time-series data
- Prices, trades, positions
- Fast range queries
- ACID compliance
- Proven at scale

**Vector DB:** Required for AI/semantic search
- Company embeddings
- Similar M&A target discovery
- Document search (SEC filings)
- Better than SQL for similarity

**Graph DB:** Optimal for relationships
- M&A networks
- Supply chain dependencies
- Risk contagion analysis
- Correlation clustering

**Redis:** Essential for real-time
- Market data cache (<1ms)
- Session state
- Rate limiting
- Pub/sub for live updates

### Implementation Priority

**Phase 1 (Immediate):**
1. PostgreSQL setup
2. Redis caching
3. Schema design

**Phase 2 (Next 2 months):**
1. Vector DB for M&A search
2. Graph DB for relationships

**Phase 3 (6 months):**
1. Advanced querying
2. Data lake integration
3. Historical archive

---

## 🚀 REAL-TIME DATA REQUIREMENTS

### Critical: Sub-Second Data Delivery

**Primary: Webhooks** (Push-based, fastest)
```python
# Exchange sends data directly to us
POST /webhooks/market-data
{
  "symbol": "AAPL",
  "price": 175.43,
  "timestamp": "2025-10-23T00:00:00.123Z",
  "volume": 1000
}

# We process in <1ms
# Update database
# Recalculate VaR (<10ms)
# Check limits (<1ms)
# Send alerts if needed (<100ms)
# Total: <112ms end-to-end
```

**Secondary: WebSockets** (Streaming)
```python
# Persistent connection
websocket.connect("wss://exchange.com/stream")

# Continuous data flow
async for message in websocket:
    price_update = parse(message)  # <1ms
    await update_position(price_update)  # <5ms
    await recalculate_metrics()  # <10ms
```

**Tertiary: Polling** (Fallback only)
```python
# Only if webhooks/websockets unavailable
# Poll every 100ms (slow but works)
```

### Message Queue Architecture

**Kafka/RabbitMQ for High Throughput:**
```
Producer: Market Data API
   ↓
Kafka Topic: market.prices (partitioned by symbol)
   ↓
Consumers:
  - VaR Calculator (processes AAPL, MSFT)
  - Portfolio Updater (processes GOOGL, AMZN)
  - Risk Monitor (processes all for alerts)

Throughput: 100,000+ messages/second
Latency: <5ms producer to consumer
```

---

## 🔬 MICROSERVICES DEEP DIVE

### Every Component Must Be:

**1. Independently Deployable**
```yaml
# docker-compose.yml
services:
  var-service:
    image: axiom-var:latest
    ports: ["8010:8010"]
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://...
    deploy:
      replicas: 3  # Auto-scale
      resources:
        limits: {cpus: '2', memory: '4G'}
        
  portfolio-service:
    image: axiom-portfolio:latest
    ports: ["8011:8011"]
    # Independent lifecycle
```

**2. Minimal Dependencies**
```python
# VaR service requirements:
numpy>=1.24.0
scipy>=1.10.0
fastapi>=0.100.0
# That's it! No unnecessary dependencies
```

**3. API-First Interface**
```python
# FastAPI for each service
from fastapi import FastAPI
app = FastAPI()

@app.post("/api/v1/var/calculate")
async def calculate_var(request: VaRRequest) -> VaRResponse:
    # Inputs: JSON
    # Outputs: JSON
    # Can call from any language
```

**4. Observable**
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

var_calculations = Counter('var_calculations_total')
var_latency = Histogram('var_calculation_seconds')

@var_latency.time()
def calculate_var(...):
    var_calculations.inc()
    # Calculation
```

---

## 📊 TESTING REQUIREMENTS

### 100% Test Coverage Mandatory

**Unit Tests:**
```python
def test_var_parametric():
    """Test Parametric VaR with known inputs."""
    result = ParametricVaR.calculate(...)
    assert result.var_amount == pytest.approx(expected)
    assert result.method == VaRMethod.PARAMETRIC
```

**Integration Tests:**
```python
def test_var_with_real_data():
    """Test VaR with actual market data."""
    prices = fetch_yahoo_finance("AAPL", "1y")
    var = calculate_var(prices)
    assert var.var_amount > 0
    assert var.calculation_time_ms < 10
```

**Performance Tests:**
```python
def test_var_performance_benchmark():
    """VaR must calculate in <10ms."""
    start = time.time()
    var = calculate_var(portfolio, returns)
    elapsed = (time.time() - start) * 1000
    assert elapsed < 10, f"VaR took {elapsed}ms, must be <10ms"
```

**Load Tests:**
```python
def test_concurrent_var_calculations():
    """Handle 1000 concurrent VaR calculations."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(calculate_var, ...) for _ in range(1000)]
        results = [f.result() for f in futures]
    
    assert all(r.var_amount > 0 for r in results)
    assert max_latency < 100  # Even under load
```

---

## 🎯 DEVELOPMENT WORKFLOW

### Before Writing Any Code

**1. Check Strategic Alignment:**
- Is this algorithm best-in-class?
- Does it beat Bloomberg/FactSet?
- Is it used by institutions?

**2. Design for Containers:**
- Minimal dependencies?
- Clean interface?
- API-ready?

**3. Configuration-First:**
- All parameters exposed?
- Environment variables?
- Runtime adjustable?

**4. Performance Target:**
- What's acceptable latency?
- Benchmark against competitors?
- GPU acceleration beneficial?

### Code Review Checklist

- [ ] Containerizable (separate service possible)
- [ ] Configurable (20+ parameters exposed)
- [ ] Performant (<10ms for critical paths)
- [ ] Tested (unit + integration + performance)
- [ ] Documented (API + usage + architecture)
- [ ] Observable (metrics + logging)
- [ ] Secure (input validation + audit trails)
- [ ] Type-safe (full typing with Pydantic)

---

## 📈 CURRENT STATUS (October 2025)

### Completed ✅

**Quantitative Finance:**
- VaR Models: 3 methods, <10ms, containerizable
- Portfolio Optimization: 6 methods, 8 strategies, <100ms
- Time Series: ARIMA, GARCH, EWMA, working with real data
- Financial Data: 8 providers, 2 FREE unlimited

**Investment Banking:**
- M&A Workflows: 11 complete workflows
- GitHub Actions: 6 automated pipelines
- Deal Pipeline: End-to-end automation

**Infrastructure:**
- Tests: 114/114 passing (100%)
- Configuration: 47+ options
- Documentation: Complete
- Real Data Integration: Working

### Next Priorities 📋

**Immediate (This Week):**
1. PostgreSQL schema design
2. Redis caching layer
3. Webhook handlers skeleton
4. Performance benchmarking

**Short-Term (2-3 Months):**
1. Vector DB integration
2. Graph DB setup
3. Microservices deployment
4. Real-time streaming

**Medium-Term (6-12 Months):**
1. Derivatives pricing
2. Credit risk models
3. Alternative data
4. Enterprise SSO

---

## 🎓 CRITICAL REMINDERS FOR AI THREADS

### Always Remember:

1. **We're competing with Bloomberg ($24K/year)**
   - Everything must be better
   - Performance is non-negotiable
   - Cost advantage is our weapon

2. **Institutional-grade only**
   - No shortcuts
   - No "good enough"
   - Production-ready from day one

3. **Microservices from day one**
   - Every module containerizable
   - Clean interfaces
   - Independent deployment

4. **Extreme customizability**
   - 200+ config options target
   - Every parameter exposed
   - Multiple profiles

5. **Real-time capable**
   - Webhooks > Polling
   - <1s end-to-end
   - Sub-10ms critical paths

6. **Database-driven**
   - PostgreSQL for structure
   - Vector for AI
   - Graph for relationships
   - Redis for speed

7. **Best algorithms only**
   - Don't implement everything
   - Implement the top 10% perfectly

---

## 📚 REQUIRED READING

Before starting ANY development:

1. [`docs/STRATEGIC_VISION.md`](STRATEGIC_VISION.md) - Overall strategy
2. [`docs/architecture/MICROSERVICES_ARCHITECTURE_ANALYSIS.md`](architecture/MICROSERVICES_ARCHITECTURE_ANALYSIS.md) - Architecture decisions
3. [`docs/FINAL_PROJECT_STATUS.md`](FINAL_PROJECT_STATUS.md) - Current status
4. This document (MASTER_CONTEXT.md) - Complete context

---

## 🎯 SUCCESS DEFINITION

**We succeed when:**
- Hedge funds choose Axiom over Bloomberg
- <10ms VaR is the standard
- 200+ configuration options available
- Every algorithm is containerized
- PostgreSQL + Vector + Graph DBs integrated
- Webhooks deliver data in <50ms
- We're 100-1000x faster than competitors
- We're 99% cheaper than competitors
- We have features Bloomberg doesn't (AI, DSPy)

**This is not a hobby project. This is a commercial platform to dominate institutional finance.**

---

**Document Version:** 1.0  
**Last Updated:** October 23, 2025  
**Status:** Active Development - Production Path  
**Next Review:** When adding major new components

**All future AI development must align with these principles. No exceptions.**