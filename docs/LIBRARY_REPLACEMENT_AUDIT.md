# Axiom Platform - Library/Package Replacement Audit

## Executive Summary

**Objective**: Identify all custom code that could be replaced with existing packages, containers, or MCP servers to reduce maintenance burden.

**Audit Date**: 2025-10-24
**Auditor**: Axiom Development Team

---

## Audit Methodology

For each custom implementation, we ask:
1. Does a Python package exist that does this? (PyPI search)
2. Does a Docker container exist? (Docker Hub search)
3. Does an MCP server exist? (GitHub search)
4. Is the existing solution actively maintained?
5. Would integration be simpler than custom code?

---

## Category 1: Quantitative Finance Models

### Current Custom Implementation
- Options pricing (6 models, ~4,000 lines)
- Credit risk (7 models, ~5,000 lines)
- Portfolio optimization (2 models, ~1,400 lines)
- Time series (3 models, ~1,900 lines)
- Market microstructure (6 models, ~4,600 lines)
- Fixed income (11 models, ~6,000 lines)
- M&A models (6 models, ~4,000 lines)

### ✅ Already Using External Libraries
- `scipy` - Core numerical operations
- `numpy` - Array operations
- `pandas` - Data manipulation
- `statsmodels` - Time series (ARIMA, etc.)
- `QuantLib` - Bond pricing (integrated)
- `PyPortfolioOpt` - Portfolio optimization (integrated)
- `arch` - GARCH models (integrated)

### 🔄 Potential Replacements

**Options Pricing**:
- Package: `py_vollib` - Volatility and Greeks calculations
- Package: `mibian` - Options pricing library
- Keep our implementation: ✅ Already optimized, well-tested

**Credit Risk**:
- Package: `creditrisk` - Basic credit risk models
- Keep our implementation: ✅ Basel III compliance, comprehensive

**Fixed Income**:
- Package: `QuantLib` - Already integrated ✅
- Action: Expand QuantLib usage, reduce custom bond code

**Time Series**:
- Package: `pmdarima` - Auto-ARIMA (better than our implementation)
- Package: `arch` - Already integrated for GARCH ✅
- Action: Replace custom ARIMA with pmdarima

**Market Microstructure**:
- No good existing packages
- Keep our implementation: ✅ Unique, Bloomberg-level

**Recommendation**: 
- ✅ Keep most quant models (unique implementations)
- 🔄 Replace ARIMA with `pmdarima`
- ✅ Already using best external libraries where they exist

---

## Category 2: Data Providers

### Current Custom Implementation
- Alpha Vantage provider (~180 lines)
- FMP provider (~200 lines)
- Finnhub provider (~150 lines)
- Yahoo Finance provider (~200 lines)
- IEX Cloud provider (~150 lines)
- Tavily provider (~100 lines)

### ✅ External MCPs Available
- **OpenBB MCP** - INTEGRATED ✅ (replaces 5 providers)
- Polygon MCP - Already using ✅
- Yahoo Finance MCP - Already using ✅
- Firecrawl MCP - Already using ✅

### 🔄 To Replace
- All 5 REST providers → Use OpenBB MCP (DONE)
- Tavily → Keep (specialized search, no MCP alternative)

**Recommendation**: 
- ✅ Migration complete to OpenBB MCP
- 🗑️ Remove deprecated REST provider files

---

## Category 3: Database & Storage

### Current Custom Implementation
- PostgreSQL models (~600 lines)
- Vector store (~700 lines)
- Database integrations (~650 lines)
- Session management (~100 lines)

### 🔄 Potential Replacements

**ORM**:
- Package: `sqlalchemy` - Already using ✅
- Package: `sqlmodel` - Better integration with Pydantic
- Action: Consider migrating to SQLModel

**Migrations**:
- Package: `alembic` - Already using ✅

**Connection Pooling**:
- Package: `asyncpg` - For PostgreSQL
- Package: `aiopg` - Alternative
- Action: Integrate asyncpg for better async support

**Vector Database**:
- Container: `chromadb/chroma:latest` - Lightweight vector DB
- Container: `qdrant/qdrant:latest` - High-performance vector DB
- Container: `weaviate/weaviate:latest` - Production vector DB
- Action: Use container instead of custom vector_store.py

**Recommendation**:
- 🔄 Replace custom vector_store.py with ChromaDB container
- 🔄 Add asyncpg for better PostgreSQL performance
- ✅ Keep SQLAlchemy models (industry standard)

---

## Category 4: Real-Time Streaming

### Current Custom Implementation
- WebSocket manager (~400 lines)
- Market data streamer (~500 lines)
- Event processor (~350 lines)
- Portfolio tracker (~600 lines)
- Risk monitor (~450 lines)

### 🔄 Potential Replacements

**Message Queue**:
- Container: `redis/redis-stack:latest` - Includes TimeSeries, JSON, Search
- Container: `apache/kafka:latest` - Enterprise message queue
- Container: `rabbitmq:management` - Message broker
- Package: `celery` - Distributed task queue
- Action: Use Redis Stack container (includes all features we need)

**Stream Processing**:
- Package: `faust-streaming` - Kafka stream processing
- Package: `streamz` - Real-time stream processing
- Action: Consider faust for complex event processing

**WebSocket**:
- Package: `websockets` - Already using ✅
- Package: `socket.io` - Alternative with more features
- Keep current: ✅ Works well

**Recommendation**:
- ✅ Keep streaming infrastructure (already using external libraries)
- 🔄 Use Redis Stack container instead of plain Redis

---

## Category 5: API Layer

### Current Custom Implementation
- FastAPI routes (~4,000 lines)
- Pydantic models (~700 lines)
- Auth system (~400 lines)
- Rate limiting (~150 lines)

### ✅ Already Using Best External Solutions
- `fastapi` - Best Python API framework ✅
- `uvicorn` - Best ASGI server ✅
- `pydantic` - Best validation library ✅
- `python-jose` - Best JWT library ✅
- `slowapi` - Best rate limiting ✅

### Potential Enhancements

**API Gateway**:
- Container: `kong/kong-gateway:latest` - Enterprise API gateway
- Container: `traefik:latest` - Modern reverse proxy
- Action: Consider Kong for production API management

**Authentication**:
- Container: `keycloak/keycloak:latest` - Enterprise SSO
- Package: `authlib` - OAuth2/OpenID Connect
- Keep current: ✅ JWT is sufficient for now

**Recommendation**:
- ✅ API layer is already optimal (using best packages)
- 🔄 Consider Kong gateway for production (optional)

---

## Category 6: AI/ML Infrastructure

### Current Custom Implementation
- DSPy modules (~600 lines)
- Model optimization (~400 lines)

### 🔄 Potential Replacements

**ML Framework**:
- Package: `dspy-ai` - Already using ✅
- Container: `ray-project/ray:latest` - Distributed ML
- Container: `mlflow/mlflow:latest` - ML lifecycle
- Action: Use MLflow container for experiment tracking

**Model Serving**:
- Container: `bentoml/bentoml:latest` - Model serving
- Container: `seldon/seldon-core:latest` - K8s-native serving
- Container: `ray-project/ray-serve:latest` - Ray Serve
- Action: Use BentoML container for production model serving

**Recommendation**:
- 🔄 Add MLflow container (already planned in MCP)
- 🔄 Add BentoML for model serving
- ✅ Keep DSPy integration (no better alternative)

---

## Category 7: Testing & CI/CD

### Current Custom Implementation
- Test helpers (~100 lines)
- Test runners

### ✅ Already Using External Solutions
- `pytest` - Test framework ✅
- `pytest-cov` - Coverage ✅
- `pytest-asyncio` - Async testing ✅
- `pytest-benchmark` - Performance tests ✅

### 🔄 Additional Tools to Use

**CI/CD**:
- Use GitHub Actions (built-in, free) instead of custom scripts
- Use pre-commit hooks package instead of custom quality checks
- Use tox for multi-environment testing

**Code Quality**:
- Container: `pycqa/pylint:latest`
- Container: `cytopia/black:latest`
- Action: Use containers in CI/CD instead of local tools

**Recommendation**:
- ✅ Testing is already optimal
- 🔄 Move to GitHub Actions for CI/CD
- 🔄 Use Docker containers for linting in CI

---

## Summary of Findings

### ✅ Already Optimal (Keep As-Is)
1. Quantitative models - Unique implementations, well-tested
2. API layer - Using FastAPI (best available)
3. Testing - Using pytest ecosystem (industry standard)
4. Streaming - Using websockets, redis-py (optimal)

### 🔄 Should Replace with External Solutions

| Custom Code | Replace With | Lines Saved | Effort |
|-------------|--------------|-------------|--------|
| Custom ARIMA | `pmdarima` package | ~300 | Low |
| Vector store implementation | ChromaDB container | ~700 | Medium |
| Redis usage | Redis Stack container | 0 (enhancement) | Low |
| MLflow integration | MLflow container | ~200 | Low |
| Model serving | BentoML container | ~400 | Medium |
| CI/CD scripts | GitHub Actions | ~200 | Low |
| **Total** | | **~1,800 lines** | |

### 🗑️ Already Deprecated (Remove)

| File | Replaced By | Lines to Remove |
|------|-------------|-----------------|
| alpha_vantage_provider.py | OpenBB MCP | 180 |
| fmp_provider.py | OpenBB MCP | 200 |
| finnhub_provider.py | OpenBB MCP | 150 |
| iex_cloud_provider.py | OpenBB MCP | 150 |
| **Total** | | **680 lines** |

---

## Replacement Roadmap

### Phase 1: Immediate (This Week)
1. ✅ Integrate OpenBB MCP (DONE)
2. ✅ Integrate SEC Edgar MCP (DONE)
3. ✅ Integrate FRED MCP (DONE)
4. 🔄 Remove deprecated REST providers (680 lines)

### Phase 2: High Priority (Next Week)
5. 🔄 Replace custom ARIMA with pmdarima (300 lines saved)
6. 🔄 Replace vector_store.py with ChromaDB container (700 lines saved)
7. 🔄 Add Redis Stack container (enhanced features, 0 lines but better)

### Phase 3: Medium Priority (Month 2)
8. 🔄 Add MLflow container for experiment tracking (200 lines saved)
9. 🔄 Add BentoML container for model serving (400 lines saved)
10. 🔄 Migrate CI to GitHub Actions (200 lines saved)

### Phase 4: Future (Month 3+)
11. Consider Kong API Gateway (optional)
12. Consider Keycloak for enterprise SSO (optional)
13. Consider additional external services as they become available

---

## Total Potential Savings

**Immediate**: 680 lines (deprecated REST providers)
**Near-term**: 1,800 lines (external package replacements)
**Total**: ~2,500 lines of custom code can be eliminated

**Plus**: Continuous improvements from community-maintained packages!

---

## Recommendations

1. **DO THIS NOW**:
   - Remove deprecated REST provider files (680 lines)
   - Integrate pmdarima for ARIMA (300 lines saved)
   - Use ChromaDB container (700 lines saved)

2. **DO NEXT**:
   - Add MLflow + BentoML containers
   - Migrate to GitHub Actions CI/CD
   - Standardize on Redis Stack

3. **PRINCIPLE GOING FORWARD**:
   - Always search PyPI/Docker Hub/GitHub first
   - Only implement custom if nothing exists or existing solutions are inadequate
   - Prefer containers over packages (easier deployment)
   - Prefer MCP servers over REST APIs (less maintenance)

---

**Next Action**: Should I proceed with removing the 680 lines of deprecated REST provider code and integrating pmdarima/ChromaDB?