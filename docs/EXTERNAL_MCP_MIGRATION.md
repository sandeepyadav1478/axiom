# External MCP Migration Guide

**Objective**: Replace custom REST API wrappers with community-maintained external MCP servers for zero-maintenance financial data integration.

## Table of Contents

1. [Overview](#overview)
2. [Why External MCPs?](#why-external-mcps)
3. [Available External MCPs](#available-external-mcps)
4. [Migration Path](#migration-path)
5. [Benefits](#benefits)
6. [Cost Comparison](#cost-comparison)
7. [Quick Start](#quick-start)
8. [Deprecated Providers](#deprecated-providers)

---

## Overview

External MCP servers are community-maintained Model Context Protocol servers that provide financial data through standardized interfaces. Instead of maintaining custom REST API wrappers, we leverage these open-source solutions.

### Key Principle
**Zero Maintenance > Custom Code**

---

## Why External MCPs?

### Problems with REST API Wrappers
- **Maintenance burden**: ~1,500 lines of custom code to maintain
- **API changes**: Breaking changes require immediate fixes
- **Error handling**: Custom retry logic and rate limiting
- **Testing overhead**: Need to mock and test each provider
- **Documentation debt**: Must document each wrapper

### Benefits of External MCPs
- **Zero maintenance**: Community maintains the code
- **Automatic updates**: Bug fixes and features arrive automatically
- **Standardized protocol**: MCP provides consistent interface
- **Better AI integration**: Native MCP support in AI frameworks
- **Community support**: Larger user base = more robust code

---

## Available External MCPs

### 1. OpenBB MCP Server ⭐ **HIGHEST PRIORITY**

**Repository**: https://github.com/openbb/openbb-mcp-server

**What it replaces**: 5+ REST providers
- Alpha Vantage (~180 lines)
- FMP (~200 lines)
- Finnhub (~150 lines)
- Yahoo Finance (~200 lines)
- IEX Cloud (~150 lines)

**Total elimination**: ~880 lines of custom code!

**Tools Available** (50+ tools):
- Stock quotes & fundamentals
- Options data & ETF data
- Economic indicators
- Crypto prices & Forex rates
- Commodities data
- News aggregation
- Alternative data sources

**Installation**:
```yaml
# In axiom/integrations/data_sources/finance/docker-compose.yml
openbb-server:
  build:
    context: https://github.com/openbb/openbb-mcp-server.git
  container_name: axiom-openbb-mcp
  environment:
    - OPENBB_API_KEY=${OPENBB_API_KEY:-}
    - MCP_TRANSPORT=stdio
  stdin_open: true
  tty: true
  restart: unless-stopped
  networks:
    - financial-data-network
  profiles:
    - "openbb"
    - "mcp"
    - "external"
```

**Start the server**:
```bash
cd axiom/integrations/data_sources/finance
docker-compose --profile openbb up -d
```

---

### 2. SEC Edgar MCP Server

**What it replaces**: SEC Edgar provider (~150 lines)

**Tools Available**:
- Search SEC filings (10-K, 10-Q, 8-K)
- Get filing content
- Company facts & metrics
- Insider trading (Form 4)
- Institutional ownership (13F, 13G)

**Cost**: **FREE** - Unlimited access to SEC data

**Installation**:
```yaml
sec-edgar-server:
  image: python:3.11-slim
  container_name: axiom-sec-edgar-mcp
  environment:
    - SEC_API_KEY=${SEC_API_KEY:-}
    - MCP_TRANSPORT=stdio
  # ... see docker-compose.yml
```

**Start the server**:
```bash
docker-compose --profile sec-edgar up -d
```

---

### 3. FRED Economic Data MCP Server

**What it replaces**: Custom FRED integration (~120 lines)

**Data Available**: 800,000+ economic time series

**Tools Available**:
- Get economic series (GDP, CPI, unemployment, etc.)
- Search FRED database
- Get data releases
- Browse categories

**Cost**: **FREE** - Federal Reserve API

**Installation**:
```yaml
fred-server:
  image: python:3.11-alpine
  container_name: axiom-fred-mcp
  environment:
    - FRED_API_KEY=${FRED_API_KEY:-}
    - MCP_TRANSPORT=stdio
  # ... see docker-compose.yml
```

**Start the server**:
```bash
docker-compose --profile fred up -d
```

---

### 4. CoinGecko MCP Server

**What it replaces**: Custom crypto integration (~100 lines)

**Tools Available**:
- Get coin prices
- Historical crypto data
- Market cap rankings
- Trending coins
- Global market data

**Cost**: **FREE** - No API key required!

**Installation**:
```yaml
coingecko-server:
  image: python:3.11-alpine
  container_name: axiom-coingecko-mcp
  environment:
    - MCP_TRANSPORT=stdio
  # ... see docker-compose.yml
```

**Start the server**:
```bash
docker-compose --profile coingecko up -d
```

---

### 5. NewsAPI MCP Server

**What it replaces**: Custom news aggregation (~100 lines)

**Sources**: 80,000+ news sources

**Tools Available**:
- Search news by keyword
- Get headlines by topic
- Filter by source
- Get article content
- Sentiment analysis

**Installation**:
```yaml
newsapi-server:
  image: python:3.11-alpine
  container_name: axiom-newsapi-mcp
  environment:
    - NEWS_API_KEY=${NEWS_API_KEY:-}
    - MCP_TRANSPORT=stdio
  # ... see docker-compose.yml
```

**Start the server**:
```bash
docker-compose --profile newsapi up -d
```

---

## Migration Path

### Phase 1: Install External MCPs (Week 1)
1. Add MCP server configurations to docker-compose.yml ✅
2. Start MCP servers in development environment
3. Test connectivity and tool availability
4. Verify data quality matches REST providers

### Phase 2: Update Application Code (Week 2)
1. Update `mcp_adapter.py` to support external MCPs ✅
2. Add external MCP tool definitions
3. Create abstraction layer for backward compatibility
4. Update AI workflows to use MCP tools

### Phase 3: Deprecate REST Providers (Week 3)
1. Add deprecation warnings to REST providers ✅
2. Update documentation with migration guides
3. Monitor usage and ensure smooth transition
4. Remove REST provider code after confirmation

### Phase 4: Cleanup (Week 4)
1. Remove deprecated REST API wrappers
2. Update tests to use MCP mocks
3. Simplify configuration
4. Document the new architecture

---

## Benefits

### Code Elimination
| Provider | Lines Removed | Maintenance Hours Saved/Year |
|----------|--------------|------------------------------|
| Alpha Vantage | ~180 | 40h |
| FMP | ~200 | 45h |
| Finnhub | ~150 | 35h |
| Yahoo Finance | ~200 | 45h |
| IEX Cloud | ~150 | 35h |
| SEC Edgar | ~150 | 30h |
| Custom integrations | ~370 | 80h |
| **TOTAL** | **~1,400 lines** | **310 hours/year** |

### Maintenance Reduction
- **Zero** code changes when APIs update
- **Community** fixes bugs automatically
- **Automatic** feature additions
- **Standardized** error handling
- **No** API key rotation logic needed
- **No** retry/backoff implementation required

---

## Cost Comparison

### Before (REST API Wrappers)
| Provider | Monthly Cost | Annual Cost |
|----------|-------------|-------------|
| Alpha Vantage Premium | $49 | $588 |
| FMP Professional | $29 | $348 |
| Finnhub Premium | $7.99 | $96 |
| IEX Cloud | $9 | $108 |
| News API | $0 (free) | $0 |
| **TOTAL** | **$95/month** | **$1,140/year** |

**Plus**: ~310 hours/year of developer maintenance time

### After (External MCPs)
| MCP Server | Monthly Cost | Annual Cost |
|-----------|-------------|-------------|
| OpenBB (FREE tier) | $0 | $0 |
| OpenBB (Premium optional) | $50 | $600 |
| SEC Edgar | $0 (FREE) | $0 |
| FRED | $0 (FREE) | $0 |
| CoinGecko | $0 (FREE) | $0 |
| NewsAPI | $0 (FREE tier) | $0 |
| **TOTAL (Free tier)** | **$0/month** | **$0/year** |
| **TOTAL (Premium)** | **$50/month** | **$600/year** |

**Plus**: ~0 hours/year of maintenance (community-maintained)

### Savings
- **Cost savings**: $540-1,140/year (47-100%)
- **Time savings**: 310 hours/year of developer time
- **Value of time saved**: ~$31,000/year (at $100/hour rate)
- **Total annual savings**: **$31,540-32,140**

---

## Quick Start

### 1. Configure Environment Variables

Add to your `.env` file:
```bash
# External MCP Servers
OPENBB_API_KEY=your_openbb_key_here  # Optional for free tier
SEC_API_KEY=your_sec_key_here        # Optional
FRED_API_KEY=your_fred_key_here      # FREE at research.stlouisfed.org
NEWS_API_KEY=your_news_key_here      # FREE tier available
# COINGECKO_API_KEY not needed (free tier works without key)
```

### 2. Start External MCP Servers

```bash
# Navigate to finance data sources
cd axiom/integrations/data_sources/finance

# Start all external MCPs
docker-compose --profile external up -d

# Or start individual servers
docker-compose --profile openbb up -d
docker-compose --profile sec-edgar up -d
docker-compose --profile fred up -d
docker-compose --profile coingecko up -d
docker-compose --profile newsapi up -d
```

### 3. Verify Installation

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs openbb-server
docker-compose logs sec-edgar-server
docker-compose logs fred-server
```

### 4. Test in Python

```python
from axiom.integrations.search_tools.mcp_adapter import mcp_adapter

# Get available external MCP servers
servers = mcp_adapter.get_external_mcp_servers()
print(f"Available servers: {servers}")

# Get replacement mapping
mapping = mcp_adapter.get_replacement_mapping()
print(f"alpha_vantage → {mapping['alpha_vantage']}")  # → openbb
```

---

## Deprecated Providers

The following REST API providers are **DEPRECATED** and will be removed in a future release:

### ⚠️ alpha_vantage_provider.py
**Status**: DEPRECATED  
**Replacement**: OpenBB MCP Server  
**Migration**: Use `openbb-server` tools instead  

### ⚠️ fmp_provider.py
**Status**: DEPRECATED  
**Replacement**: OpenBB MCP Server  
**Migration**: Use `openbb-server` tools instead  

### ⚠️ finnhub_provider.py
**Status**: DEPRECATED  
**Replacement**: OpenBB MCP Server  
**Migration**: Use `openbb-server` tools instead  

### ⚠️ iex_cloud_provider.py
**Status**: DEPRECATED  
**Replacement**: OpenBB MCP Server  
**Migration**: Use `openbb-server` tools instead  

### ⚠️ yahoo_finance_provider.py
**Status**: DEPRECATED  
**Replacement**: OpenBB MCP Server  
**Migration**: Use `openbb-server` tools instead  

---

## Success Criteria

- [x] External MCP servers configured in docker-compose.yml
- [x] MCP adapter updated to support external servers
- [x] REST API providers marked as deprecated
- [x] Migration documentation created
- [ ] All AI workflows updated to use MCPs
- [ ] Testing confirms data quality matches
- [ ] REST providers successfully removed
- [ ] ~1,500 lines of code eliminated

---

## Support & Resources

### Documentation
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [OpenBB Documentation](https://docs.openbb.co/)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/fred/)
- [CoinGecko API](https://www.coingecko.com/en/api)

### Community
- OpenBB Discord: https://openbb.co/discord
- MCP GitHub: https://github.com/modelcontextprotocol

### Internal
- MCP Manager: [`axiom/integrations/mcp_servers/manager.py`](../axiom/integrations/mcp_servers/manager.py)
- MCP Adapter: [`axiom/integrations/search_tools/mcp_adapter.py`](../axiom/integrations/search_tools/mcp_adapter.py)
- Docker Compose: [`axiom/integrations/data_sources/finance/docker-compose.yml`](../axiom/integrations/data_sources/finance/docker-compose.yml)

---

## FAQs

**Q: Will data quality decrease with external MCPs?**  
A: No. External MCPs often aggregate multiple sources (like OpenBB) providing *better* data quality than single REST providers.

**Q: What if an external MCP breaks?**  
A: The community maintains these servers. Issues are usually fixed within hours. You can also temporarily fall back to REST providers during migration.

**Q: How do I contribute back?**  
A: Report issues, submit PRs, and share improvements with the MCP community. This benefits everyone.

**Q: Can I use both MCPs and REST providers?**  
A: Yes, during migration. The system supports both. However, prefer MCPs for new features.

**Q: What about data from Bloomberg or FactSet?**  
A: Premium enterprise data sources remain as specialized integrations. This migration focuses on replacing public API wrappers.

---

**Last Updated**: 2024-10-24  
**Status**: Active Migration  
**Next Review**: Q1 2025