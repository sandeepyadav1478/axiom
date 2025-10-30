# Axiom MCP - Professional Model Context Protocol Architecture

## 🎯 Overview

This directory contains Axiom's complete MCP (Model Context Protocol) implementation with clear separation between clients and servers.

## 📁 Professional Structure

```
axiom/mcp_final/                    # Root MCP directory
│
├── README.md                        # This file - master guide
├── docker-compose-public.yml        # 9 public servers (Claude Desktop)
├── DOCKER_STARTUP_GUIDE.md          # Operations guide
│
├── clients/                         # MCP CLIENTS (we consume external MCPs)
│   ├── external_integrations/       # External MCP servers we use
│   │   ├── storage/                 # Redis, Postgres, Vector DB
│   │   ├── devops/                  # Git, Docker, Kubernetes
│   │   ├── cloud/                   # AWS, GCP
│   │   ├── communication/           # Slack, Notifications
│   │   ├── documents/               # PDF, Excel processing
│   │   ├── analytics/               # SQL analytics
│   │   ├── research/                # ArXiv
│   │   ├── mlops/                   # Model serving
│   │   ├── monitoring/              # Prometheus
│   │   ├── filesystem/              # File operations
│   │   └── code_quality/            # Linting
│   │
│   └── data_sources/                # Market data MCP clients
│       ├── derivatives_data_mcp.py
│       └── market_data_integrations.py
│
└── servers/                         # MCP SERVERS (we expose to clients)
    │
    ├── public/                      # PUBLIC (9 servers) - For Claude Desktop
    │   ├── trading/                 # 5 Trading Servers
    │   │   ├── pricing_greeks/      # Options Greeks (<1ms)
    │   │   ├── portfolio_risk/      # Portfolio VaR (<5ms)
    │   │   ├── strategy_gen/        # AI strategies
    │   │   ├── execution/           # Smart routing
    │   │   └── hedging/             # DRL hedging
    │   │
    │   ├── analytics/               # 3 Analytics Servers
    │   │   ├── performance/         # P&L attribution
    │   │   ├── market_data/         # NBBO quotes
    │   │   └── volatility/          # AI vol forecasting
    │   │
    │   └── compliance/              # 1 Compliance Server
    │       └── regulatory/          # SEC/FINRA compliance
    │
    ├── internal/                    # INTERNAL (3 servers) - System use only
    │   ├── monitoring/
    │   │   └── system_health/       # Health monitoring
    │   ├── safety/
    │   │   └── guardrails/          # AI safety
    │   └── client/
    │       └── interface/           # Orchestration
    │
    └── shared/                      # MCP Infrastructure
        ├── mcp_base.py              # Base MCP implementation
        ├── mcp_protocol.py          # JSON-RPC 2.0 + MCP 1.0.0
        └── mcp_transport.py         # STDIO/HTTP/SSE transports
```

## 🌍 PUBLIC vs INTERNAL

### PUBLIC SERVERS (9) - Safe for Claude Desktop ✅
**Purpose**: Expose to external clients  
**Location**: `servers/public/`  
**Deploy**: `docker-compose -f docker-compose-public.yml up -d`

**These are user-facing features:**
- Calculate option Greeks
- Assess portfolio risk
- Generate trading strategies
- Route orders
- Calculate hedges
- Analyze P&L
- Get market data
- Forecast volatility
- Check regulatory compliance

### INTERNAL SERVERS (3) - System Infrastructure Only ⚠️
**Purpose**: Internal system operations  
**Location**: `servers/internal/`  
**Deploy**: For system use, not Claude Desktop

**These are admin-level:**
- System health monitoring
- AI safety validation (veto authority)
- Multi-MCP orchestration

## 🚀 Quick Start

### For Claude Desktop Integration
```bash
cd axiom/mcp_final
docker-compose -f docker-compose-public.yml up -d
docker ps  # Should show 9 public MCP servers
```

### Test a Server
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","clientInfo":{"name":"test","version":"1.0"}}}' | \
docker run -i --rm pricing-greeks-mcp
```

## 📖 Import Patterns

### Server Imports (in axiom/mcp_final/servers/)
```python
from axiom.mcp_final.servers.shared.mcp_base import BaseMCPServer
from axiom.mcp_final.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp_final.servers.shared.mcp_transport import STDIOTransport
```

### Client Imports (in axiom/mcp_final/clients/)
```python
from axiom.mcp_final.clients.external_integrations.storage.redis_server import RedisMCPServer
from axiom.mcp_final.clients.data_sources.market_data_integrations import MarketDataAggregator
```

## ✅ What's Working

- ✅ All 9 public servers tested
- ✅ All 3 internal servers tested
- ✅ MCP protocol v1.0.0 compliant
- ✅ Docker builds successful
- ✅ ARM-compatible
- ✅ Minimal dependencies
- ✅ Production-ready

## 📚 Additional Documentation

- [DOCKER_STARTUP_GUIDE.md](DOCKER_STARTUP_GUIDE.md) - Complete Docker guide
- [README_PUBLIC_VS_INTERNAL.md](../axiom/mcp_servers/README_PUBLIC_VS_INTERNAL.md) - Security separation
- [MCP_SERVERS_EXPLAINED.md](../MCP_SERVERS_EXPLAINED.md) - What each server does

## 🎯 Key Benefits of This Structure

1. **Clear Separation**: Clients vs Servers, Public vs Internal
2. **Professional Naming**: No confusion about what's where
3. **Security**: Clear boundary between public and internal
4. **Scalability**: Easy to add new servers/clients
5. **Maintainability**: Logical organization
6. **Documentation**: Self-explaining structure

## 🔧 Development

### Adding a New Public Server
1. Create directory in `servers/public/<category>/<name>/`
2. Add server.py, Dockerfile, __init__.py
3. Update docker-compose-public.yml
4. Test and deploy

### Adding a New Client
1. Create directory in `clients/external_integrations/<category>/`
2. Add client implementation
3. Register in registry.py
4. Use in your code

## ✨ This is Production-Grade Architecture

Built following:
- ✅ Microservice best practices
- ✅ Clear separation of concerns
- ✅ Industry-standard naming
- ✅ Security-first design
- ✅ Comprehensive documentation

---

**For questions or issues**, see the documentation files or check the individual server README files.