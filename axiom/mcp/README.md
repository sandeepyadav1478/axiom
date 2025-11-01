# Axiom MCP (Model Context Protocol) Infrastructure

## 🎯 Overview

This directory contains ALL Model Context Protocol (MCP) infrastructure for the Axiom platform, providing a unified, professional structure for both MCP clients and servers.

## 📁 Directory Structure

```
axiom/mcp/
├── clients/                    # MCP Client Implementations
│   ├── external/              # External service integrations
│   │   ├── communication/     # Slack, email, notifications
│   │   ├── devops/           # Git, Docker, K8s, CI/CD
│   │   ├── filesystem/       # File operations
│   │   ├── monitoring/       # Monitoring & observability
│   │   ├── research/         # Research & analysis tools
│   │   └── storage/          # Database & storage systems
│   │
│   └── internal/             # Internal system clients
│       ├── derivatives_data_mcp.py
│       └── market_data_integrations.py
│
└── servers/                   # MCP Server Implementations
    ├── shared/               # Shared MCP infrastructure
    │   ├── mcp_base.py      # Base server class
    │   ├── mcp_protocol.py  # Protocol definitions
    │   └── mcp_transport.py # Transport layers
    │
    ├── public/              # Public-facing MCP servers
    │   ├── trading/         # Trading cluster (5 servers)
    │   │   ├── pricing_greeks/
    │   │   ├── portfolio_risk/
    │   │   ├── strategy_gen/
    │   │   ├── execution/
    │   │   └── hedging/
    │   ├── analytics/       # Analytics cluster (3 servers)
    │   │   ├── performance/
    │   │   ├── market_data/
    │   │   └── volatility/
    │   └── compliance/      # Compliance cluster (1 server)
    │       └── regulatory/
    │
    └── internal/            # Internal system servers
        ├── monitoring/
        │   └── system_health/
        ├── safety/
        │   └── guardrails/
        └── client/
            └── interface/
```

## 🔧 Components

### MCP Clients (External)

**Communication**
- Slack integration for notifications
- Email services
- Real-time messaging

**DevOps**
- Git operations and version control
- Docker container management
- Kubernetes orchestration
- CI/CD pipeline integration

**Filesystem**
- File operations and management
- Directory traversal
- File search and indexing

**Monitoring**
- System health monitoring
- Performance metrics
- Alert management

**Research**
- Research paper analysis
- Data gathering
- Documentation tools

**Storage**
- PostgreSQL integration
- Redis caching
- S3-compatible storage

### MCP Clients (Internal)

**Market Data**
- Real-time market data aggregation
- Historical data access
- Multi-source data integration

**Derivatives Data**
- Options data
- Futures data
- Greeks calculation data

### MCP Servers (Public)

**Trading Cluster**
1. **pricing_greeks** - Real-time Greeks calculation
2. **portfolio_risk** - Portfolio risk analytics
3. **strategy_gen** - Strategy generation and optimization
4. **execution** - Trade execution management
5. **hedging** - Dynamic hedging strategies

**Analytics Cluster**
1. **performance** - Real-time P&L and performance
2. **market_data** - Market data aggregation
3. **volatility** - AI-powered volatility prediction

**Compliance Cluster**
1. **regulatory** - Regulatory reporting and compliance

### MCP Servers (Internal)

**System Health**
- Real-time system monitoring
- Health checks
- Performance metrics

**Guardrails**
- AI safety controls
- Risk limits enforcement
- Trading guardrails

**Interface**
- Client interface management
- Session handling
- Request routing

## 🚀 Usage

### Importing MCP Servers

```python
# Shared infrastructure
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport

# Public servers
from axiom.mcp.servers.public.trading.pricing_greeks.server import PricingGreeksServer
from axiom.mcp.servers.public.analytics.performance.server import PerformanceServer

# Internal servers
from axiom.mcp.servers.internal.monitoring.system_health.server import SystemHealthServer
```

### Importing MCP Clients

```python
# External clients
from axiom.mcp.clients.external.communication.slack_server import SlackMCPServer
from axiom.mcp.clients.external.devops.git_server import GitMCPServer

# Internal clients
from axiom.mcp.clients.internal.derivatives_data_mcp import DerivativesDataClient
from axiom.mcp.clients.internal.market_data_integrations import MarketDataAggregator
```

## 🐳 Docker Deployment

### Start All Public Servers
```bash
cd axiom/mcp
docker-compose up -d
```

### Start Specific Server
```bash
docker-compose up -d pricing_greeks
```

### View Server Logs
```bash
docker-compose logs -f pricing_greeks
```

### Stop All Servers
```bash
docker-compose down
```

## 📊 MCP Server Statistics

### Public Servers
- **Trading Cluster**: 5 servers
- **Analytics Cluster**: 3 servers
- **Compliance Cluster**: 1 server
- **Total Public**: 9 servers

### Internal Servers
- **Monitoring**: 1 server
- **Safety**: 1 server
- **Client Interface**: 1 server
- **Total Internal**: 3 servers

### External Clients
- **Communication**: Slack, Email
- **DevOps**: Git, Docker, K8s
- **Filesystem**: File operations
- **Monitoring**: System metrics
- **Research**: Analysis tools
- **Storage**: PostgreSQL, Redis, S3

### Internal Clients
- **Market Data**: Real-time & historical
- **Derivatives**: Options, futures, Greeks

## 🔒 Security

All MCP servers implement:
- Authentication via API keys
- Rate limiting
- Request validation
- Error handling with proper error codes
- Audit logging

## 📈 Performance

- **STDIO Transport**: Low-latency local communication
- **HTTP Transport**: RESTful API access
- **Async Processing**: Non-blocking operations
- **Caching**: Redis-backed caching layer
- **Connection Pooling**: Efficient resource usage

## 🧪 Testing

### Test All Servers
```bash
python -m pytest tests/test_mcp_*.py
```

### Test Specific Server
```bash
python -m pytest tests/test_mcp_pricing_greeks.py
```

### Integration Tests
```bash
python -m pytest tests/test_mcp_integration.py
```

## 📝 Development

### Creating a New MCP Server

1. Create server directory:
```bash
mkdir -p axiom/mcp/servers/public/new_category/new_server
```

2. Create `server.py`:
```python
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer

class NewServer(BaseMCPServer):
    def __init__(self):
        super().__init__(
            name="new-server",
            version="1.0.0",
            description="Description of new server"
        )
    
    async def handle_tool_call(self, tool_name: str, args: dict):
        # Implementation
        pass
```

3. Add to docker-compose.yml
4. Create tests
5. Update documentation

## 🔄 Migration Notes

This unified structure consolidates:
- `axiom/mcp/` (old external clients)
- `axiom/mcp_clients/` (old internal clients)
- `axiom/mcp_servers/` (old servers)
- `axiom/mcp_professional/` (previous structure)

All imports have been updated to use the new paths:
- `axiom.mcp.servers.*` for servers
- `axiom.mcp.clients.external.*` for external clients
- `axiom.mcp.clients.internal.*` for internal clients

## 📚 Additional Documentation

- [MCP Protocol Specification](./docs/mcp_protocol.md)
- [Server Development Guide](./docs/server_development.md)
- [Client Integration Guide](./docs/client_integration.md)
- [Docker Deployment Guide](./docs/docker_deployment.md)

## 🤝 Contributing

1. Follow the established directory structure
2. Use absolute imports (`from axiom.mcp...`)
3. Include comprehensive docstrings
4. Add tests for new functionality
5. Update this README for new servers/clients

## 📞 Support

For issues or questions:
- Check existing documentation
- Review test files for examples
- Open an issue on GitHub

---

**Last Updated**: November 1, 2025  
**Version**: 2.0.0 (Unified Structure)  
**Status**: ✅ Production Ready