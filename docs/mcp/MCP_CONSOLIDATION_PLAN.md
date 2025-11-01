# MCP Infrastructure Consolidation Plan

## 🎯 Objective
Consolidate all MCP-related code from 4 separate directories into ONE unified, professional structure.

## 📊 Current State Analysis

### Existing Directories:
1. **`axiom/mcp/`** - External MCP client integrations
   - `client/communication/` - Slack integration
   - `client/devops/` - K8s, GitHub integration  
   - `client/filesystem/` - Filesystem operations
   - `client/monitoring/` - Monitoring tools
   - `client/research/` - Research tools
   - `client/storage/` - Storage systems

2. **`axiom/mcp_clients/`** - Internal MCP clients
   - `derivatives_data_mcp.py`
   - `market_data_integrations.py`

3. **`axiom/mcp_servers/`** - Old server structure
   - `trading/`, `analytics/`, `compliance/`, `internal/`, `shared/`
   - Multiple docker-compose files
   - Documentation files

4. **`axiom/mcp_professional/`** - Current professional structure
   - `servers/public/` (trading, analytics, compliance)
   - `servers/internal/`
   - `servers/shared/`
   - `clients/`

## ✅ Target Structure

```
axiom/mcp/                              # SINGLE unified MCP directory
├── __init__.py
├── README.md                           # Comprehensive MCP documentation
├── docker-compose.yml                  # Unified docker configuration
├── requirements.txt                    # MCP dependencies
│
├── clients/                            # ALL MCP clients
│   ├── __init__.py
│   ├── README.md
│   │
│   ├── external/                       # External service integrations
│   │   ├── __init__.py
│   │   ├── communication/              # Slack, email, etc.
│   │   ├── devops/                     # K8s, GitHub, CI/CD
│   │   ├── filesystem/                 # File operations
│   │   ├── monitoring/                 # Monitoring systems
│   │   ├── research/                   # Research tools
│   │   └── storage/                    # Storage systems
│   │
│   └── internal/                       # Internal system clients
│       ├── __init__.py
│       ├── derivatives_data_mcp.py
│       └── market_data_integrations.py
│
└── servers/                            # ALL MCP servers
    ├── __init__.py
    ├── README.md
    ├── requirements.txt
    │
    ├── shared/                         # Shared MCP infrastructure
    │   ├── __init__.py
    │   ├── mcp_base.py
    │   ├── mcp_protocol.py
    │   └── mcp_transport.py
    │
    ├── public/                         # Public-facing MCP servers
    │   ├── __init__.py
    │   ├── trading/                    # Trading cluster
    │   │   ├── pricing_greeks/
    │   │   ├── portfolio_risk/
    │   │   ├── strategy_gen/
    │   │   ├── execution/
    │   │   └── hedging/
    │   ├── analytics/                  # Analytics cluster
    │   │   ├── performance/
    │   │   ├── market_data/
    │   │   └── volatility/
    │   └── compliance/                 # Compliance cluster
    │       └── regulatory/
    │
    └── internal/                       # Internal system servers
        ├── __init__.py
        ├── system_health/
        ├── guardrails/
        └── interface/
```

## 🔄 Migration Steps

### Phase 1: Prepare New Structure
1. ✅ Create base `axiom/mcp/` directories
2. ✅ Create `clients/external/` and `clients/internal/`
3. ✅ Create `servers/public/`, `servers/internal/`, `servers/shared/`

### Phase 2: Move Clients
1. Move `axiom/mcp/client/*` → `axiom/mcp/clients/external/*`
2. Move `axiom/mcp_clients/*` → `axiom/mcp/clients/internal/*`

### Phase 3: Move Servers
1. Move `axiom/mcp_professional/servers/public/*` → `axiom/mcp/servers/public/*`
2. Move `axiom/mcp_professional/servers/internal/*` → `axiom/mcp/servers/internal/*`
3. Move `axiom/mcp_professional/servers/shared/*` → `axiom/mcp/servers/shared/*`

### Phase 4: Update Imports
1. Update all imports from `axiom.mcp_servers.*` → `axiom.mcp.servers.*`
2. Update all imports from `axiom.mcp_clients.*` → `axiom.mcp.clients.internal.*`
3. Update all imports from `axiom.mcp.client.*` → `axiom.mcp.clients.external.*`

### Phase 5: Update Docker
1. Consolidate all docker-compose files
2. Update paths in Dockerfiles
3. Update environment variables

### Phase 6: Cleanup
1. Remove `axiom/mcp_servers/` (old)
2. Remove `axiom/mcp_clients/` (old)
3. Remove `axiom/mcp_professional/` (old)
4. Update documentation

## 📝 Import Pattern Changes

### Before:
```python
from axiom.mcp_servers.shared.mcp_base import BaseMCPServer
from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksServer
from axiom.mcp_clients.derivatives_data_mcp import DerivativesDataClient
```

### After:
```python
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer
from axiom.mcp.servers.public.trading.pricing_greeks.server import PricingGreeksServer
from axiom.mcp.clients.internal.derivatives_data_mcp import DerivativesDataClient
```

## 🎯 Benefits

1. **Single Source of Truth**: All MCP code in one place
2. **Clear Separation**: Clients vs Servers, External vs Internal
3. **Professional Structure**: Industry-standard organization
4. **Easy Discovery**: Developers know exactly where to find MCP code
5. **Simplified Imports**: Consistent import patterns
6. **Better Docker**: Unified docker-compose configuration
7. **Cleaner Root**: Remove 3 top-level directories

## 🚀 Execution Order

1. Create new structure (no breaking changes yet)
2. Copy files to new locations
3. Update imports systematically
4. Test thoroughly
5. Remove old directories
6. Update all documentation

## ⚠️ Risk Mitigation

- Work on a feature branch ✅ (Already on feature/20251031-1903-next-milestone)
- Test after each major step
- Keep old directories until everything works
- Document all changes
- Create rollback plan if needed

---

**Status**: Ready to execute
**Branch**: feature/20251031-1903-next-milestone
**Estimated Files to Move**: ~100+
**Estimated Imports to Update**: ~50+