# MCP Servers - Public vs Internal

## 🌍 PUBLIC MCP SERVERS (9 servers)

**Purpose**: Exposed to external clients (Claude Desktop, Cline, etc.)  
**Location**: `axiom/mcp_servers/docker-compose-public.yml`  
**Deployment**: `docker-compose -f docker-compose-public.yml up -d`

### Trading (5 servers)
1. **pricing-greeks-mcp** - Options pricing and Greeks
2. **portfolio-risk-mcp** - Portfolio VaR and risk metrics
3. **strategy-gen-mcp** - AI trading strategy generation
4. **execution-mcp** - Smart order routing
5. **hedging-mcp** - Optimal hedging via Deep RL

### Analytics (3 servers)  
6. **performance-mcp** - P&L attribution
7. **market-data-mcp** - Market quotes aggregation
8. **volatility-mcp** - AI volatility forecasting

### Compliance (1 server)
9. **regulatory-mcp** - SEC/FINRA compliance checking

**These are what users/clients interact with via Claude Desktop!**

---

## 🏢 INTERNAL MCP SERVERS (3 servers)

**Purpose**: Internal infrastructure - NOT for public exposure  
**Location**: `axiom/mcp_servers/docker-compose-internal.yml`  
**Deployment**: `docker-compose -f docker-compose-internal.yml up -d`

### Infrastructure (3 servers)
1. **system-health-mcp** - System monitoring (internal only)
2. **guardrails-mcp** - AI safety validation (internal only)
3. **interface-mcp** - Orchestration layer (internal only)

**These support the platform but aren't directly exposed to clients!**

---

## 🤔 Why the Separation?

### Public Servers (9):
- ✅ Safe to expose to Claude Desktop
- ✅ Provide user-facing features
- ✅ Read-only or validated operations
- ✅ Compliance-checked
- ✅ Example: "Calculate Greeks" or "Get quotes"

### Internal Servers (3):
- ⚠️ System infrastructure
- ⚠️ Should not be directly accessible
- ⚠️ May have admin-level access
- ⚠️ Used by other services internally
- ⚠️ Example: "Shut down agent" or "Override guardrails"

---

## 🚀 Deployment Guide

### For Client Use (Recommended)
```bash
# Start only public-facing servers
cd axiom/mcp_servers
docker-compose -f docker-compose-public.yml up -d

# Verify
docker ps | grep mcp
# Should show 9 containers
```

### For Full System (Development/Testing)
```bash
# Start ALL servers (public + internal)
cd axiom/mcp_servers
docker-compose -f docker-compose.yml up -d

# Verify
docker ps | grep mcp
# Should show 12 containers
```

### For Internal Only (System Admin)
```bash
# Start only internal infrastructure
cd axiom/mcp_servers
docker-compose -f docker-compose-internal.yml up -d

# Verify
docker ps | grep mcp
# Should show 3 containers
```

---

## 📁 Directory Structure

```
axiom/mcp_servers/
├── trading/              # PUBLIC: 5 trading servers
│   ├── pricing_greeks/
│   ├── portfolio_risk/
│   ├── strategy_gen/
│   ├── execution/
│   └── hedging/
│
├── analytics/            # PUBLIC: 3 analytics servers
│   ├── performance/
│   ├── market_data/
│   └── volatility/
│
├── compliance/           # PUBLIC: 1 compliance server
│   └── regulatory/
│
└── internal/             # INTERNAL: 3 infrastructure servers
    ├── monitoring/
    │   └── system_health/
    ├── safety/
    │   └── guardrails/
    └── client/
        └── interface/
```

---

## 🔐 Security Implications

### Public Servers:
- ✅ Designed for external access
- ✅ Input validation
- ✅ Rate limiting ready
- ✅ Audit logging
- ✅ Safe for Claude Desktop

### Internal Servers:
- ⚠️ Admin-level access
- ⚠️ Can modify system behavior
- ⚠️ Should be on private network
- ⚠️ Not for direct client access
- ⚠️ Use via internal APIs only

---

## 📊 Recommended Setup

### Production:
```bash
# Public network (internet-facing)
docker-compose -f docker-compose-public.yml up -d

# Private network (internal only)
docker-compose -f docker-compose-internal.yml up -d --network private
```

### Development:
```bash
# Everything on one network for testing
docker-compose up -d
```

---

## ✅ Current Status

**Public Servers**: 9/9 running ✅  
**Internal Servers**: 3/3 running ✅  
**Total**: 12/12 operational

**For Claude Desktop Integration**: Use only the 9 public servers!

---

**See [MCP_SERVERS_EXPLAINED.md](../../MCP_SERVERS_EXPLAINED.md) for detailed explanation of each server's purpose.**