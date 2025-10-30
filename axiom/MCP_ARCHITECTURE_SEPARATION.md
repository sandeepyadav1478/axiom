# MCP Architecture - Clear Separation

## 🎯 TWO DISTINCT MCP SYSTEMS (PROPERLY SEPARATED)

---

## 📁 FOLDER STRUCTURE (CLEARLY SEPARATED)

```
axiom/
│
├── derivatives/mcp/              # ⬇️ INCOMING: External MCP Clients
│   │                             # (We CONSUME external data)
│   ├── market_data_integrations.py
│   ├── polygon_mcp_query.py      # Polygon.io MCP client
│   ├── yahoo_comp_mcp_query.py   # Yahoo Finance MCP client  
│   ├── firecrawl_integration.py  # Firecrawl MCP client
│   └── tavily_integration.py     # Tavily MCP client
│   │
│   └── Purpose: Fetch data FROM external MCP servers
│
│
├── mcp_servers/                  # ⬆️ OUTGOING: Our MCP Servers
│   │                             # (We EXPOSE our capabilities)
│   │
│   ├── shared/                   # MCP server infrastructure
│   │   ├── mcp_base.py
│   │   ├── mcp_protocol.py
│   │   └── mcp_transport.py
│   │
│   ├── trading/                  # Our trading MCPs
│   │   ├── pricing_greeks/       # ✅ COMPLETE
│   │   ├── portfolio_risk/       # TODO
│   │   ├── strategy_gen/         # TODO
│   │   ├── execution/            # TODO
│   │   └── hedging/              # TODO
│   │
│   ├── analytics/                # Our analytics MCPs
│   │   ├── performance/          # TODO
│   │   ├── market_data/          # TODO
│   │   └── volatility/           # TODO
│   │
│   ├── compliance/               # Our compliance MCPs
│   │   └── regulatory/           # TODO
│   │
│   ├── monitoring/               # Our monitoring MCPs
│   │   └── system_health/        # TODO
│   │
│   ├── safety/                   # Our safety MCPs
│   │   └── guardrails/           # TODO
│   │
│   ├── client/                   # Our client MCPs
│   │   └── interface/            # TODO
│   │
│   └── Purpose: Expose OUR capabilities TO external clients
│
│
└── ai_layer/                     # Internal Processing (Not MCP)
    ├── agents/professional/      # 12 internal agents
    ├── domain/                   # Domain objects
    └── infrastructure/           # Core patterns
    
    Purpose: Internal processing layer
```

---

## 🔄 ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTERNAL WORLD                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Polygon MCP  │  │ Yahoo MCP    │  │ Claude Desktop│     │
│  │ (market data)│  │ (historical) │  │ (our client)  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬────────┘     │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │ INCOMING         │ INCOMING          │ OUTGOING
          ▼                  ▼                  ▲
┌─────────────────────────────────────────────┼──────────────┐
│              AXIOM SYSTEM                    │              │
│                                              │              │
│  ┌─────────────────────────────────────┐   │              │
│  │  axiom/derivatives/mcp/              │   │              │
│  │  (External MCP Clients - Incoming)   │   │              │
│  │  - Polygon client                    │   │              │
│  │  - Yahoo client                      │   │              │
│  │  - Firecrawl client                  │   │              │
│  └─────────────┬───────────────────────┘   │              │
│                │                            │              │
│                ▼                            │              │
│  ┌─────────────────────────────────────┐   │              │
│  │  axiom/ai_layer/                     │   │              │
│  │  (Internal Agents - Processing)      │   │              │
│  │  - 12 Professional Agents            │   │              │
│  │  - Domain objects                    │───┘              │
│  │  - Infrastructure                    │                  │
│  └─────────────┬───────────────────────┘                  │
│                │                                            │
│                ▼                                            │
│  ┌─────────────────────────────────────┐                  │
│  │  axiom/mcp_servers/                  │                  │
│  │  (Our MCP Servers - Outgoing)        │                  │
│  │  - pricing-greeks-mcp-server         │──────────────────┘
│  │  - portfolio-risk-mcp-server         │
│  │  - (12 total MCP servers)            │
│  └──────────────────────────────────────┘
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## ✅ CURRENT STATUS

**1. INCOMING (axiom/derivatives/mcp/)** ✅
- Already exists
- External MCP clients working
- Polygon, Yahoo, Firecrawl integrated
- **Purpose:** Get data FROM external sources

**2. OUTGOING (axiom/mcp_servers/)** 🚧
- Folder created ✅
- Infrastructure complete ✅
- Template server complete ✅
- **Purpose:** Expose OUR services TO clients
- **Status:** 1/12 complete (pricing-greeks)

**3. INTERNAL (axiom/ai_layer/)** ✅
- All 12 agents complete
- Professional quality
- **Purpose:** Internal processing

---

## 🎯 SEPARATION IS CORRECT

**Yes, the folders are properly separated:**
- `axiom/derivatives/mcp/` - External MCPs we consume ⬇️
- `axiom/mcp_servers/` - Our MCPs we expose ⬆️
- `axiom/ai_layer/` - Internal processing (not MCP)

**Different purposes, different folders, clean architecture.**

---

## 🐳 DOCKER TESTING STATUS

**What's Ready:**
- ✅ Dockerfile for pricing-greeks MCP
- ✅ docker-compose.test.yml
- ✅ test_mcp_via_docker.sh (executable)

**What's NOT Done:**
- ❌ Haven't run Docker tests yet
- ❌ Haven't validated in container
- ❌ Haven't tested with Claude Desktop

**Ready to test when you want:**
```bash
./axiom/mcp_servers/test_mcp_via_docker.sh
```

---

## 🚀 NEXT STEPS

1. **Test current MCP via Docker** (validates container works)
2. **Create remaining 11 MCPs** (following template)
3. **Test all 12 MCPs** (Docker validation)
4. **Deploy to production** (each MCP separately)

---

**The architecture is properly separated. MCPs are Docker-ready but not tested yet. Ready to proceed when you want to test or continue building remaining MCPs.**