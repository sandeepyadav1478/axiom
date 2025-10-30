# MCP Server Testing Guide

## 🎯 Testing MCP Servers via Docker

This guide shows how to test our MCP servers in Docker containers.

---

## 📁 ARCHITECTURE CLARIFICATION

### **Folder Structure (Separate & Clear)**

```
axiom/
├── ai_layer/                    # Professional Agents (Internal Backend)
│   ├── agents/professional/     # 12 production agents
│   ├── domain/                  # 50+ value objects
│   └── infrastructure/          # Core patterns
│
└── mcp_servers/                 # MCP Servers (External Interface) ⭐
    ├── shared/                  # MCP infrastructure
    │   ├── mcp_base.py          # Base MCP implementation
    │   ├── mcp_protocol.py      # JSON-RPC 2.0 + MCP
    │   └── mcp_transport.py     # STDIO/HTTP/SSE
    │
    ├── trading/                 # Trading MCPs
    │   └── pricing_greeks/      # COMPLETE ✅
    │       ├── server.py        # MCP server
    │       ├── config.json      # MCP config
    │       ├── README.md        # Docs
    │       ├── Dockerfile       # Container
    │       ├── requirements.txt # Dependencies
    │       └── __init__.py      # Package
    │
    ├── docker-compose.test.yml  # Docker testing
    └── test_mcp_via_docker.sh   # Test script
```

**Key Points:**
- ✅ MCPs are **separate** from agents (`mcp_servers/` vs `ai_layer/`)
- ✅ Each MCP is **self-contained** with all files
- ✅ Each MCP is **Docker-deployable** independently
- ✅ MCPs **use** the professional agents as backend

---

## 🐳 HOW MCPs WORK WITH DOCKER

### **Architecture:**

```
┌─────────────────┐
│  Claude Desktop │  or  Cline  or  Any MCP Client
└────────┬────────┘
         │ MCP Protocol (JSON-RPC 2.0)
         │ via STDIO/HTTP/SSE
         ▼
┌─────────────────────────────────────┐
│    MCP Server (in Docker)           │
│  ┌──────────────────────────────┐   │
│  │ pricing-greeks-mcp-server    │   │
│  │  - Tools (calculate_greeks)  │   │
│  │  - Resources (cache, stats)  │   │
│  │  - Prompts (explain_greeks)  │   │
│  └─────────┬────────────────────┘   │
│            │                         │
│            ▼                         │
│  ┌──────────────────────────────┐   │
│  │ Professional Agent (Backend) │   │
│  │  - ProfessionalPricingAgent  │   │
│  │  - Domain value objects      │   │
│  │  - Infrastructure patterns   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

**Flow:**
1. Client sends MCP request to Docker container (STDIO)
2. MCP server receives via transport layer
3. MCP server calls professional agent
4. Agent does calculation (using domain + infrastructure)
5. MCP server formats response per protocol
6. Returns to client via STDIO

---

## 🧪 RUNNING TESTS

### **Option 1: Quick Test (Shell Script)**

```bash
# Run complete test suite
./axiom/mcp_servers/test_mcp_via_docker.sh
```

**Tests:**
- ✅ Docker image builds
- ✅ Container starts
- ✅ MCP protocol works
- ✅ Server initializes
- ✅ Tools discoverable
- ✅ Greeks calculation works

### **Option 2: Docker Compose**

```bash
# Start MCP server
cd axiom/mcp_servers
docker-compose -f docker-compose.test.yml up -d pricing-greeks-mcp

# Test interactively
docker exec -it pricing-greeks-mcp-test python

# Send MCP message
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  docker exec -i pricing-greeks-mcp-test python -m pricing_greeks.server
```

### **Option 3: Manual Docker**

```bash
# Build
docker build -t pricing-greeks-mcp -f axiom/mcp_servers/trading/pricing_greeks/Dockerfile .

# Run interactively
docker run -it pricing-greeks-mcp

# Send MCP request via STDIN
{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}
```

---

## 📊 EXPECTED TEST RESULTS

### **Initialize Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "1.0.0",
    "serverInfo": {
      "name": "pricing-greeks-mcp-server",
      "version": "1.0.0"
    },
    "capabilities": {
      "tools": true,
      "resources": true,
      "prompts": true
    }
  }
}
```

### **Tools List Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "calculate_greeks",
        "description": "Calculate option Greeks...",
        "inputSchema": {...}
      },
      {
        "name": "batch_greeks",
        ...
      },
      {
        "name": "validate_greeks",
        ...
      }
    ]
  }
}
```

### **Greeks Calculation Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{
      "type": "text",
      "text": "{\"success\": true, \"greeks\": {\"delta\": 0.52, ...}, \"price\": 10.45, \"calculation_time_us\": 850.2}"
    }]
  }
}
```

---

## 🔍 TROUBLESHOOTING

### **Issue: Docker build fails**
```bash
# Check Dockerfile
cat axiom/mcp_servers/trading/pricing_greeks/Dockerfile

# Build with verbose output
docker build --progress=plain -t pricing-greeks-mcp -f axiom/mcp_servers/trading/pricing_greeks/Dockerfile .
```

### **Issue: MCP server doesn't respond**
```bash
# Check logs
docker logs pricing-greeks-mcp-test

# Run with debug logging
docker run -e LOG_LEVEL=DEBUG -it pricing-greeks-mcp
```

### **Issue: Protocol errors**
- Ensure JSON-RPC 2.0 format exactly
- Check `jsonrpc: "2.0"` is present
- Validate message structure

---

## ✅ VALIDATION CHECKLIST

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] MCP server responds to initialize
- [ ] Tools are listed correctly
- [ ] Greeks calculation works
- [ ] Response format is valid JSON-RPC 2.0
- [ ] Performance meets <1ms target
- [ ] Error handling works

---

## 🚀 NEXT STEPS

1. **Test current MCP:** Run `./test_mcp_via_docker.sh`
2. **Fix any issues** found in testing
3. **Create remaining 11 MCPs** following same template
4. **Test all MCPs** via Docker
5. **Deploy to production** (each MCP separately)

---

**The MCP servers are Docker-ready and Claude Desktop compatible. Testing validates they work in containerized production environment.**