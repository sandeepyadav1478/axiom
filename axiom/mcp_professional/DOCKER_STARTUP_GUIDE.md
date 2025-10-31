# MCP Docker Startup Guide

## 🚀 Quick Start - Run All 12 MCP Servers

### Prerequisites
- Docker installed and running
- Docker Compose installed
- Project dependencies installed (`pip install -e .`)

### Start All Containers

```bash
# From project root
cd axiom/mcp_servers

# Build and start all 12 MCP servers
docker-compose up -d

# Verify all containers are running
docker ps
```

Expected output: 12 running containers

## 📦 Individual Server Management

### Start Specific Server
```bash
docker-compose up -d pricing-greeks-mcp
```

### Stop Specific Server
```bash
docker-compose stop pricing-greeks-mcp
```

### View Logs
```bash
docker-compose logs -f pricing-greeks-mcp
```

### Restart Server
```bash
docker-compose restart pricing-greeks-mcp
```

## 🔍 Troubleshooting

### Check Container Status
```bash
docker-compose ps
```

### View All Logs
```bash
docker-compose logs
```

### Rebuild After Code Changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Remove All Containers and Volumes
```bash
docker-compose down -v
```

## 📊 The 12 MCP Servers

### Trading Cluster (5)
1. `pricing-greeks-mcp` - Options pricing and Greeks (<1ms)
2. `portfolio-risk-mcp` - Portfolio risk management (<5ms)
3. `strategy-gen-mcp` - AI trading strategy generation
4. `execution-mcp` - Smart order routing
5. `hedging-mcp` - DRL-based hedging

### Analytics Cluster (3)
6. `performance-mcp` - P&L attribution
7. `market-data-mcp` - NBBO aggregation
8. `volatility-mcp` - AI volatility forecasting

### Support Cluster (4)
9. `regulatory-mcp` - Regulatory compliance
10. `system-health-mcp` - Platform monitoring
11. `guardrails-mcp` - AI safety validation
12. `interface-mcp` - Client orchestration

## 🧪 Testing

### Test Single Server
```bash
# Test pricing-greeks
./test_mcp_via_docker.sh
```

### Test MCP Protocol
```bash
# Send test request to server
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","clientInfo":{"name":"test","version":"1.0"}}}' | \
docker run -i --rm pricing-greeks-mcp
```

### Validate All Servers
```bash
# From project root
python3 scripts/validate_all_mcp_servers.py
```

## 🌐 Network

All containers are connected via `axiom-mcp-network`:
- Bridge driver for inter-container communication
- Isolated from other Docker networks
- Containers can communicate by service name

## 📝 Common Commands

```bash
# See what's running
docker-compose ps

# Stop all
docker-compose stop

# Start all
docker-compose start

# Restart all
docker-compose restart

# View resource usage
docker stats

# Clean up everything
docker-compose down -v --remove-orphans
```

## ⚠️ Troubleshooting

### Container Won't Start
1. Check logs: `docker-compose logs [service-name]`
2. Check for port conflicts
3. Rebuild: `docker-compose build --no-cache [service-name]`

### Import Errors
1. Ensure you're in project root when building
2. Check PYTHONPATH is set correctly
3. Verify all dependencies in requirements.txt

### Performance Issues
1. Allocate more memory to Docker
2. Use `USE_GPU=true` if GPU available
3. Check system resources with `docker stats`

## 🎯 Expected Behavior

When all 12 containers are running:
```bash
$ docker ps
CONTAINER ID   IMAGE                      STATUS
abc123...      pricing-greeks-mcp        Up 2 minutes
def456...      portfolio-risk-mcp        Up 2 minutes
... (10 more containers)
```

Each container:
- ✅ Listens on STDIO for MCP protocol
- ✅ Responds to initialize, tools/list, tools/call
- ✅ Compatible with Claude Desktop
- ✅ Includes health checks
- ✅ Auto-restarts on failure

## 📚 Additional Resources

- [MCP Architecture Plan](MCP_ARCHITECTURE_PLAN.md)
- [MCP Testing Guide](MCP_TESTING_GUIDE.md)
- [MCP Implementation Status](MCP_IMPLEMENTATION_STATUS.md)
- [Validation Status](../MCP_VALIDATION_STATUS.md)

---

**Need help?** Check logs first, then refer to troubleshooting section above.