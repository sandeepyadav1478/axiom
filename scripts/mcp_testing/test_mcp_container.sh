#!/bin/bash
# Test running MCP container

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONTAINER="pricing-greeks-mcp"

echo "=========================================="
echo "MCP CONTAINER TESTING"
echo "=========================================="
echo ""

# Test 1: Initialize
echo -e "${YELLOW}Test 1: Initialize${NC}"
docker exec $CONTAINER python << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, "/app")
from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer
import asyncio
import json

async def test():
    server = PricingGreeksMCPServer()
    req = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","clientInfo":{"name":"test","version":"1.0"}}}
    result = await server.handle_message(req)
    print(json.dumps(result, indent=2))

asyncio.run(test())
PYTHON_SCRIPT
echo ""

# Test 2: List Tools
echo -e "${YELLOW}Test 2: List Tools${NC}"
docker exec $CONTAINER python << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, "/app")
from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer
import asyncio
import json

async def test():
    server = PricingGreeksMCPServer()
    req = {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
    result = await server.handle_message(req)
    tools = result.get("result", {}).get("tools", [])
    print(f"Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

asyncio.run(test())
PYTHON_SCRIPT
echo ""

# Test 3: Calculate Greeks
echo -e "${YELLOW}Test 3: Calculate Greeks${NC}"
docker exec $CONTAINER python << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, "/app")
from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer
import asyncio
import json

async def test():
    server = PricingGreeksMCPServer()
    req = {
        "jsonrpc":"2.0",
        "id":3,
        "method":"tools/call",
        "params":{
            "name":"calculate_greeks",
            "arguments":{
                "spot":100.0,
                "strike":100.0,
                "time_to_maturity":1.0,
                "risk_free_rate":0.03,
                "volatility":0.25,
                "option_type":"call"
            }
        }
    }
    result = await server.handle_message(req)
    print(json.dumps(result, indent=2))

asyncio.run(test())
PYTHON_SCRIPT
echo ""

echo -e "${GREEN}✅ All tests complete!${NC}"
echo ""
echo "Container status:"
docker ps | grep pricing-greeks