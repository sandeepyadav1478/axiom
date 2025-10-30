#!/bin/bash
# Test MCP server with various queries

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "MCP SERVER QUERY TESTING"
echo "=========================================="
echo ""

CONTAINER="pricing-greeks-mcp"

# Test 1: Initialize
echo -e "${YELLOW}Test 1: Initialize${NC}"
INIT_REQUEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","clientInfo":{"name":"test-client","version":"1.0.0"}}}'
echo "$INIT_REQUEST" | docker exec -i $CONTAINER python -c "import sys, json; msg=sys.stdin.read(); print(msg)" | \
    docker exec -i $CONTAINER sh -c 'cat > /tmp/req.json && python -c "import sys; sys.path.insert(0, \"/app\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.load(open(\"/tmp/req.json\")); result = asyncio.run(server.handle_message(req)); print(json.dumps(result, indent=2))"'

echo ""

# Test 2: List Tools
echo -e "${YELLOW}Test 2: List Tools${NC}"
TOOLS_REQUEST='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
echo "$TOOLS_REQUEST" | docker exec -i $CONTAINER sh -c 'cat > /tmp/tools.json && python -c "import sys; sys.path.insert(0, \"/app\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.load(open(\"/tmp/tools.json\")); result = asyncio.run(server.handle_message(req)); print(json.dumps(result, indent=2))"'

echo ""

# Test 3: Calculate Greeks
echo -e "${YELLOW}Test 3: Calculate Greeks${NC}"
GREEKS_REQUEST='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculate_greeks","arguments":{"spot":100.0,"strike":100.0,"time_to_maturity":1.0,"risk_free_rate":0.03,"volatility":0.25,"option_type":"call"}}}'
echo "$GREEKS_REQUEST" | docker exec -i $CONTAINER sh -c 'cat > /tmp/greeks.json && python -c "import sys; sys.path.insert(0, \"/app\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.load(open(\"/tmp/greeks.json\")); result = asyncio.run(server.handle_message(req)); print(json.dumps(result, indent=2))"'

echo ""
echo -e "${GREEN}All tests complete!${NC}"