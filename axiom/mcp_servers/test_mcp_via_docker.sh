#!/bin/bash

# Test MCP Servers via Docker
# Tests that MCP servers work correctly in containerized environment

set -e

echo "=================================================="
echo "MCP SERVER DOCKER TESTING"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test Pricing Greeks MCP Server
echo -e "\n${GREEN}→ Testing Pricing Greeks MCP Server${NC}"

# Build Docker image
echo "Building Docker image..."
cd ../..
docker build -t pricing-greeks-mcp-test -f axiom/mcp_servers/trading/pricing_greeks/Dockerfile .

# Test MCP server via STDIO
echo -e "\n${GREEN}→ Testing MCP Protocol via STDIO${NC}"

# Create test request (MCP initialize)
TEST_REQUEST='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"1.0.0","clientInfo":{"name":"test-client","version":"1.0.0"}}}'

# Send to container and get response
echo "Sending MCP initialize request..."
echo "$TEST_REQUEST" | docker run -i --rm pricing-greeks-mcp-test > /tmp/mcp_response.json

# Check response
if grep -q "protocolVersion" /tmp/mcp_response.json; then
    echo -e "${GREEN}✓ MCP server initialized successfully${NC}"
    cat /tmp/mcp_response.json | python -m json.tool
else
    echo -e "${RED}✗ MCP initialization failed${NC}"
    cat /tmp/mcp_response.json
    exit 1
fi

# Test tools/list
echo -e "\n${GREEN}→ Testing tools/list${NC}"
TOOLS_REQUEST='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

echo "$TOOLS_REQUEST" | docker run -i --rm pricing-greeks-mcp-test > /tmp/mcp_tools.json

if grep -q "calculate_greeks" /tmp/mcp_tools.json; then
    echo -e "${GREEN}✓ Tools listed successfully${NC}"
    echo "Available tools:"
    cat /tmp/mcp_tools.json | python -m json.tool | grep '"name"'
else
    echo -e "${RED}✗ Tools listing failed${NC}"
    exit 1
fi

# Test actual Greeks calculation
echo -e "\n${GREEN}→ Testing calculate_greeks tool${NC}"
GREEKS_REQUEST='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"calculate_greeks","arguments":{"spot":100.0,"strike":100.0,"time_to_maturity":1.0,"risk_free_rate":0.03,"volatility":0.25,"option_type":"call"}}}'

echo "$GREEKS_REQUEST" | docker run -i --rm pricing-greeks-mcp-test > /tmp/mcp_greeks.json

if grep -q "delta" /tmp/mcp_greeks.json; then
    echo -e "${GREEN}✓ Greeks calculated successfully${NC}"
    cat /tmp/mcp_greeks.json | python -m json.tool
else
    echo -e "${RED}✗ Greeks calculation failed${NC}"
    cat /tmp/mcp_greeks.json
    exit 1
fi

# Cleanup
rm -f /tmp/mcp_*.json

echo -e "\n=================================================="
echo -e "${GREEN}ALL MCP DOCKER TESTS PASSED${NC}"
echo "=================================================="
echo ""
echo "✓ Docker image builds successfully"
echo "✓ MCP server starts in container"
echo "✓ MCP protocol works via STDIO"
echo "✓ Server responds to initialize"
echo "✓ Tools are discoverable"
echo "✓ Greeks calculation works"
echo ""
echo "MCP server is Docker-ready and Claude Desktop compatible!"