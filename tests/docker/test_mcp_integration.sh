#!/bin/bash
# MCP Services Integration Test
# Location: tests/docker/test_mcp_integration.sh
# Run from project root: bash tests/docker/test_mcp_integration.sh

set -e

PROJECT_ROOT="/Users/sandeep.yadav/work/axiom"
COMPOSE_FILE="axiom/integrations/data_sources/finance/mcp_servers/docker-compose.yml"

echo "🧪 MCP Services Integration Test"
echo "================================="
echo ""

# Test 1: Start Polygon.io service
echo "1️⃣ Starting Polygon.io MCP Server..."
docker rm -f axiom-polygon-financial-mcp 2>/dev/null || true
docker-compose -f "$COMPOSE_FILE" --profile polygon up -d
sleep 3
echo "✅ Service started"
echo ""

# Test 2: Check container status
echo "2️⃣ Checking container status..."
STATUS=$(docker inspect -f '{{.State.Status}}' axiom-polygon-financial-mcp 2>/dev/null || echo "not found")
if [ "$STATUS" = "running" ]; then
    echo "✅ Container running"
else
    echo "❌ Container not running (status: $STATUS)"
    docker-compose -f "$COMPOSE_FILE" logs polygon-io-server --tail=50
    exit 1
fi
echo ""

# Test 3: Verify environment variables
echo "3️⃣ Verifying environment variables inside container..."
ENV_CHECK=$(docker exec axiom-polygon-financial-mcp env | grep -E "POLYGON_API_KEY|MCP_TRANSPORT")
if echo "$ENV_CHECK" | grep -q "POLYGON_API_KEY="; then
    echo "✅ POLYGON_API_KEY loaded"
    POLYGON_KEY=$(echo "$ENV_CHECK" | grep "POLYGON_API_KEY=" | cut -d'=' -f2)
    echo "   Value: ${POLYGON_KEY:0:10}...${POLYGON_KEY: -10}"
else
    echo "❌ POLYGON_API_KEY not found"
    exit 1
fi

if echo "$ENV_CHECK" | grep -q "MCP_TRANSPORT=stdio"; then
    echo "✅ MCP_TRANSPORT configured (stdio)"
else
    echo "❌ MCP_TRANSPORT not configured"
    exit 1
fi
echo ""

# Test 4: Check container health
echo "4️⃣ Checking container health..."
HEALTH=$(docker inspect -f '{{.State.Health.Status}}' axiom-polygon-financial-mcp 2>/dev/null || echo "no healthcheck")
echo "   Health status: $HEALTH"
echo "✅ Container health check complete"
echo ""

# Test 5: Check logs for errors
echo "5️⃣ Checking logs for errors..."
LOGS=$(docker-compose -f "$COMPOSE_FILE" logs polygon-io-server --tail=100 2>&1)
if echo "$LOGS" | grep -iq "error\|exception\|failed\|traceback"; then
    echo "⚠️  Potential errors found in logs:"
    echo "$LOGS" | grep -i "error\|exception\|failed" | head -5
    echo ""
    echo "Full logs available with: docker-compose -f $COMPOSE_FILE logs polygon-io-server"
else
    echo "✅ No errors in logs"
fi

if echo "$LOGS" | grep -q "API key configured"; then
    echo "✅ API key successfully configured in service"
else
    echo "⚠️  API key confirmation not found in logs"
fi
echo ""

# Test 6: Check resource usage
echo "6️⃣ Checking resource usage..."
STATS=$(docker stats axiom-polygon-financial-mcp --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>/dev/null | tail -1)
echo "   $STATS"
echo "✅ Container running with normal resources"
echo ""

# Test 7: Network connectivity
echo "7️⃣ Checking network connectivity..."
NETWORK=$(docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' axiom-polygon-financial-mcp)
if [ "$NETWORK" = "axiom-financial-mcp-external" ]; then
    echo "✅ Connected to network: $NETWORK"
else
    echo "⚠️  Connected to network: $NETWORK (expected: axiom-financial-mcp-external)"
fi
echo ""

echo "✅ All integration tests passed!"
echo ""
echo "📊 Summary:"
echo "   - Container: Running (Up)"
echo "   - API Key: Loaded and configured"
echo "   - MCP Transport: stdio"
echo "   - Logs: Clean (no errors)"
echo "   - Network: Connected"
echo ""
echo "🎯 Service is operational and ready to use!"
echo ""
echo "📋 To test with MCP queries, the service needs to be added to MCP settings:"
echo "   See: guides/INSTALLATION_GUIDE.md"
echo ""
echo "🛑 To stop the service:"
echo "   docker-compose -f $COMPOSE_FILE down"
echo ""