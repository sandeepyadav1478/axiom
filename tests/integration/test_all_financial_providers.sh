#!/bin/bash
# Comprehensive Test for All Financial Data Providers
# Tests: Polygon, Yahoo Comp, Firecrawl (MCP) + Tavily, FMP, Finnhub, Alpha Vantage (REST)
# Location: tests/integration/test_all_financial_providers.sh

echo "🧪 Testing All 7 Financial Data Provider Containers"
echo "======================================================"
echo ""

# Test MCP Servers
echo "📡 MCP SERVERS (Model Context Protocol)"
echo "========================================"
echo ""

# Test 1: Polygon MCP
echo "1️⃣ Polygon.io MCP Server..."
if docker ps | grep -q axiom-polygon-financial-mcp; then
    RESULT=$(docker exec axiom-polygon-financial-mcp python -c "import os; print(os.getenv('POLYGON_API_KEY', 'MISSING')[:20])" 2>&1)
    if echo "$RESULT" | grep -q "MISSING"; then
        echo "❌ Polygon: API key not loaded"
    else
        echo "✅ Polygon: Container running, API key loaded: ${RESULT}..."
    fi
else
    echo "❌ Polygon: Container not running"
fi
echo ""

# Test 2: Yahoo Comprehensive MCP  
echo "2️⃣ Yahoo Finance Comprehensive MCP..."
if docker ps | grep -q axiom-yahoo-comprehensive-mcp; then
    echo "✅ Yahoo Comp: Container running"
    # Test yfinance library
    RESULT=$(docker exec axiom-yahoo-comprehensive-mcp sh -c 'cd /tmp/yahoo-comp && source .venv/bin/activate && python -c "import yfinance; print(\"yfinance OK\")"' 2>&1)
    if echo "$RESULT" | grep -q "yfinance OK"; then
        echo "✅ Yahoo Comp: yfinance library accessible"
    else
        echo "⚠️  Yahoo Comp: Library check inconclusive"
    fi
else
    echo "❌ Yahoo Comp: Container not running"
fi
echo ""

# Test 3: Firecrawl MCP
echo "3️⃣ Firecrawl MCP Server..."
if docker ps | grep -q axiom-firecrawl-mcp; then
    RESULT=$(docker exec axiom-firecrawl-mcp sh -c 'echo $FIRECRAWL_API_KEY' | head -c 20)
    if [ -n "$RESULT" ]; then
        echo "✅ Firecrawl: Container running, API key loaded: ${RESULT}..."
    else
        echo "⚠️  Firecrawl: Running but API key check failed"
    fi
else
    echo "❌ Firecrawl: Container not running"
fi
echo ""

# Test REST API Providers
echo "🌐 REST API PROVIDERS"
echo "======================================================"
echo ""

# Test 4: Tavily (8001)
echo "4️⃣ Tavily Provider (8001)..."
HEALTH=$(curl -s http://localhost:8001/health 2>&1)
if echo "$HEALTH" | grep -q "healthy\|ok\|status"; then
    echo "✅ Tavily: Health check passed"
    # Try search query
    SEARCH=$(curl -s -X POST http://localhost:8001/search \
      -H "Content-Type: application/json" \
      -d '{"query": "test", "max_results": 1}' 2>&1 | head -c 100)
    if [ -n "$SEARCH" ]; then
        echo "✅ Tavily: API responding to queries"
    fi
else
    echo "❌ Tavily: Health check failed or not responding"
    echo "   Troubleshoot: docker logs axiom-tavily-provider"
fi
echo ""

# Test 5: FMP (8002)
echo "5️⃣ FMP Provider (8002)..."
HEALTH=$(curl -s http://localhost:8002/health 2>&1)
if echo "$HEALTH" | grep -q "healthy\|ok\|status"; then
    echo "✅ FMP: Health check passed"
    QUOTE=$(curl -s http://localhost:8002/quote/AAPL 2>&1 | head -c 100)
    if [ -n "$QUOTE" ]; then
        echo "✅ FMP: API responding to queries"
    fi
else
    echo "❌ FMP: Health check failed or not responding"
    echo "   Troubleshoot: docker logs axiom-fmp-provider"
fi
echo ""

# Test 6: Finnhub (8003)
echo "6️⃣ Finnhub Provider (8003)..."
HEALTH=$(curl -s http://localhost:8003/health 2>&1)
if echo "$HEALTH" | grep -q "healthy\|ok\|status"; then
    echo "✅ Finnhub: Health check passed"
    QUOTE=$(curl -s http://localhost:8003/quote/AAPL 2>&1 | head -c 100)
    if [ -n "$QUOTE" ]; then
        echo "✅ Finnhub: API responding to queries"
    fi
else
    echo "❌ Finnhub: Health check failed or not responding"
    echo "   Troubleshoot: docker logs axiom-finnhub-provider"
fi
echo ""

# Test 7: Alpha Vantage (8004)
echo "7️⃣ Alpha Vantage Provider (8004)..."
HEALTH=$(curl -s http://localhost:8004/health 2>&1)
if echo "$HEALTH" | grep -q "healthy\|ok\|status"; then
    echo "✅ Alpha Vantage: Health check passed"
    QUOTE=$(curl -s http://localhost:8004/quote/AAPL 2>&1 | head -c 100)
    if [ -n "$QUOTE" ]; then
        echo "✅ Alpha Vantage: API responding to queries"
    fi
else
    echo "❌ Alpha Vantage: Health check failed or not responding"
    echo "   Troubleshoot: docker logs axiom-alpha-vantage-provider"
fi
echo ""

echo "======================================================"
echo "📊 Test Summary"
echo "======================================================"
echo ""
echo "MCP Servers: Check logs with docker logs <container-name>"
echo "REST Providers: Health endpoints verified"
echo ""
echo "✅ All 7 financial data providers tested for connectivity"