#!/bin/bash
# Test All Financial Data Providers with Real Queries
# Location: tests/integration/test_all_provider_queries.sh

echo "🧪 Testing All Financial Data Provider Containers"
echo "=================================================="
echo ""

# Test 1: Tavily (8001)
echo "1️⃣ Testing Tavily Provider (8001)..."
RESPONSE=$(curl -s -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple Inc financial performance", "max_results": 3}' | head -c 200)
if echo "$RESPONSE" | grep -q "results\|data"; then
    echo "✅ Tavily: Connected and responding"
    echo "   Response: ${RESPONSE:0:100}..."
else
    echo "❌ Tavily: Not responding"
fi
echo ""

# Test 2: FMP (8002)
echo "2️⃣ Testing FMP Provider (8002)..."
RESPONSE=$(curl -s http://localhost:8002/quote/AAPL | head -c 200)
if echo "$RESPONSE" | grep -q "symbol\|price\|AAPL"; then
    echo "✅ FMP: Connected and responding"
    echo "   Response: ${RESPONSE:0:100}..."
else
    echo "❌ FMP: Not responding"
fi
echo ""

# Test 3: Finnhub (8003)
echo "3️⃣ Testing Finnhub Provider (8003)..."
RESPONSE=$(curl -s http://localhost:8003/quote/AAPL | head -c 200)
if echo "$RESPONSE" | grep -q "c\|price\|current"; then
    echo "✅ Finnhub: Connected and responding"
    echo "   Response: ${RESPONSE:0:100}..."
else
    echo "❌ Finnhub: Not responding"
fi
echo ""

# Test 4: Alpha Vantage (8004)
echo "4️⃣ Testing Alpha Vantage Provider (8004)..."
RESPONSE=$(curl -s http://localhost:8004/quote/AAPL | head -c 200)
if echo "$RESPONSE" | grep -q "symbol\|price\|AAPL"; then
    echo "✅ Alpha Vantage: Connected and responding"
    echo "   Response: ${RESPONSE:0:100}..."
else
    echo "❌ Alpha Vantage: Not responding"
fi
echo ""

echo "=================================================="
echo "✅ Provider Container Connectivity Test Complete"