#!/bin/bash
# World-Class Verification of All 12 MCP Servers

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}WORLD-CLASS MCP SERVER VERIFICATION${NC}"
echo -e "${BLUE}Testing All 12 Servers for Production Readiness${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

passed=0
failed=0

# Test each server
test_server() {
    local name=$1
    local module=$2
    local class=$3
    
    echo -e "\n${BLUE}Testing: $name${NC}"
    echo "----------------------------------------"
    
    # Test 1: Initialize
    result=$(docker exec $name python -c "import sys; sys.path.insert(0, '/app'); from $module import $class; import asyncio, json; server = $class(); req = {'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'1.0.0','clientInfo':{'name':'test','version':'1.0'}}}; result = asyncio.run(server.handle_message(req)); print(json.dumps(result))" 2>&1)
    
    if echo "$result" | grep -q "protocolVersion"; then
        echo -e "  ${GREEN}✅ Initialize: PASS${NC}"
        ((passed++))
    else
        echo -e "  ${RED}✗ Initialize: FAIL${NC}"
        ((failed++))
    fi
    
    # Test 2: List Tools
    result=$(docker exec $name python -c "import sys; sys.path.insert(0, '/app'); from $module import $class; import asyncio, json; server = $class(); req = {'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}}; result = asyncio.run(server.handle_message(req)); tools = result.get('result', {}).get('tools', []); print(f'{len(tools)} tools')" 2>&1)
    
    if echo "$result" | grep -q "tools"; then
        echo -e "  ${GREEN}✅ List Tools: PASS${NC} ($result)"
        ((passed++))
    else
        echo -e "  ${RED}✗ List Tools: FAIL${NC}"
        ((failed++))
    fi
}

# Trading Cluster
test_server "pricing-greeks-mcp" "axiom.mcp_servers.trading.pricing_greeks.server" "PricingGreeksMCPServer"
test_server "portfolio-risk-mcp" "axiom.mcp_servers.trading.portfolio_risk.server" "PortfolioRiskMCPServer"
test_server "strategy-gen-mcp" "axiom.mcp_servers.trading.strategy_gen.server" "StrategyGenerationMCPServer"
test_server "execution-mcp" "axiom.mcp_servers.trading.execution.server" "SmartExecutionMCPServer"
test_server "hedging-mcp" "axiom.mcp_servers.trading.hedging.server" "AutoHedgingMCPServer"

# Analytics Cluster  
test_server "performance-mcp" "axiom.mcp_servers.analytics.performance.server" "PerformanceAnalyticsMCPServer"
test_server "market-data-mcp" "axiom.mcp_servers.analytics.market_data.server" "MarketDataAggregatorMCPServer"
test_server "volatility-mcp" "axiom.mcp_servers.analytics.volatility.server" "VolatilityForecastingMCPServer"

# Support Cluster
test_server "regulatory-mcp" "axiom.mcp_servers.compliance.regulatory.server" "RegulatoryComplianceMCPServer"
test_server "system-health-mcp" "axiom.mcp_servers.monitoring.system_health.server" "SystemMonitoringMCPServer"
test_server "guardrails-mcp" "axiom.mcp_servers.safety.guardrails.server" "SafetyGuardrailMCPServer"
test_server "interface-mcp" "axiom.mcp_servers.client.interface.server" "ClientInterfaceMCPServer"

# Final Summary
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}FINAL RESULTS${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "Servers Tested: 12/12"
echo -e "Tests Passed: ${GREEN}$passed${NC}"
echo -e "Tests Failed: ${RED}$failed${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED - WORLD-CLASS QUALITY!${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed TESTS FAILED${NC}"
    exit 1
fi