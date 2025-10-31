#!/usr/bin/env python3
"""Quick test of MCP server imports"""
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

print("Testing MCP Server Imports...\n")

# Test 1: Pricing Greeks
print("1. Testing pricing_greeks server...")
try:
    from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer
    print("   ✓ PASS\n")
except Exception as e:
    print(f"   ✗ FAIL: {e}\n")

# Test 2: Portfolio Risk  
print("2. Testing portfolio_risk server...")
try:
    from axiom.mcp_servers.trading.portfolio_risk.server import PortfolioRiskMCPServer
    print("   ✓ PASS\n")
except Exception as e:
    print(f"   ✗ FAIL: {e}\n")

# Test 3: Strategy Gen
print("3. Testing strategy_gen server...")
try:
    from axiom.mcp_servers.trading.strategy_gen.server import StrategyGenerationMCPServer
    print("   ✓ PASS\n")
except Exception as e:
    print(f"   ✗ FAIL: {e}\n")

print("Test complete!")