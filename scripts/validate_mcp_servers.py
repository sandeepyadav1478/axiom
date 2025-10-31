#!/usr/bin/env python3
"""
Validate all MCP servers for import errors and basic functionality
"""
import sys
import importlib
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Define all 12 MCP servers
MCP_SERVERS = [
    # Trading Cluster (5)
    ("axiom.mcp_servers.trading.pricing_greeks.server", "PricingGreeksMCPServer"),
    ("axiom.mcp_servers.trading.portfolio_risk.server", "PortfolioRiskMCPServer"),
    ("axiom.mcp_servers.trading.strategy_gen.server", "StrategyGenerationMCPServer"),
    ("axiom.mcp_servers.trading.execution.server", "ExecutionMCPServer"),
    ("axiom.mcp_servers.trading.hedging.server", "HedgingMCPServer"),
    
    # Analytics Cluster (3)
    ("axiom.mcp_servers.analytics.performance.server", "PerformanceAnalyticsMCPServer"),
    ("axiom.mcp_servers.analytics.market_data.server", "MarketDataMCPServer"),
    ("axiom.mcp_servers.analytics.volatility.server", "VolatilityAnalysisMCPServer"),
    
    # Support Cluster (4)
    ("axiom.mcp_servers.compliance.regulatory.server", "RegulatoryComplianceMCPServer"),
    ("axiom.mcp_servers.monitoring.system_health.server", "SystemHealthMCPServer"),
    ("axiom.mcp_servers.safety.guardrails.server", "GuardrailsMCPServer"),
    ("axiom.mcp_servers.client.interface.server", "ClientInterfaceMCPServer"),
]

def test_server_import(module_path: str, class_name: str) -> Tuple[bool, str]:
    """Test if a server module can be imported"""
    try:
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            return False, f"Module imported but class '{class_name}' not found"
        return True, "OK"
    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def main():
    print("="*80)
    print("MCP SERVER VALIDATION")
    print("="*80)
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for module_path, class_name in MCP_SERVERS:
        server_name = module_path.split('.')[-2]  # Get server name from path
        print(f"Testing: {server_name:30s} ... ", end='', flush=True)
        
        success, message = test_server_import(module_path, class_name)
        
        if success:
            print(f"✓ PASS")
            passed += 1
        else:
            print(f"✗ FAIL")
            print(f"  Error: {message}")
            failed += 1
        
        results.append((server_name, success, message))
    
    print()
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Total Servers: {len(MCP_SERVERS)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("FAILED SERVERS:")
        print("-"*80)
        for name, success, message in results:
            if not success:
                print(f"  {name}: {message}")
        print()
        sys.exit(1)
    else:
        print("✓ ALL MCP SERVERS VALIDATED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    main()