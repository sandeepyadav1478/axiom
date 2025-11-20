#!/usr/bin/env python3
"""
Axiom System Check - Comprehensive health verification
Checks all components: databases, MCP servers, GPU, API keys, etc.
"""

import sys
import importlib
import subprocess
from pathlib import Path

# Colors
G = '\033[92m'  # Green
R = '\033[91m'  # Red
Y = '\033[93m'  # Yellow
B = '\033[94m'  # Blue
BOLD = '\033[1m'
RESET = '\033[0m'

def print_section(title):
    print(f"\n{B}{BOLD}{'='*70}{RESET}")
    print(f"{B}{BOLD}{title:^70}{RESET}")
    print(f"{B}{BOLD}{'='*70}{RESET}\n")

def check_databases():
    """Check all 4 databases"""
    print_section("DATABASE HEALTH CHECK")
    
    databases = [
        ("PostgreSQL", "docker exec -it axiom_postgres pg_isready -U axiom"),
        ("Redis", "docker exec -it axiom_redis redis-cli -a axiom_redis ping"),
        ("Neo4j", "docker exec -it axiom_neo4j cypher-shell -u neo4j -p axiom_neo4j 'RETURN 1;'"),
        ("ChromaDB", "curl -s http://localhost:8000/api/v1/heartbeat"),
    ]
    
    results = []
    for name, cmd in databases:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 or "PONG" in result.stdout or "ready" in result.stdout:
                print(f"{G}✅{RESET} {name:15} - HEALTHY")
                results.append(True)
            else:
                print(f"{R}❌{RESET} {name:15} - FAILED")
                results.append(False)
        except Exception as e:
            print(f"{R}❌{RESET} {name:15} - ERROR: {str(e)[:30]}")
            results.append(False)
    
    return all(results)

def check_mcp_servers():
    """Check all 12 MCP servers"""
    print_section("MCP SERVERS CHECK")
    
    servers = [
        ('pricing_greeks', 'axiom.mcp.servers.public.trading.pricing_greeks.server'),
        ('portfolio_risk', 'axiom.mcp.servers.public.trading.portfolio_risk.server'),
        ('strategy_gen', 'axiom.mcp.servers.public.trading.strategy_gen.server'),
        ('execution', 'axiom.mcp.servers.public.trading.execution.server'),
        ('hedging', 'axiom.mcp.servers.public.trading.hedging.server'),
        ('performance', 'axiom.mcp.servers.public.analytics.performance.server'),
        ('market_data', 'axiom.mcp.servers.public.analytics.market_data.server'),
        ('volatility', 'axiom.mcp.servers.public.analytics.volatility.server'),
        ('regulatory', 'axiom.mcp.servers.public.compliance.regulatory.server'),
        ('system_health', 'axiom.mcp.servers.internal.monitoring.system_health.server'),
        ('guardrails', 'axiom.mcp.servers.internal.safety.guardrails.server'),
        ('interface', 'axiom.mcp.servers.internal.client.interface.server'),
    ]
    
    passed = 0
    for name, path in servers:
        try:
            importlib.import_module(path)
            print(f"{G}✅{RESET} {name:20}")
            passed += 1
        except Exception as e:
            print(f"{R}❌{RESET} {name:20} - {str(e)[:40]}")
    
    print(f"\n{BOLD}Result: {passed}/12 servers operational{RESET}")
    return passed == 12

def check_mcp_clients():
    """Check MCP clients"""
    print_section("MCP CLIENTS CHECK")
    
    try:
        from axiom.mcp.clients.internal.market_data_integrations import MarketDataAggregator
        print(f"{G}✅{RESET} MarketDataAggregator")
        client1 = True
    except Exception as e:
        print(f"{R}❌{RESET} MarketDataAggregator - {str(e)[:40]}")
        client1 = False
    
    try:
        from axiom.mcp.clients.internal.derivatives_data_mcp import DerivativesDataMCP
        print(f"{G}✅{RESET} DerivativesDataMCP")
        client2 = True
    except Exception as e:
        print(f"{R}❌{RESET} DerivativesDataMCP - {str(e)[:40]}")
        client2 = False
    
    return client1 and client2

def check_gpu():
    """Check GPU availability"""
    print_section("GPU CHECK")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"{G}✅{RESET} GPU: {gpu_name}")
            print(f"   VRAM: {memory:.2f} GB")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   PyTorch: {torch.__version__}")
            return True
        else:
            print(f"{Y}⚠️{RESET}  No GPU available (CPU mode)")
            return False
    except Exception as e:
        print(f"{R}❌{RESET} GPU check failed: {e}")
        return False

def check_api_keys():
    """Check API keys configuration"""
    print_section("API KEYS CHECK")
    
    import os
    
    keys = [
        ("Claude", "CLAUDE_API_KEY"),
        ("Tavily", "TAVILY_API_KEY"),
        ("Firecrawl", "FIRECRAWL_API_KEY"),
        ("Polygon", "POLYGON_API_KEY"),
        ("Alpha Vantage", "ALPHA_VANTAGE_API_KEY"),
        ("Finnhub", "FINNHUB_API_KEY"),
        ("LangChain", "LANGCHAIN_API_KEY"),
    ]
    
    configured = 0
    for name, key in keys:
        value = os.getenv(key)
        if value and not value.startswith("your") and not value.endswith("here"):
            print(f"{G}✅{RESET} {name:20} - configured")
            configured += 1
        else:
            print(f"{Y}⚠️{RESET}  {name:20} - not configured")
    
    print(f"\n{BOLD}Result: {configured}/{len(keys)} API keys configured{RESET}")
    return configured > 0

def check_dspy():
    """Check DSPy framework"""
    print_section("DSPy FRAMEWORK CHECK")
    
    try:
        import dspy
        print(f"{G}✅{RESET} DSPy version: {dspy.__version__}")
        
        from axiom.dspy_modules.optimizer import InvestmentBankingOptimizer
        print(f"{G}✅{RESET} InvestmentBankingOptimizer")
        
        from axiom.dspy_modules.hyde import InvestmentBankingHyDEModule
        print(f"{G}✅{RESET} InvestmentBankingHyDEModule")
        
        from axiom.dspy_modules.multi_query import InvestmentBankingMultiQueryModule
        print(f"{G}✅{RESET} InvestmentBankingMultiQueryModule")
        
        return True
    except Exception as e:
        print(f"{R}❌{RESET} DSPy error: {e}")
        return False

def main():
    """Run complete system check"""
    print(f"\n{BOLD}{B}{'='*70}")
    print(f"AXIOM SYSTEM HEALTH CHECK")
    print(f"{'='*70}{RESET}\n")
    
    results = {}
    
    # Run checks
    results['databases'] = check_databases()
    results['mcp_servers'] = check_mcp_servers()
    results['mcp_clients'] = check_mcp_clients()
    results['gpu'] = check_gpu()
    results['api_keys'] = check_api_keys()
    results['dspy'] = check_dspy()
    
    # Summary
    print_section("SYSTEM STATUS SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    
    for component, status in results.items():
        status_icon = f"{G}✅{RESET}" if status else f"{R}❌{RESET}"
        print(f"{status_icon} {component.replace('_', ' ').title():20} - {'PASS' if status else 'FAIL'}")
    
    print(f"\n{BOLD}Overall: {passed}/{total} components operational{RESET}")
    
    if passed == total:
        print(f"\n{G}{BOLD}🎉 ALL SYSTEMS OPERATIONAL!{RESET}")
        print(f"{G}Ready for development and production use.{RESET}\n")
        return 0
    elif passed >= total * 0.8:
        print(f"\n{Y}{BOLD}⚠️  MOSTLY OPERATIONAL ({passed}/{total}){RESET}")
        print(f"{Y}Some components need attention.{RESET}\n")
        return 1
    else:
        print(f"\n{R}{BOLD}❌ SYSTEM ISSUES DETECTED{RESET}")
        print(f"{R}Multiple components need attention.{RESET}\n")
        return 2

if __name__ == "__main__":
    sys.exit(main())