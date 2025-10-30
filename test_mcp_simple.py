#!/usr/bin/env python3
"""Simple test of running MCP container"""
import subprocess
import json

def test_mcp_server(container_name="pricing-greeks-mcp"):
    """Test MCP server with simple queries"""
    
    print("=" * 60)
    print("MCP SERVER TESTING")
    print("=" * 60)
    print()
    
    # Test 1: Initialize
    print("Test 1: Initialize Request")
    print("-" * 60)
    init_req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "1.0.0",
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }
    
    cmd = f'echo \'{json.dumps(init_req)}\' | docker exec -i {container_name} python -c "import sys; sys.path.insert(0, \\"/app\\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.loads(sys.stdin.read()); result = asyncio.run(server.handle_message(req)); print(json.dumps(result, indent=2))"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    print()
    
    # Test 2: List Tools  
    print("Test 2: List Tools")
    print("-" * 60)
    tools_req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    cmd = f'echo \'{json.dumps(tools_req)}\' | docker exec -i {container_name} python -c "import sys; sys.path.insert(0, \\"/app\\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.loads(sys.stdin.read()); result = asyncio.run(server.handle_message(req)); tools = result.get(\\"result\\", {}).get(\\"tools\\", []); print(f\\"Found {len(tools)} tools:\\"); [print(f\\"  - {t[\\"name\\"]}: {t[\\"description\\"]}\\") for t in tools]"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print()
    
    # Test 3: Calculate Greeks
    print("Test 3: Calculate Greeks")
    print("-" * 60)
    greeks_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_greeks",
            "arguments": {
                "spot": 100.0,
                "strike": 100.0,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.03,
                "volatility": 0.25,
                "option_type": "call"
            }
        }
    }
    
    cmd = f'echo \'{json.dumps(greeks_req)}\' | docker exec -i {container_name} python -c "import sys; sys.path.insert(0, \\"/app\\"); from axiom.mcp_servers.trading.pricing_greeks.server import PricingGreeksMCPServer; import asyncio, json; server = PricingGreeksMCPServer(); req = json.loads(sys.stdin.read()); result = asyncio.run(server.handle_message(req)); print(json.dumps(result, indent=2))"'
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    print()
    
    print("=" * 60)
    print("✅ All tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_mcp_server()