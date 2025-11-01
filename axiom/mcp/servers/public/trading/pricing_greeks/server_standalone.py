"""
Pricing Greeks MCP Server - Standalone Simplified

Simple MCP server that exposes ultra-fast Greeks calculation.
Does NOT import complex axiom internals - stays lightweight.

This is a thin API layer over the pricing functionality.
"""

import asyncio
import sys
import json
from typing import Dict
import logging

# Simple base MCP (no complex dependencies)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from shared.mcp_protocol import MCPRequest, MCPResponse, MCPErrorCode


class PricingGreeksMCPServer:
    """
    Lightweight MCP server for Greeks calculation
    
    Exposes tools via MCP protocol without heavy dependencies.
    Calls actual pricing engine only when needed.
    """
    
    def __init__(self):
        self.name = "pricing-greeks-mcp-server"
        self.version = "1.0.0"
        
        # Lazy loading - only import when actually called
        self._pricing_engine = None
    
    def _get_pricing_engine(self):
        """Lazy load pricing engine"""
        if self._pricing_engine is None:
            # Import only when needed
            from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
            self._pricing_engine = UltraFastGreeksEngine(use_gpu=False)
        return self._pricing_engine
    
    async def handle_message(self, message: Dict) -> Dict:
        """Handle MCP message"""
        try:
            request = MCPRequest.from_dict(message)
            
            if request.method == "initialize":
                result = {
                    "protocolVersion": "1.0.0",
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    },
                    "capabilities": {
                        "tools": True
                    }
                }
                return MCPResponse.success(request.id, result).to_dict()
            
            elif request.method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "calculate_greeks",
                            "description": "Calculate option Greeks (<1ms)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "spot": {"type": "number"},
                                    "strike": {"type": "number"},
                                    "time_to_maturity": {"type": "number"},
                                    "risk_free_rate": {"type": "number"},
                                    "volatility": {"type": "number"},
                                    "option_type": {"type": "string", "enum": ["call", "put"]}
                                },
                                "required": ["spot", "strike", "time_to_maturity", "risk_free_rate", "volatility"]
                            }
                        }
                    ]
                }
                return MCPResponse.success(request.id, result).to_dict()
            
            elif request.method == "tools/call":
                tool_name = request.params.get("name")
                arguments = request.params.get("arguments", {})
                
                if tool_name == "calculate_greeks":
                    # Call actual pricing engine
                    engine = self._get_pricing_engine()
                    result_greeks = engine.calculate_greeks(
                        spot=float(arguments['spot']),
                        strike=float(arguments['strike']),
                        time_to_maturity=arguments['time_to_maturity'],
                        risk_free_rate=arguments['risk_free_rate'],
                        volatility=arguments['volatility'],
                        option_type=arguments.get('option_type', 'call')
                    )
                    
                    result = {
                        "content": [{
                            "type": "text",
                            "text": json.dumps({
                                "success": True,
                                "greeks": {
                                    "delta": result_greeks.delta,
                                    "gamma": result_greeks.gamma,
                                    "theta": result_greeks.theta,
                                    "vega": result_greeks.vega,
                                    "rho": result_greeks.rho
                                },
                                "price": result_greeks.price,
                                "calculation_time_us": result_greeks.calculation_time_us
                            }, indent=2)
                        }]
                    }
                    return MCPResponse.success(request.id, result).to_dict()
            
            else:
                return MCPResponse.error(
                    request.id,
                    MCPErrorCode.METHOD_NOT_FOUND.value,
                    f"Method not found: {request.method}"
                ).to_dict()
        
        except Exception as e:
            return MCPResponse.error(
                request.id if 'request' in locals() else None,
                MCPErrorCode.INTERNAL_ERROR.value,
                str(e)
            ).to_dict()
    
    async def run_stdio(self):
        """Run via STDIO"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                message = json.loads(line)
                response = await self.handle_message(message)
                
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                break


if __name__ == "__main__":
    async def main():
        server = PricingGreeksMCPServer()
        await server.run_stdio()
    
    asyncio.run(main())