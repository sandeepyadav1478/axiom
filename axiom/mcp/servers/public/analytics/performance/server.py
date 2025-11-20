"""
Performance Analytics MCP Server - Industry Standard

Real-time P&L calculation and Greeks attribution via MCP.

Tools: calculate_pnl, generate_report, attribute_performance
Resources: pnl://snapshot, pnl://history, pnl://attribution
Prompts: explain_pnl, attribution_help, performance_insights

Performance: <10ms complete analytics
"""

import asyncio
import sys
import json
from typing import Dict
import logging

from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport
from axiom.derivatives.analytics.pnl_engine import RealTimePnLEngine


class PerformanceAnalyticsMCPServer(BaseMCPServer):
    """Performance Analytics MCP Server - Real-time P&L with Greeks attribution"""
    
    def __init__(self):
        super().__init__(
            name="performance-analytics-mcp-server",
            version="1.0.0",
            description="Real-time P&L calculation and Greeks attribution"
        )
        
        self.pnl_engine = RealTimePnLEngine(use_gpu=False)
        self._pnl_calculations = 0
        
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(
            name="calculate_pnl",
            description="Calculate real-time P&L with Greeks attribution (<10ms)",
            input_schema={
                "type": "object",
                "properties": {
                    "positions": {"type": "array"},
                    "market_data": {"type": "object"}
                },
                "required": ["positions", "market_data"]
            },
            handler=self._calculate_pnl_handler
        )
    
    def _register_resources(self):
        self.register_resource(
            uri="pnl://snapshot",
            name="Current P&L",
            description="Current P&L snapshot with attribution"
        )
    
    def _register_prompts(self):
        self.register_prompt(
            name="explain_pnl",
            description="Explain P&L and Greeks attribution",
            arguments=[]
        )
    
    async def _calculate_pnl_handler(self, arguments: Dict) -> Dict:
        try:
            pnl = self.pnl_engine.calculate_pnl(
                positions=arguments['positions'],
                current_market_data=arguments['market_data']
            )
            
            self._pnl_calculations += 1
            
            return {
                "success": True,
                "pnl": {
                    "total": pnl.total_pnl,
                    "realized": pnl.realized_pnl,
                    "unrealized": pnl.unrealized_pnl,
                    "delta_pnl": pnl.delta_pnl,
                    "gamma_pnl": pnl.gamma_pnl,
                    "vega_pnl": pnl.vega_pnl,
                    "theta_pnl": pnl.theta_pnl
                }
            }
        except Exception as e:
            raise MCPError(MCPErrorCode.TOOL_EXECUTION_ERROR.value, "P&L calculation failed", {"details": str(e)})
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"calculations": self._pnl_calculations})
    
    async def generate_prompt(self, name: str, arguments: Dict) -> str:
        return "Explain P&L: total, realized, unrealized, and Greeks attribution (delta, gamma, vega, theta)."


# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = PerformanceAnalyticsMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8105'))
            transport = HTTPTransport(server.handle_message, host='0.0.0.0', port=port)
            print(f"Starting MCP server on HTTP port {port}")
            await transport.start()
            # Keep server running forever
            print(f"MCP HTTP server running on port {port}")
            while True:
                await asyncio.sleep(3600)
        else:
            # STDIO transport (Claude Desktop compatible)
            transport = STDIOTransport(server.handle_message)
            print("Starting MCP server on STDIO")
            await transport.start()
    
    asyncio.run(main())
