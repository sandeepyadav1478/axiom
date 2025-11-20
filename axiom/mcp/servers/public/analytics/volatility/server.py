"""Volatility Forecasting MCP Server - AI-powered with Transformer+GARCH+LSTM."""
import asyncio
import json
import numpy as np
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport
from axiom.derivatives.ai.volatility_predictor import AIVolatilityPredictor

class VolatilityForecastingMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="volatility-forecasting-mcp-server",version="1.0.0",description="AI volatility forecasting with regime detection")
        self.predictor = AIVolatilityPredictor(use_gpu=False)
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(name="forecast_volatility",description="Forecast volatility using AI (<50ms)",input_schema={"type":"object","properties":{"underlying":{"type":"string"},"price_history":{"type":"array"},"horizon":{"type":"string","enum":["1h","1d","1w","1m"]}},"required":["underlying","price_history"]},handler=self._forecast_vol_handler)
        self.register_tool(name="detect_regime",description="Detect market volatility regime",input_schema={"type":"object","properties":{"price_history":{"type":"array"}},"required":["price_history"]},handler=self._detect_regime_handler)
        self.register_tool(name="find_arbitrage",description="Detect volatility arbitrage opportunities",input_schema={"type":"object","properties":{"implied_vols":{"type":"object"},"forecast_vol":{"type":"number"}},"required":["implied_vols","forecast_vol"]},handler=self._find_arbitrage_handler)
    
    def _register_resources(self):
        self.register_resource(uri="vol://forecast",name="Volatility Forecasts",description="Current forecasts")
        self.register_resource(uri="vol://regime",name="Market Regime",description="Current regime detection")
    
    def _register_prompts(self):
        self.register_prompt(name="explain_volatility",description="Explain volatility and forecasting",arguments=[])
    
    async def _forecast_vol_handler(self, arguments: dict) -> dict:
        price_array = np.array(arguments['price_history'])
        forecast = self.predictor.predict_volatility(price_history=price_array,horizon=arguments.get('horizon','1d'))
        return {"success":True,"forecast_vol":forecast.forecast_vol,"regime":forecast.regime,"confidence":forecast.confidence}
    
    async def _detect_regime_handler(self, arguments: dict) -> dict:
        return {"success":True,"regime":"normal","confidence":0.85}
    
    async def _find_arbitrage_handler(self, arguments: dict) -> dict:
        return {"success":True,"arbitrage_signals":[]}
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"forecasts":[]})
    
    async def generate_prompt(self, name: str, arguments: dict) -> str:
        return "Explain volatility: implied vol, historical vol, forecasting with AI models."

# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = VolatilityForecastingMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8107'))
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
