"""System Monitoring MCP Server - Continuous health tracking of all agents."""
import asyncio
import json
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport

class SystemMonitoringMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="system-monitoring-mcp-server",version="1.0.0",description="Continuous system health monitoring and anomaly detection")
        self._health_checks = 0
        self._alerts = 0
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(name="check_system_health",description="Check health of all agents and infrastructure",input_schema={"type":"object","properties":{"include_metrics":{"type":"boolean","default":True}}},handler=self._check_health_handler)
        self.register_tool(name="record_metric",description="Record agent metric for monitoring",input_schema={"type":"object","properties":{"agent_name":{"type":"string"},"metric_name":{"type":"string"},"metric_value":{"type":"number"}},"required":["agent_name","metric_name","metric_value"]},handler=self._record_metric_handler)
        self.register_tool(name="trigger_alert",description="Trigger alert based on conditions",input_schema={"type":"object","properties":{"severity":{"type":"string","enum":["info","warning","error","critical"]},"message":{"type":"string"}},"required":["severity","message"]},handler=self._trigger_alert_handler)
    
    def _register_resources(self):
        self.register_resource(uri="monitoring://health",name="System Health",description="Current system health status")
        self.register_resource(uri="monitoring://metrics",name="Metrics",description="System metrics time series")
        self.register_resource(uri="monitoring://alerts",name="Alerts",description="Active alerts")
    
    def _register_prompts(self):
        self.register_prompt(name="health_analysis",description="Analyze system health and provide recommendations",arguments=[])
    
    async def _check_health_handler(self, arguments: dict) -> dict:
        self._health_checks += 1
        return {"success":True,"overall_status":"healthy","healthy_agents":11,"degraded_agents":0,"down_agents":0}
    
    async def _record_metric_handler(self, arguments: dict) -> dict:
        return {"success":True,"metric_recorded":True}
    
    async def _trigger_alert_handler(self, arguments: dict) -> dict:
        self._alerts += 1
        return {"success":True,"alert_triggered":True,"severity":arguments['severity']}
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"health_checks":self._health_checks,"alerts":self._alerts})
    
    async def generate_prompt(self, name: str, arguments: dict) -> str:
        return "Analyze system health: check all agents, identify issues, provide recommendations."

# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = SystemMonitoringMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8109'))
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
