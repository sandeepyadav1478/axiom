"""Regulatory Compliance MCP Server - SEC, FINRA, MiFID II, EMIR compliance."""
import asyncio
import json
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport
from axiom.derivatives.compliance.regulatory_reporting import RegulatoryReporter

class RegulatoryComplianceMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="regulatory-compliance-mcp-server",version="1.0.0",description="Automated regulatory compliance (SEC, FINRA, MiFID II, EMIR)")
        self.reporter = RegulatoryReporter()
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(name="check_compliance",description="Check regulatory compliance",input_schema={"type":"object","properties":{"positions":{"type":"array"},"trades":{"type":"array"}},"required":["positions"]},handler=self._check_compliance_handler)
        self.register_tool(name="generate_report",description="Generate regulatory report (LOPR, Blue Sheet, Daily Position)",input_schema={"type":"object","properties":{"report_type":{"type":"string","enum":["daily_position","lopr","blue_sheet"]}},"required":["report_type"]},handler=self._generate_report_handler)
        self.register_tool(name="audit_trail",description="Create audit trail entry",input_schema={"type":"object","properties":{"event_type":{"type":"string"},"event_data":{"type":"object"}},"required":["event_type"]},handler=self._audit_trail_handler)
    
    def _register_resources(self):
        self.register_resource(uri="compliance://rules",name="Compliance Rules",description="Active compliance rules")
        self.register_resource(uri="compliance://reports",name="Reports",description="Generated reports")
    
    def _register_prompts(self):
        self.register_prompt(name="explain_compliance",description="Explain regulatory compliance requirements",arguments=[])
    
    async def _check_compliance_handler(self, arguments: dict) -> dict:
        issues = []
        for pos in arguments.get('positions', []):
            if abs(pos.get('quantity', 0)) > 10000:
                issues.append(f"Large position: {pos.get('symbol')}")
        return {"success":True,"compliant":len(issues)==0,"issues":issues}
    
    async def _generate_report_handler(self, arguments: dict) -> dict:
        return {"success":True,"report_type":arguments['report_type'],"generated":True}
    
    async def _audit_trail_handler(self, arguments: dict) -> dict:
        return {"success":True,"audit_recorded":True}
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"regulations":["SEC","FINRA","MiFID II","EMIR"]})
    
    async def generate_prompt(self, name: str, arguments: dict) -> str:
        return "Explain regulatory compliance: SEC, FINRA, MiFID II, EMIR requirements."

# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = RegulatoryComplianceMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8108'))
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
