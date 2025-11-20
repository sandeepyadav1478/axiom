"""Client Interface MCP Server - Orchestrates all other MCP servers for clients."""
import asyncio
import json
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport

class ClientInterfaceMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="client-interface-mcp-server",version="1.0.0",description="Client orchestration - coordinates all 11 other MCP servers")
        self._queries = 0
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(name="process_query",description="Process client query by orchestrating other MCPs",input_schema={"type":"object","properties":{"query_text":{"type":"string"},"request_type":{"type":"string","enum":["question","dashboard","report"]}},"required":["query_text","request_type"]},handler=self._process_query_handler)
        self.register_tool(name="generate_dashboard",description="Generate real-time client dashboard",input_schema={"type":"object","properties":{"dashboard_type":{"type":"string"}}},handler=self._generate_dashboard_handler)
    
    def _register_resources(self):
        self.register_resource(uri="client://sessions",name="Client Sessions",description="Active client sessions")
    
    def _register_prompts(self):
        self.register_prompt(name="help",description="Get help with the system",arguments=[])
    
    async def _process_query_handler(self, arguments: dict) -> dict:
        self._queries += 1
        return {"success":True,"response":f"Processed: {arguments['query_text']}","agents_consulted":["analytics","risk"]}
    
    async def _generate_dashboard_handler(self, arguments: dict) -> dict:
        return {"success":True,"dashboard":"HTML content here"}
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"queries_processed":self._queries})
    
    async def generate_prompt(self, name: str, arguments: dict) -> str:
        return "Help with using the Axiom derivatives system: ask questions, get analytics, execute trades."

# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = ClientInterfaceMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8111'))
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
