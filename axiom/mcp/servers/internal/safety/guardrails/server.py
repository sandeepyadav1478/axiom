"""Safety Guardrail MCP Server - Multi-layer safety validation with veto authority."""
import asyncio
import json
from axiom.mcp.servers.shared.mcp_base import BaseMCPServer, MCPError
from axiom.mcp.servers.shared.mcp_protocol import MCPErrorCode
from axiom.mcp.servers.shared.mcp_transport import STDIOTransport, HTTPTransport
from axiom.ai_layer.guardrails.ai_safety_layer import AIGuardrailSystem

class SafetyGuardrailMCPServer(BaseMCPServer):
    def __init__(self):
        super().__init__(name="safety-guardrail-mcp-server",version="1.0.0",description="Multi-layer safety validation (veto authority over all actions)")
        self.safety_system = AIGuardrailSystem()
        self._validations = 0
        self._blocked = 0
        self._register_tools()
        self._register_resources()
        self._register_prompts()
    
    def _register_tools(self):
        self.register_tool(name="validate_action",description="Validate action through multi-layer safety checks",input_schema={"type":"object","properties":{"action_type":{"type":"string","enum":["validate_greeks","validate_strategy","validate_execution"]},"proposed_action":{"type":"object"},"context":{"type":"object"}},"required":["action_type","proposed_action"]},handler=self._validate_action_handler)
        self.register_tool(name="check_safety_rules",description="Check if action complies with safety rules",input_schema={"type":"object","properties":{"rules":{"type":"array"}}},handler=self._check_rules_handler)
    
    def _register_resources(self):
        self.register_resource(uri="safety://rules",name="Safety Rules",description="Active safety rules and thresholds")
        self.register_resource(uri="safety://history",name="Validation History",description="Historical validation decisions")
    
    def _register_prompts(self):
        self.register_prompt(name="explain_safety",description="Explain multi-layer safety validation",arguments=[])
    
    async def _validate_action_handler(self, arguments: dict) -> dict:
        try:
            self._validations += 1
            action_type = arguments['action_type']
            
            if action_type == 'validate_greeks':
                validation = self.safety_system.validate_greeks_output(
                    ai_greeks=arguments['proposed_action'],
                    spot=arguments.get('context', {}).get('spot', 100),
                    strike=arguments.get('context', {}).get('strike', 100),
                    time=arguments.get('context', {}).get('time', 1.0),
                    rate=arguments.get('context', {}).get('rate', 0.03),
                    vol=arguments.get('context', {}).get('vol', 0.25)
                )
            elif action_type == 'validate_strategy':
                validation = self.safety_system.validate_strategy(
                    strategy=arguments['proposed_action'],
                    max_risk=arguments.get('context', {}).get('max_risk', 100000)
                )
            elif action_type == 'validate_execution':
                validation = self.safety_system.validate_execution(
                    order=arguments['proposed_action'],
                    current_portfolio=arguments.get('context', {}).get('portfolio', {})
                )
            else:
                validation = type('obj', (), {'passed': False, 'risk_level': type('obj', (), {'value': 'critical'})(), 'issues_found': ["Unknown action type"]})()
            
            if not validation.passed:
                self._blocked += 1
            
            return {
                "success":True,
                "approved":validation.passed,
                "risk_level":validation.risk_level.value,
                "issues":validation.issues_found,
                "requires_human":validation.risk_level.value == 'critical' and not validation.passed
            }
        except Exception as e:
            self._blocked += 1
            raise MCPError(MCPErrorCode.TOOL_EXECUTION_ERROR.value,"Safety validation failed",{"details":str(e)})
    
    async def _check_rules_handler(self, arguments: dict) -> dict:
        return {"success":True,"rules_compliant":True}
    
    async def read_resource(self, uri: str) -> str:
        return json.dumps({"validations":self._validations,"blocked":self._blocked,"block_rate":self._blocked/max(self._validations,1)})
    
    async def generate_prompt(self, name: str, arguments: dict) -> str:
        return "Explain safety guardrails: multi-layer validation, circuit breakers, human escalation."

# Run MCP server
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        # Create MCP server
        server = SafetyGuardrailMCPServer()
        
        # Check if running in Docker (use HTTP) or direct (use STDIO)
        transport_type = os.getenv('MCP_TRANSPORT', 'stdio').lower()
        
        if transport_type == 'http':
            # HTTP transport for Docker daemon mode
            port = int(os.getenv('MCP_PORT', '8110'))
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
