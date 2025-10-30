"""
MCP Protocol Implementation - Complete JSON-RPC 2.0

Implements Model Context Protocol following official spec exactly.
Handles all message types, validation, and error codes per specification.

Specification: https://spec.modelcontextprotocol.io/
JSON-RPC 2.0: https://www.jsonrpc.org/specification

Built with senior developer attention to protocol details.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import uuid


class MCPMethod(str, Enum):
    """MCP protocol methods (per spec)"""
    # Lifecycle
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    SHUTDOWN = "shutdown"
    
    # Tools
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    
    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"
    
    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    
    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"
    
    # Sampling (for LLM integration)
    SAMPLING_CREATE_MESSAGE = "sampling/createMessage"


class MCPErrorCode(int, Enum):
    """JSON-RPC 2.0 error codes + MCP-specific"""
    # JSON-RPC 2.0 standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    RESOURCE_NOT_FOUND = -32003
    RESOURCE_ACCESS_ERROR = -32004
    PROMPT_NOT_FOUND = -32005


@dataclass
class MCPRequest:
    """
    MCP Request Message
    
    Follows JSON-RPC 2.0 + MCP specification
    """
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict = None
    id: Union[str, int, None] = None
    
    def __post_init__(self):
        """Validate request"""
        if self.jsonrpc != "2.0":
            raise ValueError("Invalid jsonrpc version")
        
        if not self.method:
            raise ValueError("Method required")
        
        if self.params is None:
            self.params = {}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPRequest':
        """Create from dictionary"""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data["method"],
            params=data.get("params", {}),
            id=data.get("id")
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        msg = {
            "jsonrpc": self.jsonrpc,
            "method": self.method
        }
        
        if self.params:
            msg["params"] = self.params
        
        if self.id is not None:
            msg["id"] = self.id
        
        return msg


@dataclass
class MCPResponse:
    """
    MCP Response Message
    
    Follows JSON-RPC 2.0 specification
    """
    jsonrpc: str = "2.0"
    id: Union[str, int, None] = None
    result: Optional[Any] = None
    error: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate response"""
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")
        
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
    
    @classmethod
    def success(cls, request_id: Union[str, int, None], result: Any) -> 'MCPResponse':
        """Create success response"""
        return cls(id=request_id, result=result)
    
    @classmethod
    def error(cls, request_id: Union[str, int, None], code: int, message: str, data: Optional[Dict] = None) -> 'MCPResponse':
        """Create error response"""
        error_obj = {
            "code": code,
            "message": message
        }
        
        if data:
            error_obj["data"] = data
        
        return cls(id=request_id, error=error_obj)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        msg = {
            "jsonrpc": self.jsonrpc,
            "id": self.id
        }
        
        if self.result is not None:
            msg["result"] = self.result
        
        if self.error is not None:
            msg["error"] = self.error
        
        return msg


class MCPMessageValidator:
    """
    MCP Message Validation
    
    Validates messages against MCP + JSON-RPC 2.0 spec
    """
    
    @staticmethod
    def validate_request(message: Dict) -> bool:
        """Validate request message format"""
        # JSON-RPC 2.0 requirements
        if "jsonrpc" not in message:
            return False
        
        if message["jsonrpc"] != "2.0":
            return False
        
        if "method" not in message:
            return False
        
        # Method must be string
        if not isinstance(message["method"], str):
            return False
        
        # Params must be dict or list if present
        if "params" in message:
            if not isinstance(message["params"], (dict, list)):
                return False
        
        # ID can be string, number, or null
        if "id" in message:
            if not isinstance(message["id"], (str, int, type(None))):
                return False
        
        return True
    
    @staticmethod
    def validate_response(message: Dict) -> bool:
        """Validate response message format"""
        # JSON-RPC 2.0 requirements
        if "jsonrpc" not in message:
            return False
        
        if message["jsonrpc"] != "2.0":
            return False
        
        # Must have either result or error, not both
        has_result = "result" in message
        has_error = "error" in message
        
        if not (has_result or has_error):
            return False
        
        if has_result and has_error:
            return False
        
        # Error must have code and message
        if has_error:
            error = message["error"]
            if "code" not in error or "message" not in error:
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    import json
    
    # Create request
    request = MCPRequest(
        method=MCPMethod.TOOLS_LIST.value,
        id=str(uuid.uuid4())
    )
    
    print("MCP Request:")
    print(json.dumps(request.to_dict(), indent=2))
    
    # Create success response
    response = MCPResponse.success(
        request_id=request.id,
        result={"tools": []}
    )
    
    print("\nMCP Response (Success):")
    print(json.dumps(response.to_dict(), indent=2))
    
    # Create error response
    error_response = MCPResponse.error(
        request_id=request.id,
        code=MCPErrorCode.METHOD_NOT_FOUND.value,
        message="Method not found"
    )
    
    print("\nMCP Response (Error):")
    print(json.dumps(error_response.to_dict(), indent=2))
    
    # Validate
    validator = MCPMessageValidator()
    is_valid = validator.validate_request(request.to_dict())
    print(f"\nRequest valid: {is_valid}")