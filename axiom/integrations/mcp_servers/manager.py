"""Unified MCP Manager for Axiom Platform.

Manages ALL MCP servers across categories:
- Data providers (financial, market data)
- Storage (PostgreSQL, Redis, Vector DBs)
- File system (filesystem, PDF, Excel, Markdown)
- DevOps (Git, Docker, Kubernetes, CI/CD)
- Cloud (AWS, GCP, Azure)
- Communication (Slack, Email, SMS)
- Monitoring (Prometheus, Grafana, Logging)
- ML Operations (model serving, training, MLflow)
- Code quality (linting, testing, security)
- Business intelligence (analytics, reporting)
- Research (papers, patents, legal)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MCPCategory(Enum):
    """MCP server categories."""

    DATA = "data"  # Financial data providers
    STORAGE = "storage"  # Databases, caches
    FILESYSTEM = "filesystem"  # File operations
    DEVOPS = "devops"  # Git, Docker, CI/CD
    CLOUD = "cloud"  # AWS, GCP, Azure
    COMMUNICATION = "communication"  # Slack, Email, SMS
    MONITORING = "monitoring"  # Prometheus, Grafana
    ML_OPS = "ml_ops"  # Model serving, training
    CODE_QUALITY = "code_quality"  # Linting, testing
    BUSINESS_INTEL = "business_intel"  # Analytics, BI
    RESEARCH = "research"  # Papers, patents, legal


class MCPServerStatus(Enum):
    """MCP server health status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    category: MCPCategory
    server_name: str
    handler: Optional[Callable] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResource:
    """MCP resource definition."""

    uri: str
    name: str
    description: str
    mime_type: str
    category: MCPCategory
    server_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServer:
    """MCP server registration."""

    name: str
    category: MCPCategory
    description: str
    tools: list[MCPTool] = field(default_factory=list)
    resources: list[MCPResource] = field(default_factory=list)
    status: MCPServerStatus = MCPServerStatus.INACTIVE
    connection_url: Optional[str] = None
    api_key: Optional[str] = None
    config: dict[str, Any] = field(default_factory=dict)
    health_check_interval: int = 60  # seconds
    last_health_check: Optional[datetime] = None
    error_count: int = 0
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedMCPManager:
    """Unified MCP server manager for all categories."""

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}
        self.category_index: dict[MCPCategory, list[str]] = {
            category: [] for category in MCPCategory
        }
        self._health_check_tasks: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def register_server(self, server: MCPServer) -> bool:
        """Register an MCP server.

        Args:
            server: MCPServer instance to register

        Returns:
            True if registration successful
        """
        try:
            if server.name in self.servers:
                logger.warning(f"Server {server.name} already registered, updating")

            self.servers[server.name] = server
            self.category_index[server.category].append(server.name)
            self._locks[server.name] = asyncio.Lock()

            # Register tools
            for tool in server.tools:
                tool_key = f"{server.name}.{tool.name}"
                self.tools[tool_key] = tool

            # Register resources
            for resource in server.resources:
                resource_key = f"{server.name}.{resource.uri}"
                self.resources[resource_key] = resource

            logger.info(
                f"Registered MCP server: {server.name} ({server.category.value}) "
                f"with {len(server.tools)} tools and {len(server.resources)} resources"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to register server {server.name}: {e}")
            return False

    def unregister_server(self, server_name: str) -> bool:
        """Unregister an MCP server.

        Args:
            server_name: Name of server to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if server_name not in self.servers:
                logger.warning(f"Server {server_name} not found")
                return False

            server = self.servers[server_name]

            # Stop health checks
            if server_name in self._health_check_tasks:
                self._health_check_tasks[server_name].cancel()
                del self._health_check_tasks[server_name]

            # Remove from category index
            self.category_index[server.category].remove(server_name)

            # Remove tools
            tools_to_remove = [
                key for key in self.tools if key.startswith(f"{server_name}.")
            ]
            for key in tools_to_remove:
                del self.tools[key]

            # Remove resources
            resources_to_remove = [
                key for key in self.resources if key.startswith(f"{server_name}.")
            ]
            for key in resources_to_remove:
                del self.resources[key]

            # Remove server
            del self.servers[server_name]
            del self._locks[server_name]

            logger.info(f"Unregistered MCP server: {server_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister server {server_name}: {e}")
            return False

    async def connect_server(self, server_name: str) -> bool:
        """Connect to an MCP server.

        Args:
            server_name: Name of server to connect

        Returns:
            True if connection successful
        """
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found")
            return False

        server = self.servers[server_name]
        async with self._locks[server_name]:
            try:
                server.status = MCPServerStatus.CONNECTING

                # Perform connection logic (to be implemented per server type)
                # This is a placeholder for actual connection implementation
                await asyncio.sleep(0.1)  # Simulate connection

                server.status = MCPServerStatus.ACTIVE
                server.last_health_check = datetime.now()
                server.error_count = 0

                # Start health checks
                if server_name not in self._health_check_tasks:
                    task = asyncio.create_task(self._health_check_loop(server_name))
                    self._health_check_tasks[server_name] = task

                logger.info(f"Connected to MCP server: {server_name}")
                return True

            except Exception as e:
                server.status = MCPServerStatus.ERROR
                server.error_count += 1
                logger.error(f"Failed to connect to server {server_name}: {e}")
                return False

    async def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from an MCP server.

        Args:
            server_name: Name of server to disconnect

        Returns:
            True if disconnection successful
        """
        if server_name not in self.servers:
            logger.error(f"Server {server_name} not found")
            return False

        server = self.servers[server_name]
        async with self._locks[server_name]:
            try:
                # Stop health checks
                if server_name in self._health_check_tasks:
                    self._health_check_tasks[server_name].cancel()
                    del self._health_check_tasks[server_name]

                # Perform disconnection logic
                server.status = MCPServerStatus.DISCONNECTED

                logger.info(f"Disconnected from MCP server: {server_name}")
                return True

            except Exception as e:
                logger.error(f"Failed to disconnect from server {server_name}: {e}")
                return False

    async def call_tool(
        self, server_name: str, tool_name: str, **parameters
    ) -> dict[str, Any]:
        """Call an MCP tool.

        Args:
            server_name: Name of server hosting the tool
            tool_name: Name of tool to call
            **parameters: Tool parameters

        Returns:
            Tool execution result
        """
        tool_key = f"{server_name}.{tool_name}"

        if tool_key not in self.tools:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found in server {server_name}",
                "available_tools": self.get_server_tools(server_name),
            }

        if server_name not in self.servers:
            return {
                "success": False,
                "error": f"Server {server_name} not registered",
            }

        server = self.servers[server_name]

        if server.status != MCPServerStatus.ACTIVE:
            return {
                "success": False,
                "error": f"Server {server_name} is not active (status: {server.status.value})",
            }

        tool = self.tools[tool_key]

        try:
            # Validate parameters
            validation = self._validate_tool_parameters(tool, parameters)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": validation["error"],
                    "tool": tool_name,
                }

            # Call tool handler
            if tool.handler:
                result = await tool.handler(**parameters)
            else:
                result = {
                    "success": False,
                    "error": f"No handler defined for tool {tool_name}",
                }

            # Add metadata
            if result.get("success"):
                result["server"] = server_name
                result["tool"] = tool_name
                result["category"] = tool.category.value
                result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            server.error_count += 1
            logger.error(f"Tool execution failed for {tool_key}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "server": server_name,
            }

    async def access_resource(
        self, server_name: str, resource_uri: str
    ) -> dict[str, Any]:
        """Access an MCP resource.

        Args:
            server_name: Name of server hosting the resource
            resource_uri: URI of resource to access

        Returns:
            Resource content and metadata
        """
        resource_key = f"{server_name}.{resource_uri}"

        if resource_key not in self.resources:
            return {
                "success": False,
                "error": f"Resource {resource_uri} not found in server {server_name}",
                "available_resources": self.get_server_resources(server_name),
            }

        if server_name not in self.servers:
            return {
                "success": False,
                "error": f"Server {server_name} not registered",
            }

        server = self.servers[server_name]

        if server.status != MCPServerStatus.ACTIVE:
            return {
                "success": False,
                "error": f"Server {server_name} is not active",
            }

        resource = self.resources[resource_key]

        try:
            # Access resource (to be implemented per resource type)
            # Placeholder implementation
            result = {
                "success": True,
                "uri": resource_uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
                "content": None,  # To be fetched
                "server": server_name,
                "category": resource.category.value,
                "timestamp": datetime.now().isoformat(),
            }

            return result

        except Exception as e:
            server.error_count += 1
            logger.error(f"Resource access failed for {resource_key}: {e}")
            return {
                "success": False,
                "error": f"Resource access failed: {str(e)}",
                "uri": resource_uri,
                "server": server_name,
            }

    def get_server_tools(self, server_name: str) -> list[str]:
        """Get list of tools for a server.

        Args:
            server_name: Name of server

        Returns:
            List of tool names
        """
        return [
            tool.name
            for key, tool in self.tools.items()
            if key.startswith(f"{server_name}.")
        ]

    def get_server_resources(self, server_name: str) -> list[str]:
        """Get list of resources for a server.

        Args:
            server_name: Name of server

        Returns:
            List of resource URIs
        """
        return [
            resource.uri
            for key, resource in self.resources.items()
            if key.startswith(f"{server_name}.")
        ]

    def get_servers_by_category(self, category: MCPCategory) -> list[MCPServer]:
        """Get all servers in a category.

        Args:
            category: MCP category

        Returns:
            List of servers
        """
        server_names = self.category_index.get(category, [])
        return [self.servers[name] for name in server_names if name in self.servers]

    def get_available_categories(self) -> list[MCPCategory]:
        """Get categories with registered servers.

        Returns:
            List of active categories
        """
        return [
            category
            for category, servers in self.category_index.items()
            if len(servers) > 0
        ]

    def get_server_status(self, server_name: str) -> dict[str, Any]:
        """Get status information for a server.

        Args:
            server_name: Name of server

        Returns:
            Server status information
        """
        if server_name not in self.servers:
            return {"error": f"Server {server_name} not found"}

        server = self.servers[server_name]

        return {
            "name": server.name,
            "category": server.category.value,
            "status": server.status.value,
            "description": server.description,
            "tools_count": len(server.tools),
            "resources_count": len(server.resources),
            "last_health_check": (
                server.last_health_check.isoformat()
                if server.last_health_check
                else None
            ),
            "error_count": server.error_count,
            "connection_url": server.connection_url,
            "metadata": server.metadata,
        }

    def get_ecosystem_status(self) -> dict[str, Any]:
        """Get overall ecosystem status.

        Returns:
            Comprehensive ecosystem status
        """
        total_servers = len(self.servers)
        active_servers = sum(
            1 for s in self.servers.values() if s.status == MCPServerStatus.ACTIVE
        )
        total_tools = len(self.tools)
        total_resources = len(self.resources)

        category_breakdown = {
            category.value: {
                "servers": len(server_names),
                "active": sum(
                    1
                    for name in server_names
                    if name in self.servers
                    and self.servers[name].status == MCPServerStatus.ACTIVE
                ),
            }
            for category, server_names in self.category_index.items()
        }

        return {
            "total_servers": total_servers,
            "active_servers": active_servers,
            "total_tools": total_tools,
            "total_resources": total_resources,
            "categories": category_breakdown,
            "health": "healthy" if active_servers == total_servers else "degraded",
            "timestamp": datetime.now().isoformat(),
        }

    def _validate_tool_parameters(
        self, tool: MCPTool, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate tool parameters against schema.

        Args:
            tool: Tool definition
            parameters: Parameters to validate

        Returns:
            Validation result
        """
        required_params = tool.parameters.get("required", [])
        missing_params = [p for p in required_params if p not in parameters]

        if missing_params:
            return {
                "valid": False,
                "error": f"Missing required parameters: {missing_params}",
                "required": required_params,
                "provided": list(parameters.keys()),
            }

        return {"valid": True}

    async def _health_check_loop(self, server_name: str):
        """Background health check loop for a server.

        Args:
            server_name: Name of server to monitor
        """
        while True:
            try:
                if server_name not in self.servers:
                    break

                server = self.servers[server_name]

                # Perform health check
                await asyncio.sleep(server.health_check_interval)

                # Simple ping check (to be enhanced per server type)
                server.last_health_check = datetime.now()

                if server.error_count >= server.max_retries:
                    logger.warning(
                        f"Server {server_name} exceeded max retries, marking as error"
                    )
                    server.status = MCPServerStatus.ERROR

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check failed for {server_name}: {e}")
                if server_name in self.servers:
                    self.servers[server_name].error_count += 1

    async def shutdown(self):
        """Shutdown all MCP servers and cleanup."""
        logger.info("Shutting down MCP manager")

        # Cancel all health check tasks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Disconnect all servers
        for server_name in list(self.servers.keys()):
            await self.disconnect_server(server_name)

        logger.info("MCP manager shutdown complete")


# Global manager instance
mcp_manager = UnifiedMCPManager()