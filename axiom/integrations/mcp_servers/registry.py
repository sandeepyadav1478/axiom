"""MCP Server Registry and Discovery System.

Provides automatic discovery, registration, and lifecycle management
for all MCP servers in the ecosystem.
"""

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any, Optional

from axiom.integrations.mcp_servers.config import mcp_settings
from axiom.integrations.mcp_servers.manager import (
    MCPCategory,
    MCPResource,
    MCPServer,
    MCPServerStatus,
    MCPTool,
    UnifiedMCPManager,
)

logger = logging.getLogger(__name__)


class MCPRegistry:
    """Registry for discovering and managing MCP servers."""

    def __init__(self, manager: UnifiedMCPManager):
        self.manager = manager
        self.settings = mcp_settings
        self._discovered_servers: dict[str, dict[str, Any]] = {}
        self._server_modules: dict[str, Any] = {}

    async def discover_servers(
        self, search_paths: Optional[list[str]] = None
    ) -> dict[str, dict[str, Any]]:
        """Discover MCP servers in specified paths.

        Args:
            search_paths: Paths to search for MCP servers

        Returns:
            Dictionary of discovered servers
        """
        if search_paths is None:
            # Default search paths
            base_path = Path(__file__).parent
            search_paths = [
                str(base_path / "implementations"),
                str(base_path / "data_providers"),
                str(base_path / "storage"),
                str(base_path / "filesystem"),
                str(base_path / "devops"),
                str(base_path / "communication"),
            ]

        discovered = {}

        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue

            # Look for server modules
            for py_file in path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    # Try to import and discover server
                    server_info = await self._discover_server_from_file(py_file)
                    if server_info:
                        discovered[server_info["name"]] = server_info
                        logger.info(f"Discovered MCP server: {server_info['name']}")

                except Exception as e:
                    logger.debug(f"Could not discover server from {py_file}: {e}")

        self._discovered_servers = discovered
        return discovered

    async def _discover_server_from_file(
        self, file_path: Path
    ) -> Optional[dict[str, Any]]:
        """Discover MCP server from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Server information if found
        """
        try:
            # Convert file path to module path
            module_path = str(file_path.relative_to(Path.cwd())).replace("/", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]

            # Import module
            module = importlib.import_module(module_path)

            # Look for server definition
            if hasattr(module, "get_server_definition"):
                server_def = module.get_server_definition()
                return server_def

            if hasattr(module, "SERVER_DEFINITION"):
                return module.SERVER_DEFINITION

            return None

        except Exception as e:
            logger.debug(f"Failed to discover server from {file_path}: {e}")
            return None

    async def register_all_enabled(self) -> dict[str, bool]:
        """Register all enabled MCP servers.

        Returns:
            Dictionary of registration results
        """
        results = {}
        enabled_servers = self.settings.get_enabled_servers()

        logger.info(f"Registering {len(enabled_servers)} enabled MCP servers")

        for server_name in enabled_servers:
            try:
                success = await self.register_server(server_name)
                results[server_name] = success

                if success:
                    logger.info(f"Successfully registered: {server_name}")
                else:
                    logger.warning(f"Failed to register: {server_name}")

            except Exception as e:
                logger.error(f"Error registering {server_name}: {e}")
                results[server_name] = False

        return results

    async def register_server(self, server_name: str) -> bool:
        """Register a specific MCP server.

        Args:
            server_name: Name of server to register

        Returns:
            True if registration successful
        """
        # Check if enabled
        if not self.settings.is_server_enabled(server_name):
            logger.info(f"Server {server_name} is not enabled")
            return False

        # Get server configuration
        config = self.settings.get_server_config(server_name)

        # Get server definition
        server_def = await self._get_server_definition(server_name)
        if not server_def:
            logger.warning(f"No definition found for server: {server_name}")
            return False

        # Create MCPServer instance
        server = self._create_server_instance(server_name, server_def, config)

        # Register with manager
        success = self.manager.register_server(server)

        if success:
            # Attempt to connect
            await self.manager.connect_server(server_name)

        return success

    async def _get_server_definition(
        self, server_name: str
    ) -> Optional[dict[str, Any]]:
        """Get server definition from discovered servers or module.

        Args:
            server_name: Name of server

        Returns:
            Server definition if found
        """
        # Check discovered servers
        if server_name in self._discovered_servers:
            return self._discovered_servers[server_name]

        # Try to load from predefined modules
        module_map = {
            # Week 1-2 servers
            "filesystem": "axiom.integrations.mcp_servers.filesystem.server",
            "git": "axiom.integrations.mcp_servers.devops.git_server",
            "postgres": "axiom.integrations.mcp_servers.storage.postgres_server",
            "redis": "axiom.integrations.mcp_servers.storage.redis_server",
            "slack": "axiom.integrations.mcp_servers.communication.slack_server",
            "email": "axiom.integrations.mcp_servers.communication.email_server",
            "docker": "axiom.integrations.mcp_servers.devops.docker_server",
            "prometheus": "axiom.integrations.mcp_servers.monitoring.prometheus_server",
            "pdf": "axiom.integrations.mcp_servers.documents.pdf_server",
            "excel": "axiom.integrations.mcp_servers.documents.excel_server",
            # Week 3 servers
            "aws": "axiom.integrations.mcp_servers.cloud.aws_server",
            "gcp": "axiom.integrations.mcp_servers.cloud.gcp_server",
            "notification": "axiom.integrations.mcp_servers.communication.notification_server",
            "vector_db": "axiom.integrations.mcp_servers.storage.vector_db_server",
            "kubernetes": "axiom.integrations.mcp_servers.devops.kubernetes_server",
        }

        module_path = module_map.get(server_name)
        if not module_path:
            return None

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "get_server_definition"):
                return module.get_server_definition()
            if hasattr(module, "SERVER_DEFINITION"):
                return module.SERVER_DEFINITION
        except ImportError as e:
            logger.debug(f"Could not import module {module_path}: {e}")

        return None

    def _create_server_instance(
        self, server_name: str, server_def: dict[str, Any], config: dict[str, Any]
    ) -> MCPServer:
        """Create MCPServer instance from definition.

        Args:
            server_name: Name of server
            server_def: Server definition
            config: Server configuration

        Returns:
            MCPServer instance
        """
        # Parse category
        category_str = server_def.get("category", "data")
        try:
            category = MCPCategory(category_str)
        except ValueError:
            category = MCPCategory.DATA

        # Create tools
        tools = []
        for tool_def in server_def.get("tools", []):
            tool = MCPTool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                parameters=tool_def.get("parameters", {}),
                category=category,
                server_name=server_name,
                handler=tool_def.get("handler"),
                metadata=tool_def.get("metadata", {}),
            )
            tools.append(tool)

        # Create resources
        resources = []
        for resource_def in server_def.get("resources", []):
            resource = MCPResource(
                uri=resource_def["uri"],
                name=resource_def.get("name", ""),
                description=resource_def.get("description", ""),
                mime_type=resource_def.get("mime_type", "application/octet-stream"),
                category=category,
                server_name=server_name,
                metadata=resource_def.get("metadata", {}),
            )
            resources.append(resource)

        # Create server
        server = MCPServer(
            name=server_name,
            category=category,
            description=server_def.get("description", ""),
            tools=tools,
            resources=resources,
            connection_url=server_def.get("connection_url"),
            config=config,
            metadata=server_def.get("metadata", {}),
        )

        return server

    async def unregister_all(self) -> dict[str, bool]:
        """Unregister all registered servers.

        Returns:
            Dictionary of unregistration results
        """
        results = {}
        server_names = list(self.manager.servers.keys())

        for server_name in server_names:
            try:
                await self.manager.disconnect_server(server_name)
                success = self.manager.unregister_server(server_name)
                results[server_name] = success
            except Exception as e:
                logger.error(f"Error unregistering {server_name}: {e}")
                results[server_name] = False

        return results

    async def reload_server(self, server_name: str) -> bool:
        """Reload a specific server.

        Args:
            server_name: Name of server to reload

        Returns:
            True if reload successful
        """
        try:
            # Disconnect and unregister
            await self.manager.disconnect_server(server_name)
            self.manager.unregister_server(server_name)

            # Re-register
            success = await self.register_server(server_name)
            return success

        except Exception as e:
            logger.error(f"Error reloading server {server_name}: {e}")
            return False

    def get_registry_status(self) -> dict[str, Any]:
        """Get registry status.

        Returns:
            Registry status information
        """
        return {
            "discovered_servers": len(self._discovered_servers),
            "registered_servers": len(self.manager.servers),
            "enabled_servers": len(self.settings.get_enabled_servers()),
            "active_servers": sum(
                1
                for s in self.manager.servers.values()
                if s.status == MCPServerStatus.ACTIVE
            ),
            "server_list": list(self.manager.servers.keys()),
            "enabled_list": self.settings.get_enabled_servers(),
        }


class MCPServerFactory:
    """Factory for creating standardized MCP server implementations."""

    @staticmethod
    def create_filesystem_server() -> dict[str, Any]:
        """Create filesystem MCP server definition.

        Returns:
            Server definition dictionary
        """
        return {
            "name": "filesystem",
            "category": "filesystem",
            "description": "File system operations (read, write, search, watch)",
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read",
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding",
                                "default": "utf-8",
                            },
                        },
                        "required": ["path"],
                    },
                },
                {
                    "name": "write_file",
                    "description": "Write contents to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to write",
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write",
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding",
                                "default": "utf-8",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
                {
                    "name": "list_directory",
                    "description": "List contents of a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "List recursively",
                                "default": False,
                            },
                        },
                        "required": ["path"],
                    },
                },
                {
                    "name": "search_files",
                    "description": "Search for files matching pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory to search",
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Search pattern (glob or regex)",
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Search recursively",
                                "default": True,
                            },
                        },
                        "required": ["path", "pattern"],
                    },
                },
            ],
            "resources": [],
            "metadata": {
                "version": "1.0.0",
                "priority": "critical",
            },
        }

    @staticmethod
    def create_git_server() -> dict[str, Any]:
        """Create Git MCP server definition.

        Returns:
            Server definition dictionary
        """
        return {
            "name": "git",
            "category": "devops",
            "description": "Git operations (commit, push, pull, branch management)",
            "tools": [
                {
                    "name": "git_status",
                    "description": "Get Git repository status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Repository path",
                            }
                        },
                        "required": ["repo_path"],
                    },
                },
                {
                    "name": "git_commit",
                    "description": "Commit changes to repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Repository path",
                            },
                            "message": {
                                "type": "string",
                                "description": "Commit message",
                            },
                            "files": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Files to commit",
                            },
                        },
                        "required": ["repo_path", "message"],
                    },
                },
                {
                    "name": "git_branch",
                    "description": "Create or switch branch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Repository path",
                            },
                            "branch_name": {
                                "type": "string",
                                "description": "Branch name",
                            },
                            "create": {
                                "type": "boolean",
                                "description": "Create new branch",
                                "default": False,
                            },
                        },
                        "required": ["repo_path", "branch_name"],
                    },
                },
            ],
            "resources": [],
            "metadata": {
                "version": "1.0.0",
                "priority": "critical",
            },
        }

    @staticmethod
    def create_slack_server() -> dict[str, Any]:
        """Create Slack MCP server definition.

        Returns:
            Server definition dictionary
        """
        return {
            "name": "slack",
            "category": "communication",
            "description": "Slack messaging and notifications",
            "tools": [
                {
                    "name": "send_message",
                    "description": "Send message to Slack channel",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "description": "Channel name or ID",
                            },
                            "message": {
                                "type": "string",
                                "description": "Message text",
                            },
                            "thread_ts": {
                                "type": "string",
                                "description": "Thread timestamp for replies",
                            },
                        },
                        "required": ["channel", "message"],
                    },
                },
                {
                    "name": "send_alert",
                    "description": "Send alert notification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "description": "Channel name or ID",
                            },
                            "title": {
                                "type": "string",
                                "description": "Alert title",
                            },
                            "message": {
                                "type": "string",
                                "description": "Alert message",
                            },
                            "level": {
                                "type": "string",
                                "enum": ["info", "warning", "error", "critical"],
                                "description": "Alert level",
                                "default": "info",
                            },
                        },
                        "required": ["channel", "title", "message"],
                    },
                },
            ],
            "resources": [],
            "metadata": {
                "version": "1.0.0",
                "priority": "critical",
            },
        }

    @staticmethod
    def create_postgres_server() -> dict[str, Any]:
        """Create PostgreSQL MCP server definition.

        Returns:
            Server definition dictionary
        """
        return {
            "name": "postgres",
            "category": "storage",
            "description": "PostgreSQL database operations",
            "tools": [
                {
                    "name": "execute_query",
                    "description": "Execute SQL query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "SQL query",
                            },
                            "parameters": {
                                "type": "array",
                                "description": "Query parameters",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_schema",
                    "description": "Get database schema information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Table name",
                            }
                        },
                    },
                },
            ],
            "resources": [],
            "metadata": {
                "version": "1.0.0",
                "priority": "critical",
            },
        }


# Global registry instance
def create_registry(manager: Optional[UnifiedMCPManager] = None) -> MCPRegistry:
    """Create MCP registry instance.

    Args:
        manager: Optional UnifiedMCPManager instance

    Returns:
        MCPRegistry instance
    """
    from axiom.integrations.mcp_servers.manager import mcp_manager

    if manager is None:
        manager = mcp_manager

    return MCPRegistry(manager)