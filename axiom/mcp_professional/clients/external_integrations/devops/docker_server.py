"""Docker MCP Server Implementation.

Provides Docker container and image management through MCP protocol:
- Container lifecycle (list, start, stop, restart, remove)
- Image management (build, pull, push, remove)
- Container logs and statistics
- Network and volume management
- Health monitoring
"""

import asyncio
import logging
from typing import Any, Optional

try:
    import docker
    from docker.errors import DockerException, APIError, NotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
    DockerException = Exception
    APIError = Exception
    NotFound = Exception

logger = logging.getLogger(__name__)


class DockerMCPServer:
    """Docker MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not DOCKER_AVAILABLE:
            raise ImportError(
                "docker library not installed. "
                "Install with: pip install docker"
            )
        
        self.config = config
        self.socket = config.get("socket", "unix:///var/run/docker.sock")
        self.registry_url = config.get("registry_url")
        self.registry_user = config.get("registry_user")
        self.registry_password = config.get("registry_password")
        
        self._client: Optional["docker.DockerClient"] = None

    def _ensure_client(self) -> "docker.DockerClient":
        """Ensure Docker client is initialized.

        Returns:
            Docker client instance
        """
        if self._client is None:
            try:
                self._client = docker.DockerClient(base_url=self.socket)
                # Test connection
                self._client.ping()
                logger.info(f"Connected to Docker daemon at {self.socket}")
            except DockerException as e:
                logger.error(f"Failed to connect to Docker: {e}")
                raise
        
        return self._client

    def close(self):
        """Close Docker client connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Docker client connection closed")

    async def list_containers(
        self,
        all: bool = False,
        filters: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """List Docker containers.

        Args:
            all: Show all containers (default shows just running)
            filters: Filters to apply (e.g., {"status": "running"})

        Returns:
            List of containers
        """
        try:
            client = self._ensure_client()
            containers = client.containers.list(all=all, filters=filters)
            
            container_list = []
            for container in containers:
                container_list.append({
                    "id": container.short_id,
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else container.image.short_id,
                    "status": container.status,
                    "created": container.attrs["Created"],
                    "ports": container.ports,
                    "labels": container.labels,
                })
            
            return {
                "success": True,
                "containers": container_list,
                "count": len(container_list),
            }

        except DockerException as e:
            logger.error(f"Failed to list containers: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
            }
        except Exception as e:
            logger.error(f"Failed to list containers: {e}")
            return {
                "success": False,
                "error": f"Failed to list containers: {str(e)}",
            }

    async def start_container(self, container_id: str) -> dict[str, Any]:
        """Start a Docker container.

        Args:
            container_id: Container ID or name

        Returns:
            Operation result
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            container.start()
            
            return {
                "success": True,
                "container_id": container.short_id,
                "name": container.name,
                "status": container.status,
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to start container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to start container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to start container: {str(e)}",
                "container_id": container_id,
            }

    async def stop_container(
        self,
        container_id: str,
        timeout: int = 10,
    ) -> dict[str, Any]:
        """Stop a Docker container.

        Args:
            container_id: Container ID or name
            timeout: Timeout in seconds before killing

        Returns:
            Operation result
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            container.stop(timeout=timeout)
            
            return {
                "success": True,
                "container_id": container.short_id,
                "name": container.name,
                "status": "stopped",
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to stop container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to stop container: {str(e)}",
                "container_id": container_id,
            }

    async def restart_container(
        self,
        container_id: str,
        timeout: int = 10,
    ) -> dict[str, Any]:
        """Restart a Docker container.

        Args:
            container_id: Container ID or name
            timeout: Timeout in seconds

        Returns:
            Operation result
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            container.restart(timeout=timeout)
            
            return {
                "success": True,
                "container_id": container.short_id,
                "name": container.name,
                "status": container.status,
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to restart container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to restart container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to restart container: {str(e)}",
                "container_id": container_id,
            }

    async def remove_container(
        self,
        container_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Remove a Docker container.

        Args:
            container_id: Container ID or name
            force: Force removal even if running

        Returns:
            Operation result
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            name = container.name
            container.remove(force=force)
            
            return {
                "success": True,
                "container_id": container_id,
                "name": name,
                "removed": True,
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to remove container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to remove container {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to remove container: {str(e)}",
                "container_id": container_id,
            }

    async def build_image(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        buildargs: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Build a Docker image from Dockerfile.

        Args:
            path: Path to build context
            tag: Image tag
            dockerfile: Dockerfile name
            buildargs: Build arguments

        Returns:
            Build result
        """
        try:
            client = self._ensure_client()
            
            image, build_logs = client.images.build(
                path=path,
                tag=tag,
                dockerfile=dockerfile,
                buildargs=buildargs or {},
                rm=True,
            )
            
            # Collect build logs
            logs = []
            for log in build_logs:
                if 'stream' in log:
                    logs.append(log['stream'].strip())
            
            return {
                "success": True,
                "image_id": image.short_id,
                "tag": tag,
                "logs": logs[-10:] if len(logs) > 10 else logs,  # Last 10 lines
            }

        except DockerException as e:
            logger.error(f"Failed to build image {tag}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "tag": tag,
            }
        except Exception as e:
            logger.error(f"Failed to build image {tag}: {e}")
            return {
                "success": False,
                "error": f"Failed to build image: {str(e)}",
                "tag": tag,
            }

    async def pull_image(
        self,
        repository: str,
        tag: str = "latest",
    ) -> dict[str, Any]:
        """Pull a Docker image from registry.

        Args:
            repository: Image repository
            tag: Image tag

        Returns:
            Pull result
        """
        try:
            client = self._ensure_client()
            
            image = client.images.pull(repository, tag=tag)
            
            return {
                "success": True,
                "image_id": image.short_id,
                "repository": repository,
                "tag": tag,
                "tags": image.tags,
            }

        except DockerException as e:
            logger.error(f"Failed to pull image {repository}:{tag}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "repository": repository,
                "tag": tag,
            }
        except Exception as e:
            logger.error(f"Failed to pull image {repository}:{tag}: {e}")
            return {
                "success": False,
                "error": f"Failed to pull image: {str(e)}",
                "repository": repository,
                "tag": tag,
            }

    async def push_image(
        self,
        repository: str,
        tag: str = "latest",
    ) -> dict[str, Any]:
        """Push a Docker image to registry.

        Args:
            repository: Image repository
            tag: Image tag

        Returns:
            Push result
        """
        try:
            client = self._ensure_client()
            
            # Login to registry if credentials provided
            if self.registry_url and self.registry_user and self.registry_password:
                client.login(
                    username=self.registry_user,
                    password=self.registry_password,
                    registry=self.registry_url,
                )
            
            # Push image
            push_log = client.images.push(repository, tag=tag)
            
            return {
                "success": True,
                "repository": repository,
                "tag": tag,
                "log": push_log,
            }

        except DockerException as e:
            logger.error(f"Failed to push image {repository}:{tag}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "repository": repository,
                "tag": tag,
            }
        except Exception as e:
            logger.error(f"Failed to push image {repository}:{tag}: {e}")
            return {
                "success": False,
                "error": f"Failed to push image: {str(e)}",
                "repository": repository,
                "tag": tag,
            }

    async def get_logs(
        self,
        container_id: str,
        tail: int = 100,
        follow: bool = False,
        timestamps: bool = False,
    ) -> dict[str, Any]:
        """Get container logs.

        Args:
            container_id: Container ID or name
            tail: Number of lines to show
            follow: Follow log output
            timestamps: Show timestamps

        Returns:
            Container logs
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            
            logs = container.logs(
                tail=tail,
                follow=follow,
                timestamps=timestamps,
            )
            
            # Decode logs
            if isinstance(logs, bytes):
                logs = logs.decode('utf-8')
            
            return {
                "success": True,
                "container_id": container.short_id,
                "name": container.name,
                "logs": logs,
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to get logs for {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to get logs for {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get logs: {str(e)}",
                "container_id": container_id,
            }

    async def get_stats(
        self,
        container_id: str,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Get container statistics.

        Args:
            container_id: Container ID or name
            stream: Stream stats continuously

        Returns:
            Container statistics
        """
        try:
            client = self._ensure_client()
            container = client.containers.get(container_id)
            
            stats = container.stats(stream=stream)
            
            # Get one stats sample
            if stream:
                stat = next(stats)
            else:
                stat = stats
            
            # Extract key metrics
            cpu_stats = stat.get('cpu_stats', {})
            memory_stats = stat.get('memory_stats', {})
            
            return {
                "success": True,
                "container_id": container.short_id,
                "name": container.name,
                "stats": {
                    "cpu_usage": cpu_stats.get('cpu_usage', {}).get('total_usage', 0),
                    "memory_usage": memory_stats.get('usage', 0),
                    "memory_limit": memory_stats.get('limit', 0),
                    "networks": stat.get('networks', {}),
                },
                "raw_stats": stat,
            }

        except NotFound:
            return {
                "success": False,
                "error": f"Container not found: {container_id}",
            }
        except DockerException as e:
            logger.error(f"Failed to get stats for {container_id}: {e}")
            return {
                "success": False,
                "error": f"Docker error: {str(e)}",
                "container_id": container_id,
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {container_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get stats: {str(e)}",
                "container_id": container_id,
            }


def get_server_definition() -> dict[str, Any]:
    """Get Docker MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "docker",
        "category": "devops",
        "description": "Docker container and image management (lifecycle, builds, logs, stats)",
        "tools": [
            {
                "name": "list_containers",
                "description": "List Docker containers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "all": {
                            "type": "boolean",
                            "description": "Show all containers (default shows just running)",
                            "default": False,
                        },
                        "filters": {
                            "type": "object",
                            "description": "Filters to apply (e.g., {'status': 'running'})",
                        },
                    },
                },
            },
            {
                "name": "start_container",
                "description": "Start a Docker container",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        }
                    },
                    "required": ["container_id"],
                },
            },
            {
                "name": "stop_container",
                "description": "Stop a Docker container",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds before killing",
                            "default": 10,
                        },
                    },
                    "required": ["container_id"],
                },
            },
            {
                "name": "restart_container",
                "description": "Restart a Docker container",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 10,
                        },
                    },
                    "required": ["container_id"],
                },
            },
            {
                "name": "remove_container",
                "description": "Remove a Docker container",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force removal even if running",
                            "default": False,
                        },
                    },
                    "required": ["container_id"],
                },
            },
            {
                "name": "build_image",
                "description": "Build a Docker image from Dockerfile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to build context",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Image tag",
                        },
                        "dockerfile": {
                            "type": "string",
                            "description": "Dockerfile name",
                            "default": "Dockerfile",
                        },
                        "buildargs": {
                            "type": "object",
                            "description": "Build arguments",
                        },
                    },
                    "required": ["path", "tag"],
                },
            },
            {
                "name": "pull_image",
                "description": "Pull a Docker image from registry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repository": {
                            "type": "string",
                            "description": "Image repository",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Image tag",
                            "default": "latest",
                        },
                    },
                    "required": ["repository"],
                },
            },
            {
                "name": "push_image",
                "description": "Push a Docker image to registry",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repository": {
                            "type": "string",
                            "description": "Image repository",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Image tag",
                            "default": "latest",
                        },
                    },
                    "required": ["repository"],
                },
            },
            {
                "name": "get_logs",
                "description": "Get container logs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        },
                        "tail": {
                            "type": "integer",
                            "description": "Number of lines to show",
                            "default": 100,
                        },
                        "follow": {
                            "type": "boolean",
                            "description": "Follow log output",
                            "default": False,
                        },
                        "timestamps": {
                            "type": "boolean",
                            "description": "Show timestamps",
                            "default": False,
                        },
                    },
                    "required": ["container_id"],
                },
            },
            {
                "name": "get_stats",
                "description": "Get container statistics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "container_id": {
                            "type": "string",
                            "description": "Container ID or name",
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Stream stats continuously",
                            "default": False,
                        },
                    },
                    "required": ["container_id"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "devops",
            "requires": ["docker>=7.0.0"],
            "performance_target": "<100ms for list, <5s for builds",
        },
    }