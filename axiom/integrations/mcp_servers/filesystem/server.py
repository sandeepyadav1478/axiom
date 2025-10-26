"""Filesystem MCP Server Implementation.

Provides file system operations through MCP protocol:
- Read/write files
- List directories
- Search files
- Watch file changes
- File permissions
"""

import asyncio
import glob
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FilesystemMCPServer:
    """Filesystem MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.root_path = Path(config.get("root_path", "/"))
        self.allowed_paths = config.get("allowed_paths")
        self.max_file_size = config.get("max_file_size", 104857600)  # 100MB
        self._watchers: dict[str, asyncio.Task] = {}

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve file path.

        Args:
            path: File path to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path is invalid or not allowed
        """
        resolved = Path(path).resolve()

        # Check if path is within allowed paths
        if self.allowed_paths:
            allowed = False
            for allowed_path in self.allowed_paths:
                if resolved.is_relative_to(Path(allowed_path).resolve()):
                    allowed = True
                    break
            
            if not allowed:
                raise ValueError(f"Path {path} is not in allowed paths")

        return resolved

    async def read_file(self, path: str, encoding: str = "utf-8") -> dict[str, Any]:
        """Read file contents.

        Args:
            path: File path
            encoding: File encoding

        Returns:
            File contents and metadata
        """
        try:
            file_path = self._validate_path(path)

            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}",
                }

            if not file_path.is_file():
                return {
                    "success": False,
                    "error": f"Not a file: {path}",
                }

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                return {
                    "success": False,
                    "error": f"File too large: {file_size} bytes (max: {self.max_file_size})",
                }

            # Read file
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            return {
                "success": True,
                "path": str(file_path),
                "content": content,
                "size": file_size,
                "encoding": encoding,
                "modified": file_path.stat().st_mtime,
            }

        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to read file: {str(e)}",
                "path": path,
            }

    async def write_file(
        self, path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
    ) -> dict[str, Any]:
        """Write content to file.

        Args:
            path: File path
            content: Content to write
            encoding: File encoding
            create_dirs: Create parent directories if they don't exist

        Returns:
            Write operation result
        """
        try:
            file_path = self._validate_path(path)

            # Create parent directories if needed
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(file_path, "w", encoding=encoding) as f:
                f.write(content)

            file_size = file_path.stat().st_size

            return {
                "success": True,
                "path": str(file_path),
                "size": file_size,
                "encoding": encoding,
            }

        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}",
                "path": path,
            }

    async def list_directory(
        self, path: str, recursive: bool = False, pattern: Optional[str] = None
    ) -> dict[str, Any]:
        """List directory contents.

        Args:
            path: Directory path
            recursive: List recursively
            pattern: Optional glob pattern filter

        Returns:
            Directory listing
        """
        try:
            dir_path = self._validate_path(path)

            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {path}",
                }

            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Not a directory: {path}",
                }

            entries = []

            if recursive:
                # Recursive listing
                if pattern:
                    files = dir_path.rglob(pattern)
                else:
                    files = dir_path.rglob("*")
            else:
                # Top-level listing
                if pattern:
                    files = dir_path.glob(pattern)
                else:
                    files = dir_path.glob("*")

            for entry in files:
                stat = entry.stat()
                entries.append({
                    "name": entry.name,
                    "path": str(entry),
                    "type": "directory" if entry.is_dir() else "file",
                    "size": stat.st_size if entry.is_file() else 0,
                    "modified": stat.st_mtime,
                    "permissions": oct(stat.st_mode)[-3:],
                })

            return {
                "success": True,
                "path": str(dir_path),
                "entries": entries,
                "count": len(entries),
                "recursive": recursive,
            }

        except Exception as e:
            logger.error(f"Failed to list directory {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to list directory: {str(e)}",
                "path": path,
            }

    async def search_files(
        self, path: str, pattern: str, recursive: bool = True, content_search: bool = False
    ) -> dict[str, Any]:
        """Search for files matching pattern.

        Args:
            path: Directory to search
            pattern: Search pattern (glob or regex)
            recursive: Search recursively
            content_search: Search file contents

        Returns:
            Search results
        """
        try:
            dir_path = self._validate_path(path)

            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {path}",
                }

            matches = []

            if content_search:
                # Search file contents
                if recursive:
                    files = dir_path.rglob("*")
                else:
                    files = dir_path.glob("*")

                for file_path in files:
                    if file_path.is_file():
                        try:
                            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                                if pattern in content:
                                    matches.append({
                                        "path": str(file_path),
                                        "type": "content_match",
                                    })
                        except:
                            continue
            else:
                # Search by filename pattern
                if recursive:
                    files = dir_path.rglob(pattern)
                else:
                    files = dir_path.glob(pattern)

                for file_path in files:
                    matches.append({
                        "path": str(file_path),
                        "type": "directory" if file_path.is_dir() else "file",
                        "size": file_path.stat().st_size if file_path.is_file() else 0,
                    })

            return {
                "success": True,
                "search_path": str(dir_path),
                "pattern": pattern,
                "matches": matches,
                "count": len(matches),
                "recursive": recursive,
                "content_search": content_search,
            }

        except Exception as e:
            logger.error(f"Failed to search files in {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to search files: {str(e)}",
                "path": path,
            }

    async def delete_file(self, path: str) -> dict[str, Any]:
        """Delete a file.

        Args:
            path: File path to delete

        Returns:
            Delete operation result
        """
        try:
            file_path = self._validate_path(path)

            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}",
                }

            if file_path.is_dir():
                return {
                    "success": False,
                    "error": f"Cannot delete directory with this method: {path}",
                }

            file_path.unlink()

            return {
                "success": True,
                "path": str(file_path),
                "operation": "deleted",
            }

        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to delete file: {str(e)}",
                "path": path,
            }

    async def get_file_info(self, path: str) -> dict[str, Any]:
        """Get file metadata.

        Args:
            path: File path

        Returns:
            File metadata
        """
        try:
            file_path = self._validate_path(path)

            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}",
                }

            stat = file_path.stat()

            return {
                "success": True,
                "path": str(file_path),
                "name": file_path.name,
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size,
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "permissions": oct(stat.st_mode)[-3:],
                "owner": stat.st_uid,
            }

        except Exception as e:
            logger.error(f"Failed to get file info for {path}: {e}")
            return {
                "success": False,
                "error": f"Failed to get file info: {str(e)}",
                "path": path,
            }


def get_server_definition() -> dict[str, Any]:
    """Get filesystem MCP server definition.

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
                        "create_dirs": {
                            "type": "boolean",
                            "description": "Create parent directories",
                            "default": True,
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
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern filter",
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
                        "content_search": {
                            "type": "boolean",
                            "description": "Search file contents",
                            "default": False,
                        },
                    },
                    "required": ["path", "pattern"],
                },
            },
            {
                "name": "delete_file",
                "description": "Delete a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to delete",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "get_file_info",
                "description": "Get file metadata and information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path",
                        }
                    },
                    "required": ["path"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "filesystem",
        },
    }