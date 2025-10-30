"""Git MCP Server Implementation.

Provides Git operations through MCP protocol:
- Repository status
- Commit changes
- Branch management
- Push/pull operations
- Diff and log
- Tag management
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class GitMCPServer:
    """Git MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.default_branch = config.get("default_branch", "main")
        self.user_name = config.get("user_name")
        self.user_email = config.get("user_email")
        self.ssh_key_path = config.get("ssh_key_path")

    def _run_git_command(
        self, repo_path: str, command: list[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a Git command.

        Args:
            repo_path: Repository path
            command: Git command and arguments
            check: Raise exception on error

        Returns:
            Completed process
        """
        full_command = ["git", "-C", repo_path] + command
        
        env = None
        if self.ssh_key_path:
            env = {"GIT_SSH_COMMAND": f"ssh -i {self.ssh_key_path}"}

        return subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=check,
            env=env,
        )

    async def git_status(self, repo_path: str) -> dict[str, Any]:
        """Get repository status.

        Args:
            repo_path: Repository path

        Returns:
            Repository status
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Get status
            result = self._run_git_command(repo_path, ["status", "--porcelain"])
            
            # Parse status
            modified = []
            staged = []
            untracked = []
            
            for line in result.stdout.splitlines():
                if not line:
                    continue
                status = line[:2]
                file_path = line[3:]
                
                if status[0] != " ":
                    staged.append(file_path)
                elif status[1] == "M":
                    modified.append(file_path)
                elif status == "??":
                    untracked.append(file_path)

            # Get current branch
            branch_result = self._run_git_command(
                repo_path, ["branch", "--show-current"]
            )
            current_branch = branch_result.stdout.strip()

            # Get remote status
            try:
                self._run_git_command(repo_path, ["fetch", "--dry-run"])
                behind_result = self._run_git_command(
                    repo_path, ["rev-list", "--count", f"HEAD..origin/{current_branch}"]
                )
                ahead_result = self._run_git_command(
                    repo_path, ["rev-list", "--count", f"origin/{current_branch}..HEAD"]
                )
                behind = int(behind_result.stdout.strip())
                ahead = int(ahead_result.stdout.strip())
            except:
                behind = 0
                ahead = 0

            return {
                "success": True,
                "repo_path": str(repo),
                "branch": current_branch,
                "modified": modified,
                "staged": staged,
                "untracked": untracked,
                "ahead": ahead,
                "behind": behind,
                "clean": len(modified) == 0 and len(staged) == 0 and len(untracked) == 0,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git status failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git status failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to get git status for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to get git status: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_commit(
        self,
        repo_path: str,
        message: str,
        files: Optional[list[str]] = None,
        add_all: bool = False,
    ) -> dict[str, Any]:
        """Commit changes.

        Args:
            repo_path: Repository path
            message: Commit message
            files: Specific files to commit (None for staged files)
            add_all: Add all modified files

        Returns:
            Commit result
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Configure user if provided
            if self.user_name:
                self._run_git_command(
                    repo_path, ["config", "user.name", self.user_name]
                )
            if self.user_email:
                self._run_git_command(
                    repo_path, ["config", "user.email", self.user_email]
                )

            # Add files
            if add_all:
                self._run_git_command(repo_path, ["add", "-A"])
            elif files:
                self._run_git_command(repo_path, ["add"] + files)

            # Commit
            result = self._run_git_command(repo_path, ["commit", "-m", message])

            # Get commit hash
            hash_result = self._run_git_command(repo_path, ["rev-parse", "HEAD"])
            commit_hash = hash_result.stdout.strip()

            return {
                "success": True,
                "repo_path": str(repo),
                "commit_hash": commit_hash,
                "message": message,
                "output": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git commit failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git commit failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to commit for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to commit: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_branch(
        self, repo_path: str, branch_name: str, create: bool = False, switch: bool = True
    ) -> dict[str, Any]:
        """Create or switch branch.

        Args:
            repo_path: Repository path
            branch_name: Branch name
            create: Create new branch
            switch: Switch to branch

        Returns:
            Branch operation result
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            if create:
                # Create new branch
                self._run_git_command(repo_path, ["branch", branch_name])

            if switch:
                # Switch to branch
                self._run_git_command(repo_path, ["checkout", branch_name])

            # Get current branch
            result = self._run_git_command(repo_path, ["branch", "--show-current"])
            current_branch = result.stdout.strip()

            return {
                "success": True,
                "repo_path": str(repo),
                "branch": current_branch,
                "created": create,
                "switched": switch,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git branch failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git branch failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to manage branch for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to manage branch: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_push(
        self, repo_path: str, remote: str = "origin", branch: Optional[str] = None
    ) -> dict[str, Any]:
        """Push commits to remote.

        Args:
            repo_path: Repository path
            remote: Remote name
            branch: Branch name (None for current branch)

        Returns:
            Push result
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Get current branch if not specified
            if branch is None:
                branch_result = self._run_git_command(
                    repo_path, ["branch", "--show-current"]
                )
                branch = branch_result.stdout.strip()

            # Push
            result = self._run_git_command(repo_path, ["push", remote, branch])

            return {
                "success": True,
                "repo_path": str(repo),
                "remote": remote,
                "branch": branch,
                "output": result.stdout or result.stderr,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git push failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git push failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to push for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to push: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_pull(
        self, repo_path: str, remote: str = "origin", branch: Optional[str] = None
    ) -> dict[str, Any]:
        """Pull commits from remote.

        Args:
            repo_path: Repository path
            remote: Remote name
            branch: Branch name (None for current branch)

        Returns:
            Pull result
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Get current branch if not specified
            if branch is None:
                branch_result = self._run_git_command(
                    repo_path, ["branch", "--show-current"]
                )
                branch = branch_result.stdout.strip()

            # Pull
            result = self._run_git_command(repo_path, ["pull", remote, branch])

            return {
                "success": True,
                "repo_path": str(repo),
                "remote": remote,
                "branch": branch,
                "output": result.stdout,
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git pull failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git pull failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to pull for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to pull: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_log(
        self, repo_path: str, max_count: int = 10, oneline: bool = False
    ) -> dict[str, Any]:
        """Get commit log.

        Args:
            repo_path: Repository path
            max_count: Maximum number of commits
            oneline: Show one line per commit

        Returns:
            Commit log
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Build log command
            cmd = ["log", f"--max-count={max_count}"]
            if oneline:
                cmd.append("--oneline")
            else:
                cmd.append("--format=%H|%an|%ae|%ad|%s")

            result = self._run_git_command(repo_path, cmd)

            # Parse commits
            commits = []
            for line in result.stdout.splitlines():
                if not line:
                    continue
                if oneline:
                    parts = line.split(" ", 1)
                    commits.append({
                        "hash": parts[0],
                        "message": parts[1] if len(parts) > 1 else "",
                    })
                else:
                    parts = line.split("|")
                    if len(parts) == 5:
                        commits.append({
                            "hash": parts[0],
                            "author": parts[1],
                            "email": parts[2],
                            "date": parts[3],
                            "message": parts[4],
                        })

            return {
                "success": True,
                "repo_path": str(repo),
                "commits": commits,
                "count": len(commits),
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"Git log failed for {repo_path}: {e.stderr}")
            return {
                "success": False,
                "error": f"Git log failed: {e.stderr}",
                "repo_path": repo_path,
            }
        except Exception as e:
            logger.error(f"Failed to get log for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to get log: {str(e)}",
                "repo_path": repo_path,
            }

    async def git_diff(
        self, repo_path: str, staged: bool = False, file_path: Optional[str] = None
    ) -> dict[str, Any]:
        """Get diff of changes.

        Args:
            repo_path: Repository path
            staged: Show staged changes
            file_path: Specific file to diff

        Returns:
            Diff output
        """
        try:
            repo = Path(repo_path)
            if not repo.exists():
                return {
                    "success": False,
                    "error": f"Repository not found: {repo_path}",
                }

            # Build diff command
            cmd = ["diff"]
            if staged:
                cmd.append("--cached")
            if file_path:
                cmd.append(file_path)

            result = self._run_git_command(repo_path, cmd, check=False)

            return {
                "success": True,
                "repo_path": str(repo),
                "diff": result.stdout,
                "staged": staged,
                "file_path": file_path,
            }

        except Exception as e:
            logger.error(f"Failed to get diff for {repo_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to get diff: {str(e)}",
                "repo_path": repo_path,
            }


def get_server_definition() -> dict[str, Any]:
    """Get Git MCP server definition.

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
                            "description": "Specific files to commit",
                        },
                        "add_all": {
                            "type": "boolean",
                            "description": "Add all modified files",
                            "default": False,
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
                        "switch": {
                            "type": "boolean",
                            "description": "Switch to branch",
                            "default": True,
                        },
                    },
                    "required": ["repo_path", "branch_name"],
                },
            },
            {
                "name": "git_push",
                "description": "Push commits to remote",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path",
                        },
                        "remote": {
                            "type": "string",
                            "description": "Remote name",
                            "default": "origin",
                        },
                        "branch": {
                            "type": "string",
                            "description": "Branch name (current branch if not specified)",
                        },
                    },
                    "required": ["repo_path"],
                },
            },
            {
                "name": "git_pull",
                "description": "Pull commits from remote",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path",
                        },
                        "remote": {
                            "type": "string",
                            "description": "Remote name",
                            "default": "origin",
                        },
                        "branch": {
                            "type": "string",
                            "description": "Branch name (current branch if not specified)",
                        },
                    },
                    "required": ["repo_path"],
                },
            },
            {
                "name": "git_log",
                "description": "Get commit log",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path",
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "Maximum number of commits",
                            "default": 10,
                        },
                        "oneline": {
                            "type": "boolean",
                            "description": "Show one line per commit",
                            "default": False,
                        },
                    },
                    "required": ["repo_path"],
                },
            },
            {
                "name": "git_diff",
                "description": "Get diff of changes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Repository path",
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "Show staged changes",
                            "default": False,
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Specific file to diff",
                        },
                    },
                    "required": ["repo_path"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "critical",
            "category": "devops",
        },
    }