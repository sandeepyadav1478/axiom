"""
Linting & Formatting MCP Server

Provides automated code quality checks including linting, formatting, type checking,
security scanning, and complexity analysis. Maintains Bloomberg-level code quality.
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LintingMCPServer:
    """MCP Server for code quality automation."""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize Linting MCP Server.

        Args:
            project_root: Root directory of the project (default: current directory)
        """
        self.server = Server("linting-code-quality")
        self.project_root = Path(project_root or os.getcwd())

        # Tool configurations
        self.tools_config = {
            "pylint": {"enabled": True, "config": None},
            "flake8": {"enabled": True, "config": None},
            "ruff": {"enabled": True, "config": None},
            "black": {"enabled": True, "config": None},
            "isort": {"enabled": True, "config": None},
            "mypy": {"enabled": True, "config": None},
            "bandit": {"enabled": True, "config": None}
        }

        self._register_handlers()
        logger.info("Linting & Formatting MCP Server initialized")

    def _register_handlers(self):
        """Register all tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available linting and formatting tools."""
            return [
                Tool(
                    name="lint_python",
                    description="Run Python linters (pylint, flake8, ruff) on specified files or directories. "
                                "Returns detailed linting results with line numbers and suggestions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to lint (relative to project root)"
                            },
                            "linter": {
                                "type": "string",
                                "enum": ["pylint", "flake8", "ruff", "all"],
                                "description": "Linter to use (default: all)",
                                "default": "all"
                            },
                            "strict": {
                                "type": "boolean",
                                "description": "Enable strict mode with all checks (default: false)",
                                "default": False
                            },
                            "ignore_errors": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of error codes to ignore (e.g., ['E501', 'W503'])"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="format_python",
                    description="Auto-format Python code with black and isort. "
                                "Formats code to PEP 8 standards and organizes imports.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to format"
                            },
                            "check_only": {
                                "type": "boolean",
                                "description": "Only check formatting without making changes (default: false)",
                                "default": False
                            },
                            "line_length": {
                                "type": "integer",
                                "description": "Maximum line length (default: 88 for black)",
                                "default": 88
                            },
                            "skip_string_normalization": {
                                "type": "boolean",
                                "description": "Skip string quote normalization (default: false)",
                                "default": False
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="type_check",
                    description="Run mypy static type checking to find type-related bugs. "
                                "Checks type annotations and catches type errors before runtime.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to type check"
                            },
                            "strict": {
                                "type": "boolean",
                                "description": "Enable strict mode (default: false)",
                                "default": False
                            },
                            "show_error_codes": {
                                "type": "boolean",
                                "description": "Show error codes in output (default: true)",
                                "default": True
                            },
                            "ignore_missing_imports": {
                                "type": "boolean",
                                "description": "Ignore missing imports (default: false)",
                                "default": False
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="security_scan",
                    description="Run bandit security scanner to find common security issues in Python code. "
                                "Identifies vulnerabilities like SQL injection, hardcoded passwords, etc.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to scan"
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "all"],
                                "description": "Minimum severity level (default: medium)",
                                "default": "medium"
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "all"],
                                "description": "Minimum confidence level (default: medium)",
                                "default": "medium"
                            },
                            "format": {
                                "type": "string",
                                "enum": ["json", "text", "html"],
                                "description": "Output format (default: json)",
                                "default": "json"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="complexity_analysis",
                    description="Analyze code complexity using McCabe complexity metric. "
                                "Identifies complex functions that may need refactoring.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to analyze"
                            },
                            "max_complexity": {
                                "type": "integer",
                                "description": "Maximum allowed complexity (default: 10)",
                                "default": 10
                            },
                            "show_complexity": {
                                "type": "boolean",
                                "description": "Show complexity scores for all functions (default: true)",
                                "default": True
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="import_optimization",
                    description="Optimize and organize Python imports using isort. "
                                "Groups and sorts imports according to PEP 8 and project standards.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to optimize"
                            },
                            "check_only": {
                                "type": "boolean",
                                "description": "Only check without making changes (default: false)",
                                "default": False
                            },
                            "profile": {
                                "type": "string",
                                "enum": ["black", "django", "pycharm", "google", "open_stack"],
                                "description": "Import style profile (default: black)",
                                "default": "black"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="docstring_validation",
                    description="Validate docstring presence and format using pydocstyle. "
                                "Ensures all public functions, classes, and modules have proper documentation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to validate"
                            },
                            "convention": {
                                "type": "string",
                                "enum": ["google", "numpy", "pep257"],
                                "description": "Docstring convention (default: google)",
                                "default": "google"
                            },
                            "match_dir": {
                                "type": "string",
                                "description": "Pattern for directories to check (default: all)"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="dead_code_detection",
                    description="Find unused code, imports, and variables using vulture. "
                                "Helps identify and remove dead code to improve maintainability.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to analyze"
                            },
                            "min_confidence": {
                                "type": "integer",
                                "description": "Minimum confidence (0-100, default: 60)",
                                "default": 60
                            },
                            "exclude": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Patterns to exclude (e.g., ['tests/*', '__init__.py'])"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="auto_fix",
                    description="Automatically fix common code quality issues. "
                                "Runs formatters and auto-fixers to resolve linting issues.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File or directory path to fix"
                            },
                            "tools": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["black", "isort", "ruff", "autopep8", "all"]
                                },
                                "description": "Tools to use for fixing (default: ['black', 'isort', 'ruff'])",
                                "default": ["black", "isort", "ruff"]
                            },
                            "safe_mode": {
                                "type": "boolean",
                                "description": "Only apply safe fixes (default: true)",
                                "default": True
                            }
                        },
                        "required": ["path"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "lint_python":
                    result = await self._lint_python(**arguments)
                elif name == "format_python":
                    result = await self._format_python(**arguments)
                elif name == "type_check":
                    result = await self._type_check(**arguments)
                elif name == "security_scan":
                    result = await self._security_scan(**arguments)
                elif name == "complexity_analysis":
                    result = await self._complexity_analysis(**arguments)
                elif name == "import_optimization":
                    result = await self._import_optimization(**arguments)
                elif name == "docstring_validation":
                    result = await self._docstring_validation(**arguments)
                elif name == "dead_code_detection":
                    result = await self._dead_code_detection(**arguments)
                elif name == "auto_fix":
                    result = await self._auto_fix(**arguments)
                else:
                    result = f"Unknown tool: {name}"

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

    async def _run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run a shell command and return the result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd or self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout.decode("utf-8", errors="ignore"),
                "stderr": stderr.decode("utf-8", errors="ignore"),
                "command": " ".join(cmd)
            }

        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Command not found: {cmd[0]}. Please install it first.",
                "command": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(cmd)
            }

    async def _lint_python(
        self,
        path: str,
        linter: str = "all",
        strict: bool = False,
        ignore_errors: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run Python linters."""
        target_path = self.project_root / path
        results = {"path": path, "linters": {}}

        # Run pylint
        if linter in ["pylint", "all"]:
            cmd = ["pylint", str(target_path)]
            if ignore_errors:
                cmd.extend(["--disable", ",".join(ignore_errors)])
            if not strict:
                cmd.append("--errors-only")

            result = await self._run_command(cmd)
            results["linters"]["pylint"] = {
                "success": result["success"],
                "output": result.get("stdout", ""),
                "issues_found": "rated at" in result.get("stdout", "").lower()
            }

        # Run flake8
        if linter in ["flake8", "all"]:
            cmd = ["flake8", str(target_path)]
            if ignore_errors:
                cmd.extend(["--ignore", ",".join(ignore_errors)])
            if strict:
                cmd.append("--max-complexity=10")

            result = await self._run_command(cmd)
            results["linters"]["flake8"] = {
                "success": result["success"],
                "output": result.get("stdout", ""),
                "issues_count": len(result.get("stdout", "").split("\n")) - 1
            }

        # Run ruff
        if linter in ["ruff", "all"]:
            cmd = ["ruff", "check", str(target_path)]
            if ignore_errors:
                cmd.extend(["--ignore", ",".join(ignore_errors)])
            if strict:
                cmd.append("--select=ALL")

            result = await self._run_command(cmd)
            results["linters"]["ruff"] = {
                "success": result["success"],
                "output": result.get("stdout", ""),
                "can_autofix": "--fix" in result.get("stdout", "")
            }

        # Overall summary
        all_success = all(
            linter_result["success"]
            for linter_result in results["linters"].values()
        )
        results["overall_success"] = all_success
        results["recommendation"] = (
            "All linters passed!" if all_success
            else "Issues found. Review output and consider running auto_fix."
        )

        return results

    async def _format_python(
        self,
        path: str,
        check_only: bool = False,
        line_length: int = 88,
        skip_string_normalization: bool = False
    ) -> Dict[str, Any]:
        """Format Python code with black and isort."""
        target_path = self.project_root / path
        results = {"path": path, "formatters": {}}

        # Run black
        cmd = ["black", str(target_path), f"--line-length={line_length}"]
        if check_only:
            cmd.append("--check")
        if skip_string_normalization:
            cmd.append("--skip-string-normalization")

        black_result = await self._run_command(cmd)
        results["formatters"]["black"] = {
            "success": black_result["success"],
            "formatted": not check_only and black_result["success"],
            "output": black_result.get("stdout", ""),
            "files_changed": black_result.get("stdout", "").count("reformatted") if not check_only else 0
        }

        # Run isort
        cmd = ["isort", str(target_path), "--profile=black"]
        if check_only:
            cmd.append("--check-only")

        isort_result = await self._run_command(cmd)
        results["formatters"]["isort"] = {
            "success": isort_result["success"],
            "formatted": not check_only and isort_result["success"],
            "output": isort_result.get("stdout", "")
        }

        results["check_only"] = check_only
        results["overall_success"] = all(
            fmt["success"] for fmt in results["formatters"].values()
        )

        return results

    async def _type_check(
        self,
        path: str,
        strict: bool = False,
        show_error_codes: bool = True,
        ignore_missing_imports: bool = False
    ) -> Dict[str, Any]:
        """Run mypy type checking."""
        target_path = self.project_root / path

        cmd = ["mypy", str(target_path)]
        if strict:
            cmd.append("--strict")
        if show_error_codes:
            cmd.append("--show-error-codes")
        if ignore_missing_imports:
            cmd.append("--ignore-missing-imports")

        result = await self._run_command(cmd)

        # Parse mypy output
        output = result.get("stdout", "")
        errors = [line for line in output.split("\n") if line.strip() and "error:" in line.lower()]

        return {
            "path": path,
            "success": result["success"],
            "total_errors": len(errors),
            "errors": errors[:20],  # Limit to first 20 for readability
            "full_output": output,
            "type_coverage": self._calculate_type_coverage(output)
        }

    def _calculate_type_coverage(self, mypy_output: str) -> Optional[float]:
        """Calculate type annotation coverage from mypy output."""
        import re
        match = re.search(r"(\d+)% type coverage", mypy_output)
        if match:
            return float(match.group(1))
        return None

    async def _security_scan(
        self,
        path: str,
        severity: str = "medium",
        confidence: str = "medium",
        format: str = "json"
    ) -> Dict[str, Any]:
        """Run bandit security scanner."""
        target_path = self.project_root / path

        cmd = ["bandit", "-r", str(target_path), f"-f{format}"]

        # Add severity filter
        severity_map = {"low": "L", "medium": "M", "high": "H"}
        if severity != "all":
            cmd.extend(["-ll", f"-i{severity_map[severity]}"])

        # Add confidence filter
        if confidence != "all":
            cmd.extend(["-iii", f"-c{confidence_map[confidence]}" if confidence in (confidence_map := {"low": "L", "medium": "M", "high": "H"}) else ""])

        result = await self._run_command(cmd)

        if format == "json" and result.get("stdout"):
            try:
                security_data = json.loads(result["stdout"])
                return {
                    "path": path,
                    "success": result["success"],
                    "total_issues": len(security_data.get("results", [])),
                    "issues_by_severity": self._group_by_severity(security_data.get("results", [])),
                    "critical_issues": [
                        issue for issue in security_data.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ][:10],
                    "metrics": security_data.get("metrics", {})
                }
            except json.JSONDecodeError:
                pass

        return {
            "path": path,
            "success": result["success"],
            "output": result.get("stdout", ""),
            "format": format
        }

    def _group_by_severity(self, issues: List[Dict]) -> Dict[str, int]:
        """Group security issues by severity."""
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for issue in issues:
            severity = issue.get("issue_severity", "LOW")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        return severity_counts

    async def _complexity_analysis(
        self,
        path: str,
        max_complexity: int = 10,
        show_complexity: bool = True
    ) -> Dict[str, Any]:
        """Analyze code complexity."""
        target_path = self.project_root / path

        cmd = ["radon", "cc", str(target_path), "-a", "-j"]
        if show_complexity:
            cmd.append("-s")

        result = await self._run_command(cmd)

        try:
            if result.get("stdout"):
                complexity_data = json.loads(result["stdout"])

                # Find complex functions
                complex_functions = []
                for file_path, functions in complexity_data.items():
                    for func in functions:
                        if func.get("complexity", 0) > max_complexity:
                            complex_functions.append({
                                "file": file_path,
                                "function": func.get("name"),
                                "complexity": func.get("complexity"),
                                "line": func.get("lineno")
                            })

                return {
                    "path": path,
                    "max_complexity_threshold": max_complexity,
                    "total_functions": sum(len(funcs) for funcs in complexity_data.values()),
                    "complex_functions_count": len(complex_functions),
                    "complex_functions": complex_functions,
                    "average_complexity": self._calculate_average_complexity(complexity_data),
                    "recommendation": (
                        "Good code complexity!" if not complex_functions
                        else f"Found {len(complex_functions)} functions exceeding complexity threshold. Consider refactoring."
                    )
                }
        except (json.JSONDecodeError, KeyError):
            pass

        return {
            "path": path,
            "success": result["success"],
            "output": result.get("stdout", "")
        }

    def _calculate_average_complexity(self, complexity_data: Dict) -> float:
        """Calculate average complexity across all functions."""
        total_complexity = 0
        total_functions = 0

        for functions in complexity_data.values():
            for func in functions:
                total_complexity += func.get("complexity", 0)
                total_functions += 1

        return round(total_complexity / total_functions, 2) if total_functions > 0 else 0

    async def _import_optimization(
        self,
        path: str,
        check_only: bool = False,
        profile: str = "black"
    ) -> Dict[str, Any]:
        """Optimize imports with isort."""
        target_path = self.project_root / path

        cmd = ["isort", str(target_path), f"--profile={profile}"]
        if check_only:
            cmd.append("--check-only")
        cmd.append("--diff")  # Always show diff

        result = await self._run_command(cmd)

        return {
            "path": path,
            "profile": profile,
            "check_only": check_only,
            "success": result["success"],
            "changes_needed": not result["success"],
            "diff": result.get("stdout", ""),
            "files_modified": result.get("stdout", "").count("Fixing") if not check_only else 0
        }

    async def _docstring_validation(
        self,
        path: str,
        convention: str = "google",
        match_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate docstrings."""
        target_path = self.project_root / path

        cmd = ["pydocstyle", str(target_path), f"--convention={convention}"]
        if match_dir:
            cmd.extend(["--match-dir", match_dir])

        result = await self._run_command(cmd)

        output = result.get("stdout", "")
        violations = [line for line in output.split("\n") if line.strip()]

        return {
            "path": path,
            "convention": convention,
            "success": result["success"],
            "total_violations": len(violations),
            "violations": violations[:20],  # Limit output
            "full_output": output
        }

    async def _dead_code_detection(
        self,
        path: str,
        min_confidence: int = 60,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect dead code."""
        target_path = self.project_root / path

        cmd = ["vulture", str(target_path), f"--min-confidence={min_confidence}"]
        if exclude:
            for pattern in exclude:
                cmd.extend(["--exclude", pattern])

        result = await self._run_command(cmd)

        output = result.get("stdout", "")
        dead_code_items = [line for line in output.split("\n") if line.strip() and "unused" in line.lower()]

        return {
            "path": path,
            "min_confidence": min_confidence,
            "success": result["success"],
            "dead_code_found": len(dead_code_items) > 0,
            "total_items": len(dead_code_items),
            "items": dead_code_items[:20],
            "recommendation": (
                "No dead code detected!" if not dead_code_items
                else f"Found {len(dead_code_items)} potentially unused items. Review and remove if appropriate."
            )
        }

    async def _auto_fix(
        self,
        path: str,
        tools: List[str] = ["black", "isort", "ruff"],
        safe_mode: bool = True
    ) -> Dict[str, Any]:
        """Auto-fix code quality issues."""
        target_path = self.project_root / path
        results = {"path": path, "tools": {}, "safe_mode": safe_mode}

        if "all" in tools:
            tools = ["black", "isort", "ruff", "autopep8"]

        # Run black
        if "black" in tools:
            cmd = ["black", str(target_path)]
            result = await self._run_command(cmd)
            results["tools"]["black"] = {
                "success": result["success"],
                "files_reformatted": result.get("stdout", "").count("reformatted")
            }

        # Run isort
        if "isort" in tools:
            cmd = ["isort", str(target_path), "--profile=black"]
            result = await self._run_command(cmd)
            results["tools"]["isort"] = {
                "success": result["success"],
                "files_fixed": "Fixed" in result.get("stdout", "")
            }

        # Run ruff
        if "ruff" in tools:
            cmd = ["ruff", "check", str(target_path), "--fix"]
            if safe_mode:
                cmd.append("--safe")
            result = await self._run_command(cmd)
            results["tools"]["ruff"] = {
                "success": result["success"],
                "fixes_applied": "Fixed" in result.get("stdout", "")
            }

        # Run autopep8
        if "autopep8" in tools:
            cmd = ["autopep8", "--in-place", "--recursive", str(target_path)]
            if not safe_mode:
                cmd.append("--aggressive")
            result = await self._run_command(cmd)
            results["tools"]["autopep8"] = {
                "success": result["success"]
            }

        results["overall_success"] = all(
            tool_result["success"]
            for tool_result in results["tools"].values()
        )
        results["recommendation"] = (
            "All fixes applied successfully! Run lint_python to verify."
            if results["overall_success"]
            else "Some fixes failed. Check individual tool outputs."
        )

        return results

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = LintingMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())