"""LangSmith tracing integration."""

import uuid
from functools import wraps
from typing import Any, Callable, Dict, Optional
from langsmith import Client, traceable

from axiom.config.settings import settings


# Initialize LangSmith client if API key is provided
if settings.langchain_api_key:
    langsmith_client = Client(
        api_url=settings.langchain_endpoint,
        api_key=settings.langchain_api_key
    )
else:
    langsmith_client = None


def trace_node(node_name: str):
    """Decorator for tracing LangGraph nodes."""

    def decorator(func: Callable) -> Callable:
        if not langsmith_client:
            return func

        @traceable(
            name=f"axiom_node_{node_name}",
            project_name=settings.langchain_project,
            client=langsmith_client
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Add node-specific metadata
            if langsmith_client:
                # Get current run context and add metadata
                pass  # LangSmith handles this automatically with traceable

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def trace_tool(tool_name: str):
    """Decorator for tracing tool calls."""

    def decorator(func: Callable) -> Callable:
        if not langsmith_client:
            return func

        @traceable(
            name=f"axiom_tool_{tool_name}",
            project_name=settings.langchain_project,
            client=langsmith_client
        )
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def create_trace_id() -> str:
    """Create a unique trace ID for a research session."""
    return str(uuid.uuid4())


def get_trace_url(trace_id: str) -> Optional[str]:
    """Get the LangSmith trace URL for a given trace ID."""
    if not langsmith_client:
        return None

    return f"{settings.langchain_endpoint}/o/{settings.langchain_project}/traces/{trace_id}"
