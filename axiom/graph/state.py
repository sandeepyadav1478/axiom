"""LangGraph state management for Axiom Investment Banking Analytics Platform."""

from typing import Annotated, TypedDict

from langgraph.graph import add_messages

from axiom.config.schemas import (
    CrawlResult,
    Evidence,
    ResearchBrief,
    SearchQuery,
    SearchResult,
    TaskPlan,
)


class AxiomState(TypedDict):
    """Typed state for LangGraph workflow."""

    # Core input/output
    query: str
    brief: ResearchBrief | None

    # Planning phase
    task_plans: list[TaskPlan]
    enriched_queries: list[SearchQuery]

    # Execution phase
    search_results: list[SearchResult]
    crawl_results: list[CrawlResult]

    # Analysis phase
    evidence: list[Evidence]

    # Messages and tracing
    messages: Annotated[list, add_messages]
    step_count: int
    error_messages: list[str]
    trace_id: str | None


def create_initial_state(query: str, trace_id: str | None = None) -> AxiomState:
    """Create initial state for a research query."""
    return AxiomState(
        query=query,
        brief=None,
        task_plans=[],
        enriched_queries=[],
        search_results=[],
        crawl_results=[],
        evidence=[],
        messages=[],
        step_count=0,
        error_messages=[],
        trace_id=trace_id,
    )
