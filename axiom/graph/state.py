"""LangGraph state management for Axiom research agent."""

from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages
from axiom.config.schemas import (
    TaskPlan, SearchQuery, SearchResult, CrawlResult, 
    Evidence, ResearchBrief
)


class AxiomState(TypedDict):
    """Typed state for LangGraph workflow."""

    # Core input/output
    query: str
    brief: Optional[ResearchBrief]

    # Planning phase
    task_plans: List[TaskPlan]
    enriched_queries: List[SearchQuery]

    # Execution phase  
    search_results: List[SearchResult]
    crawl_results: List[CrawlResult]

    # Analysis phase
    evidence: List[Evidence]

    # Messages and tracing
    messages: Annotated[List, add_messages]
    step_count: int
    error_messages: List[str]
    trace_id: Optional[str]


def create_initial_state(query: str, trace_id: Optional[str] = None) -> AxiomState:
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
        trace_id=trace_id
    )
