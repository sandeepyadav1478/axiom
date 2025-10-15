"""Pydantic schemas for Axiom Investment Banking Analytics Platform."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """A search query with metadata."""

    query: str = Field(..., description="The search query string")
    query_type: Literal["original", "expanded", "hyde"] = Field(
        "original", description="Type of query"
    )
    priority: int = Field(1, description="Query priority (1=highest)")


class SearchResult(BaseModel):
    """A search result from Tavily or other sources."""

    title: str = Field(..., description="Result title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(..., description="Text snippet")
    score: float = Field(..., description="Relevance score")
    timestamp: datetime | None = Field(None, description="Publication timestamp")


class CrawlResult(BaseModel):
    """Full page content from Firecrawl."""

    url: str = Field(..., description="Crawled URL")
    title: str = Field(..., description="Page title")
    content: str = Field(..., description="Full page content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class TaskPlan(BaseModel):
    """A decomposed research task."""

    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    queries: list[SearchQuery] = Field(..., description="Search queries for this task")
    estimated_priority: int = Field(1, description="Task priority")


class Evidence(BaseModel):
    """A piece of evidence with citation."""

    content: str = Field(..., description="Evidence content")
    source_url: str = Field(..., description="Source URL")
    source_title: str = Field(..., description="Source title")
    confidence: float = Field(..., description="Confidence in this evidence")
    relevance_score: float = Field(..., description="Relevance to the query")


class Citation(BaseModel):
    """A citation reference."""

    source_url: str = Field(..., description="Source URL")
    source_title: str = Field(..., description="Source title")
    snippet: str = Field(..., description="Relevant snippet")
    access_date: datetime = Field(
        default_factory=datetime.now, description="When this was accessed"
    )


class ResearchBrief(BaseModel):
    """Final structured research brief."""

    topic: str = Field(..., description="Research topic")
    questions_answered: list[str] = Field(
        ..., description="Questions that were answered"
    )
    key_findings: list[str] = Field(..., description="Key findings and insights")
    evidence: list[Evidence] = Field(..., description="Supporting evidence")
    citations: list[Citation] = Field(..., description="Source citations")
    remaining_gaps: list[str] = Field(
        default_factory=list, description="Unanswered questions"
    )
    confidence: float = Field(..., description="Overall confidence in findings (0-1)")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this brief was generated"
    )


class GraphState(BaseModel):
    """State object for LangGraph workflow."""

    # Input
    query: str = Field(..., description="Original research query")

    # Planning phase
    task_plans: list[TaskPlan] = Field(
        default_factory=list, description="Decomposed tasks"
    )
    enriched_queries: list[SearchQuery] = Field(
        default_factory=list, description="DSPy-enriched queries"
    )

    # Execution phase
    search_results: list[SearchResult] = Field(
        default_factory=list, description="Search results"
    )
    crawl_results: list[CrawlResult] = Field(
        default_factory=list, description="Crawled content"
    )

    # Analysis phase
    evidence: list[Evidence] = Field(
        default_factory=list, description="Extracted evidence"
    )

    # Output
    brief: ResearchBrief | None = Field(None, description="Final research brief")

    # Metadata
    step_count: int = Field(0, description="Current step in workflow")
    error_messages: list[str] = Field(
        default_factory=list, description="Any errors encountered"
    )
    trace_id: str | None = Field(None, description="LangSmith trace ID")


class DSPyMetrics(BaseModel):
    """Metrics for DSPy optimization."""

    recall_at_k: float = Field(..., description="Recall@K metric")
    precision_at_k: float = Field(..., description="Precision@K metric")
    citation_completeness: float = Field(..., description="Citation completeness score")
    faithfulness: float = Field(..., description="Faithfulness to sources")
    query_diversity: float = Field(..., description="Diversity of generated queries")
