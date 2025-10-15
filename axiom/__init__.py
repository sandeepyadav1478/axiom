"""Axiom Investment Banking Analytics Platform."""

__version__ = "0.1.0"
__author__ = "Axiom Team"
__description__ = "Investment Banking Analytics Platform — AI-Powered Due Diligence, M&A Analysis, and Financial Intelligence"

from axiom.config.schemas import Citation, Evidence, ResearchBrief
from axiom.config.settings import settings
from axiom.graph.graph import create_research_graph, run_research
from axiom.main import research_query

__all__ = [
    "run_research",
    "create_research_graph",
    "research_query",
    "settings",
    "ResearchBrief",
    "Evidence",
    "Citation",
]
