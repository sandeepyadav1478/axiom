"""Axiom Investment Banking Analytics Platform."""

__version__ = "0.1.0"
__author__ = "Axiom Team"
__description__ = "Investment Banking Analytics Platform — AI-Powered Due Diligence, M&A Analysis, and Financial Intelligence"

from axiom.graph.graph import run_research, create_research_graph
from axiom.config.settings import settings
from axiom.config.schemas import ResearchBrief, Evidence, Citation
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
