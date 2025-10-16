"""Axiom Investment Banking Analytics Platform."""

__version__ = "0.1.0"
__author__ = "Axiom Team"
__description__ = "Investment Banking Analytics Platform — AI-Powered Due Diligence, M&A Analysis, and Financial Intelligence"

from axiom.config.schemas import Citation, Evidence, ResearchBrief
from axiom.config.settings import settings
from axiom.graph.graph import create_research_graph, run_research

from axiom.workflows.due_diligence import (
    CommercialDDResult,
    ComprehensiveDDResult,
    FinancialDDResult,
    MADueDiligenceWorkflow,
    OperationalDDResult,
    run_comprehensive_dd,
    run_financial_dd,
)

# M&A Workflow Imports
from axiom.workflows.target_screening import (
    MATargetScreeningWorkflow,
    ScreeningResult,
    TargetCriteria,
    TargetProfile,
    run_target_screening,
)
from axiom.workflows.valuation import (
    ComparableAnalysis,
    DCFAnalysis,
    MAValuationWorkflow,
    PrecedentAnalysis,
    SynergyAnalysis,
    ValuationSummary,
    run_comprehensive_valuation,
    run_dcf_valuation,
)

__all__ = [
    # Core Platform
    "run_research",
    "create_research_graph",
    "settings",
    "ResearchBrief",
    "Evidence",
    "Citation",
    # M&A Target Screening
    "MATargetScreeningWorkflow",
    "TargetCriteria",
    "TargetProfile",
    "ScreeningResult",
    "run_target_screening",
    # M&A Due Diligence
    "MADueDiligenceWorkflow",
    "FinancialDDResult",
    "CommercialDDResult",
    "OperationalDDResult",
    "ComprehensiveDDResult",
    "run_financial_dd",
    "run_comprehensive_dd",
    # M&A Valuation
    "MAValuationWorkflow",
    "DCFAnalysis",
    "ComparableAnalysis",
    "PrecedentAnalysis",
    "SynergyAnalysis",
    "ValuationSummary",
    "run_dcf_valuation",
    "run_comprehensive_valuation",
]
