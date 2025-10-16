"""
Axiom Core Business Logic Module
Contains the essential components for Investment Banking Analytics
"""

# Core orchestration components
from .orchestration.graph import create_research_graph, run_research
from .orchestration.state import AxiomState, create_initial_state

# Analysis engines (M&A workflows)
from .analysis_engines.due_diligence import (
    ComprehensiveDDResult,
    FinancialDDResult,
    MADueDiligenceWorkflow,
    run_comprehensive_dd,
    run_financial_dd,
)
from .analysis_engines.target_screening import (
    MATargetScreeningWorkflow,
    ScreeningResult,
    TargetCriteria,
    TargetProfile,
    run_target_screening,
)
from .analysis_engines.valuation import (
    ComparableAnalysis,
    DCFAnalysis,
    MAValuationWorkflow,
    SynergyAnalysis,
    ValuationSummary,
    run_comprehensive_valuation,
    run_dcf_valuation,
)

# Validation and error handling
from .validation.error_handling import (
    AxiomError,
    ComplianceError,
    ErrorHandler,
    FinancialDataError,
    global_error_handler,
)
from .validation.validation import (
    ComplianceValidator,
    FinancialValidator,
    validate_investment_banking_workflow,
)

__all__ = [
    # Orchestration
    "create_research_graph",
    "run_research", 
    "AxiomState",
    "create_initial_state",
    
    # Analysis Engines
    "MATargetScreeningWorkflow",
    "MADueDiligenceWorkflow", 
    "MAValuationWorkflow",
    "TargetCriteria",
    "TargetProfile",
    "ScreeningResult",
    "FinancialDDResult",
    "ComprehensiveDDResult",
    "DCFAnalysis",
    "ComparableAnalysis",
    "SynergyAnalysis", 
    "ValuationSummary",
    "run_target_screening",
    "run_financial_dd",
    "run_comprehensive_dd",
    "run_dcf_valuation",
    "run_comprehensive_valuation",
    
    # Validation
    "AxiomError",
    "FinancialDataError",
    "ComplianceError",
    "ErrorHandler",
    "global_error_handler",
    "FinancialValidator",
    "ComplianceValidator", 
    "validate_investment_banking_workflow",
]