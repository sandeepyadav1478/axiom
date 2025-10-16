"""
Axiom Investment Banking M&A Workflows Package

Complete M&A lifecycle management with specialized workflows for:
- Target Identification & Screening
- Due Diligence (Financial, Commercial, Legal, Operational)
- Valuation & Deal Structure
- Risk Assessment & Management
- Post-Merger Integration Planning
- Regulatory & Antitrust Analysis
- Deal Execution Support
"""

from .due_diligence import (
    CommercialDDResult,
    ComprehensiveDDResult,
    FinancialDDResult,
    MADueDiligenceWorkflow,
    OperationalDDResult,
    run_comprehensive_dd,
    run_financial_dd,
)
from .target_screening import (
    MATargetScreeningWorkflow,
    ScreeningResult,
    TargetCriteria,
    TargetProfile,
    run_target_screening,
)
from .valuation import (
    ComparableAnalysis,
    DCFAnalysis,
    MAValuationWorkflow,
    PrecedentAnalysis,
    SynergyAnalysis,
    ValuationSummary,
    run_comprehensive_valuation,
    run_dcf_valuation,
)
from .risk_assessment import (
    MAAdvancedRiskAssessment,
    RiskCategory,
    RiskAssessmentResult,
    run_advanced_risk_assessment,
)
from .regulatory_compliance import (
    MARegulatoryComplianceWorkflow,
    HSRAnalysis,
    InternationalClearance,
    RegulatoryComplianceResult,
    run_regulatory_compliance_analysis,
    run_hsr_analysis,
)

__all__ = [
    # Target Screening
    "MATargetScreeningWorkflow",
    "TargetCriteria",
    "TargetProfile",
    "ScreeningResult",
    "run_target_screening",
    # Due Diligence
    "MADueDiligenceWorkflow",
    "FinancialDDResult",
    "CommercialDDResult",
    "OperationalDDResult",
    "ComprehensiveDDResult",
    "run_financial_dd",
    "run_comprehensive_dd",
    # Valuation
    "MAValuationWorkflow",
    "DCFAnalysis",
    "ComparableAnalysis",
    "PrecedentAnalysis",
    "SynergyAnalysis",
    "ValuationSummary",
    "run_dcf_valuation",
    "run_comprehensive_valuation",
    
    # Advanced Risk Assessment
    "MAAdvancedRiskAssessment",
    "RiskCategory",
    "RiskAssessmentResult",
    "run_advanced_risk_assessment",
    
    # Regulatory Compliance
    "MARegulatoryComplianceWorkflow",
    "HSRAnalysis",
    "InternationalClearance",
    "RegulatoryComplianceResult",
    "run_regulatory_compliance_analysis",
    "run_hsr_analysis",
]
