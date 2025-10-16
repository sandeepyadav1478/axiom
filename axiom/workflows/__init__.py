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
from .pmi_planning import (
    MAPMIPlanningWorkflow,
    PMIExecutionPlan,
    IntegrationWorkstream,
    Day1ReadinessPlan,
    SynergyRealizationPlan,
    run_pmi_planning,
    run_day1_planning,
)
from .advanced_modeling import (
    MAAdvancedModelingWorkflow,
    MonteCarloResult,
    StressTestResult,
    ScenarioAnalysis,
    AdvancedModelingResult,
    run_monte_carlo_valuation,
    run_comprehensive_stress_testing,
    run_advanced_financial_modeling,
)
from .market_intelligence import (
    MAMarketIntelligenceWorkflow,
    CompetitorProfile,
    MarketTrendAnalysis,
    DisruptionAssessment,
    MarketIntelligenceResult,
    run_market_intelligence_analysis,
    run_competitive_analysis,
    run_disruption_assessment,
)
from .executive_dashboards import (
    MAExecutiveDashboardWorkflow,
    ExecutiveDashboardResult,
    PortfolioAnalytics,
    DealPerformanceMetrics,
    SynergyRealizationDashboard,
    generate_executive_ma_dashboard,
    track_portfolio_synergies,
    analyze_portfolio_performance,
)
from .esg_analysis import (
    MAESGAnalysisWorkflow,
    ESGAssessmentResult,
    EnvironmentalAssessment,
    SocialAssessment,
    GovernanceAssessment,
    run_esg_analysis,
    assess_esg_investment_impact,
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
