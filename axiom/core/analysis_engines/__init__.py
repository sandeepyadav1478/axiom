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

from .advanced_modeling import (
    AdvancedModelingResult,
    MAAdvancedModelingWorkflow,
    MonteCarloResult,
    ScenarioAnalysis,
    StressTestResult,
    run_advanced_financial_modeling,
    run_comprehensive_stress_testing,
    run_monte_carlo_valuation,
)
from .due_diligence import (
    CommercialDDResult,
    ComprehensiveDDResult,
    FinancialDDResult,
    MADueDiligenceWorkflow,
    OperationalDDResult,
    run_comprehensive_dd,
    run_financial_dd,
)
from .esg_analysis import (
    EnvironmentalAssessment,
    ESGAssessmentResult,
    GovernanceAssessment,
    MAESGAnalysisWorkflow,
    SocialAssessment,
    assess_esg_investment_impact,
    run_esg_analysis,
)
from .executive_dashboards import (
    DealPerformanceMetrics,
    ExecutiveDashboardResult,
    MAExecutiveDashboardWorkflow,
    PortfolioAnalytics,
    SynergyRealizationDashboard,
    analyze_portfolio_performance,
    generate_executive_ma_dashboard,
    track_portfolio_synergies,
)
from .market_intelligence import (
    CompetitorProfile,
    DisruptionAssessment,
    MAMarketIntelligenceWorkflow,
    MarketIntelligenceResult,
    MarketTrendAnalysis,
    run_competitive_analysis,
    run_disruption_assessment,
    run_market_intelligence_analysis,
)
from .pmi_planning import (
    Day1ReadinessPlan,
    IntegrationWorkstream,
    MAPMIPlanningWorkflow,
    PMIExecutionPlan,
    SynergyRealizationPlan,
    run_day1_planning,
    run_pmi_planning,
)
from .regulatory_compliance import (
    HSRAnalysis,
    InternationalClearance,
    MARegulatoryComplianceWorkflow,
    RegulatoryComplianceResult,
    run_hsr_analysis,
    run_regulatory_compliance_analysis,
)
from .risk_assessment import (
    MAAdvancedRiskAssessment,
    RiskAssessmentResult,
    RiskCategory,
    run_advanced_risk_assessment,
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

    # Advanced Financial Modeling
    "MAAdvancedModelingWorkflow",
    "MonteCarloResult",
    "StressTestResult",
    "ScenarioAnalysis",
    "AdvancedModelingResult",
    "run_monte_carlo_valuation",
    "run_comprehensive_stress_testing",
    "run_advanced_financial_modeling",

    # Market Intelligence
    "MAMarketIntelligenceWorkflow",
    "CompetitorProfile",
    "MarketTrendAnalysis",
    "DisruptionAssessment",
    "MarketIntelligenceResult",
    "run_market_intelligence_analysis",
    "run_competitive_analysis",
    "run_disruption_assessment",

    # Executive Dashboards
    "MAExecutiveDashboardWorkflow",
    "ExecutiveDashboardResult",
    "PortfolioAnalytics",
    "DealPerformanceMetrics",
    "SynergyRealizationDashboard",
    "generate_executive_ma_dashboard",
    "track_portfolio_synergies",
    "analyze_portfolio_performance",

    # ESG Analysis
    "MAESGAnalysisWorkflow",
    "ESGAssessmentResult",
    "EnvironmentalAssessment",
    "SocialAssessment",
    "GovernanceAssessment",
    "run_esg_analysis",
    "assess_esg_investment_impact",

    # PMI Planning
    "MAPMIPlanningWorkflow",
    "PMIExecutionPlan",
    "IntegrationWorkstream",
    "Day1ReadinessPlan",
    "SynergyRealizationPlan",
    "run_pmi_planning",
    "run_day1_planning",
]
