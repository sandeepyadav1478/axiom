"""
Axiom Credit Risk Models - Basel III Compliant Suite
====================================================

Institutional-grade credit risk modeling and analysis:

Phase 2A - Core Models:
- Merton's structural credit model with term structure
- Probability of Default (PD) estimation (KMV, Altman, Logistic, Rating curves)
- Loss Given Default (LGD) modeling (Beta, Seniority, Collateral, Downturn)
- Exposure at Default (EAD) calculation (Simple, CCF, PFE, EPE, SA-CCR)

Phase 2B - Portfolio Risk:
- Credit Value at Risk (CVaR) with multiple approaches
- Portfolio credit risk aggregation and concentration
- Default correlation modeling (Copulas, Factor models)
- Economic and regulatory capital allocation
- Risk-adjusted return metrics (RAROC, RoRWA)

All models are optimized for institutional-grade risk management with
Basel III compliance, IFRS 9/CECL support, and <500ms performance targets.
"""

# Phase 2A - Core Models
from .structural_models import (
    MertonModel,
    calculate_default_probability,
    calculate_distance_to_default,
    calculate_credit_spread,
)
from .default_probability import (
    PDApproach,
    PDType,
    PDEstimate,
    KMVMertonPD,
    AltmanZScore,
    LogisticPDModel,
    RatingAgencyPDCurve,
    PDEstimator,
    calculate_kmv_pd,
    calculate_altman_pd,
    get_rating_pd,
)
from .lgd_models import (
    LGDApproach,
    SeniorityClass,
    LGDEstimate,
    BetaLGD,
    RecoveryRateBySeniority,
    CollateralLGD,
    WorkoutLGD,
    DownturnLGD,
    LGDModel,
    calculate_lgd_by_seniority,
    calculate_beta_lgd,
    calculate_downturn_lgd,
)
from .ead_models import (
    EADApproach,
    FacilityType,
    EADEstimate,
    DerivativeExposure,
    CreditConversionFactor,
    SimpleEAD,
    PotentialFutureExposure,
    SACCR,
    EADCalculator,
    calculate_simple_ead,
    calculate_derivative_pfe,
    calculate_sa_ccr_ead,
)

# Phase 2B - Portfolio Risk
from .credit_var import (
    CVaRApproach,
    VarianceReduction,
    Obligor,
    CreditVaRResult,
    ComponentCVaR,
    AnalyticalCVaR,
    CreditMetricsCVaR,
    MonteCarloCVaR,
    CreditVaRCalculator,
    calculate_credit_var,
    calculate_expected_shortfall,
)
from .portfolio_risk import (
    CapitalApproach,
    ConcentrationMetric,
    PortfolioMetrics,
    ConcentrationAnalysis,
    RiskContribution,
    PortfolioAggregator,
    ConcentrationRisk,
    CapitalAllocation,
    RiskAdjustedMetrics,
    PortfolioRiskAnalyzer,
    analyze_credit_portfolio,
    calculate_portfolio_hhi,
)
from .correlation import (
    CopulaType,
    FactorModelType,
    CalibrationMethod,
    CorrelationResult,
    TransitionMatrix,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    OneFactorModel,
    MultiFactorModel,
    TransitionMatrixModel,
    CorrelationCalibration,
    calculate_default_correlation,
    calculate_basel_correlation,
    build_correlation_matrix,
)

__all__ = [
    # Structural Models
    "MertonModel",
    "calculate_default_probability",
    "calculate_distance_to_default",
    "calculate_credit_spread",
    
    # Default Probability (PD)
    "PDApproach",
    "PDType",
    "PDEstimate",
    "KMVMertonPD",
    "AltmanZScore",
    "LogisticPDModel",
    "RatingAgencyPDCurve",
    "PDEstimator",
    "calculate_kmv_pd",
    "calculate_altman_pd",
    "get_rating_pd",
    
    # Loss Given Default (LGD)
    "LGDApproach",
    "SeniorityClass",
    "LGDEstimate",
    "BetaLGD",
    "RecoveryRateBySeniority",
    "CollateralLGD",
    "WorkoutLGD",
    "DownturnLGD",
    "LGDModel",
    "calculate_lgd_by_seniority",
    "calculate_beta_lgd",
    "calculate_downturn_lgd",
    
    # Exposure at Default (EAD)
    "EADApproach",
    "FacilityType",
    "EADEstimate",
    "DerivativeExposure",
    "CreditConversionFactor",
    "SimpleEAD",
    "PotentialFutureExposure",
    "SACCR",
    "EADCalculator",
    "calculate_simple_ead",
    "calculate_derivative_pfe",
    "calculate_sa_ccr_ead",
    
    # Credit Value at Risk (CVaR)
    "CVaRApproach",
    "VarianceReduction",
    "Obligor",
    "CreditVaRResult",
    "ComponentCVaR",
    "AnalyticalCVaR",
    "CreditMetricsCVaR",
    "MonteCarloCVaR",
    "CreditVaRCalculator",
    "calculate_credit_var",
    "calculate_expected_shortfall",
    
    # Portfolio Credit Risk
    "CapitalApproach",
    "ConcentrationMetric",
    "PortfolioMetrics",
    "ConcentrationAnalysis",
    "RiskContribution",
    "PortfolioAggregator",
    "ConcentrationRisk",
    "CapitalAllocation",
    "RiskAdjustedMetrics",
    "PortfolioRiskAnalyzer",
    "analyze_credit_portfolio",
    "calculate_portfolio_hhi",
    
    # Default Correlation
    "CopulaType",
    "FactorModelType",
    "CalibrationMethod",
    "CorrelationResult",
    "TransitionMatrix",
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "OneFactorModel",
    "MultiFactorModel",
    "TransitionMatrixModel",
    "CorrelationCalibration",
    "calculate_default_correlation",
    "calculate_basel_correlation",
    "build_correlation_matrix",
]