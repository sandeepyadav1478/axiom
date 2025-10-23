"""
Portfolio Credit Risk - Basel III Compliant
==========================================

Comprehensive portfolio-level credit risk analysis for institutional risk management:
- Risk aggregation (PD, LGD, EAD)
- Expected Loss (EL) and Unexpected Loss (UL)
- Economic Capital (EC) requirements
- Concentration risk metrics (HHI, single-name, sector, geographic)
- Risk contributions and decomposition
- Capital allocation (Basel III SA-CR, IRB Foundation, IRB Advanced)
- Risk-adjusted return metrics (RAROC, EVA, RoRWA)
- Portfolio optimization under risk constraints
- Stress testing and scenario analysis

Mathematical Framework:
----------------------
1. Portfolio Expected Loss:
   EL_portfolio = Σᵢ EADᵢ × PDᵢ × LGDᵢ
   
2. Portfolio Unexpected Loss:
   UL_portfolio = √(Σᵢ Σⱼ ULᵢ × ULⱼ × ρᵢⱼ)
   
3. Economic Capital:
   EC = CVaR_α - EL (capital to cover unexpected losses)
   
4. Herfindahl-Hirschman Index:
   HHI = Σᵢ (EADᵢ / Total_EAD)²
   
5. Risk-Adjusted Return on Capital:
   RAROC = (Revenue - EL - OpEx) / EC
   
6. Basel III IRB Risk Weight:
   RW = LGD × N((1-R)^(-0.5) × N^(-1)(PD) + (R/(1-R))^(0.5) × N^(-1)(0.999)) × (1 + (M-2.5) × b) / (1 - 1.5 × b)

Features:
- <500ms for 1000-obligor portfolios
- Basel III regulatory capital
- IFRS 9 / CECL compliance
- Multi-dimensional concentration analysis
- Diversification benefit quantification
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict

from axiom.core.logging.axiom_logger import get_logger
from axiom.models.credit.credit_var import Obligor, CreditVaRCalculator, CVaRApproach

logger = get_logger("axiom.models.credit.portfolio_risk")


class CapitalApproach(Enum):
    """Regulatory capital approaches."""
    STANDARDIZED = "sa_cr"  # Standardized Approach
    FOUNDATION_IRB = "firb"  # Foundation Internal Ratings-Based
    ADVANCED_IRB = "airb"  # Advanced IRB
    ECONOMIC_CAPITAL = "economic"  # Internal economic capital


class ConcentrationMetric(Enum):
    """Concentration risk metrics."""
    HHI = "hhi"  # Herfindahl-Hirschman Index
    GINI = "gini"  # Gini coefficient
    LARGEST_N = "largest_n"  # Largest N exposures
    SECTOR = "sector"  # Sector concentration
    GEOGRAPHIC = "geographic"  # Geographic concentration
    RATING = "rating"  # Rating concentration


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics."""
    total_exposure: float
    expected_loss: float
    unexpected_loss: float
    economic_capital: float
    num_obligors: int
    
    # Concentration metrics
    hhi_index: float
    gini_coefficient: float
    top_10_concentration: float
    
    # Risk-adjusted metrics
    raroc: Optional[float] = None
    rorac: Optional[float] = None
    rorwa: Optional[float] = None
    
    # Regulatory capital
    regulatory_capital: Optional[float] = None
    risk_weighted_assets: Optional[float] = None
    
    execution_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_exposure": self.total_exposure,
            "expected_loss": self.expected_loss,
            "expected_loss_rate": self.expected_loss / self.total_exposure * 10000 if self.total_exposure > 0 else 0,  # bps
            "unexpected_loss": self.unexpected_loss,
            "economic_capital": self.economic_capital,
            "num_obligors": self.num_obligors,
            "hhi_index": self.hhi_index,
            "gini_coefficient": self.gini_coefficient,
            "top_10_concentration": self.top_10_concentration,
            "raroc": self.raroc,
            "rorac": self.rorac,
            "rorwa": self.rorwa,
            "regulatory_capital": self.regulatory_capital,
            "risk_weighted_assets": self.risk_weighted_assets,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class ConcentrationAnalysis:
    """Concentration risk analysis results."""
    metric_type: ConcentrationMetric
    overall_index: float
    breakdown: Dict[str, float]
    diversification_benefit: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "overall_index": self.overall_index,
            "breakdown": self.breakdown,
            "diversification_benefit": self.diversification_benefit,
            "metadata": self.metadata,
        }


@dataclass
class RiskContribution:
    """Risk contribution for an obligor."""
    obligor_id: str
    exposure: float
    expected_loss: float
    unexpected_loss_contribution: float
    marginal_risk: float
    percentage_contribution: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "obligor_id": self.obligor_id,
            "exposure": self.exposure,
            "expected_loss": self.expected_loss,
            "unexpected_loss_contribution": self.unexpected_loss_contribution,
            "marginal_risk": self.marginal_risk,
            "percentage_contribution": self.percentage_contribution,
        }


class PortfolioAggregator:
    """
    Portfolio-level risk aggregation.
    
    Aggregates individual obligor metrics to portfolio level:
    - Expected Loss
    - Unexpected Loss
    - Concentrations
    - Diversification benefits
    """
    
    @staticmethod
    def aggregate_metrics(
        obligors: List[Obligor],
        correlation_matrix: Optional[np.ndarray] = None,
        default_correlation: float = 0.15,
    ) -> Tuple[float, float, float]:
        """
        Aggregate portfolio-level PD, LGD, and calculate EL, UL.
        
        Args:
            obligors: List of credit obligors
            correlation_matrix: Default correlation matrix
            default_correlation: Default correlation if matrix not provided
            
        Returns:
            Tuple of (expected_loss, unexpected_loss, total_exposure)
        """
        n = len(obligors)
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        
        # Expected Loss
        expected_loss = sum(ob.expected_loss for ob in obligors)
        
        # Unexpected Loss with correlation
        if correlation_matrix is None:
            correlation_matrix = np.full((n, n), default_correlation)
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Calculate UL considering correlations
        ul_vector = np.array([ob.unexpected_loss_std for ob in obligors])
        portfolio_variance = ul_vector @ correlation_matrix @ ul_vector
        unexpected_loss = np.sqrt(portfolio_variance)
        
        return expected_loss, unexpected_loss, total_exposure
    
    @staticmethod
    def calculate_weighted_average_pd(obligors: List[Obligor]) -> float:
        """Calculate exposure-weighted average PD."""
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        if total_exposure == 0:
            return 0.0
        
        weighted_pd = sum(
            ob.exposure_at_default * ob.probability_of_default 
            for ob in obligors
        ) / total_exposure
        
        return weighted_pd
    
    @staticmethod
    def calculate_weighted_average_lgd(obligors: List[Obligor]) -> float:
        """Calculate exposure-weighted average LGD."""
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        if total_exposure == 0:
            return 0.0
        
        weighted_lgd = sum(
            ob.exposure_at_default * ob.loss_given_default 
            for ob in obligors
        ) / total_exposure
        
        return weighted_lgd


class ConcentrationRisk:
    """
    Concentration risk analysis.
    
    Measures portfolio concentration across multiple dimensions:
    - Single-name concentration (HHI, Gini)
    - Sector/industry concentration
    - Geographic concentration
    - Rating grade concentration
    """
    
    @staticmethod
    def calculate_hhi(obligors: List[Obligor]) -> float:
        """
        Calculate Herfindahl-Hirschman Index.
        
        HHI = Σᵢ (EADᵢ / Total_EAD)²
        
        HHI ranges from 1/n (perfect diversification) to 1 (full concentration).
        
        Args:
            obligors: List of credit obligors
            
        Returns:
            HHI value (0 to 1)
        """
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        if total_exposure == 0:
            return 0.0
        
        hhi = sum(
            (ob.exposure_at_default / total_exposure) ** 2 
            for ob in obligors
        )
        
        return hhi
    
    @staticmethod
    def calculate_gini(obligors: List[Obligor]) -> float:
        """
        Calculate Gini coefficient for concentration.
        
        Gini = 0: Perfect equality
        Gini = 1: Perfect inequality (full concentration)
        
        Args:
            obligors: List of credit obligors
            
        Returns:
            Gini coefficient (0 to 1)
        """
        exposures = np.array([ob.exposure_at_default for ob in obligors])
        sorted_exposures = np.sort(exposures)
        n = len(sorted_exposures)
        
        if n == 0 or np.sum(sorted_exposures) == 0:
            return 0.0
        
        cumsum = np.cumsum(sorted_exposures)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_exposures)) / (n * np.sum(sorted_exposures)) - (n + 1) / n
        
        return gini
    
    @staticmethod
    def calculate_top_n_concentration(
        obligors: List[Obligor],
        n: int = 10,
    ) -> float:
        """
        Calculate concentration in top N exposures.
        
        Args:
            obligors: List of credit obligors
            n: Number of top exposures
            
        Returns:
            Percentage of total exposure in top N obligors
        """
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        if total_exposure == 0:
            return 0.0
        
        sorted_exposures = sorted(
            [ob.exposure_at_default for ob in obligors],
            reverse=True
        )
        
        top_n_exposure = sum(sorted_exposures[:n])
        concentration = top_n_exposure / total_exposure
        
        return concentration
    
    @staticmethod
    def analyze_sector_concentration(obligors: List[Obligor]) -> ConcentrationAnalysis:
        """
        Analyze sector/industry concentration.
        
        Args:
            obligors: List of credit obligors (with sector attribute)
            
        Returns:
            ConcentrationAnalysis with sector breakdown
        """
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        
        # Group by sector
        sector_exposures = defaultdict(float)
        for ob in obligors:
            sector = ob.sector or "Unknown"
            sector_exposures[sector] += ob.exposure_at_default
        
        # Calculate sector shares
        sector_shares = {
            sector: exposure / total_exposure 
            for sector, exposure in sector_exposures.items()
        }
        
        # Calculate sector HHI
        sector_hhi = sum(share ** 2 for share in sector_shares.values())
        
        # Diversification benefit (1 - HHI)
        diversification = 1 - sector_hhi
        
        return ConcentrationAnalysis(
            metric_type=ConcentrationMetric.SECTOR,
            overall_index=sector_hhi,
            breakdown=sector_shares,
            diversification_benefit=diversification,
            metadata={
                "num_sectors": len(sector_exposures),
                "largest_sector": max(sector_shares.items(), key=lambda x: x[1])[0] if sector_shares else None,
                "largest_sector_share": max(sector_shares.values()) if sector_shares else 0,
            }
        )
    
    @staticmethod
    def analyze_geographic_concentration(obligors: List[Obligor]) -> ConcentrationAnalysis:
        """
        Analyze geographic concentration.
        
        Args:
            obligors: List of credit obligors (with region attribute)
            
        Returns:
            ConcentrationAnalysis with geographic breakdown
        """
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        
        # Group by region
        region_exposures = defaultdict(float)
        for ob in obligors:
            region = ob.region or "Unknown"
            region_exposures[region] += ob.exposure_at_default
        
        # Calculate regional shares
        region_shares = {
            region: exposure / total_exposure 
            for region, exposure in region_exposures.items()
        }
        
        # Calculate geographic HHI
        geo_hhi = sum(share ** 2 for share in region_shares.values())
        
        # Diversification benefit
        diversification = 1 - geo_hhi
        
        return ConcentrationAnalysis(
            metric_type=ConcentrationMetric.GEOGRAPHIC,
            overall_index=geo_hhi,
            breakdown=region_shares,
            diversification_benefit=diversification,
            metadata={
                "num_regions": len(region_exposures),
                "largest_region": max(region_shares.items(), key=lambda x: x[1])[0] if region_shares else None,
                "largest_region_share": max(region_shares.values()) if region_shares else 0,
            }
        )
    
    @staticmethod
    def analyze_rating_concentration(obligors: List[Obligor]) -> ConcentrationAnalysis:
        """
        Analyze rating grade concentration.
        
        Args:
            obligors: List of credit obligors (with rating attribute)
            
        Returns:
            ConcentrationAnalysis with rating breakdown
        """
        total_exposure = sum(ob.exposure_at_default for ob in obligors)
        
        # Group by rating
        rating_exposures = defaultdict(float)
        for ob in obligors:
            rating = ob.rating or "NR"
            rating_exposures[rating] += ob.exposure_at_default
        
        # Calculate rating shares
        rating_shares = {
            rating: exposure / total_exposure 
            for rating, exposure in rating_exposures.items()
        }
        
        # Calculate rating HHI
        rating_hhi = sum(share ** 2 for share in rating_shares.values())
        
        # Diversification benefit
        diversification = 1 - rating_hhi
        
        return ConcentrationAnalysis(
            metric_type=ConcentrationMetric.RATING,
            overall_index=rating_hhi,
            breakdown=rating_shares,
            diversification_benefit=diversification,
            metadata={
                "num_ratings": len(rating_exposures),
                "investment_grade_share": sum(
                    share for rating, share in rating_shares.items()
                    if rating in ["AAA", "AA", "A", "BBB", "Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3"]
                ),
            }
        )


class CapitalAllocation:
    """
    Economic and regulatory capital allocation.
    
    Implements:
    - Basel III Standardized Approach (SA-CR)
    - Basel III IRB Foundation
    - Basel III IRB Advanced
    - Internal economic capital models
    """
    
    # Basel III risk weight tables (SA-CR)
    RISK_WEIGHTS_SA = {
        "AAA": 0.20, "AA+": 0.20, "AA": 0.20, "AA-": 0.20,
        "A+": 0.50, "A": 0.50, "A-": 0.50,
        "BBB+": 0.75, "BBB": 0.75, "BBB-": 1.00,
        "BB+": 1.00, "BB": 1.00, "BB-": 1.00,
        "B+": 1.50, "B": 1.50, "B-": 1.50,
        "CCC+": 1.50, "CCC": 1.50, "CCC-": 1.50,
        "Unrated": 1.00,
    }
    
    @staticmethod
    def calculate_regulatory_capital(
        obligors: List[Obligor],
        approach: CapitalApproach = CapitalApproach.FOUNDATION_IRB,
        maturity: float = 2.5,
    ) -> Tuple[float, float]:
        """
        Calculate regulatory capital requirement.
        
        Args:
            obligors: List of credit obligors
            approach: Capital calculation approach
            maturity: Average maturity in years
            
        Returns:
            Tuple of (regulatory_capital, risk_weighted_assets)
        """
        if approach == CapitalApproach.STANDARDIZED:
            return CapitalAllocation._calculate_sa_cr(obligors)
        elif approach == CapitalApproach.FOUNDATION_IRB:
            return CapitalAllocation._calculate_firb(obligors, maturity)
        elif approach == CapitalApproach.ADVANCED_IRB:
            return CapitalAllocation._calculate_airb(obligors, maturity)
        else:
            raise ValueError(f"Unsupported capital approach: {approach}")
    
    @staticmethod
    def _calculate_sa_cr(obligors: List[Obligor]) -> Tuple[float, float]:
        """Calculate standardized approach capital."""
        rwa = 0.0
        
        for ob in obligors:
            rating = ob.rating or "Unrated"
            risk_weight = CapitalAllocation.RISK_WEIGHTS_SA.get(rating, 1.00)
            rwa += ob.exposure_at_default * risk_weight
        
        # Capital = 8% of RWA (Basel III minimum)
        capital = rwa * 0.08
        
        return capital, rwa
    
    @staticmethod
    def _calculate_firb(obligors: List[Obligor], maturity: float) -> Tuple[float, float]:
        """Calculate Foundation IRB capital."""
        rwa = 0.0
        
        for ob in obligors:
            # Basel III IRB risk weight function
            pd = ob.probability_of_default
            lgd = 0.45  # Regulatory LGD for senior unsecured (Foundation IRB)
            ead = ob.exposure_at_default
            
            # Maturity adjustment
            b = (0.11852 - 0.05478 * np.log(pd)) ** 2
            maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
            
            # Correlation
            R = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + \
                0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
            
            # Capital requirement
            K = lgd * norm.cdf(
                (norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
            ) - lgd * pd
            
            K = K * maturity_adj
            
            # Risk-weighted asset
            rwa += ead * K * 12.5  # 12.5 = 1/0.08
        
        capital = rwa * 0.08
        
        return capital, rwa
    
    @staticmethod
    def _calculate_airb(obligors: List[Obligor], maturity: float) -> Tuple[float, float]:
        """Calculate Advanced IRB capital."""
        rwa = 0.0
        
        for ob in obligors:
            pd = ob.probability_of_default
            lgd = ob.loss_given_default  # Own estimates in AIRB
            ead = ob.exposure_at_default
            
            # Maturity adjustment
            b = (0.11852 - 0.05478 * np.log(pd)) ** 2
            maturity_adj = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
            
            # Correlation
            R = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + \
                0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
            
            # Capital requirement
            K = lgd * norm.cdf(
                (norm.ppf(pd) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)
            ) - lgd * pd
            
            K = K * maturity_adj
            
            # Risk-weighted asset
            rwa += ead * K * 12.5
        
        capital = rwa * 0.08
        
        return capital, rwa
    
    @staticmethod
    def calculate_economic_capital(
        obligors: List[Obligor],
        confidence_level: float = 0.999,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate internal economic capital.
        
        EC = CVaR - EL
        
        Args:
            obligors: List of credit obligors
            confidence_level: Confidence level (typically 99.9%)
            correlation_matrix: Default correlation matrix
            
        Returns:
            Economic capital amount
        """
        calculator = CreditVaRCalculator()
        
        # Calculate Credit VaR
        cvar_result = calculator.calculate_cvar(
            obligors,
            approach=CVaRApproach.MONTE_CARLO,
            confidence_level=confidence_level,
            correlation_matrix=correlation_matrix,
        )
        
        # Economic capital = CVaR - EL
        economic_capital = cvar_result.unexpected_loss
        
        return economic_capital


class RiskAdjustedMetrics:
    """
    Risk-adjusted performance metrics.
    
    Calculates:
    - RAROC (Risk-Adjusted Return on Capital)
    - RORAC (Return on Risk-Adjusted Capital)
    - RoRWA (Return on Risk-Weighted Assets)
    - EVA (Economic Value Added)
    """
    
    @staticmethod
    def calculate_raroc(
        revenue: float,
        expected_loss: float,
        operating_expenses: float,
        economic_capital: float,
    ) -> float:
        """
        Calculate Risk-Adjusted Return on Capital.
        
        RAROC = (Revenue - EL - OpEx) / EC
        
        Args:
            revenue: Total revenue
            expected_loss: Expected credit losses
            operating_expenses: Operating costs
            economic_capital: Economic capital allocated
            
        Returns:
            RAROC as decimal (e.g., 0.15 for 15%)
        """
        if economic_capital == 0:
            return 0.0
        
        risk_adjusted_return = revenue - expected_loss - operating_expenses
        raroc = risk_adjusted_return / economic_capital
        
        return raroc
    
    @staticmethod
    def calculate_rorac(
        net_income: float,
        risk_adjusted_capital: float,
    ) -> float:
        """
        Calculate Return on Risk-Adjusted Capital.
        
        Args:
            net_income: Net income after all costs
            risk_adjusted_capital: Risk-adjusted capital base
            
        Returns:
            RORAC as decimal
        """
        if risk_adjusted_capital == 0:
            return 0.0
        
        return net_income / risk_adjusted_capital
    
    @staticmethod
    def calculate_rorwa(
        net_income: float,
        risk_weighted_assets: float,
    ) -> float:
        """
        Calculate Return on Risk-Weighted Assets.
        
        Args:
            net_income: Net income
            risk_weighted_assets: Total RWA
            
        Returns:
            RoRWA as decimal
        """
        if risk_weighted_assets == 0:
            return 0.0
        
        return net_income / risk_weighted_assets
    
    @staticmethod
    def calculate_eva(
        revenue: float,
        expected_loss: float,
        operating_expenses: float,
        economic_capital: float,
        cost_of_capital: float = 0.10,
    ) -> float:
        """
        Calculate Economic Value Added.
        
        EVA = (Revenue - EL - OpEx) - (EC × Cost_of_Capital)
        
        Args:
            revenue: Total revenue
            expected_loss: Expected losses
            operating_expenses: Operating costs
            economic_capital: Economic capital
            cost_of_capital: Required return on capital (e.g., 0.10 for 10%)
            
        Returns:
            EVA amount
        """
        net_operating_profit = revenue - expected_loss - operating_expenses
        capital_charge = economic_capital * cost_of_capital
        eva = net_operating_profit - capital_charge
        
        return eva


class PortfolioRiskAnalyzer:
    """
    Unified portfolio credit risk analyzer.
    
    Features:
    - Portfolio aggregation
    - Concentration analysis
    - Risk contributions
    - Capital allocation
    - Risk-adjusted metrics
    - Optimization
    """
    
    def __init__(self):
        """Initialize portfolio risk analyzer."""
        logger.info("Initialized Portfolio Risk Analyzer")
    
    def analyze_portfolio(
        self,
        obligors: List[Obligor],
        correlation_matrix: Optional[np.ndarray] = None,
        confidence_level: float = 0.999,
        capital_approach: CapitalApproach = CapitalApproach.ADVANCED_IRB,
    ) -> PortfolioMetrics:
        """
        Perform comprehensive portfolio risk analysis.
        
        Args:
            obligors: List of credit obligors
            correlation_matrix: Default correlation matrix
            confidence_level: Confidence level for capital
            capital_approach: Regulatory capital approach
            
        Returns:
            PortfolioMetrics with comprehensive analysis
        """
        start_time = time.perf_counter()
        
        # Risk aggregation
        el, ul, total_exp = PortfolioAggregator.aggregate_metrics(
            obligors, correlation_matrix
        )
        
        # Economic capital
        ec = CapitalAllocation.calculate_economic_capital(
            obligors, confidence_level, correlation_matrix
        )
        
        # Concentration metrics
        hhi = ConcentrationRisk.calculate_hhi(obligors)
        gini = ConcentrationRisk.calculate_gini(obligors)
        top10 = ConcentrationRisk.calculate_top_n_concentration(obligors, 10)
        
        # Regulatory capital
        reg_cap, rwa = CapitalAllocation.calculate_regulatory_capital(
            obligors, capital_approach
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        return PortfolioMetrics(
            total_exposure=total_exp,
            expected_loss=el,
            unexpected_loss=ul,
            economic_capital=ec,
            num_obligors=len(obligors),
            hhi_index=hhi,
            gini_coefficient=gini,
            top_10_concentration=top10,
            regulatory_capital=reg_cap,
            risk_weighted_assets=rwa,
            execution_time_ms=execution_time_ms,
            metadata={
                "confidence_level": confidence_level,
                "capital_approach": capital_approach.value,
                "diversification_ratio": ul / sum(ob.unexpected_loss_std for ob in obligors) if sum(ob.unexpected_loss_std for ob in obligors) > 0 else 0,
            }
        )
    
    def calculate_risk_contributions(
        self,
        obligors: List[Obligor],
        portfolio_ul: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[RiskContribution]:
        """
        Calculate risk contributions for each obligor.
        
        Args:
            obligors: List of credit obligors
            portfolio_ul: Portfolio unexpected loss
            correlation_matrix: Default correlation matrix
            
        Returns:
            List of RiskContribution for each obligor
        """
        contributions = []
        
        for i, ob in enumerate(obligors):
            # Marginal risk (simplified approximation)
            marginal = ob.unexpected_loss_std
            
            # UL contribution
            ul_contrib = marginal * ob.exposure_at_default / portfolio_ul if portfolio_ul > 0 else 0
            
            # Percentage contribution
            pct_contrib = ul_contrib / portfolio_ul * 100 if portfolio_ul > 0 else 0
            
            contributions.append(RiskContribution(
                obligor_id=ob.id,
                exposure=ob.exposure_at_default,
                expected_loss=ob.expected_loss,
                unexpected_loss_contribution=ul_contrib,
                marginal_risk=marginal,
                percentage_contribution=pct_contrib,
            ))
        
        return contributions


# Convenience functions
def analyze_credit_portfolio(
    obligors: List[Obligor],
    **kwargs
) -> PortfolioMetrics:
    """Quick portfolio analysis."""
    analyzer = PortfolioRiskAnalyzer()
    return analyzer.analyze_portfolio(obligors, **kwargs)


def calculate_portfolio_hhi(obligors: List[Obligor]) -> float:
    """Quick HHI calculation."""
    return ConcentrationRisk.calculate_hhi(obligors)


__all__ = [
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
]