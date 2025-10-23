"""
Base Classes for M&A Quantitative Models
=========================================

Defines abstract base classes and data structures for all M&A models,
providing a consistent interface and shared functionality.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import time

from axiom.models.base.base_model import BaseFinancialModel, ModelResult
from axiom.models.base.mixins import (
    NumericalMethodsMixin,
    ValidationMixin,
    PerformanceMixin
)


@dataclass
class SynergyEstimate:
    """
    Comprehensive synergy valuation result.
    
    Attributes:
        cost_synergies_npv: NPV of cost synergies
        revenue_synergies_npv: NPV of revenue synergies
        total_synergies_npv: Total NPV of all synergies
        integration_costs: One-time integration costs
        net_synergies: Net synergies after integration costs
        realization_schedule: Year-by-year synergy realization
        confidence_level: Statistical confidence in estimates
        key_assumptions: Critical assumptions used
        sensitivity_analysis: Sensitivity to key drivers
    """
    cost_synergies_npv: float
    revenue_synergies_npv: float
    total_synergies_npv: float
    integration_costs: float
    net_synergies: float
    realization_schedule: List[float]
    confidence_level: float
    key_assumptions: Dict[str, Any]
    sensitivity_analysis: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cost_synergies_npv": self.cost_synergies_npv,
            "revenue_synergies_npv": self.revenue_synergies_npv,
            "total_synergies_npv": self.total_synergies_npv,
            "integration_costs": self.integration_costs,
            "net_synergies": self.net_synergies,
            "realization_schedule": self.realization_schedule,
            "confidence_level": self.confidence_level,
            "key_assumptions": self.key_assumptions,
            "sensitivity_analysis": self.sensitivity_analysis or {}
        }


@dataclass
class DealFinancing:
    """
    Deal financing structure and analysis.
    
    Attributes:
        purchase_price: Total purchase price
        cash_component: Cash portion of consideration
        stock_component: Stock portion of consideration
        debt_financing: New debt raised
        equity_contribution: Equity capital required
        wacc: Weighted average cost of capital
        cost_of_debt: After-tax cost of debt
        cost_of_equity: Cost of equity (CAPM)
        eps_impact: EPS accretion/(dilution)
        accretive: Whether deal is EPS accretive
        payback_years: Years to recover dilution
        credit_ratios: Key credit metrics
        rating_impact: Pro forma credit rating
    """
    purchase_price: float
    cash_component: float
    stock_component: float
    debt_financing: float
    equity_contribution: float
    wacc: float
    cost_of_debt: float
    cost_of_equity: float
    eps_impact: float
    accretive: bool
    payback_years: Optional[float]
    credit_ratios: Dict[str, float]
    rating_impact: Optional[str] = None
    financing_sources: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "purchase_price": self.purchase_price,
            "cash_component": self.cash_component,
            "stock_component": self.stock_component,
            "debt_financing": self.debt_financing,
            "equity_contribution": self.equity_contribution,
            "wacc": self.wacc,
            "cost_of_debt": self.cost_of_debt,
            "cost_of_equity": self.cost_of_equity,
            "eps_impact": self.eps_impact,
            "accretive": self.accretive,
            "payback_years": self.payback_years,
            "credit_ratios": self.credit_ratios,
            "rating_impact": self.rating_impact,
            "financing_sources": self.financing_sources or {}
        }


@dataclass
class MergerArbPosition:
    """
    Merger arbitrage position analysis.
    
    Attributes:
        target_position: Long position value in target
        acquirer_hedge: Short hedge value in acquirer
        deal_spread: Current deal spread (%)
        annualized_return: Annualized spread return
        implied_probability: Market-implied probability of close
        expected_return: Probability-weighted expected return
        position_size: Recommended position size
        kelly_optimal_size: Kelly criterion optimal size
        risk_metrics: Key risk measures
        hedge_ratio: Optimal hedge ratio
        break_even_prob: Minimum probability for positive return
    """
    target_position: float
    acquirer_hedge: float
    deal_spread: float
    annualized_return: float
    implied_probability: float
    expected_return: float
    position_size: float
    kelly_optimal_size: float
    risk_metrics: Dict[str, float]
    hedge_ratio: float
    break_even_prob: float
    days_to_close: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_position": self.target_position,
            "acquirer_hedge": self.acquirer_hedge,
            "deal_spread": self.deal_spread,
            "annualized_return": self.annualized_return,
            "implied_probability": self.implied_probability,
            "expected_return": self.expected_return,
            "position_size": self.position_size,
            "kelly_optimal_size": self.kelly_optimal_size,
            "risk_metrics": self.risk_metrics,
            "hedge_ratio": self.hedge_ratio,
            "break_even_prob": self.break_even_prob,
            "days_to_close": self.days_to_close
        }


@dataclass
class LBOAnalysis:
    """
    Leveraged buyout analysis result.
    
    Attributes:
        entry_price: Purchase price (entry valuation)
        exit_price: Exit valuation
        equity_contribution: Sponsor equity investment
        debt_financing: Total debt raised
        irr: Internal rate of return
        cash_on_cash: Cash-on-cash multiple (MoM)
        holding_period: Years held
        exit_multiple: Exit EBITDA multiple
        debt_paydown: Debt reduction during hold
        operational_improvements: Value from ops improvements
        multiple_expansion: Value from multiple expansion
        dividend_recap: Interim dividend recaps
    """
    entry_price: float
    exit_price: float
    equity_contribution: float
    debt_financing: float
    irr: float
    cash_on_cash: float
    holding_period: int
    exit_multiple: float
    debt_paydown: float
    operational_improvements: Dict[str, float]
    multiple_expansion: float
    dividend_recap: float = 0.0
    sensitivity_matrix: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "equity_contribution": self.equity_contribution,
            "debt_financing": self.debt_financing,
            "irr": self.irr,
            "cash_on_cash": self.cash_on_cash,
            "holding_period": self.holding_period,
            "exit_multiple": self.exit_multiple,
            "debt_paydown": self.debt_paydown,
            "operational_improvements": self.operational_improvements,
            "multiple_expansion": self.multiple_expansion,
            "dividend_recap": self.dividend_recap
        }
        if self.sensitivity_matrix:
            result["sensitivity_matrix"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.sensitivity_matrix.items()
            }
        return result


@dataclass
class DealScreeningResult:
    """
    Quantitative deal screening result.
    
    Attributes:
        deal_id: Unique deal identifier
        strategic_fit_score: Strategic fit rating (0-100)
        financial_attractiveness: Financial score (0-100)
        risk_score: Risk assessment score (0-100, lower is better)
        synergy_potential: Synergy value estimate
        integration_difficulty: Integration complexity score
        overall_score: Composite overall score
        recommendation: Buy/Hold/Pass recommendation
        key_metrics: Summary metrics
        comparable_deals: Similar precedent transactions
    """
    deal_id: str
    strategic_fit_score: float
    financial_attractiveness: float
    risk_score: float
    synergy_potential: float
    integration_difficulty: float
    overall_score: float
    recommendation: str
    key_metrics: Dict[str, float]
    comparable_deals: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deal_id": self.deal_id,
            "strategic_fit_score": self.strategic_fit_score,
            "financial_attractiveness": self.financial_attractiveness,
            "risk_score": self.risk_score,
            "synergy_potential": self.synergy_potential,
            "integration_difficulty": self.integration_difficulty,
            "overall_score": self.overall_score,
            "recommendation": self.recommendation,
            "key_metrics": self.key_metrics,
            "comparable_deals": self.comparable_deals or []
        }


class BaseMandAModel(BaseFinancialModel, NumericalMethodsMixin, ValidationMixin, PerformanceMixin):
    """
    Abstract base class for all M&A quantitative models.
    
    Provides common M&A-specific functionality:
    - IRR calculation
    - NPV calculation  
    - Risk assessment
    - Deal value calculation
    - Synergy modeling
    - Deal comparison
    
    All M&A models inherit from this class.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize M&A model.
        
        Args:
            config: M&A-specific configuration
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(config, enable_logging, enable_performance_tracking)
    
    @abstractmethod
    def calculate_value(self, **kwargs) -> float:
        """
        Calculate deal value or return.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            Calculated value (price, NPV, IRR, etc.)
        """
        pass
    
    def calculate_irr(
        self,
        cash_flows: List[float],
        initial_investment: Optional[float] = None,
        guess: float = 0.10
    ) -> Tuple[float, bool]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.
        
        Args:
            cash_flows: List of cash flows (negative = outflow, positive = inflow)
            initial_investment: Initial investment (if not in cash_flows[0])
            guess: Initial IRR guess
            
        Returns:
            Tuple of (IRR, converged)
        """
        # Prepend initial investment if provided
        if initial_investment is not None:
            cash_flows = [-abs(initial_investment)] + list(cash_flows)
        
        # NPV function
        def npv(rate: float) -> float:
            return sum(cf / (1 + rate) ** i for i, cf in enumerate(cash_flows))
        
        # NPV derivative
        def npv_derivative(rate: float) -> float:
            return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cash_flows))
        
        # Use Newton-Raphson from mixin
        irr, converged, iterations = self.newton_raphson(
            npv,
            npv_derivative,
            guess,
            tolerance=1e-6,
            max_iterations=100
        )
        
        return irr, converged
    
    def calculate_npv(
        self,
        cash_flows: List[float],
        discount_rate: float,
        initial_investment: Optional[float] = None
    ) -> float:
        """
        Calculate Net Present Value.
        
        Args:
            cash_flows: List of future cash flows
            discount_rate: Discount rate (WACC, required return, etc.)
            initial_investment: Initial investment (if not in cash_flows[0])
            
        Returns:
            NPV
        """
        if initial_investment is not None:
            cash_flows = [-abs(initial_investment)] + list(cash_flows)
        
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        return npv
    
    def calculate_wacc(
        self,
        equity_value: float,
        debt_value: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital.
        
        Args:
            equity_value: Market value of equity
            debt_value: Market value of debt
            cost_of_equity: Cost of equity (CAPM)
            cost_of_debt: Pre-tax cost of debt
            tax_rate: Corporate tax rate
            
        Returns:
            WACC
        """
        total_value = equity_value + debt_value
        
        if total_value == 0:
            return cost_of_equity  # All equity
        
        equity_weight = equity_value / total_value
        debt_weight = debt_value / total_value
        
        wacc = (equity_weight * cost_of_equity + 
                debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc
    
    def calculate_capm(
        self,
        risk_free_rate: float,
        beta: float,
        market_return: float
    ) -> float:
        """
        Calculate cost of equity using CAPM.
        
        Args:
            risk_free_rate: Risk-free rate
            beta: Equity beta
            market_return: Expected market return
            
        Returns:
            Cost of equity
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    def calculate_levered_beta(
        self,
        unlevered_beta: float,
        debt_to_equity: float,
        tax_rate: float
    ) -> float:
        """
        Calculate levered beta from unlevered beta.
        
        Args:
            unlevered_beta: Asset beta
            debt_to_equity: D/E ratio
            tax_rate: Corporate tax rate
            
        Returns:
            Levered equity beta
        """
        return unlevered_beta * (1 + (1 - tax_rate) * debt_to_equity)
    
    def calculate_synergy_pv(
        self,
        annual_synergies: List[float],
        discount_rate: float,
        realization_years: Optional[List[int]] = None
    ) -> float:
        """
        Calculate present value of synergies.
        
        Args:
            annual_synergies: Annual synergy amounts
            discount_rate: Discount rate for synergies
            realization_years: Years in which synergies are realized
            
        Returns:
            Present value of synergies
        """
        if realization_years is None:
            realization_years = list(range(1, len(annual_synergies) + 1))
        
        pv = sum(
            synergy / (1 + discount_rate) ** year
            for synergy, year in zip(annual_synergies, realization_years)
        )
        
        return pv
    
    def assess_risk(
        self,
        probability_of_success: float,
        regulatory_risk: float = 0.0,
        financing_risk: float = 0.0,
        integration_risk: float = 0.0,
        **kwargs
    ) -> float:
        """
        Quantify deal risk.
        
        Args:
            probability_of_success: Base probability of deal closing
            regulatory_risk: Regulatory risk factor (0-1)
            financing_risk: Financing risk factor (0-1)
            integration_risk: Integration risk factor (0-1)
            **kwargs: Additional risk factors
            
        Returns:
            Adjusted probability of success
        """
        # Combine risks multiplicatively
        adjusted_prob = probability_of_success
        adjusted_prob *= (1 - regulatory_risk)
        adjusted_prob *= (1 - financing_risk)
        adjusted_prob *= (1 - integration_risk)
        
        # Apply any additional risk factors
        for risk_factor in kwargs.values():
            if isinstance(risk_factor, (int, float)) and 0 <= risk_factor <= 1:
                adjusted_prob *= (1 - risk_factor)
        
        return max(0.0, min(1.0, adjusted_prob))
    
    def calculate_breakeven_synergies(
        self,
        purchase_price: float,
        standalone_value: float,
        discount_rate: float,
        years: int = 5
    ) -> float:
        """
        Calculate annual synergies needed to breakeven.
        
        Args:
            purchase_price: Deal purchase price
            standalone_value: Standalone valuation
            discount_rate: Discount rate
            years: Number of years to realize synergies
            
        Returns:
            Required annual synergies
        """
        premium_paid = purchase_price - standalone_value
        
        if premium_paid <= 0:
            return 0.0
        
        # Calculate annuity factor
        annuity_factor = sum(1 / (1 + discount_rate) ** i for i in range(1, years + 1))
        
        # Required annual synergy
        required_synergy = premium_paid / annuity_factor
        
        return required_synergy
    
    def calculate_control_premium(
        self,
        offer_price: float,
        pre_announcement_price: float
    ) -> float:
        """
        Calculate control premium paid.
        
        Args:
            offer_price: Offer price per share
            pre_announcement_price: Pre-announcement trading price
            
        Returns:
            Control premium as percentage
        """
        if pre_announcement_price <= 0:
            raise ValueError("Pre-announcement price must be positive")
        
        premium = (offer_price - pre_announcement_price) / pre_announcement_price
        return premium
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate common M&A model inputs.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Validate positive values
        for param in ['purchase_price', 'equity_value', 'debt_value', 'ebitda']:
            if param in kwargs:
                self.validate_positive(kwargs[param], param)
        
        # Validate probabilities
        for param in ['probability', 'confidence_level']:
            if param in kwargs:
                self.validate_probability(kwargs[param], param)
        
        # Validate discount rates
        for param in ['discount_rate', 'wacc', 'cost_of_equity', 'cost_of_debt']:
            if param in kwargs:
                rate = kwargs[param]
                if not -1 < rate < 2:  # Reasonable range for rates
                    raise ValueError(f"{param} must be between -100% and 200%")
        
        return True


__all__ = [
    "BaseMandAModel",
    "SynergyEstimate",
    "DealFinancing",
    "MergerArbPosition",
    "LBOAnalysis",
    "DealScreeningResult",
]