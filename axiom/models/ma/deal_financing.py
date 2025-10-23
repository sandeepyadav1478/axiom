"""
Deal Financing Optimization Models
===================================

Comprehensive deal financing framework including:
- Capital structure optimization (debt/equity mix)
- WACC minimization
- EPS accretion/dilution analysis  
- Credit ratio analysis
- Cost of capital calculation (debt and equity)
- Rating agency impact assessment
- Tax shield optimization

Performance target: <30ms for financing optimization
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time
from scipy import optimize

from axiom.models.ma.base_model import BaseMandAModel, DealFinancing
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class FinancingSource:
    """Individual financing source."""
    name: str
    amount: float
    cost: float  # Interest rate or cost of capital
    type: str  # debt, equity, preferred, convertible
    maturity: Optional[int] = None  # Years to maturity (for debt)


class DealFinancingModel(BaseMandAModel):
    """
    Comprehensive deal financing optimization model.
    
    Features:
    - Optimal debt/equity mix calculation
    - WACC minimization
    - EPS accretion/dilution analysis
    - Credit ratio monitoring (Debt/EBITDA, Interest Coverage)
    - Rating agency impact estimation
    - Tax shield benefits
    - Multiple financing instrument support
    - Financing constraint optimization
    
    Performance: <30ms for full financing analysis
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize deal financing model.
        
        Args:
            config: M&A configuration
            enable_logging: Enable detailed logging
            enable_performance_tracking: Track execution time
        """
        super().__init__(
            config=config.__dict__ if config and hasattr(config, '__dict__') else (config or {}),
            enable_logging=enable_logging,
            enable_performance_tracking=enable_performance_tracking
        )
        
        # Set default config values
        self.config.setdefault('target_debt_ebitda', 5.0)
        self.config.setdefault('min_interest_coverage', 2.0)
        self.config.setdefault('default_tax_rate', 0.21)
        self.config.setdefault('default_risk_free_rate', 0.04)
        self.config.setdefault('default_market_return', 0.10)
        self.config.setdefault('optimal_cash_stock_mix', 'npv_maximizing')
    
    def calculate(
        self,
        purchase_price: float,
        target_ebitda: float,
        acquirer_market_cap: float,
        acquirer_shares_outstanding: float,
        acquirer_eps: float,
        target_eps: Optional[float] = None,
        acquirer_beta: float = 1.0,
        credit_spread: float = 0.02,
        tax_rate: Optional[float] = None,
        synergies: float = 0.0,
        cash_available: float = 0.0,
        max_leverage: Optional[float] = None,
        optimization_objective: str = "wacc"  # wacc, eps_accretion, rating_neutral
    ) -> ModelResult[DealFinancing]:
        """
        Calculate optimal deal financing structure.
        
        Args:
            purchase_price: Total purchase price
            target_ebitda: Target company EBITDA
            acquirer_market_cap: Acquirer's market capitalization
            acquirer_shares_outstanding: Acquirer's shares outstanding
            acquirer_eps: Acquirer's current EPS
            target_eps: Target's EPS (if available)
            acquirer_beta: Acquirer's equity beta
            credit_spread: Credit spread over risk-free rate
            tax_rate: Corporate tax rate
            synergies: Expected synergies (annual)
            cash_available: Cash available for deal
            max_leverage: Maximum Debt/EBITDA allowed
            optimization_objective: Optimization goal
            
        Returns:
            ModelResult containing DealFinancing
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(
            purchase_price=purchase_price,
            target_ebitda=target_ebitda,
            acquirer_market_cap=acquirer_market_cap
        )
        
        # Use config defaults
        tax_rate = tax_rate or self.config['default_tax_rate']
        max_leverage = max_leverage or self.config['target_debt_ebitda']
        risk_free_rate = self.config['default_risk_free_rate']
        market_return = self.config['default_market_return']
        
        # Calculate cost of equity (CAPM)
        cost_of_equity = self.calculate_capm(risk_free_rate, acquirer_beta, market_return)
        
        # Calculate cost of debt
        cost_of_debt = risk_free_rate + credit_spread
        
        # Optimize financing structure
        if optimization_objective == "wacc":
            optimal_debt, optimal_equity = self._optimize_for_wacc(
                purchase_price, target_ebitda, cost_of_debt, cost_of_equity,
                tax_rate, max_leverage, cash_available
            )
        elif optimization_objective == "eps_accretion":
            optimal_debt, optimal_equity = self._optimize_for_eps_accretion(
                purchase_price, target_ebitda, acquirer_eps, acquirer_shares_outstanding,
                cost_of_debt, cost_of_equity, tax_rate, synergies, max_leverage, cash_available
            )
        else:  # rating_neutral
            optimal_debt, optimal_equity = self._optimize_for_rating_neutral(
                purchase_price, target_ebitda, max_leverage, cash_available
            )
        
        # Calculate cash vs stock split
        cash_component = min(cash_available, optimal_debt + optimal_equity)
        stock_component = purchase_price - cash_component - optimal_debt
        
        # Adjust if stock component is negative
        if stock_component < 0:
            cash_component = purchase_price - optimal_debt
            stock_component = 0
        
        # Calculate WACC
        wacc = self.calculate_wacc(
            equity_value=optimal_equity + stock_component,
            debt_value=optimal_debt,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            tax_rate=tax_rate
        )
        
        # Calculate EPS impact
        eps_impact, accretive, payback_years = self._calculate_eps_impact(
            purchase_price, target_ebitda, optimal_debt, stock_component,
            acquirer_shares_outstanding, acquirer_market_cap, acquirer_eps,
            cost_of_debt, tax_rate, synergies
        )
        
        # Calculate credit ratios
        credit_ratios = self._calculate_credit_ratios(
            optimal_debt, target_ebitda, cost_of_debt, synergies
        )
        
        # Estimate rating impact
        rating_impact = self._estimate_rating_impact(credit_ratios)
        
        # Create financing sources breakdown
        financing_sources = {
            'senior_debt': optimal_debt * 0.7 if optimal_debt > 0 else 0,
            'subordinated_debt': optimal_debt * 0.3 if optimal_debt > 0 else 0,
            'cash': cash_component,
            'stock': stock_component,
            'total_consideration': purchase_price
        }
        
        # Create result
        financing = DealFinancing(
            purchase_price=purchase_price,
            cash_component=cash_component,
            stock_component=stock_component,
            debt_financing=optimal_debt,
            equity_contribution=optimal_equity,
            wacc=wacc,
            cost_of_debt=cost_of_debt * (1 - tax_rate),  # After-tax
            cost_of_equity=cost_of_equity,
            eps_impact=eps_impact,
            accretive=accretive,
            payback_years=payback_years,
            credit_ratios=credit_ratios,
            rating_impact=rating_impact,
            financing_sources=financing_sources
        )
        
        # Track performance
        execution_time_ms = self._track_performance("deal_financing", start_time)
        
        # Create metadata
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(
            value=financing,
            metadata=metadata,
            success=True
        )
    
    def _optimize_for_wacc(
        self,
        purchase_price: float,
        target_ebitda: float,
        cost_of_debt: float,
        cost_of_equity: float,
        tax_rate: float,
        max_leverage: float,
        cash_available: float
    ) -> Tuple[float, float]:
        """
        Optimize debt/equity mix to minimize WACC.
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            cost_of_debt: Pre-tax cost of debt
            cost_of_equity: Cost of equity
            tax_rate: Tax rate
            max_leverage: Max Debt/EBITDA
            cash_available: Cash available
            
        Returns:
            Tuple of (optimal_debt, optimal_equity)
        """
        # Maximum debt based on leverage constraint
        max_debt = min(
            target_ebitda * max_leverage,
            purchase_price - cash_available
        )
        
        # Objective function: minimize WACC
        def wacc_objective(debt):
            debt = debt[0]
            equity = purchase_price - cash_available - debt
            
            if equity < 0:
                return 1000  # Penalty for infeasible solution
            
            total_value = debt + equity
            if total_value == 0:
                return 1000
            
            wacc = (debt / total_value) * cost_of_debt * (1 - tax_rate) + \
                   (equity / total_value) * cost_of_equity
            return wacc
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},  # Debt >= 0
            {'type': 'ineq', 'fun': lambda x: max_debt - x[0]},  # Debt <= max_debt
            {'type': 'ineq', 'fun': lambda x: purchase_price - cash_available - x[0]}  # Equity >= 0
        ]
        
        # Optimize
        initial_guess = np.array([max_debt * 0.5])
        result = optimize.minimize(
            wacc_objective,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, max_debt)]
        )
        
        optimal_debt = result.x[0] if result.success else max_debt * 0.5
        optimal_equity = purchase_price - cash_available - optimal_debt
        
        return optimal_debt, max(0, optimal_equity)
    
    def _optimize_for_eps_accretion(
        self,
        purchase_price: float,
        target_ebitda: float,
        acquirer_eps: float,
        acquirer_shares: float,
        cost_of_debt: float,
        cost_of_equity: float,
        tax_rate: float,
        synergies: float,
        max_leverage: float,
        cash_available: float
    ) -> Tuple[float, float]:
        """
        Optimize financing mix to maximize EPS accretion.
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            acquirer_eps: Current acquirer EPS
            acquirer_shares: Acquirer shares outstanding
            cost_of_debt: Cost of debt
            cost_of_equity: Cost of equity
            tax_rate: Tax rate
            synergies: Expected synergies
            max_leverage: Max leverage ratio
            cash_available: Cash available
            
        Returns:
            Tuple of (optimal_debt, optimal_equity)
        """
        # Maximum debt
        max_debt = min(
            target_ebitda * max_leverage,
            purchase_price - cash_available
        )
        
        # Objective: maximize EPS (minimize negative EPS impact)
        def eps_objective(debt):
            debt = debt[0]
            stock_value = max(0, purchase_price - cash_available - debt)
            
            # New shares issued (simplified)
            new_shares = stock_value / (acquirer_eps * 15)  # Assume P/E of 15
            
            # Pro forma earnings
            target_earnings = target_ebitda * (1 - tax_rate)
            interest_expense = debt * cost_of_debt * (1 - tax_rate)
            pro_forma_earnings = (acquirer_eps * acquirer_shares) + target_earnings + \
                               synergies * (1 - tax_rate) - interest_expense
            
            # Pro forma EPS
            pro_forma_shares = acquirer_shares + new_shares
            pro_forma_eps = pro_forma_earnings / pro_forma_shares if pro_forma_shares > 0 else 0
            
            # Return negative of EPS increase (to minimize)
            return -(pro_forma_eps - acquirer_eps)
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: max_debt - x[0]},
            {'type': 'ineq', 'fun': lambda x: purchase_price - cash_available - x[0]}
        ]
        
        # Optimize
        initial_guess = np.array([max_debt * 0.6])  # Start with more debt for EPS
        result = optimize.minimize(
            eps_objective,
            initial_guess,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, max_debt)]
        )
        
        optimal_debt = result.x[0] if result.success else max_debt * 0.6
        optimal_equity = purchase_price - cash_available - optimal_debt
        
        return optimal_debt, max(0, optimal_equity)
    
    def _optimize_for_rating_neutral(
        self,
        purchase_price: float,
        target_ebitda: float,
        max_leverage: float,
        cash_available: float
    ) -> Tuple[float, float]:
        """
        Optimize to maintain credit rating (rating-neutral approach).
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            max_leverage: Target leverage ratio
            cash_available: Cash available
            
        Returns:
            Tuple of (optimal_debt, optimal_equity)
        """
        # Use conservative leverage ratio (e.g., 3.5x for investment grade)
        conservative_leverage = min(max_leverage, 3.5)
        optimal_debt = min(
            target_ebitda * conservative_leverage,
            purchase_price - cash_available
        )
        optimal_equity = purchase_price - cash_available - optimal_debt
        
        return max(0, optimal_debt), max(0, optimal_equity)
    
    def _calculate_eps_impact(
        self,
        purchase_price: float,
        target_ebitda: float,
        debt: float,
        stock_value: float,
        acquirer_shares: float,
        acquirer_market_cap: float,
        acquirer_eps: float,
        cost_of_debt: float,
        tax_rate: float,
        synergies: float
    ) -> Tuple[float, bool, Optional[float]]:
        """
        Calculate EPS accretion/dilution.
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            debt: Debt amount
            stock_value: Stock consideration
            acquirer_shares: Acquirer shares outstanding
            acquirer_market_cap: Acquirer market cap
            acquirer_eps: Current EPS
            cost_of_debt: Cost of debt
            tax_rate: Tax rate
            synergies: Expected synergies
            
        Returns:
            Tuple of (eps_impact, accretive, payback_years)
        """
        # New shares issued
        share_price = acquirer_market_cap / acquirer_shares
        new_shares = stock_value / share_price if share_price > 0 else 0
        
        # Pro forma earnings
        acquirer_earnings = acquirer_eps * acquirer_shares
        target_earnings = target_ebitda * (1 - tax_rate)
        synergy_earnings = synergies * (1 - tax_rate)
        interest_expense = debt * cost_of_debt * (1 - tax_rate)
        
        pro_forma_earnings = acquirer_earnings + target_earnings + synergy_earnings - interest_expense
        
        # Pro forma EPS
        pro_forma_shares = acquirer_shares + new_shares
        pro_forma_eps = pro_forma_earnings / pro_forma_shares if pro_forma_shares > 0 else 0
        
        # EPS impact
        eps_impact = pro_forma_eps - acquirer_eps
        accretive = eps_impact > 0
        
        # Payback period (years to recover dilution)
        if eps_impact < 0 and synergies > 0:
            # Assuming synergies ramp up over 3 years
            annual_synergy_contribution = (synergies * (1 - tax_rate)) / 3 / pro_forma_shares
            payback_years = abs(eps_impact) / annual_synergy_contribution if annual_synergy_contribution > 0 else None
        else:
            payback_years = 0.0 if accretive else None
        
        return eps_impact, accretive, payback_years
    
    def _calculate_credit_ratios(
        self,
        debt: float,
        ebitda: float,
        cost_of_debt: float,
        synergies: float
    ) -> Dict[str, float]:
        """
        Calculate key credit ratios.
        
        Args:
            debt: Total debt
            ebitda: EBITDA
            cost_of_debt: Cost of debt
            synergies: Expected synergies
            
        Returns:
            Dictionary of credit ratios
        """
        # Pro forma EBITDA (including synergies)
        pro_forma_ebitda = ebitda + synergies
        
        # Interest expense
        interest_expense = debt * cost_of_debt
        
        ratios = {
            'debt_to_ebitda': debt / pro_forma_ebitda if pro_forma_ebitda > 0 else 999,
            'ebitda_to_interest': pro_forma_ebitda / interest_expense if interest_expense > 0 else 999,
            'debt_service_coverage': pro_forma_ebitda / (interest_expense * 1.3) if interest_expense > 0 else 999  # Assumes principal = 30% of interest
        }
        
        return ratios
    
    def _estimate_rating_impact(
        self,
        credit_ratios: Dict[str, float]
    ) -> Optional[str]:
        """
        Estimate credit rating based on ratios.
        
        Args:
            credit_ratios: Dictionary of credit ratios
            
        Returns:
            Estimated credit rating
        """
        debt_to_ebitda = credit_ratios.get('debt_to_ebitda', 999)
        interest_coverage = credit_ratios.get('ebitda_to_interest', 0)
        
        # Simplified rating estimation
        if debt_to_ebitda < 1.5 and interest_coverage > 8:
            return "AAA/AA"
        elif debt_to_ebitda < 2.5 and interest_coverage > 5:
            return "A"
        elif debt_to_ebitda < 3.5 and interest_coverage > 3:
            return "BBB"
        elif debt_to_ebitda < 5.0 and interest_coverage > 2:
            return "BB"
        elif debt_to_ebitda < 6.0 and interest_coverage > 1.5:
            return "B"
        else:
            return "CCC or below"
    
    def calculate_value(
        self,
        purchase_price: float,
        target_ebitda: float,
        **kwargs
    ) -> float:
        """
        Calculate optimal WACC.
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            **kwargs: Additional parameters
            
        Returns:
            Optimal WACC
        """
        result = self.calculate(purchase_price, target_ebitda, **kwargs)
        return result.value.wacc
    
    def calculate_tax_shield_value(
        self,
        debt: float,
        cost_of_debt: float,
        tax_rate: float,
        years: int = 10
    ) -> float:
        """
        Calculate present value of tax shield from debt.
        
        Args:
            debt: Debt amount
            cost_of_debt: Cost of debt
            tax_rate: Corporate tax rate
            years: Number of years to calculate
            
        Returns:
            PV of tax shield
        """
        annual_tax_shield = debt * cost_of_debt * tax_rate
        
        # PV of tax shield (annuity)
        if cost_of_debt > 0:
            pv_factor = (1 - (1 + cost_of_debt) ** -years) / cost_of_debt
            tax_shield_pv = annual_tax_shield * pv_factor
        else:
            tax_shield_pv = annual_tax_shield * years
        
        return tax_shield_pv
    
    def calculate_breakeven_synergies(
        self,
        purchase_price: float,
        target_ebitda: float,
        acquirer_eps: float,
        acquirer_shares: float,
        acquirer_market_cap: float,
        debt_ratio: float = 0.5,
        cost_of_debt: float = 0.05,
        tax_rate: float = 0.21
    ) -> float:
        """
        Calculate synergies needed for EPS neutrality.
        
        Args:
            purchase_price: Purchase price
            target_ebitda: Target EBITDA
            acquirer_eps: Current EPS
            acquirer_shares: Shares outstanding
            acquirer_market_cap: Market cap
            debt_ratio: Debt as % of purchase price
            cost_of_debt: Cost of debt
            tax_rate: Tax rate
            
        Returns:
            Required annual synergies for EPS neutrality
        """
        debt = purchase_price * debt_ratio
        stock_value = purchase_price * (1 - debt_ratio)
        
        # Solve for synergies that make EPS impact = 0
        eps_impact, _, _ = self._calculate_eps_impact(
            purchase_price, target_ebitda, debt, stock_value,
            acquirer_shares, acquirer_market_cap, acquirer_eps,
            cost_of_debt, tax_rate, 0
        )
        
        if eps_impact >= 0:
            return 0.0  # Already accretive
        
        # Binary search for breakeven synergies
        low, high = 0, target_ebitda * 2
        tolerance = 0.01
        
        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            eps_impact, _, _ = self._calculate_eps_impact(
                purchase_price, target_ebitda, debt, stock_value,
                acquirer_shares, acquirer_market_cap, acquirer_eps,
                cost_of_debt, tax_rate, mid
            )
            
            if abs(eps_impact) < tolerance:
                return mid
            elif eps_impact < 0:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate deal financing inputs.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid
        """
        # Call parent validation
        super().validate_inputs(**kwargs)
        
        # Additional financing-specific validation
        if 'target_ebitda' in kwargs and kwargs['target_ebitda'] <= 0:
            raise ValueError("Target EBITDA must be positive")
        
        if 'acquirer_shares_outstanding' in kwargs and kwargs['acquirer_shares_outstanding'] <= 0:
            raise ValueError("Shares outstanding must be positive")
        
        return True


__all__ = [
    "DealFinancingModel",
    "FinancingSource",
]