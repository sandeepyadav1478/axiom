"""
LBO (Leveraged Buyout) Modeling
================================

Comprehensive LBO analysis framework for private equity transactions:
- IRR and cash-on-cash returns calculation
- Debt sizing and amortization schedules
- Operational improvement modeling (EBITDA margin expansion)
- Multiple exit strategy scenarios (strategic sale, IPO, secondary buyout)
- Sensitivity analysis (entry/exit multiples, leverage, growth)
- Management equity and option dilution
- Dividend recapitalization modeling

Performance target: <60ms for full LBO model
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from axiom.models.ma.base_model import BaseMandAModel, LBOAnalysis
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class OperationalImprovements:
    """Operational improvements assumptions."""
    revenue_growth_rate: float  # Annual revenue growth
    ebitda_margin_expansion: float  # EBITDA margin improvement (bps)
    working_capital_improvement: float  # % of revenue
    capex_reduction: float  # % reduction in capex


@dataclass
class DebtStructure:
    """LBO debt structure."""
    senior_debt: float
    subordinated_debt: float
    total_debt: float
    senior_rate: float
    subordinated_rate: float
    amortization_years: int
    interest_only_years: int = 0


class LBOModel(BaseMandAModel):
    """
    Comprehensive Leveraged Buyout analysis model.
    
    Features:
    - IRR calculation for equity sponsors
    - Cash-on-cash returns (multiple of money)
    - Debt sizing based on EBITDA multiples
    - Debt amortization schedules
    - Interest coverage and leverage ratio monitoring
    - Operational improvements (EBITDA growth, margin expansion)
    - Working capital and capex modeling
    - Multiple exit scenarios (strategic sale, IPO, secondary LBO)
    - Exit multiple sensitivity analysis
    - Management equity and dilution
    - Dividend recapitalization
    - Sensitivity tables and tornado charts
    
    Performance: <60ms for full LBO model with sensitivity analysis
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize LBO model.
        
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
        self.config.setdefault('target_irr', 0.20)
        self.config.setdefault('holding_period_years', 5)
        self.config.setdefault('max_leverage_multiple', 6.0)
        self.config.setdefault('min_equity_contribution_pct', 0.30)
        self.config.setdefault('exit_multiple_method', 'entry_multiple')
        self.config.setdefault('default_tax_rate', 0.21)
    
    def calculate(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        exit_multiple: Optional[float] = None,
        holding_period: Optional[int] = None,
        leverage_multiple: float = 5.0,
        senior_debt_rate: float = 0.06,
        subordinated_debt_rate: float = 0.10,
        senior_debt_pct: float = 0.70,
        operational_improvements: Optional[OperationalImprovements] = None,
        tax_rate: Optional[float] = None,
        transaction_fees_pct: float = 0.02,
        management_equity_pct: float = 0.10,
        exit_strategy: str = "strategic_sale"  # strategic_sale, ipo, secondary_lbo
    ) -> ModelResult[LBOAnalysis]:
        """
        Calculate comprehensive LBO analysis.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry EV/EBITDA multiple
            exit_multiple: Exit EV/EBITDA multiple (defaults to entry)
            holding_period: Holding period in years
            leverage_multiple: Total Debt/EBITDA multiple
            senior_debt_rate: Senior debt interest rate
            subordinated_debt_rate: Subordinated debt interest rate
            senior_debt_pct: Senior debt as % of total debt
            operational_improvements: Operational improvement assumptions
            tax_rate: Corporate tax rate
            transaction_fees_pct: Transaction fees as % of enterprise value
            management_equity_pct: Management equity stake
            exit_strategy: Exit strategy type
            
        Returns:
            ModelResult containing LBOAnalysis
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(
            entry_ebitda=entry_ebitda,
            entry_multiple=entry_multiple
        )
        
        # Use config defaults
        holding_period = holding_period or self.config['holding_period_years']
        tax_rate = tax_rate or self.config['default_tax_rate']
        exit_multiple = exit_multiple or entry_multiple  # Default to entry multiple
        
        # Calculate entry price
        entry_ev = entry_ebitda * entry_multiple
        transaction_fees = entry_ev * transaction_fees_pct
        entry_price = entry_ev + transaction_fees
        
        # Size debt
        total_debt = entry_ebitda * leverage_multiple
        senior_debt = total_debt * senior_debt_pct
        subordinated_debt = total_debt * (1 - senior_debt_pct)
        
        debt_structure = DebtStructure(
            senior_debt=senior_debt,
            subordinated_debt=subordinated_debt,
            total_debt=total_debt,
            senior_rate=senior_debt_rate,
            subordinated_rate=subordinated_debt_rate,
            amortization_years=holding_period,
            interest_only_years=1  # Typical structure
        )
        
        # Calculate equity contribution
        equity_contribution = entry_price - total_debt
        sponsor_equity = equity_contribution * (1 - management_equity_pct)
        
        # Project cash flows and EBITDA
        if operational_improvements is None:
            operational_improvements = OperationalImprovements(
                revenue_growth_rate=0.05,  # 5% annual growth
                ebitda_margin_expansion=100,  # 100 bps improvement
                working_capital_improvement=0.02,  # 2% of revenue
                capex_reduction=0.10  # 10% capex reduction
            )
        
        # Build cash flow projection
        cash_flows, exit_ebitda = self._project_cash_flows(
            entry_ebitda,
            holding_period,
            operational_improvements,
            debt_structure,
            tax_rate
        )
        
        # Calculate exit value
        exit_ev = exit_ebitda * exit_multiple
        remaining_debt = self._calculate_remaining_debt(
            debt_structure, cash_flows, holding_period
        )
        exit_equity_value = exit_ev - remaining_debt
        
        # Adjust for management equity
        sponsor_exit_value = exit_equity_value * (1 - management_equity_pct)
        
        # Calculate returns
        irr, converged = self.calculate_irr(
            [-sponsor_equity] + [0] * (holding_period - 1) + [sponsor_exit_value]
        )
        
        cash_on_cash = sponsor_exit_value / sponsor_equity if sponsor_equity > 0 else 0.0
        
        # Debt paydown
        debt_paydown = total_debt - remaining_debt
        
        # Value creation attribution
        operational_value = self._calculate_operational_value_creation(
            entry_ebitda, exit_ebitda, exit_multiple, tax_rate
        )
        
        multiple_expansion_value = (exit_multiple - entry_multiple) * exit_ebitda
        
        debt_paydown_value = debt_paydown
        
        # Operational improvements breakdown
        ops_improvements = {
            'revenue_growth': entry_ebitda * operational_improvements.revenue_growth_rate * holding_period,
            'margin_expansion': entry_ebitda * (operational_improvements.ebitda_margin_expansion / 10000) * holding_period,
            'working_capital': entry_ebitda * operational_improvements.working_capital_improvement,
            'capex_optimization': entry_ebitda * operational_improvements.capex_reduction * 0.5
        }
        
        # Dividend recap calculation (if applicable)
        dividend_recap = self._calculate_dividend_recap_potential(
            exit_ebitda, leverage_multiple, remaining_debt
        )
        
        # Sensitivity analysis
        sensitivity_matrix = self._run_sensitivity_analysis(
            entry_ebitda, entry_multiple, equity_contribution,
            holding_period, operational_improvements, debt_structure, tax_rate
        ) if self.config.get('enable_sensitivity_analysis', True) else None
        
        # Create result
        analysis = LBOAnalysis(
            entry_price=entry_price,
            exit_price=exit_ev,
            equity_contribution=equity_contribution,
            debt_financing=total_debt,
            irr=irr if converged else 0.0,
            cash_on_cash=cash_on_cash,
            holding_period=holding_period,
            exit_multiple=exit_multiple,
            debt_paydown=debt_paydown,
            operational_improvements=ops_improvements,
            multiple_expansion=multiple_expansion_value,
            dividend_recap=dividend_recap,
            sensitivity_matrix=sensitivity_matrix
        )
        
        # Track performance
        execution_time_ms = self._track_performance("lbo_modeling", start_time)
        
        # Create metadata
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(
            value=analysis,
            metadata=metadata,
            success=True
        )
    
    def _project_cash_flows(
        self,
        entry_ebitda: float,
        holding_period: int,
        ops_improvements: OperationalImprovements,
        debt_structure: DebtStructure,
        tax_rate: float
    ) -> Tuple[List[float], float]:
        """
        Project annual cash flows and final EBITDA.
        
        Args:
            entry_ebitda: Starting EBITDA
            holding_period: Years to hold
            ops_improvements: Operational improvements
            debt_structure: Debt structure
            tax_rate: Tax rate
            
        Returns:
            Tuple of (cash_flows list, exit EBITDA)
        """
        cash_flows = []
        current_ebitda = entry_ebitda
        current_debt = debt_structure.total_debt
        
        for year in range(1, holding_period + 1):
            # EBITDA growth
            current_ebitda = current_ebitda * (1 + ops_improvements.revenue_growth_rate)
            current_ebitda += entry_ebitda * (ops_improvements.ebitda_margin_expansion / 10000)
            
            # Interest expense
            senior_interest = debt_structure.senior_debt * debt_structure.senior_rate
            sub_interest = debt_structure.subordinated_debt * debt_structure.subordinated_rate
            total_interest = senior_interest + sub_interest
            
            # Taxable income
            taxable_income = current_ebitda - total_interest
            taxes = max(0, taxable_income * tax_rate)
            
            # Free cash flow
            fcf = current_ebitda - total_interest - taxes
            
            # Debt amortization (after interest-only period)
            if year > debt_structure.interest_only_years:
                mandatory_amortization = debt_structure.total_debt / (holding_period - debt_structure.interest_only_years)
                fcf -= mandatory_amortization
                current_debt -= mandatory_amortization
            
            cash_flows.append(fcf)
        
        return cash_flows, current_ebitda
    
    def _calculate_remaining_debt(
        self,
        debt_structure: DebtStructure,
        cash_flows: List[float],
        holding_period: int
    ) -> float:
        """
        Calculate remaining debt at exit.
        
        Args:
            debt_structure: Initial debt structure
            cash_flows: Projected cash flows
            holding_period: Holding period
            
        Returns:
            Remaining debt
        """
        remaining_debt = debt_structure.total_debt
        
        # Assume excess cash flow goes to debt paydown
        for year, fcf in enumerate(cash_flows, 1):
            if year > debt_structure.interest_only_years:
                # Mandatory amortization
                mandatory = debt_structure.total_debt / (holding_period - debt_structure.interest_only_years)
                
                # Optional paydown from excess FCF
                optional_paydown = max(0, fcf * 0.5)  # Use 50% of FCF for debt paydown
                
                total_paydown = mandatory + optional_paydown
                remaining_debt = max(0, remaining_debt - total_paydown)
        
        return remaining_debt
    
    def _calculate_operational_value_creation(
        self,
        entry_ebitda: float,
        exit_ebitda: float,
        exit_multiple: float,
        tax_rate: float
    ) -> float:
        """
        Calculate value created from operational improvements.
        
        Args:
            entry_ebitda: Entry EBITDA
            exit_ebitda: Exit EBITDA
            exit_multiple: Exit multiple
            tax_rate: Tax rate
            
        Returns:
            Operational value creation
        """
        ebitda_growth = exit_ebitda - entry_ebitda
        value_created = ebitda_growth * exit_multiple * (1 - tax_rate)
        
        return value_created
    
    def _calculate_dividend_recap_potential(
        self,
        current_ebitda: float,
        max_leverage: float,
        current_debt: float
    ) -> float:
        """
        Calculate potential dividend recapitalization.
        
        Args:
            current_ebitda: Current EBITDA
            max_leverage: Maximum leverage multiple
            current_debt: Current debt level
            
        Returns:
            Potential dividend recap amount
        """
        # Maximum debt capacity
        max_debt = current_ebitda * max_leverage
        
        # Available capacity
        available_capacity = max(0, max_debt - current_debt)
        
        # Typical dividend recap is 50-75% of available capacity
        dividend_recap = available_capacity * 0.60
        
        return dividend_recap
    
    def _run_sensitivity_analysis(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        equity_contribution: float,
        holding_period: int,
        ops_improvements: OperationalImprovements,
        debt_structure: DebtStructure,
        tax_rate: float
    ) -> Dict[str, np.ndarray]:
        """
        Run sensitivity analysis on key variables.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry multiple
            equity_contribution: Equity contributed
            holding_period: Holding period
            ops_improvements: Operational improvements
            debt_structure: Debt structure
            tax_rate: Tax rate
            
        Returns:
            Sensitivity matrices
        """
        # Exit multiple sensitivity (rows) vs EBITDA growth sensitivity (columns)
        exit_multiples = np.linspace(entry_multiple * 0.8, entry_multiple * 1.2, 5)
        growth_rates = np.linspace(0.0, 0.10, 5)
        
        irr_matrix = np.zeros((len(exit_multiples), len(growth_rates)))
        moic_matrix = np.zeros((len(exit_multiples), len(growth_rates)))
        
        for i, exit_mult in enumerate(exit_multiples):
            for j, growth_rate in enumerate(growth_rates):
                # Adjust operational improvements
                adj_ops = OperationalImprovements(
                    revenue_growth_rate=growth_rate,
                    ebitda_margin_expansion=ops_improvements.ebitda_margin_expansion,
                    working_capital_improvement=ops_improvements.working_capital_improvement,
                    capex_reduction=ops_improvements.capex_reduction
                )
                
                # Project cash flows
                cash_flows, exit_ebitda = self._project_cash_flows(
                    entry_ebitda, holding_period, adj_ops, debt_structure, tax_rate
                )
                
                # Calculate exit value
                exit_ev = exit_ebitda * exit_mult
                remaining_debt = self._calculate_remaining_debt(
                    debt_structure, cash_flows, holding_period
                )
                exit_equity_value = exit_ev - remaining_debt
                
                # Calculate returns
                irr, _ = self.calculate_irr(
                    [-equity_contribution] + [0] * (holding_period - 1) + [exit_equity_value]
                )
                moic = exit_equity_value / equity_contribution if equity_contribution > 0 else 0.0
                
                irr_matrix[i, j] = irr
                moic_matrix[i, j] = moic
        
        return {
            'irr_matrix': irr_matrix,
            'moic_matrix': moic_matrix,
            'exit_multiples': exit_multiples,
            'growth_rates': growth_rates
        }
    
    def calculate_value(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        **kwargs
    ) -> float:
        """
        Calculate LBO IRR.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry multiple
            **kwargs: Additional parameters
            
        Returns:
            IRR
        """
        result = self.calculate(entry_ebitda, entry_multiple, **kwargs)
        return result.value.irr
    
    def calculate_minimum_exit_multiple(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        target_irr: Optional[float] = None,
        holding_period: Optional[int] = None,
        leverage_multiple: float = 5.0,
        **kwargs
    ) -> float:
        """
        Calculate minimum exit multiple needed to achieve target IRR.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry multiple
            target_irr: Target IRR (default from config)
            holding_period: Holding period
            leverage_multiple: Leverage multiple
            **kwargs: Additional parameters
            
        Returns:
            Minimum exit multiple
        """
        target_irr = target_irr or self.config['target_irr']
        holding_period = holding_period or self.config['holding_period_years']
        
        # Binary search for exit multiple
        low, high = entry_multiple * 0.5, entry_multiple * 2.0
        tolerance = 0.01
        
        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            
            result = self.calculate(
                entry_ebitda, entry_multiple,
                exit_multiple=mid,
                holding_period=holding_period,
                leverage_multiple=leverage_multiple,
                **kwargs
            )
            
            achieved_irr = result.value.irr
            
            if abs(achieved_irr - target_irr) < tolerance:
                return mid
            elif achieved_irr < target_irr:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2
    
    def calculate_maximum_purchase_price(
        self,
        entry_ebitda: float,
        target_irr: Optional[float] = None,
        exit_multiple: float = 10.0,
        holding_period: Optional[int] = None,
        leverage_multiple: float = 5.0,
        **kwargs
    ) -> float:
        """
        Calculate maximum purchase price multiple for target IRR.
        
        Args:
            entry_ebitda: Entry EBITDA
            target_irr: Target IRR
            exit_multiple: Expected exit multiple
            holding_period: Holding period
            leverage_multiple: Leverage multiple
            **kwargs: Additional parameters
            
        Returns:
            Maximum entry multiple
        """
        target_irr = target_irr or self.config['target_irr']
        holding_period = holding_period or self.config['holding_period_years']
        
        # Binary search for entry multiple
        low, high = 5.0, 20.0
        tolerance = 0.01
        
        for _ in range(50):
            mid = (low + high) / 2
            
            result = self.calculate(
                entry_ebitda, mid,
                exit_multiple=exit_multiple,
                holding_period=holding_period,
                leverage_multiple=leverage_multiple,
                **kwargs
            )
            
            achieved_irr = result.value.irr
            
            if abs(achieved_irr - target_irr) < tolerance:
                return mid
            elif achieved_irr < target_irr:
                high = mid  # Need lower entry price
            else:
                low = mid  # Can pay more
        
        return (low + high) / 2
    
    def analyze_leverage_impact(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        leverage_range: Tuple[float, float] = (3.0, 7.0),
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Analyze impact of different leverage levels on returns.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry multiple
            leverage_range: Range of leverage multiples to test
            **kwargs: Additional parameters
            
        Returns:
            Leverage analysis results
        """
        leverage_multiples = np.linspace(leverage_range[0], leverage_range[1], 9)
        
        irrs = []
        moics = []
        interest_coverages = []
        
        for leverage in leverage_multiples:
            result = self.calculate(
                entry_ebitda, entry_multiple,
                leverage_multiple=leverage,
                **kwargs
            )
            
            irrs.append(result.value.irr)
            moics.append(result.value.cash_on_cash)
            
            # Calculate average interest coverage
            total_debt = entry_ebitda * leverage
            avg_rate = 0.07  # Simplified
            interest = total_debt * avg_rate
            coverage = entry_ebitda / interest if interest > 0 else 999
            interest_coverages.append(coverage)
        
        return {
            'leverage_multiples': leverage_multiples.tolist(),
            'irrs': irrs,
            'cash_on_cash': moics,
            'interest_coverage': interest_coverages
        }
    
    def model_management_incentives(
        self,
        equity_contribution: float,
        management_equity_pct: float,
        option_pool_pct: float,
        exit_value: float
    ) -> Dict[str, float]:
        """
        Model management equity incentives and dilution.
        
        Args:
            equity_contribution: Total equity contributed
            management_equity_pct: Management equity stake
            option_pool_pct: Option pool as % of equity
            exit_value: Exit equity value
            
        Returns:
            Management incentive analysis
        """
        # Initial equity split
        sponsor_equity = equity_contribution * (1 - management_equity_pct - option_pool_pct)
        management_equity = equity_contribution * management_equity_pct
        option_pool = equity_contribution * option_pool_pct
        
        # Exit value split (assuming no dilution)
        sponsor_exit = exit_value * (1 - management_equity_pct - option_pool_pct)
        management_exit = exit_value * management_equity_pct
        option_pool_value = exit_value * option_pool_pct
        
        # Returns
        sponsor_moic = sponsor_exit / sponsor_equity if sponsor_equity > 0 else 0
        management_moic = management_exit / management_equity if management_equity > 0 else 0
        
        return {
            'sponsor_equity': sponsor_equity,
            'management_equity': management_equity,
            'option_pool': option_pool,
            'sponsor_exit_value': sponsor_exit,
            'management_exit_value': management_exit,
            'option_pool_value': option_pool_value,
            'sponsor_moic': sponsor_moic,
            'management_moic': management_moic,
            'management_return_dollars': management_exit - management_equity
        }
    
    def compare_exit_strategies(
        self,
        entry_ebitda: float,
        entry_multiple: float,
        holding_period: int,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different exit strategy scenarios.
        
        Args:
            entry_ebitda: Entry EBITDA
            entry_multiple: Entry multiple
            holding_period: Holding period
            **kwargs: Additional parameters
            
        Returns:
            Exit strategy comparison
        """
        strategies = {}
        
        # Strategic sale (premium exit multiple)
        strategic_result = self.calculate(
            entry_ebitda, entry_multiple,
            exit_multiple=entry_multiple * 1.1,
            holding_period=holding_period,
            **kwargs
        )
        strategies['strategic_sale'] = {
            'irr': strategic_result.value.irr,
            'moic': strategic_result.value.cash_on_cash,
            'exit_value': strategic_result.value.exit_price
        }
        
        # IPO (market multiple, potential discount)
        ipo_result = self.calculate(
            entry_ebitda, entry_multiple,
            exit_multiple=entry_multiple * 0.95,
            holding_period=holding_period,
            **kwargs
        )
        strategies['ipo'] = {
            'irr': ipo_result.value.irr,
            'moic': ipo_result.value.cash_on_cash,
            'exit_value': ipo_result.value.exit_price
        }
        
        # Secondary LBO (PE buyer, entry multiple)
        secondary_result = self.calculate(
            entry_ebitda, entry_multiple,
            exit_multiple=entry_multiple,
            holding_period=holding_period,
            **kwargs
        )
        strategies['secondary_lbo'] = {
            'irr': secondary_result.value.irr,
            'moic': secondary_result.value.cash_on_cash,
            'exit_value': secondary_result.value.exit_price
        }
        
        return strategies
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate LBO model inputs.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid
        """
        # Call parent validation
        super().validate_inputs(**kwargs)
        
        # LBO-specific validation
        if 'entry_ebitda' in kwargs and kwargs['entry_ebitda'] <= 0:
            raise ValueError("Entry EBITDA must be positive")
        
        if 'entry_multiple' in kwargs and kwargs['entry_multiple'] <= 0:
            raise ValueError("Entry multiple must be positive")
        
        if 'leverage_multiple' in kwargs:
            lev = kwargs['leverage_multiple']
            if lev < 0 or lev > self.config['max_leverage_multiple']:
                raise ValueError(f"Leverage multiple must be between 0 and {self.config['max_leverage_multiple']}")
        
        return True


__all__ = [
    "LBOModel",
    "OperationalImprovements",
    "DebtStructure",
]