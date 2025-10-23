"""
Synergy Valuation Models
========================

Comprehensive synergy valuation framework for M&A transactions including:
- Cost synergies: Operating leverage, economies of scale, procurement
- Revenue synergies: Cross-selling, market expansion, pricing power
- NPV analysis with realization schedules
- Monte Carlo simulation and sensitivity analysis
- Integration cost modeling

Performance target: <50ms for comprehensive synergy valuation
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from axiom.models.ma.base_model import BaseMandAModel, SynergyEstimate
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class CostSynergy:
    """Individual cost synergy component."""
    name: str
    annual_amount: float
    realization_year: int
    probability: float = 1.0
    one_time_cost: float = 0.0
    category: str = "other"  # operating, procurement, overhead, facilities, technology


@dataclass
class RevenueSynergy:
    """Individual revenue synergy component."""
    name: str
    annual_amount: float
    realization_year: int
    probability: float = 1.0
    investment_required: float = 0.0
    category: str = "other"  # cross_sell, market_expansion, pricing, channel, retention


class SynergyValuationModel(BaseMandAModel):
    """
    Comprehensive synergy valuation model.
    
    Features:
    - Cost synergy modeling (operating leverage, economies of scale, etc.)
    - Revenue synergy modeling (cross-selling, market expansion, etc.)
    - NPV calculation with risk-adjusted discount rates
    - Realization timeline modeling (typically 1-3 years for cost, 1-5 for revenue)
    - Integration cost estimation
    - Monte Carlo simulation for uncertainty
    - Sensitivity analysis for key drivers
    
    Performance: <50ms for full synergy analysis
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """
        Initialize synergy valuation model.
        
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
        
        # Set default config values if not provided
        self.config.setdefault('cost_synergy_realization_years', 3)
        self.config.setdefault('revenue_synergy_realization_years', 5)
        self.config.setdefault('synergy_discount_rate', 0.12)
        self.config.setdefault('integration_cost_multiple', 0.15)
        self.config.setdefault('synergy_tax_rate', 0.21)
        self.config.setdefault('enable_monte_carlo', True)
        self.config.setdefault('monte_carlo_scenarios', 10000)
    
    def calculate(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy],
        discount_rate: Optional[float] = None,
        tax_rate: Optional[float] = None,
        integration_cost_override: Optional[float] = None,
        run_monte_carlo: bool = True,
        run_sensitivity: bool = True
    ) -> ModelResult[SynergyEstimate]:
        """
        Calculate comprehensive synergy valuation.
        
        Args:
            cost_synergies: List of cost synergy components
            revenue_synergies: List of revenue synergy components
            discount_rate: Discount rate for NPV (default from config)
            tax_rate: Tax rate for synergies (default from config)
            integration_cost_override: Override integration cost calculation
            run_monte_carlo: Whether to run Monte Carlo simulation
            run_sensitivity: Whether to run sensitivity analysis
            
        Returns:
            ModelResult containing SynergyEstimate
        """
        start_time = time.perf_counter()
        
        # Validate inputs
        self.validate_inputs(
            discount_rate=discount_rate or self.config['synergy_discount_rate'],
            tax_rate=tax_rate or self.config['synergy_tax_rate']
        )
        
        # Use config defaults if not provided
        discount_rate = discount_rate or self.config['synergy_discount_rate']
        tax_rate = tax_rate or self.config['synergy_tax_rate']
        
        # Calculate cost synergies NPV
        cost_synergies_npv, cost_schedule = self._calculate_cost_synergies_npv(
            cost_synergies, discount_rate, tax_rate
        )
        
        # Calculate revenue synergies NPV
        revenue_synergies_npv, revenue_schedule = self._calculate_revenue_synergies_npv(
            revenue_synergies, discount_rate, tax_rate
        )
        
        # Total synergies
        total_synergies_npv = cost_synergies_npv + revenue_synergies_npv
        
        # Calculate integration costs
        if integration_cost_override is not None:
            integration_costs = integration_cost_override
        else:
            integration_costs = self._estimate_integration_costs(
                cost_synergies, revenue_synergies
            )
        
        # Net synergies after integration costs
        net_synergies = total_synergies_npv - integration_costs
        
        # Combined realization schedule
        realization_schedule = self._combine_schedules(cost_schedule, revenue_schedule)
        
        # Key assumptions
        key_assumptions = {
            'discount_rate': discount_rate,
            'tax_rate': tax_rate,
            'cost_realization_years': self.config['cost_synergy_realization_years'],
            'revenue_realization_years': self.config['revenue_synergy_realization_years'],
            'integration_cost_multiple': self.config['integration_cost_multiple'],
            'total_cost_synergies': len(cost_synergies),
            'total_revenue_synergies': len(revenue_synergies)
        }
        
        # Run optional analyses
        sensitivity_analysis = None
        confidence_level = 1.0  # Base case
        
        if run_sensitivity and self.config.get('enable_sensitivity_analysis', True):
            sensitivity_analysis = self._run_sensitivity_analysis(
                cost_synergies, revenue_synergies, discount_rate, tax_rate
            )
        
        if run_monte_carlo and self.config.get('enable_monte_carlo', True):
            confidence_level = self._run_monte_carlo_simulation(
                cost_synergies, revenue_synergies, discount_rate, tax_rate
            )
        
        # Create result
        estimate = SynergyEstimate(
            cost_synergies_npv=cost_synergies_npv,
            revenue_synergies_npv=revenue_synergies_npv,
            total_synergies_npv=total_synergies_npv,
            integration_costs=integration_costs,
            net_synergies=net_synergies,
            realization_schedule=realization_schedule,
            confidence_level=confidence_level,
            key_assumptions=key_assumptions,
            sensitivity_analysis=sensitivity_analysis
        )
        
        # Track performance
        execution_time_ms = self._track_performance("synergy_valuation", start_time)
        
        # Create metadata
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(
            value=estimate,
            metadata=metadata,
            success=True
        )
    
    def _calculate_cost_synergies_npv(
        self,
        cost_synergies: List[CostSynergy],
        discount_rate: float,
        tax_rate: float
    ) -> Tuple[float, List[float]]:
        """
        Calculate NPV of cost synergies.
        
        Args:
            cost_synergies: List of cost synergy components
            discount_rate: Discount rate
            tax_rate: Tax rate
            
        Returns:
            Tuple of (NPV, realization schedule)
        """
        max_years = self.config['cost_synergy_realization_years'] + 10  # Include perpetuity
        schedule = np.zeros(max_years)
        
        for synergy in cost_synergies:
            # After-tax synergy value
            after_tax_amount = synergy.annual_amount * (1 - tax_rate) * synergy.probability
            
            # Add to schedule starting from realization year
            for year in range(synergy.realization_year, max_years):
                schedule[year] += after_tax_amount
        
        # Calculate NPV
        discount_factors = np.array([(1 + discount_rate) ** -i for i in range(max_years)])
        npv = np.sum(schedule * discount_factors)
        
        # Subtract one-time costs (PV)
        one_time_costs = sum(
            synergy.one_time_cost / (1 + discount_rate) ** synergy.realization_year
            for synergy in cost_synergies
        )
        npv -= one_time_costs
        
        return npv, schedule[:self.config['cost_synergy_realization_years']].tolist()
    
    def _calculate_revenue_synergies_npv(
        self,
        revenue_synergies: List[RevenueSynergy],
        discount_rate: float,
        tax_rate: float
    ) -> Tuple[float, List[float]]:
        """
        Calculate NPV of revenue synergies.
        
        Args:
            revenue_synergies: List of revenue synergy components
            discount_rate: Discount rate
            tax_rate: Tax rate
            
        Returns:
            Tuple of (NPV, realization schedule)
        """
        max_years = self.config['revenue_synergy_realization_years'] + 10
        schedule = np.zeros(max_years)
        
        for synergy in revenue_synergies:
            # After-tax synergy value (revenue synergies generate taxable income)
            after_tax_amount = synergy.annual_amount * (1 - tax_rate) * synergy.probability
            
            # Add to schedule starting from realization year
            for year in range(synergy.realization_year, max_years):
                schedule[year] += after_tax_amount
        
        # Calculate NPV
        discount_factors = np.array([(1 + discount_rate) ** -i for i in range(max_years)])
        npv = np.sum(schedule * discount_factors)
        
        # Subtract investment required (PV)
        investments = sum(
            synergy.investment_required / (1 + discount_rate) ** synergy.realization_year
            for synergy in revenue_synergies
        )
        npv -= investments
        
        return npv, schedule[:self.config['revenue_synergy_realization_years']].tolist()
    
    def _estimate_integration_costs(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy]
    ) -> float:
        """
        Estimate integration costs as percentage of synergies.
        
        Args:
            cost_synergies: Cost synergy components
            revenue_synergies: Revenue synergy components
            
        Returns:
            Estimated integration costs
        """
        # Total annual synergies (first year fully realized)
        total_cost = sum(s.annual_amount for s in cost_synergies)
        total_revenue = sum(s.annual_amount for s in revenue_synergies)
        total_annual = total_cost + total_revenue
        
        # Integration costs as multiple of annual synergies
        integration_multiple = self.config['integration_cost_multiple']
        integration_costs = total_annual * integration_multiple
        
        # Add explicit one-time costs
        explicit_costs = sum(s.one_time_cost for s in cost_synergies)
        explicit_investments = sum(s.investment_required for s in revenue_synergies)
        
        return integration_costs + explicit_costs + explicit_investments
    
    def _combine_schedules(
        self,
        cost_schedule: List[float],
        revenue_schedule: List[float]
    ) -> List[float]:
        """
        Combine cost and revenue realization schedules.
        
        Args:
            cost_schedule: Cost synergy schedule
            revenue_schedule: Revenue synergy schedule
            
        Returns:
            Combined schedule
        """
        max_len = max(len(cost_schedule), len(revenue_schedule))
        combined = [0.0] * max_len
        
        for i in range(len(cost_schedule)):
            combined[i] += cost_schedule[i]
        
        for i in range(len(revenue_schedule)):
            combined[i] += revenue_schedule[i]
        
        return combined
    
    def _run_sensitivity_analysis(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy],
        base_discount_rate: float,
        base_tax_rate: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Run sensitivity analysis on key drivers.
        
        Args:
            cost_synergies: Cost synergy components
            revenue_synergies: Revenue synergy components
            base_discount_rate: Base discount rate
            base_tax_rate: Base tax rate
            
        Returns:
            Sensitivity results dictionary
        """
        sensitivity_results = {}
        
        # Get base case NPV
        base_cost_npv, _ = self._calculate_cost_synergies_npv(
            cost_synergies, base_discount_rate, base_tax_rate
        )
        base_revenue_npv, _ = self._calculate_revenue_synergies_npv(
            revenue_synergies, base_discount_rate, base_tax_rate
        )
        base_total = base_cost_npv + base_revenue_npv
        
        # Discount rate sensitivity (-/+ 20%)
        sensitivity_results['discount_rate'] = {}
        for adjustment in [-0.20, -0.10, 0.10, 0.20]:
            adjusted_rate = base_discount_rate * (1 + adjustment)
            cost_npv, _ = self._calculate_cost_synergies_npv(
                cost_synergies, adjusted_rate, base_tax_rate
            )
            revenue_npv, _ = self._calculate_revenue_synergies_npv(
                revenue_synergies, adjusted_rate, base_tax_rate
            )
            total_npv = cost_npv + revenue_npv
            sensitivity_results['discount_rate'][f'{adjustment:+.0%}'] = total_npv - base_total
        
        # Synergy amount sensitivity (-/+ 50%)
        sensitivity_results['synergy_amount'] = {}
        for adjustment in [-0.50, -0.25, 0.25, 0.50]:
            # Adjust synergies
            adj_cost_synergies = [
                CostSynergy(
                    s.name, s.annual_amount * (1 + adjustment), s.realization_year,
                    s.probability, s.one_time_cost, s.category
                ) for s in cost_synergies
            ]
            adj_revenue_synergies = [
                RevenueSynergy(
                    s.name, s.annual_amount * (1 + adjustment), s.realization_year,
                    s.probability, s.investment_required, s.category
                ) for s in revenue_synergies
            ]
            
            cost_npv, _ = self._calculate_cost_synergies_npv(
                adj_cost_synergies, base_discount_rate, base_tax_rate
            )
            revenue_npv, _ = self._calculate_revenue_synergies_npv(
                adj_revenue_synergies, base_discount_rate, base_tax_rate
            )
            total_npv = cost_npv + revenue_npv
            sensitivity_results['synergy_amount'][f'{adjustment:+.0%}'] = total_npv - base_total
        
        # Realization timing sensitivity (+/- 1 year)
        sensitivity_results['realization_timing'] = {}
        for adjustment in [-1, 1]:
            adj_cost_synergies = [
                CostSynergy(
                    s.name, s.annual_amount, max(1, s.realization_year + adjustment),
                    s.probability, s.one_time_cost, s.category
                ) for s in cost_synergies
            ]
            adj_revenue_synergies = [
                RevenueSynergy(
                    s.name, s.annual_amount, max(1, s.realization_year + adjustment),
                    s.probability, s.investment_required, s.category
                ) for s in revenue_synergies
            ]
            
            cost_npv, _ = self._calculate_cost_synergies_npv(
                adj_cost_synergies, base_discount_rate, base_tax_rate
            )
            revenue_npv, _ = self._calculate_revenue_synergies_npv(
                adj_revenue_synergies, base_discount_rate, base_tax_rate
            )
            total_npv = cost_npv + revenue_npv
            key = f'{adjustment:+d}_year'
            sensitivity_results['realization_timing'][key] = total_npv - base_total
        
        return sensitivity_results
    
    def _run_monte_carlo_simulation(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy],
        discount_rate: float,
        tax_rate: float
    ) -> float:
        """
        Run Monte Carlo simulation to estimate confidence level.
        
        Args:
            cost_synergies: Cost synergy components
            revenue_synergies: Revenue synergy components
            discount_rate: Discount rate
            tax_rate: Tax rate
            
        Returns:
            Confidence level (probability of achieving >0 net synergies)
        """
        n_scenarios = self.config['monte_carlo_scenarios']
        results = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            # Randomize synergy amounts (normal distribution, +/-30% std dev)
            scenario_cost_synergies = [
                CostSynergy(
                    s.name,
                    max(0, np.random.normal(s.annual_amount, s.annual_amount * 0.3)),
                    s.realization_year,
                    np.random.uniform(0.5, 1.0) if s.probability < 1.0 else 1.0,
                    s.one_time_cost,
                    s.category
                ) for s in cost_synergies
            ]
            
            scenario_revenue_synergies = [
                RevenueSynergy(
                    s.name,
                    max(0, np.random.normal(s.annual_amount, s.annual_amount * 0.4)),
                    s.realization_year,
                    np.random.uniform(0.4, 1.0) if s.probability < 1.0 else 1.0,
                    s.investment_required,
                    s.category
                ) for s in revenue_synergies
            ]
            
            # Calculate NPV for this scenario
            cost_npv, _ = self._calculate_cost_synergies_npv(
                scenario_cost_synergies, discount_rate, tax_rate
            )
            revenue_npv, _ = self._calculate_revenue_synergies_npv(
                scenario_revenue_synergies, discount_rate, tax_rate
            )
            
            # Estimate integration costs for scenario
            integration_costs = self._estimate_integration_costs(
                scenario_cost_synergies, scenario_revenue_synergies
            )
            
            results[i] = cost_npv + revenue_npv - integration_costs
        
        # Calculate confidence level (% of scenarios with positive net synergies)
        confidence_level = np.sum(results > 0) / n_scenarios
        
        return confidence_level
    
    def calculate_value(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy],
        **kwargs
    ) -> float:
        """
        Calculate total synergy value.
        
        Args:
            cost_synergies: Cost synergy components
            revenue_synergies: Revenue synergy components
            **kwargs: Additional parameters
            
        Returns:
            Total synergy NPV
        """
        result = self.calculate(cost_synergies, revenue_synergies, **kwargs)
        return result.value.total_synergies_npv
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate synergy valuation inputs.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid
        """
        # Call parent validation
        super().validate_inputs(**kwargs)
        
        # Additional synergy-specific validation
        if 'cost_synergies' in kwargs:
            for synergy in kwargs['cost_synergies']:
                if synergy.annual_amount < 0:
                    raise ValueError("Cost synergy amounts must be positive")
                if synergy.realization_year < 1:
                    raise ValueError("Realization year must be >= 1")
        
        if 'revenue_synergies' in kwargs:
            for synergy in kwargs['revenue_synergies']:
                if synergy.annual_amount < 0:
                    raise ValueError("Revenue synergy amounts must be positive")
                if synergy.realization_year < 1:
                    raise ValueError("Realization year must be >= 1")
        
        return True
    
    def calculate_breakeven_synergies(
        self,
        purchase_premium: float,
        discount_rate: Optional[float] = None,
        realization_years: int = 3
    ) -> float:
        """
        Calculate annual synergies needed to justify purchase premium.
        
        Args:
            purchase_premium: Premium paid over standalone value
            discount_rate: Discount rate (default from config)
            realization_years: Years to realize synergies
            
        Returns:
            Required annual synergies
        """
        discount_rate = discount_rate or self.config['synergy_discount_rate']
        tax_rate = self.config['synergy_tax_rate']
        
        # After-tax synergy needed
        return self.calculate_breakeven_synergies(
            purchase_premium,
            discount_rate,
            realization_years
        )
    
    def analyze_by_category(
        self,
        cost_synergies: List[CostSynergy],
        revenue_synergies: List[RevenueSynergy],
        discount_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyze synergies by category.
        
        Args:
            cost_synergies: Cost synergy components
            revenue_synergies: Revenue synergy components
            discount_rate: Discount rate
            
        Returns:
            Dictionary of NPV by category
        """
        discount_rate = discount_rate or self.config['synergy_discount_rate']
        tax_rate = self.config['synergy_tax_rate']
        
        results = {}
        
        # Group cost synergies by category
        cost_categories = {}
        for synergy in cost_synergies:
            if synergy.category not in cost_categories:
                cost_categories[synergy.category] = []
            cost_categories[synergy.category].append(synergy)
        
        # Calculate NPV for each cost category
        for category, synergies in cost_categories.items():
            npv, _ = self._calculate_cost_synergies_npv(synergies, discount_rate, tax_rate)
            results[f'cost_{category}'] = npv
        
        # Group revenue synergies by category
        revenue_categories = {}
        for synergy in revenue_synergies:
            if synergy.category not in revenue_categories:
                revenue_categories[synergy.category] = []
            revenue_categories[synergy.category].append(synergy)
        
        # Calculate NPV for each revenue category
        for category, synergies in revenue_categories.items():
            npv, _ = self._calculate_revenue_synergies_npv(synergies, discount_rate, tax_rate)
            results[f'revenue_{category}'] = npv
        
        return results


__all__ = [
    "SynergyValuationModel",
    "CostSynergy",
    "RevenueSynergy",
]