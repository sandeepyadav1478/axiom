"""
M&A Valuation Integration Framework
===================================

Integrated valuation framework combining multiple methodologies:
- DCF (Discounted Cash Flow) analysis
- Trading comparables (public companies)
- Precedent transactions (M&A deals)
- Synergy-adjusted valuations
- Control premium analysis
- Walk-away price calculation
- Fairness opinion support

Performance target: <40ms for integrated valuation
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from axiom.models.ma.base_model import BaseMandAModel
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class ValuationResult:
    """Integrated valuation result."""
    dcf_value: float
    trading_comps_value: float
    precedent_transactions_value: float
    synergy_adjusted_value: float
    recommended_value: float
    valuation_range: Tuple[float, float]
    control_premium: float
    methodology_weights: Dict[str, float]
    walk_away_price: float


class ValuationIntegrationModel(BaseMandAModel):
    """
    Integrated M&A valuation framework.
    
    Combines multiple valuation methodologies with appropriate weighting
    to arrive at a comprehensive valuation range and recommended offer price.
    
    Performance: <40ms for full integrated valuation
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize valuation integration model."""
        super().__init__(
            config=config.__dict__ if config and hasattr(config, '__dict__') else (config or {}),
            enable_logging=enable_logging,
            enable_performance_tracking=enable_performance_tracking
        )
        
        self.config.setdefault('dcf_terminal_growth', 0.02)
        self.config.setdefault('control_premium_range', (0.20, 0.40))
    
    def calculate(
        self,
        target_fcf: List[float],
        discount_rate: float,
        terminal_growth: Optional[float] = None,
        comparable_multiples: Optional[List[float]] = None,
        precedent_multiples: Optional[List[float]] = None,
        target_ebitda: Optional[float] = None,
        synergies_npv: float = 0.0,
        methodology_weights: Optional[Dict[str, float]] = None
    ) -> ModelResult[ValuationResult]:
        """
        Calculate integrated valuation.
        
        Args:
            target_fcf: Projected free cash flows
            discount_rate: Discount rate (WACC)
            terminal_growth: Terminal growth rate
            comparable_multiples: List of comparable EV/EBITDA multiples
            precedent_multiples: List of precedent transaction multiples
            target_ebitda: Target EBITDA for multiples-based valuation
            synergies_npv: NPV of synergies
            methodology_weights: Weighting for each methodology
            
        Returns:
            ModelResult containing ValuationResult
        """
        start_time = time.perf_counter()
        
        terminal_growth = terminal_growth or self.config['dcf_terminal_growth']
        
        # DCF valuation
        dcf_value = self._calculate_dcf(target_fcf, discount_rate, terminal_growth)
        
        # Trading comparables
        trading_comps_value = self._calculate_trading_comps(
            comparable_multiples, target_ebitda
        ) if comparable_multiples and target_ebitda else dcf_value
        
        # Precedent transactions
        precedent_value = self._calculate_precedent_transactions(
            precedent_multiples, target_ebitda
        ) if precedent_multiples and target_ebitda else dcf_value
        
        # Synergy-adjusted
        synergy_adjusted = dcf_value + synergies_npv
        
        # Default methodology weights
        if methodology_weights is None:
            methodology_weights = {
                'dcf': 0.40,
                'trading_comps': 0.30,
                'precedent_transactions': 0.30
            }
        
        # Calculate weighted average
        recommended_value = (
            dcf_value * methodology_weights.get('dcf', 0.40) +
            trading_comps_value * methodology_weights.get('trading_comps', 0.30) +
            precedent_value * methodology_weights.get('precedent_transactions', 0.30)
        )
        
        # Valuation range (±15%)
        valuation_range = (recommended_value * 0.85, recommended_value * 1.15)
        
        # Control premium
        control_premium_range = self.config['control_premium_range']
        avg_control_premium = (control_premium_range[0] + control_premium_range[1]) / 2
        
        # Walk-away price (max we'd pay)
        walk_away_price = synergy_adjusted * 0.95  # Leave some value for acquirer
        
        result = ValuationResult(
            dcf_value=dcf_value,
            trading_comps_value=trading_comps_value,
            precedent_transactions_value=precedent_value,
            synergy_adjusted_value=synergy_adjusted,
            recommended_value=recommended_value,
            valuation_range=valuation_range,
            control_premium=avg_control_premium,
            methodology_weights=methodology_weights,
            walk_away_price=walk_away_price
        )
        
        execution_time_ms = self._track_performance("valuation_integration", start_time)
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(value=result, metadata=metadata, success=True)
    
    def _calculate_dcf(
        self,
        cash_flows: List[float],
        discount_rate: float,
        terminal_growth: float
    ) -> float:
        """Calculate DCF valuation."""
        pv_explicit = sum(
            cf / (1 + discount_rate) ** (i + 1)
            for i, cf in enumerate(cash_flows)
        )
        
        # Terminal value
        terminal_cf = cash_flows[-1] * (1 + terminal_growth)
        terminal_value = terminal_cf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** len(cash_flows)
        
        return pv_explicit + pv_terminal
    
    def _calculate_trading_comps(
        self,
        multiples: List[float],
        target_ebitda: float
    ) -> float:
        """Calculate trading comparables valuation."""
        median_multiple = np.median(multiples)
        return target_ebitda * median_multiple
    
    def _calculate_precedent_transactions(
        self,
        multiples: List[float],
        target_ebitda: float
    ) -> float:
        """Calculate precedent transactions valuation."""
        median_multiple = np.median(multiples)
        return target_ebitda * median_multiple
    
    def calculate_value(self, **kwargs) -> float:
        """Calculate recommended valuation."""
        result = self.calculate(**kwargs)
        return result.value.recommended_value
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs."""
        super().validate_inputs(**kwargs)
        
        if 'discount_rate' in kwargs:
            self.validate_positive(kwargs['discount_rate'], 'discount_rate')
        
        return True


__all__ = ["ValuationIntegrationModel", "ValuationResult"]