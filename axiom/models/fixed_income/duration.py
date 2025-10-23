"""
Duration & Convexity Models
============================

Institutional-grade duration and convexity analytics with:
- <8ms execution for full duration/convexity analytics
- All duration measures (Macaulay, Modified, Effective, Key Rate)
- Convexity calculations (standard and effective)
- Risk metrics (DV01, PVBP, DTS)
- Hedging applications

Mathematical Formulas:
---------------------

Macaulay Duration:
D_Mac = Σ(t * CF_t * PV_t) / Price

where:
- t = Time to cash flow (years)
- CF_t = Cash flow at time t
- PV_t = Present value of CF_t
- Price = Total present value

Modified Duration:
D_Mod = D_Mac / (1 + y/n)

where:
- y = Yield to maturity
- n = Compounding frequency

Effective Duration (for bonds with embedded options):
D_Eff = (P- - P+) / (2 * P0 * Δy)

where:
- P- = Price when yield decreases by Δy
- P+ = Price when yield increases by Δy
- P0 = Current price
- Δy = Yield change (typically 0.01 or 100 bps)

Convexity:
C = Σ(t * (t+1) * CF_t * PV_t) / (Price * (1+y)²)

Effective Convexity:
C_Eff = (P+ + P- - 2*P0) / (P0 * Δy²)

Key Rate Duration:
KRD_i = (P-(shift at tenor i) - P+(shift at tenor i)) / (2 * P0 * Δy)

Dollar Value of 01 (DV01):
DV01 = -D_Mod * Price / 10000

Price Value of Basis Point (PVBP):
PVBP = D_Mod * Price / 10000

Duration Times Spread (DTS):
DTS = D_Mod * Spread
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

from axiom.models.fixed_income.base_model import (
    BaseFixedIncomeModel,
    BondSpecification,
    YieldCurve,
    ValidationError
)
from axiom.models.base.base_model import ModelResult
from axiom.core.logging.axiom_logger import get_logger


logger = get_logger("axiom.models.fixed_income.duration")


@dataclass
class DurationMetrics:
    """
    Container for all duration and convexity metrics.
    
    Attributes:
        macaulay_duration: Weighted average time to cash flows (years)
        modified_duration: Price sensitivity to yield changes
        effective_duration: Duration accounting for embedded options
        key_rate_durations: Sensitivities to specific maturity rates
        convexity: Second-order price sensitivity
        effective_convexity: Convexity for bonds with options
        dv01: Dollar value of 01 basis point
        pvbp: Price value of basis point
        duration_times_spread: Duration times spread
    """
    macaulay_duration: float
    modified_duration: float
    effective_duration: Optional[float] = None
    key_rate_durations: Optional[Dict[float, float]] = None
    convexity: float = 0.0
    effective_convexity: Optional[float] = None
    dv01: float = 0.0
    pvbp: float = 0.0
    duration_times_spread: float = 0.0
    
    # Additional metrics
    fisher_weil_duration: Optional[float] = None
    spread_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "macaulay_duration": round(self.macaulay_duration, 4),
            "modified_duration": round(self.modified_duration, 4),
            "convexity": round(self.convexity, 4),
            "dv01": round(self.dv01, 4),
            "pvbp": round(self.pvbp, 4),
        }
        
        if self.effective_duration is not None:
            result["effective_duration"] = round(self.effective_duration, 4)
        
        if self.effective_convexity is not None:
            result["effective_convexity"] = round(self.effective_convexity, 4)
        
        if self.key_rate_durations is not None:
            result["key_rate_durations"] = {
                str(k): round(v, 4) for k, v in self.key_rate_durations.items()
            }
        
        if self.fisher_weil_duration is not None:
            result["fisher_weil_duration"] = round(self.fisher_weil_duration, 4)
        
        if self.spread_duration is not None:
            result["spread_duration"] = round(self.spread_duration, 4)
        
        return result


class DurationCalculator(BaseFixedIncomeModel):
    """
    Comprehensive duration and convexity calculator.
    
    Features:
    - All duration measures (Macaulay, Modified, Effective, Key Rate)
    - Convexity calculations
    - Risk metrics (DV01, PVBP)
    - <8ms execution time
    - Support for callable/putable bonds
    
    Example:
        >>> calculator = DurationCalculator()
        >>> metrics = calculator.calculate_all_metrics(
        ...     bond=bond_spec,
        ...     price=100.0,
        ...     yield_rate=0.05,
        ...     settlement_date=datetime.now()
        ... )
        >>> print(f"Modified Duration: {metrics.modified_duration:.4f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize duration calculator.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)
        
        # Extract config
        self.shock_size_bps = self.config.get('shock_size_bps', 1.0)
        self.key_rate_tenors = self.config.get('key_rate_tenors', [1, 2, 3, 5, 7, 10, 20, 30])
        
        if self.enable_logging:
            self.logger.info("Initialized DurationCalculator")
    
    def calculate_macaulay_duration(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        yield_rate: float,
        **kwargs
    ) -> float:
        """
        Calculate Macaulay duration (weighted average time to cash flows).
        
        Formula: D_Mac = Σ(t * CF_t * PV_t) / Price
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date
            yield_rate: Yield to maturity
            **kwargs: Additional parameters
            
        Returns:
            Macaulay duration in years
        """
        # Generate cash flows
        cash_flows = self.generate_cash_flows(bond, settlement_date)
        
        if not cash_flows:
            return 0.0
        
        frequency = bond.coupon_frequency.value
        weighted_time = 0.0
        total_pv = 0.0
        
        for cf_date, cf_amount in cash_flows:
            if cf_date > settlement_date:
                # Time to cash flow in years
                time_to_cf = (cf_date - settlement_date).days / 365.25
                
                # Discount factor
                discount_factor = (1 + yield_rate / frequency) ** (frequency * time_to_cf)
                pv = cf_amount / discount_factor
                
                # Weighted time
                weighted_time += time_to_cf * pv
                total_pv += pv
        
        if total_pv == 0:
            return 0.0
        
        return weighted_time / total_pv
    
    def calculate_modified_duration(
        self,
        macaulay_duration: float,
        yield_rate: float,
        frequency: int = 2
    ) -> float:
        """
        Calculate modified duration from Macaulay duration.
        
        Formula: D_Mod = D_Mac / (1 + y/n)
        
        Args:
            macaulay_duration: Macaulay duration
            yield_rate: Yield to maturity
            frequency: Compounding frequency
            
        Returns:
            Modified duration
        """
        return macaulay_duration / (1 + yield_rate / frequency)
    
    def calculate_effective_duration(
        self,
        bond: BondSpecification,
        base_price: float,
        settlement_date: datetime,
        base_yield: float,
        shock_bps: Optional[float] = None,
        pricing_function: Optional[callable] = None,
        **kwargs
    ) -> float:
        """
        Calculate effective duration using numerical differentiation.
        
        Formula: D_Eff = (P- - P+) / (2 * P0 * Δy)
        
        Args:
            bond: Bond specification
            base_price: Current bond price
            settlement_date: Settlement date
            base_yield: Current yield
            shock_bps: Yield shock in basis points (default from config)
            pricing_function: Custom pricing function for bonds with options
            **kwargs: Additional parameters
            
        Returns:
            Effective duration
        """
        if shock_bps is None:
            shock_bps = self.shock_size_bps
        
        shock = shock_bps / 10000  # Convert bps to decimal
        
        # Calculate prices at shocked yields
        if pricing_function:
            price_down = pricing_function(bond, settlement_date, base_yield - shock, **kwargs)
            price_up = pricing_function(bond, settlement_date, base_yield + shock, **kwargs)
        else:
            # Use simple pricing
            price_down = self._price_bond_simple(bond, settlement_date, base_yield - shock)
            price_up = self._price_bond_simple(bond, settlement_date, base_yield + shock)
        
        # Effective duration formula
        effective_duration = (price_down - price_up) / (2 * base_price * shock)
        
        return effective_duration
    
    def calculate_key_rate_durations(
        self,
        bond: BondSpecification,
        base_price: float,
        settlement_date: datetime,
        yield_curve: YieldCurve,
        key_rate_tenors: Optional[List[float]] = None,
        shock_bps: Optional[float] = None,
        **kwargs
    ) -> Dict[float, float]:
        """
        Calculate key rate durations (sensitivities to specific maturity rates).
        
        Formula: KRD_i = (P-(shift at tenor i) - P+(shift at tenor i)) / (2 * P0 * Δy)
        
        Args:
            bond: Bond specification
            base_price: Current bond price
            settlement_date: Settlement date
            yield_curve: Yield curve object
            key_rate_tenors: Tenors for key rates (default from config)
            shock_bps: Yield shock in basis points
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping tenor to key rate duration
        """
        if key_rate_tenors is None:
            key_rate_tenors = self.key_rate_tenors
        
        if shock_bps is None:
            shock_bps = self.shock_size_bps
        
        shock = shock_bps / 10000
        
        krds = {}
        
        # Generate cash flows
        cash_flows = self.generate_cash_flows(bond, settlement_date)
        
        for tenor in key_rate_tenors:
            # Shift yield curve at this tenor
            shifted_curve_down = self._shift_curve_at_tenor(
                yield_curve, tenor, -shock
            )
            shifted_curve_up = self._shift_curve_at_tenor(
                yield_curve, tenor, shock
            )
            
            # Price with shifted curves
            price_down = self._price_with_curve(
                cash_flows, settlement_date, shifted_curve_down
            )
            price_up = self._price_with_curve(
                cash_flows, settlement_date, shifted_curve_up
            )
            
            # Calculate KRD
            krd = (price_down - price_up) / (2 * base_price * shock)
            krds[tenor] = krd
        
        return krds
    
    def calculate_convexity(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        yield_rate: float,
        **kwargs
    ) -> float:
        """
        Calculate convexity (second derivative of price with respect to yield).
        
        Formula: C = Σ(t * (t+1) * CF_t * PV_t) / (Price * (1+y)²)
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date
            yield_rate: Yield to maturity
            **kwargs: Additional parameters
            
        Returns:
            Convexity
        """
        # Generate cash flows
        cash_flows = self.generate_cash_flows(bond, settlement_date)
        
        if not cash_flows:
            return 0.0
        
        frequency = bond.coupon_frequency.value
        weighted_time_squared = 0.0
        total_pv = 0.0
        
        for cf_date, cf_amount in cash_flows:
            if cf_date > settlement_date:
                # Time to cash flow
                time_to_cf = (cf_date - settlement_date).days / 365.25
                
                # Discount factor
                discount_factor = (1 + yield_rate / frequency) ** (frequency * time_to_cf)
                pv = cf_amount / discount_factor
                
                # Number of periods
                n_periods = frequency * time_to_cf
                
                # Weighted time squared: t(t+1)
                weighted_time_squared += n_periods * (n_periods + 1) * pv
                total_pv += pv
        
        if total_pv == 0:
            return 0.0
        
        # Convexity formula
        convexity = weighted_time_squared / (total_pv * (1 + yield_rate / frequency) ** 2)
        
        # Adjust for frequency
        return convexity / (frequency ** 2)
    
    def calculate_effective_convexity(
        self,
        bond: BondSpecification,
        base_price: float,
        settlement_date: datetime,
        base_yield: float,
        shock_bps: Optional[float] = None,
        pricing_function: Optional[callable] = None,
        **kwargs
    ) -> float:
        """
        Calculate effective convexity using numerical differentiation.
        
        Formula: C_Eff = (P+ + P- - 2*P0) / (P0 * Δy²)
        
        Args:
            bond: Bond specification
            base_price: Current bond price
            settlement_date: Settlement date
            base_yield: Current yield
            shock_bps: Yield shock in basis points
            pricing_function: Custom pricing function
            **kwargs: Additional parameters
            
        Returns:
            Effective convexity
        """
        if shock_bps is None:
            shock_bps = self.shock_size_bps
        
        shock = shock_bps / 10000
        
        # Calculate prices at shocked yields
        if pricing_function:
            price_down = pricing_function(bond, settlement_date, base_yield - shock, **kwargs)
            price_up = pricing_function(bond, settlement_date, base_yield + shock, **kwargs)
        else:
            price_down = self._price_bond_simple(bond, settlement_date, base_yield - shock)
            price_up = self._price_bond_simple(bond, settlement_date, base_yield + shock)
        
        # Effective convexity formula
        convexity = (price_up + price_down - 2 * base_price) / (base_price * shock ** 2)
        
        return convexity
    
    def calculate_dv01(
        self,
        modified_duration: float,
        price: float
    ) -> float:
        """
        Calculate DV01 (dollar value of 01 basis point).
        
        Formula: DV01 = -D_Mod * Price / 10000
        
        Args:
            modified_duration: Modified duration
            price: Bond price
            
        Returns:
            DV01
        """
        return modified_duration * price / 10000
    
    def calculate_pvbp(
        self,
        modified_duration: float,
        price: float
    ) -> float:
        """
        Calculate PVBP (price value of basis point).
        
        Same as DV01.
        
        Args:
            modified_duration: Modified duration
            price: Bond price
            
        Returns:
            PVBP
        """
        return self.calculate_dv01(modified_duration, price)
    
    def calculate_fisher_weil_duration(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        yield_curve: YieldCurve,
        **kwargs
    ) -> float:
        """
        Calculate Fisher-Weil duration (for non-flat yield curves).
        
        Uses spot rates specific to each cash flow maturity.
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date
            yield_curve: Yield curve
            **kwargs: Additional parameters
            
        Returns:
            Fisher-Weil duration
        """
        cash_flows = self.generate_cash_flows(bond, settlement_date)
        
        if not cash_flows:
            return 0.0
        
        weighted_time = 0.0
        total_pv = 0.0
        
        for cf_date, cf_amount in cash_flows:
            if cf_date > settlement_date:
                # Time to cash flow
                time_to_cf = (cf_date - settlement_date).days / 365.25
                
                # Get spot rate from curve
                spot_rate = yield_curve.get_rate(time_to_cf)
                
                # Discount factor using spot rate
                discount_factor = (1 + spot_rate) ** time_to_cf
                pv = cf_amount / discount_factor
                
                # Derivative of discount factor
                dpv_dy = -time_to_cf * pv
                
                weighted_time += dpv_dy
                total_pv += pv
        
        if total_pv == 0:
            return 0.0
        
        return -weighted_time / total_pv
    
    def calculate_all_metrics(
        self,
        bond: BondSpecification,
        price: float,
        yield_rate: float,
        settlement_date: datetime,
        yield_curve: Optional[YieldCurve] = None,
        calculate_key_rates: bool = False,
        calculate_effective: bool = False,
        pricing_function: Optional[callable] = None,
        spread: Optional[float] = None,
        **kwargs
    ) -> DurationMetrics:
        """
        Calculate all duration and convexity metrics.
        
        Args:
            bond: Bond specification
            price: Bond price
            yield_rate: Yield to maturity
            settlement_date: Settlement date
            yield_curve: Yield curve (for key rate durations)
            calculate_key_rates: Whether to calculate key rate durations
            calculate_effective: Whether to calculate effective duration/convexity
            pricing_function: Custom pricing function for options
            spread: Credit spread for DTS calculation
            **kwargs: Additional parameters
            
        Returns:
            DurationMetrics with all calculated metrics
        """
        start_time = time.perf_counter()
        
        # Calculate Macaulay duration
        macaulay = self.calculate_macaulay_duration(
            bond, settlement_date, yield_rate
        )
        
        # Calculate modified duration
        modified = self.calculate_modified_duration(
            macaulay, yield_rate, bond.coupon_frequency.value
        )
        
        # Calculate convexity
        convexity = self.calculate_convexity(
            bond, settlement_date, yield_rate
        )
        
        # Calculate risk metrics
        dv01 = self.calculate_dv01(modified, price)
        pvbp = dv01  # Same as DV01
        
        # Optional: Effective duration and convexity
        effective_duration = None
        effective_convexity = None
        if calculate_effective:
            effective_duration = self.calculate_effective_duration(
                bond, price, settlement_date, yield_rate,
                pricing_function=pricing_function
            )
            effective_convexity = self.calculate_effective_convexity(
                bond, price, settlement_date, yield_rate,
                pricing_function=pricing_function
            )
        
        # Optional: Key rate durations
        key_rate_durations = None
        if calculate_key_rates and yield_curve is not None:
            key_rate_durations = self.calculate_key_rate_durations(
                bond, price, settlement_date, yield_curve
            )
        
        # Optional: Fisher-Weil duration
        fisher_weil = None
        if yield_curve is not None:
            fisher_weil = self.calculate_fisher_weil_duration(
                bond, settlement_date, yield_curve
            )
        
        # Duration times spread
        dts = modified * spread if spread is not None else 0.0
        
        # Spread duration (simplified - same as modified for most bonds)
        spread_duration = modified if spread is not None else None
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        if self.enable_logging:
            self.logger.debug(
                "Duration metrics calculated",
                macaulay=round(macaulay, 4),
                modified=round(modified, 4),
                convexity=round(convexity, 4),
                dv01=round(dv01, 4),
                execution_time_ms=round(execution_time_ms, 3)
            )
        
        return DurationMetrics(
            macaulay_duration=macaulay,
            modified_duration=modified,
            effective_duration=effective_duration,
            key_rate_durations=key_rate_durations,
            convexity=convexity,
            effective_convexity=effective_convexity,
            dv01=dv01,
            pvbp=pvbp,
            duration_times_spread=dts,
            fisher_weil_duration=fisher_weil,
            spread_duration=spread_duration
        )
    
    def _price_bond_simple(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        yield_rate: float
    ) -> float:
        """Simple bond pricing for numerical calculations."""
        cash_flows = self.generate_cash_flows(bond, settlement_date)
        frequency = bond.coupon_frequency.value
        
        price = 0.0
        for cf_date, cf_amount in cash_flows:
            if cf_date > settlement_date:
                time_to_cf = (cf_date - settlement_date).days / 365.25
                discount_factor = (1 + yield_rate / frequency) ** (frequency * time_to_cf)
                price += cf_amount / discount_factor
        
        return price
    
    def _shift_curve_at_tenor(
        self,
        curve: YieldCurve,
        tenor: float,
        shift: float
    ) -> YieldCurve:
        """Shift yield curve at a specific tenor."""
        # Create new rates array with shift at tenor
        new_rates = curve.rates.copy()
        
        # Find closest tenor index
        idx = np.abs(curve.tenors - tenor).argmin()
        new_rates[idx] += shift
        
        # Create new curve
        from axiom.models.fixed_income.base_model import YieldCurve as YC
        return YC(
            tenors=curve.tenors.copy(),
            rates=new_rates,
            model_type=curve.model_type,
            calibration_date=curve.calibration_date,
            parameters=curve.parameters
        )
    
    def _price_with_curve(
        self,
        cash_flows: List[Tuple[datetime, float]],
        settlement_date: datetime,
        curve: YieldCurve
    ) -> float:
        """Price bond using yield curve."""
        price = 0.0
        
        for cf_date, cf_amount in cash_flows:
            if cf_date > settlement_date:
                time_to_cf = (cf_date - settlement_date).days / 365.25
                discount_factor = curve.get_discount_factor(time_to_cf)
                price += cf_amount * discount_factor
        
        return price
    
    def calculate_price(self, **kwargs):
        """Not primary function for this model."""
        raise NotImplementedError("Use calculate_all_metrics()")
    
    def calculate_yield(self, **kwargs):
        """Not primary function for this model."""
        raise NotImplementedError("Use calculate_all_metrics()")
    
    def calculate(self, **kwargs) -> ModelResult:
        """Calculate method required by base class."""
        start_time = time.perf_counter()
        
        try:
            metrics = self.calculate_all_metrics(**kwargs)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            metadata = self._create_metadata(execution_time_ms)
            
            return ModelResult(
                value=metrics.to_dict(),
                metadata=metadata,
                success=True
            )
        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            metadata = self._create_metadata(execution_time_ms, warnings=[str(e)])
            
            return ModelResult(
                value=None,
                metadata=metadata,
                success=False,
                error_message=str(e)
            )


class DurationHedger:
    """
    Duration hedging utilities.
    
    Provides methods for:
    - Duration matching
    - Immunization strategies
    - Hedge ratio calculation
    - Barbell vs bullet analysis
    """
    
    def __init__(self):
        """Initialize hedger."""
        self.logger = get_logger("axiom.models.fixed_income.duration.hedger")
    
    def calculate_hedge_ratio(
        self,
        target_duration: float,
        hedge_duration: float,
        target_value: float
    ) -> float:
        """
        Calculate hedge ratio for duration-neutral position.
        
        Formula: Hedge_Ratio = -(Target_Duration * Target_Value) / (Hedge_Duration * Hedge_Value)
        
        Args:
            target_duration: Duration of position to hedge
            hedge_duration: Duration of hedging instrument
            target_value: Value of position to hedge
            
        Returns:
            Hedge ratio (notional of hedge / notional of target)
        """
        if hedge_duration == 0:
            raise ValueError("Hedge duration cannot be zero")
        
        ratio = -(target_duration / hedge_duration) * target_value
        
        return ratio
    
    def immunize_portfolio(
        self,
        target_horizon: float,
        available_bonds: List[Tuple[float, float]],  # (duration, yield)
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Design immunized portfolio matching target horizon.
        
        Immunization conditions:
        1. Duration = Investment horizon
        2. Convexity minimized
        3. Present value = Target value
        
        Args:
            target_horizon: Investment horizon (years)
            available_bonds: List of (duration, yield) pairs
            constraints: Portfolio constraints
            
        Returns:
            Dictionary with portfolio weights and characteristics
        """
        # Simplified immunization - find bond closest to target duration
        closest_bond = min(
            enumerate(available_bonds),
            key=lambda x: abs(x[1][0] - target_horizon)
        )
        
        bond_idx, (duration, yield_rate) = closest_bond
        
        self.logger.info(
            "Immunization portfolio",
            target_horizon=target_horizon,
            selected_duration=duration,
            selected_yield=yield_rate
        )
        
        return {
            "weights": [1.0 if i == bond_idx else 0.0 for i in range(len(available_bonds))],
            "duration": duration,
            "yield": yield_rate,
            "duration_gap": abs(duration - target_horizon)
        }
    
    def analyze_barbell_vs_bullet(
        self,
        short_duration: float,
        long_duration: float,
        bullet_duration: float,
        short_yield: float,
        long_yield: float,
        bullet_yield: float
    ) -> Dict[str, Any]:
        """
        Analyze barbell vs bullet strategy.
        
        Barbell: Combine short and long duration bonds
        Bullet: Single bond at target duration
        
        Args:
            short_duration: Duration of short bond
            long_duration: Duration of long bond
            bullet_duration: Duration of bullet bond
            short_yield: Yield of short bond
            long_yield: Yield of long bond
            bullet_yield: Yield of bullet bond
            
        Returns:
            Analysis results
        """
        # Calculate barbell weights to match bullet duration
        # w1 * D1 + w2 * D2 = D_bullet
        # w1 + w2 = 1
        # Solving: w1 = (D2 - D_bullet) / (D2 - D1)
        
        if long_duration == short_duration:
            raise ValueError("Long and short durations must be different")
        
        weight_short = (long_duration - bullet_duration) / (long_duration - short_duration)
        weight_long = 1 - weight_short
        
        # Barbell yield (weighted average)
        barbell_yield = weight_short * short_yield + weight_long * long_yield
        
        # Yield advantage
        yield_advantage = barbell_yield - bullet_yield
        
        self.logger.info(
            "Barbell vs Bullet analysis",
            barbell_yield=round(barbell_yield, 6),
            bullet_yield=round(bullet_yield, 6),
            yield_advantage_bps=round(yield_advantage * 10000, 2)
        )
        
        return {
            "barbell_weights": {
                "short": weight_short,
                "long": weight_long
            },
            "barbell_yield": barbell_yield,
            "bullet_yield": bullet_yield,
            "yield_advantage_bps": yield_advantage * 10000,
            "strategy_recommendation": "Barbell" if yield_advantage > 0 else "Bullet"
        }


# Convenience functions

def calculate_duration(
    coupon_rate: float,
    years_to_maturity: float,
    yield_rate: float,
    frequency: int = 2,
    face_value: float = 100.0
) -> Tuple[float, float]:
    """
    Quick duration calculation.
    
    Args:
        coupon_rate: Annual coupon rate
        years_to_maturity: Years to maturity
        yield_rate: Yield to maturity
        frequency: Coupon frequency
        face_value: Par value
        
    Returns:
        Tuple of (macaulay_duration, modified_duration)
    """
    from datetime import datetime, timedelta
    from axiom.models.fixed_income.base_model import BondSpecification, CompoundingFrequency
    
    settlement = datetime.now()
    maturity = settlement + timedelta(days=int(years_to_maturity * 365))
    
    bond = BondSpecification(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_date=maturity,
        issue_date=settlement,
        coupon_frequency=CompoundingFrequency(frequency)
    )
    
    calculator = DurationCalculator()
    macaulay = calculator.calculate_macaulay_duration(bond, settlement, yield_rate)
    modified = calculator.calculate_modified_duration(macaulay, yield_rate, frequency)
    
    return macaulay, modified


__all__ = [
    "DurationCalculator",
    "DurationMetrics",
    "DurationHedger",
    "calculate_duration",
]