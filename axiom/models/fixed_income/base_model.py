"""
Base Classes for Fixed Income Models
=====================================

Provides base classes and data structures for all fixed income models.

Components:
- BaseFixedIncomeModel: Abstract base class for all fixed income models
- BondSpecification: Bond characteristics data structure
- BondPrice: Pricing result with all metrics
- YieldCurve: Yield curve representation
- DayCountConvention: Enum for day count conventions
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from axiom.models.base.base_model import BasePricingModel, ModelResult, ValidationError
from axiom.models.base.mixins import NumericalMethodsMixin, ValidationMixin
from axiom.core.logging.axiom_logger import get_logger


logger = get_logger("axiom.models.fixed_income.base")


class DayCountConvention(Enum):
    """Day count conventions for accrued interest calculation."""
    THIRTY_360 = "30/360"  # 30/360 (Bond Basis)
    ACTUAL_360 = "Actual/360"  # Actual/360 (Money Market)
    ACTUAL_365 = "Actual/365"  # Actual/365 (Fixed)
    ACTUAL_ACTUAL = "Actual/Actual"  # Actual/Actual (ISDA)
    ACTUAL_365L = "Actual/365L"  # Actual/365 Leap year
    THIRTY_360E = "30E/360"  # 30E/360 (Eurobond)
    THIRTY_360_ISDA = "30/360 ISDA"  # 30/360 ISDA


class CompoundingFrequency(Enum):
    """Compounding frequency for bond calculations."""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 0  # For continuous compounding


class BondType(Enum):
    """Type of bond."""
    FIXED_RATE = "fixed_rate"
    ZERO_COUPON = "zero_coupon"
    FLOATING_RATE = "floating_rate"
    INFLATION_LINKED = "inflation_linked"
    CALLABLE = "callable"
    PUTABLE = "putable"
    CONVERTIBLE = "convertible"
    PERPETUAL = "perpetual"


@dataclass
class BondSpecification:
    """
    Bond characteristics and specifications.
    
    Attributes:
        face_value: Par value of the bond (default 100)
        coupon_rate: Annual coupon rate (e.g., 0.05 for 5%)
        maturity_date: Bond maturity date
        issue_date: Bond issue date
        coupon_frequency: Coupon payments per year
        day_count: Day count convention
        bond_type: Type of bond
        callable: Whether bond is callable
        call_price: Call price if callable
        call_date: First call date if callable
        putable: Whether bond is putable
        put_price: Put price if putable
        put_date: First put date if putable
        floating_spread: Spread over reference rate for FRN (bps)
        inflation_index: Inflation index for TIPS
    """
    face_value: float = 100.0
    coupon_rate: float = 0.05
    maturity_date: datetime = field(default_factory=lambda: datetime(2025, 12, 31))
    issue_date: datetime = field(default_factory=lambda: datetime(2020, 1, 1))
    coupon_frequency: CompoundingFrequency = CompoundingFrequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.THIRTY_360
    bond_type: BondType = BondType.FIXED_RATE
    
    # Callable/Putable features
    callable: bool = False
    call_price: Optional[float] = None
    call_date: Optional[datetime] = None
    putable: bool = False
    put_price: Optional[float] = None
    put_date: Optional[datetime] = None
    
    # Floating rate features
    floating_spread: Optional[float] = None  # in basis points
    
    # Inflation-linked features
    inflation_index: Optional[str] = None
    
    def __post_init__(self):
        """Validate bond specification."""
        if self.face_value <= 0:
            raise ValidationError(f"Face value must be positive: {self.face_value}")
        if self.coupon_rate < 0:
            raise ValidationError(f"Coupon rate cannot be negative: {self.coupon_rate}")
        if self.maturity_date <= self.issue_date:
            raise ValidationError("Maturity date must be after issue date")
        if self.callable and (self.call_price is None or self.call_date is None):
            raise ValidationError("Callable bonds must have call price and call date")
        if self.putable and (self.put_price is None or self.put_date is None):
            raise ValidationError("Putable bonds must have put price and put date")


@dataclass
class BondPrice:
    """
    Bond pricing result with comprehensive analytics.
    
    Attributes:
        clean_price: Price without accrued interest
        dirty_price: Price with accrued interest
        accrued_interest: Accrued interest amount
        ytm: Yield to maturity
        current_yield: Current yield (coupon / price)
        macaulay_duration: Macaulay duration
        modified_duration: Modified duration
        convexity: Convexity
        dv01: Dollar value of 01 (price change per bp)
        settlement_date: Settlement date for pricing
        next_coupon_date: Next coupon payment date
        time_to_maturity: Years to maturity
    """
    clean_price: float
    dirty_price: float
    accrued_interest: float
    ytm: float
    current_yield: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    settlement_date: datetime
    next_coupon_date: Optional[datetime] = None
    time_to_maturity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "clean_price": round(self.clean_price, 4),
            "dirty_price": round(self.dirty_price, 4),
            "accrued_interest": round(self.accrued_interest, 4),
            "ytm": round(self.ytm, 6),
            "current_yield": round(self.current_yield, 6),
            "macaulay_duration": round(self.macaulay_duration, 4),
            "modified_duration": round(self.modified_duration, 4),
            "convexity": round(self.convexity, 4),
            "dv01": round(self.dv01, 4),
            "settlement_date": self.settlement_date.isoformat(),
            "next_coupon_date": self.next_coupon_date.isoformat() if self.next_coupon_date else None,
            "time_to_maturity": round(self.time_to_maturity, 4),
        }


@dataclass
class YieldCurve:
    """
    Yield curve representation.
    
    Attributes:
        tenors: Time points in years
        rates: Zero rates at each tenor
        model_type: Model used for construction (nelson_siegel, svensson, etc.)
        calibration_date: Date of curve calibration
        parameters: Model-specific parameters
        discount_factors: Discount factors at each tenor
        forward_rates: Forward rates between tenors
    """
    tenors: np.ndarray
    rates: np.ndarray
    model_type: str
    calibration_date: datetime
    parameters: Dict[str, float] = field(default_factory=dict)
    discount_factors: Optional[np.ndarray] = None
    forward_rates: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate and calculate derived values."""
        if len(self.tenors) != len(self.rates):
            raise ValidationError("Tenors and rates must have same length")
        if np.any(self.tenors < 0):
            raise ValidationError("Tenors must be non-negative")
        
        # Calculate discount factors if not provided
        if self.discount_factors is None:
            self.discount_factors = np.exp(-self.rates * self.tenors)
        
        # Calculate forward rates if not provided
        if self.forward_rates is None and len(self.tenors) > 1:
            # Forward rate between t1 and t2: f(t1,t2) = (r2*t2 - r1*t1)/(t2-t1)
            self.forward_rates = np.zeros(len(self.tenors) - 1)
            for i in range(len(self.tenors) - 1):
                t1, t2 = self.tenors[i], self.tenors[i + 1]
                r1, r2 = self.rates[i], self.rates[i + 1]
                if t2 > t1:
                    self.forward_rates[i] = (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def get_rate(self, tenor: float, interpolation: str = "linear") -> float:
        """
        Get zero rate at a specific tenor using interpolation.
        
        Args:
            tenor: Time point in years
            interpolation: Interpolation method (linear, cubic)
            
        Returns:
            Zero rate at the tenor
        """
        if interpolation == "linear":
            return np.interp(tenor, self.tenors, self.rates)
        elif interpolation == "cubic":
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(self.tenors, self.rates)
            return float(cs(tenor))
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")
    
    def get_discount_factor(self, tenor: float) -> float:
        """Get discount factor at a specific tenor."""
        rate = self.get_rate(tenor)
        return np.exp(-rate * tenor)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tenors": self.tenors.tolist(),
            "rates": self.rates.tolist(),
            "model_type": self.model_type,
            "calibration_date": self.calibration_date.isoformat(),
            "parameters": self.parameters,
            "discount_factors": self.discount_factors.tolist() if self.discount_factors is not None else None,
            "forward_rates": self.forward_rates.tolist() if self.forward_rates is not None else None,
        }


class BaseFixedIncomeModel(BasePricingModel, NumericalMethodsMixin, ValidationMixin):
    """
    Abstract base class for all fixed income models.
    
    Provides common functionality for:
    - Bond pricing
    - Yield calculations
    - Duration and convexity
    - Day count calculations
    - Cash flow generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize fixed income model.
        
        Args:
            config: Model-specific configuration
            **kwargs: Additional parameters
        """
        super().__init__(config=config, **kwargs)
        self.logger = get_logger(f"axiom.models.fixed_income.{self.__class__.__name__}")
    
    @abstractmethod
    def calculate_price(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        yield_rate: Optional[float] = None,
        **kwargs
    ) -> BondPrice:
        """
        Calculate bond price and analytics.
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date for pricing
            yield_rate: Yield to maturity (if known)
            **kwargs: Additional parameters
            
        Returns:
            BondPrice with comprehensive analytics
        """
        pass
    
    @abstractmethod
    def calculate_yield(
        self,
        bond: BondSpecification,
        price: float,
        settlement_date: datetime,
        **kwargs
    ) -> float:
        """
        Calculate yield to maturity given price.
        
        Args:
            bond: Bond specification
            price: Bond price (clean or dirty)
            settlement_date: Settlement date
            **kwargs: Additional parameters
            
        Returns:
            Yield to maturity
        """
        pass
    
    def calculate_accrued_interest(
        self,
        bond: BondSpecification,
        settlement_date: datetime,
        last_coupon_date: datetime,
        next_coupon_date: Optional[datetime]
    ) -> float:
        """
        Calculate accrued interest using appropriate day count convention.
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date
            last_coupon_date: Last coupon payment date
            next_coupon_date: Next coupon payment date (None for zero-coupon bonds)
            
        Returns:
            Accrued interest amount
        """
        # Zero-coupon bonds have no accrued interest
        if bond.coupon_rate == 0.0 or next_coupon_date is None:
            return 0.0
        
        coupon_payment = bond.face_value * bond.coupon_rate / bond.coupon_frequency.value
        
        # Calculate days using day count convention
        days_accrued = self._day_count_fraction(
            last_coupon_date,
            settlement_date,
            bond.day_count
        )
        
        days_in_period = self._day_count_fraction(
            last_coupon_date,
            next_coupon_date,
            bond.day_count
        )
        
        if days_in_period == 0:
            return 0.0
        
        return coupon_payment * (days_accrued / days_in_period)
    
    def _day_count_fraction(
        self,
        start_date: datetime,
        end_date: datetime,
        convention: DayCountConvention
    ) -> float:
        """
        Calculate day count fraction using specified convention.
        
        Args:
            start_date: Start date
            end_date: End date
            convention: Day count convention
            
        Returns:
            Day count fraction
        """
        if convention == DayCountConvention.THIRTY_360:
            # 30/360 (Bond Basis)
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date cannot be None")
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            
            # Adjust day-of-month for 30/360
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 >= 30:
                d2 = 30
            
            return ((y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1))
        
        elif convention == DayCountConvention.ACTUAL_360:
            # Actual/360
            days = (end_date - start_date).days
            return days
        
        elif convention == DayCountConvention.ACTUAL_365:
            # Actual/365
            days = (end_date - start_date).days
            return days
        
        elif convention == DayCountConvention.ACTUAL_ACTUAL:
            # Actual/Actual (ISDA)
            days = (end_date - start_date).days
            # More complex calculation considering leap years
            # Simplified version here
            return days
        
        else:
            # Default to actual days
            return (end_date - start_date).days
    
    def generate_cash_flows(
        self,
        bond: BondSpecification,
        settlement_date: datetime
    ) -> List[Tuple[datetime, float]]:
        """
        Generate bond cash flows from settlement date to maturity.
        
        Args:
            bond: Bond specification
            settlement_date: Settlement date
            
        Returns:
            List of (date, amount) tuples
        """
        cash_flows = []
        
        # Generate coupon payment dates
        current_date = bond.maturity_date
        coupon_payment = bond.face_value * bond.coupon_rate / bond.coupon_frequency.value
        
        # Work backwards from maturity
        while current_date > settlement_date:
            if current_date >= settlement_date:
                # Add coupon payment
                if bond.bond_type != BondType.ZERO_COUPON:
                    cash_flows.append((current_date, coupon_payment))
            
            # Move to previous coupon date
            if bond.coupon_frequency == CompoundingFrequency.SEMI_ANNUAL:
                months = 6
            elif bond.coupon_frequency == CompoundingFrequency.QUARTERLY:
                months = 3
            elif bond.coupon_frequency == CompoundingFrequency.MONTHLY:
                months = 1
            else:  # Annual
                months = 12
            
            # Subtract months
            new_month = current_date.month - months
            new_year = current_date.year
            while new_month <= 0:
                new_month += 12
                new_year -= 1
            
            try:
                current_date = current_date.replace(year=new_year, month=new_month)
            except ValueError:
                # Handle day-of-month issues
                current_date = current_date.replace(
                    year=new_year,
                    month=new_month,
                    day=min(current_date.day, 28)
                )
        
        # Add principal repayment at maturity
        if cash_flows and cash_flows[-1][0] == bond.maturity_date:
            # Combine with last coupon
            last_date, last_amount = cash_flows[-1]
            cash_flows[-1] = (last_date, last_amount + bond.face_value)
        else:
            cash_flows.append((bond.maturity_date, bond.face_value))
        
        # Sort by date
        cash_flows.sort(key=lambda x: x[0])
        
        return cash_flows
    
    def price(self, **kwargs) -> float:
        """
        Calculate price - required by BasePricingModel.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            Bond price
        """
        result = self.calculate_price(**kwargs)
        return result.clean_price
    
    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate model inputs.
        
        Args:
            **kwargs: Model parameters to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if 'bond' in kwargs:
            bond = kwargs['bond']
            if not isinstance(bond, BondSpecification):
                raise ValidationError("bond must be BondSpecification instance")
        
        if 'yield_rate' in kwargs and kwargs['yield_rate'] is not None:
            self.validate_finite(kwargs['yield_rate'], "yield_rate")
        
        if 'price' in kwargs:
            self.validate_positive(kwargs['price'], "price")
        
        return True


__all__ = [
    "DayCountConvention",
    "CompoundingFrequency",
    "BondType",
    "BondSpecification",
    "BondPrice",
    "YieldCurve",
    "BaseFixedIncomeModel",
]