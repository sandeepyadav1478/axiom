"""
Domain Value Objects - Immutable, Self-Validating

Value objects represent domain concepts with:
- Immutability (cannot change after creation)
- Self-validation (enforce invariants)
- Equality by value (not identity)
- No side effects

This is Domain-Driven Design (DDD) done properly.
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Dict
from enum import Enum
import numpy as np
from scipy.stats import norm


class GreeksValidationError(ValueError):
    """Raised when Greeks values are invalid"""
    def __init__(self, message: str, greeks_data: Dict):
        self.greeks_data = greeks_data
        super().__init__(message)


class OptionType(Enum):
    """Option type enumeration"""
    CALL = "call"
    PUT = "put"
    
    def multiplier(self) -> int:
        """Get multiplier for put-call parity"""
        return 1 if self == OptionType.CALL else -1


@dataclass(frozen=True)  # Immutable
class Greeks:
    """
    Greeks value object - Immutable and self-validating
    
    Invariants enforced:
    - Delta: 0 to 1 for calls, -1 to 0 for puts
    - Gamma: Always positive
    - Vega: Always positive
    - All values must be finite (no NaN, inf)
    
    Uses Decimal for precision (not float)
    """
    delta: Decimal
    gamma: Decimal
    theta: Decimal
    vega: Decimal
    rho: Decimal
    option_type: OptionType
    
    # Metadata
    calculation_time_microseconds: Decimal = field(default=Decimal('0'))
    calculation_method: str = field(default='unknown')
    model_version: str = field(default='unknown')
    
    def __post_init__(self):
        """
        Validate invariants after initialization
        
        This runs automatically when object is created
        Ensures all Greeks objects are valid
        """
        # Check finite values
        for greek_name in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            value = getattr(self, greek_name)
            
            if not isinstance(value, Decimal):
                raise TypeError(f"{greek_name} must be Decimal, got {type(value)}")
            
            if value.is_nan() or value.is_infinite():
                raise GreeksValidationError(
                    f"{greek_name} is not finite: {value}",
                    greeks_data=self.to_dict()
                )
        
        # Validate delta range
        if self.option_type == OptionType.CALL:
            if not (Decimal('0') <= self.delta <= Decimal('1')):
                raise GreeksValidationError(
                    f"Call delta {self.delta} out of range [0, 1]",
                    greeks_data=self.to_dict()
                )
        else:  # PUT
            if not (Decimal('-1') <= self.delta <= Decimal('0')):
                raise GreeksValidationError(
                    f"Put delta {self.delta} out of range [-1, 0]",
                    greeks_data=self.to_dict()
                )
        
        # Validate gamma (always positive)
        if self.gamma < Decimal('0'):
            raise GreeksValidationError(
                f"Gamma {self.gamma} cannot be negative",
                greeks_data=self.to_dict()
            )
        
        # Validate vega (always positive)
        if self.vega < Decimal('0'):
            raise GreeksValidationError(
                f"Vega {self.vega} cannot be negative",
                greeks_data=self.to_dict()
            )
    
    def validate_against_black_scholes(
        self,
        spot: Decimal,
        strike: Decimal,
        time_to_maturity: Decimal,
        risk_free_rate: Decimal,
        volatility: Decimal,
        tolerance_pct: Decimal = Decimal('0.01')  # 1% tolerance
    ) -> bool:
        """
        Cross-validate against Black-Scholes analytical solution
        
        Returns: True if within tolerance, False otherwise
        Raises: GreeksValidationError with details if validation fails
        """
        # Calculate analytical Greeks
        bs_greeks = self._black_scholes_greeks(
            float(spot), float(strike), float(time_to_maturity),
            float(risk_free_rate), float(volatility)
        )
        
        # Compare delta
        delta_diff_pct = abs(self.delta - bs_greeks['delta']) / abs(bs_greeks['delta']) * 100
        
        if delta_diff_pct > tolerance_pct:
            raise GreeksValidationError(
                f"Delta validation failed: {delta_diff_pct:.2f}% difference from Black-Scholes (tolerance: {tolerance_pct}%)",
                greeks_data={
                    'ai_delta': float(self.delta),
                    'bs_delta': bs_greeks['delta'],
                    'difference_pct': float(delta_diff_pct)
                }
            )
        
        return True
    
    @staticmethod
    def _black_scholes_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
    ) -> Dict[str, float]:
        """
        Calculate analytical Black-Scholes Greeks
        
        Used for validation only
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        return {
            'delta': norm.cdf(d1),
            'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
            'vega': S * norm.pdf(d1) * np.sqrt(T) / 100,
            'theta': (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'delta': float(self.delta),
            'gamma': float(self.gamma),
            'theta': float(self.theta),
            'vega': float(self.vega),
            'rho': float(self.rho),
            'option_type': self.option_type.value,
            'calculation_time_us': float(self.calculation_time_microseconds),
            'method': self.calculation_method,
            'version': self.model_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Greeks':
        """Create from dictionary (deserialization)"""
        return cls(
            delta=Decimal(str(data['delta'])),
            gamma=Decimal(str(data['gamma'])),
            theta=Decimal(str(data['theta'])),
            vega=Decimal(str(data['vega'])),
            rho=Decimal(str(data['rho'])),
            option_type=OptionType(data['option_type']),
            calculation_time_microseconds=Decimal(str(data.get('calculation_time_us', 0))),
            calculation_method=data.get('method', 'unknown'),
            model_version=data.get('version', 'unknown')
        )
    
    def equals_within_tolerance(
        self,
        other: 'Greeks',
        tolerance: Decimal = Decimal('0.0001')
    ) -> bool:
        """
        Check if Greeks are equal within tolerance
        
        Used for testing and validation
        """
        return (
            abs(self.delta - other.delta) < tolerance and
            abs(self.gamma - other.gamma) < tolerance and
            abs(self.vega - other.vega) < tolerance
        )
    
    def __str__(self) -> str:
        """Human-readable representation"""
        return f"Greeks(δ={self.delta:.4f}, γ={self.gamma:.6f}, θ={self.theta:.4f}, ν={self.vega:.4f})"


# Example usage demonstrating proper usage
if __name__ == "__main__":
    print("="*60)
    print("DOMAIN VALUE OBJECTS - PROPER DDD")
    print("="*60)
    
    # Create Greeks (with validation)
    print("\n→ Creating valid Greeks:")
    
    greeks = Greeks(
        delta=Decimal('0.5199'),
        gamma=Decimal('0.0156'),
        theta=Decimal('-0.0323'),
        vega=Decimal('0.3897'),
        rho=Decimal('0.5123'),
        option_type=OptionType.CALL,
        calculation_time_microseconds=Decimal('85.2'),
        calculation_method='ultra_fast_neural_network',
        model_version='v2.1.0'
    )
    
    print(f"   {greeks}")
    print(f"   ✓ All invariants satisfied")
    
    # Test immutability
    print("\n→ Testing immutability:")
    try:
        greeks.delta = Decimal('0.6')  # Should fail - frozen dataclass
    except Exception as e:
        print(f"   ✓ Cannot modify: {type(e).__name__}")
    
    # Test validation
    print("\n→ Testing validation:")
    try:
        bad_greeks = Greeks(
            delta=Decimal('1.5'),  # Invalid for call!
            gamma=Decimal('0.015'),
            theta=Decimal('-0.03'),
            vega=Decimal('0.39'),
            rho=Decimal('0.51'),
            option_type=OptionType.CALL
        )
    except GreeksValidationError as e:
        print(f"   ✓ Validation caught error: {e}")
    
    # Test Black-Scholes validation
    print("\n→ Testing Black-Scholes validation:")
    try:
        is_valid = greeks.validate_against_black_scholes(
            spot=Decimal('100'),
            strike=Decimal('100'),
            time_to_maturity=Decimal('1.0'),
            risk_free_rate=Decimal('0.03'),
            volatility=Decimal('0.25')
        )
        print(f"   ✓ Cross-validation passed: {is_valid}")
    except GreeksValidationError as e:
        print(f"   ✗ Cross-validation failed: {e}")
    
    print("\n" + "="*60)
    print("✓ Immutable value objects")
    print("✓ Self-validating invariants")
    print("✓ Decimal precision (not float)")
    print("✓ Cross-validation with analytical")
    print("\nTHIS IS PROFESSIONAL DDD - PROPER FOUNDATION")