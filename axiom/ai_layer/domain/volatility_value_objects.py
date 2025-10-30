"""
Volatility Analysis Domain Value Objects

Immutable value objects for volatility forecasting and analysis domain.
Following DDD principles - these capture volatility forecasts, regimes, and surfaces.

Key concepts:
- Value objects are immutable (frozen dataclasses)
- Self-validating (fail fast on invalid forecasts)
- Rich behavior (regime analysis, arbitrage detection)
- Type-safe (using Decimal for precision, Enum for regimes)

These represent volatility as a first-class domain concept.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from enum import Enum


class VolatilityRegime(str, Enum):
    """Market volatility regime"""
    LOW_VOL = "low_vol"  # VIX < 15
    NORMAL = "normal"  # VIX 15-25
    HIGH_VOL = "high_vol"  # VIX 25-35
    CRISIS = "crisis"  # VIX > 35


class ForecastHorizon(str, Enum):
    """Volatility forecast horizon"""
    INTRADAY = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"


class VolatilityModel(str, Enum):
    """Volatility forecasting model"""
    TRANSFORMER = "transformer"
    GARCH = "garch"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class VolatilityForecast:
    """
    Immutable volatility forecast
    
    AI-powered volatility prediction with confidence
    """
    underlying: str
    forecast_vol: Decimal
    horizon: ForecastHorizon
    
    # Confidence and regime
    confidence: Decimal  # 0-1
    regime: VolatilityRegime
    
    # Components (for transparency)
    transformer_prediction: Decimal
    garch_prediction: Optional[Decimal] = None
    lstm_prediction: Optional[Decimal] = None
    
    # Sentiment impact
    sentiment_impact: Decimal = Decimal('0')  # -1 to +1
    
    # Model used
    model_type: VolatilityModel = VolatilityModel.ENSEMBLE
    
    # Metadata
    prediction_time_ms: Decimal = Decimal('0')
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate forecast"""
        if self.forecast_vol < Decimal('0.01') or self.forecast_vol > Decimal('2.0'):
            raise ValueError(f"Forecast volatility out of range: {self.forecast_vol}")
        
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
        
        if not (Decimal('-1') <= self.sentiment_impact <= Decimal('1')):
            raise ValueError("Sentiment impact must be between -1 and 1")
    
    def is_high_confidence(self, threshold: Decimal = Decimal('0.75')) -> bool:
        """Check if forecast has high confidence"""
        return self.confidence >= threshold
    
    def is_crisis_vol(self) -> bool:
        """Check if forecasting crisis-level volatility"""
        return self.regime == VolatilityRegime.CRISIS
    
    def get_regime_multiplier(self) -> Decimal:
        """Get regime adjustment multiplier"""
        multipliers = {
            VolatilityRegime.LOW_VOL: Decimal('0.8'),
            VolatilityRegime.NORMAL: Decimal('1.0'),
            VolatilityRegime.HIGH_VOL: Decimal('1.2'),
            VolatilityRegime.CRISIS: Decimal('1.5')
        }
        return multipliers[self.regime]
    
    def get_model_agreement(self) -> Decimal:
        """Calculate agreement between models (coefficient of variation)"""
        if self.garch_prediction and self.lstm_prediction:
            import statistics
            predictions = [
                float(self.transformer_prediction),
                float(self.garch_prediction),
                float(self.lstm_prediction)
            ]
            mean = statistics.mean(predictions)
            stdev = statistics.stdev(predictions)
            return Decimal(str(stdev / mean if mean > 0 else 0))
        return Decimal('0')


@dataclass(frozen=True)
class VolatilitySurfacePoint:
    """
    Single point on volatility surface
    
    Immutable (strike, expiry, vol) triple
    """
    strike: Decimal
    expiry: datetime
    implied_vol: Decimal
    
    # Market data
    bid_vol: Optional[Decimal] = None
    ask_vol: Optional[Decimal] = None
    
    # Quality
    data_quality: Decimal = Decimal('1.0')
    
    # Metadata
    source: str = "market"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate surface point"""
        if self.strike <= Decimal('0'):
            raise ValueError("Strike must be positive")
        
        if self.implied_vol <= Decimal('0') or self.implied_vol > Decimal('2.0'):
            raise ValueError(f"Implied volatility out of range: {self.implied_vol}")
        
        if not (Decimal('0') <= self.data_quality <= Decimal('1')):
            raise ValueError("Data quality must be between 0 and 1")
    
    def get_mid_vol(self) -> Decimal:
        """Calculate mid volatility if bid/ask available"""
        if self.bid_vol and self.ask_vol:
            return (self.bid_vol + self.ask_vol) / Decimal('2')
        return self.implied_vol
    
    def get_vol_spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread in vol terms"""
        if self.bid_vol and self.ask_vol:
            return self.ask_vol - self.bid_vol
        return None


@dataclass(frozen=True)
class VolatilitySurface:
    """
    Complete volatility surface
    
    Immutable collection of surface points
    """
    underlying: str
    spot_price: Decimal
    points: Tuple[VolatilitySurfacePoint, ...]
    
    # ATM vol for reference
    atm_vol: Decimal
    
    # Surface characteristics
    skew: Decimal  # Negative for typical equity options
    term_structure_slope: Decimal  # Positive for normal markets
    
    # Quality
    calibration_error: Decimal
    data_quality_score: Decimal
    
    # Metadata
    model_used: str = "SVI"  # Stochastic Volatility Inspired
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate surface"""
        if len(self.points) < 3:
            raise ValueError("Need at least 3 points for surface")
        
        if self.spot_price <= Decimal('0'):
            raise ValueError("Spot price must be positive")
        
        if self.atm_vol <= Decimal('0'):
            raise ValueError("ATM vol must be positive")
    
    def get_point_count(self) -> int:
        """Get number of surface points"""
        return len(self.points)
    
    def get_vol_at_strike(self, strike: Decimal) -> Optional[Decimal]:
        """Get interpolated volatility at specific strike"""
        # Find nearest points and interpolate
        # Simplified - would use proper interpolation in production
        closest = min(self.points, key=lambda p: abs(p.strike - strike))
        return closest.implied_vol
    
    def has_vol_smile(self) -> bool:
        """Check if surface exhibits volatility smile"""
        return abs(self.skew) > Decimal('0.05')


@dataclass(frozen=True)
class VolatilityArbitrage:
    """
    Volatility arbitrage opportunity
    
    Immutable arbitrage signal
    """
    arbitrage_id: str
    underlying: str
    
    # Mispricing
    implied_vol: Decimal
    forecast_vol: Decimal
    vol_differential: Decimal  # Difference in vol points
    vol_differential_pct: Decimal  # Percentage
    
    # Trade recommendation
    trade_type: str  # 'long_vol', 'short_vol'
    expected_profit: Decimal
    max_loss: Decimal
    probability_profit: Decimal
    
    # Quality
    confidence: Decimal
    urgency: str  # 'low', 'medium', 'high'
    
    # Metadata
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate arbitrage"""
        if not (Decimal('0') <= self.confidence <= Decimal('1')):
            raise ValueError("Confidence must be between 0 and 1")
        
        if not (Decimal('0') <= self.probability_profit <= Decimal('1')):
            raise ValueError("Probability must be between 0 and 1")
    
    def is_significant(self, min_differential: Decimal = Decimal('0.03')) -> bool:
        """Check if arbitrage is significant (>3 vol points)"""
        return abs(self.vol_differential) >= min_differential
    
    def is_actionable(
        self,
        min_confidence: Decimal = Decimal('0.75'),
        min_expected: Decimal = Decimal('1000')
    ) -> bool:
        """Check if arbitrage is actionable"""
        return (
            self.confidence >= min_confidence and
            self.expected_profit >= min_expected and
            self.is_significant()
        )


# Example usage
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("volatility_domain_test")
    
    logger.info("test_starting", test="VOLATILITY DOMAIN VALUE OBJECTS")
    
    # Create volatility forecast
    logger.info("creating_volatility_forecast")
    
    forecast = VolatilityForecast(
        underlying="SPY",
        forecast_vol=Decimal('0.25'),
        horizon=ForecastHorizon.DAILY,
        confidence=Decimal('0.85'),
        regime=VolatilityRegime.NORMAL,
        transformer_prediction=Decimal('0.24'),
        garch_prediction=Decimal('0.26'),
        lstm_prediction=Decimal('0.25'),
        sentiment_impact=Decimal('0.05'),
        model_type=VolatilityModel.ENSEMBLE,
        prediction_time_ms=Decimal('45.2')
    )
    
    logger.info(
        "forecast_created",
        forecast_vol=float(forecast.forecast_vol),
        regime=forecast.regime.value,
        high_confidence=forecast.is_high_confidence(),
        model_agreement=float(forecast.get_model_agreement())
    )
    
    # Create surface point
    logger.info("creating_surface_point")
    
    point = VolatilitySurfacePoint(
        strike=Decimal('450'),
        expiry=datetime(2024, 11, 15),
        implied_vol=Decimal('0.22'),
        bid_vol=Decimal('0.21'),
        ask_vol=Decimal('0.23'),
        data_quality=Decimal('0.95')
    )
    
    logger.info(
        "surface_point_created",
        strike=float(point.strike),
        implied_vol=float(point.implied_vol),
        mid_vol=float(point.get_mid_vol())
    )
    
    # Create volatility surface
    logger.info("creating_volatility_surface")
    
    surface = VolatilitySurface(
        underlying="SPY",
        spot_price=Decimal('450'),
        points=(point,),  # Would have many more in production
        atm_vol=Decimal('0.22'),
        skew=Decimal('-0.08'),
        term_structure_slope=Decimal('0.02'),
        calibration_error=Decimal('0.002'),
        data_quality_score=Decimal('0.92')
    )
    
    logger.info(
        "surface_created",
        point_count=surface.get_point_count(),
        atm_vol=float(surface.atm_vol),
        has_smile=surface.has_vol_smile()
    )
    
    # Create arbitrage signal
    logger.info("creating_arbitrage_signal")
    
    arb = VolatilityArbitrage(
        arbitrage_id="ARB-001",
        underlying="SPY",
        implied_vol=Decimal('0.28'),
        forecast_vol=Decimal('0.22'),
        vol_differential=Decimal('0.06'),
        vol_differential_pct=Decimal('27.3'),
        trade_type="short_vol",
        expected_profit=Decimal('2500'),
        max_loss=Decimal('1200'),
        probability_profit=Decimal('0.72'),
        confidence=Decimal('0.80'),
        urgency="high"
    )
    
    logger.info(
        "arbitrage_created",
        vol_diff=float(arb.vol_differential),
        significant=arb.is_significant(),
        actionable=arb.is_actionable()
    )
    
    logger.info(
        "test_complete",
        artifacts_created=[
            "Immutable volatility objects",
            "Self-validating forecasts",
            "Rich domain behavior",
            "Type-safe with Decimal",
            "Regime detection",
            "Arbitrage signals",
            "Proper logging (no print)"
        ]
    )