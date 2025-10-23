"""
Centralized Configuration System for All Financial Models
==========================================================

This module provides a unified configuration system for all Axiom financial models,
implementing DRY principles and enabling maximum customizability without code changes.

Features:
- Type-safe dataclass-based configuration
- Environment variable overrides
- Configuration profiles (conservative, aggressive, basel_iii, etc.)
- Runtime configuration updates
- Validation and defaults
- Configuration inheritance
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import os
import json


class RiskProfile(Enum):
    """Risk profile presets for configuration."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class TradingStyle(Enum):
    """Trading style presets."""
    INTRADAY = "intraday"
    SWING = "swing"
    POSITION = "position"
    ALGORITHMIC = "algorithmic"


@dataclass
class OptionsConfig:
    """Centralized options pricing configuration."""
    
    # Black-Scholes defaults
    default_risk_free_rate: float = 0.05
    default_dividend_yield: float = 0.0
    black_scholes_precision: float = 1e-6
    
    # Binomial tree
    binomial_steps_default: int = 100
    binomial_max_steps: int = 1000
    binomial_convergence_threshold: float = 0.01
    
    # Monte Carlo
    monte_carlo_paths_default: int = 10000
    monte_carlo_max_paths: int = 1000000
    monte_carlo_seed: Optional[int] = None
    variance_reduction: str = "antithetic"  # antithetic, importance, stratified, none
    
    # Greeks
    greeks_delta: float = 0.01
    greeks_precision: float = 1e-6
    greeks_calculate_all: bool = True  # Calculate all Greeks at once for performance
    
    # Implied Volatility
    iv_solver_method: str = "newton_raphson"  # newton_raphson, bisection, brent
    iv_max_iterations: int = 100
    iv_tolerance: float = 1e-6
    iv_initial_guess_method: str = "brenner_subrahmanyam"  # brenner_subrahmanyam, constant
    iv_constant_guess: float = 0.25
    
    # Logging and performance
    enable_logging: bool = True
    enable_performance_tracking: bool = True
    cache_results: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "OptionsConfig":
        """Create configuration from environment variables."""
        return cls(
            default_risk_free_rate=float(os.getenv("OPTIONS_RISK_FREE_RATE", "0.05")),
            default_dividend_yield=float(os.getenv("OPTIONS_DIVIDEND_YIELD", "0.0")),
            monte_carlo_paths_default=int(os.getenv("OPTIONS_MC_PATHS", "10000")),
            variance_reduction=os.getenv("OPTIONS_VARIANCE_REDUCTION", "antithetic"),
        )


@dataclass
class CreditConfig:
    """Centralized credit risk configuration."""
    
    # PD Configuration
    default_confidence_level: float = 0.99
    basel_confidence_level: float = 0.999  # Basel III requires 99.9%
    pd_approach: str = "kmv_merton"  # kmv_merton, altman_z, logistic, agency_curve
    pit_to_ttc_weight: float = 0.7  # Weight on rating-based TTC
    
    # LGD Configuration
    default_recovery_rate: float = 0.40  # 40% recovery = 60% LGD
    downturn_multiplier: float = 1.25  # Basel III downturn LGD
    use_downturn_lgd: bool = True
    collateral_haircut: float = 0.20  # 20% haircut on collateral
    
    # EAD Configuration
    default_ccf: float = 0.75  # Credit Conversion Factor
    sa_ccr_alpha: float = 1.4  # SA-CCR supervisory factor
    calculate_pfe: bool = True  # Potential Future Exposure
    
    # Credit VaR
    cvar_approach: str = "monte_carlo"  # analytical, creditmetrics, creditrisk_plus, monte_carlo
    monte_carlo_scenarios: int = 10000
    variance_reduction: str = "antithetic"
    correlation_method: str = "gaussian"  # gaussian, t_copula, clayton, gumbel
    
    # Portfolio Risk
    concentration_threshold: float = 0.10  # 10% HHI threshold
    enable_diversification_benefit: bool = True
    capital_approach: str = "ADVANCED_IRB"  # SA_CR, FIRB, AIRB
    
    # Performance
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "CreditConfig":
        """Create configuration from environment variables."""
        return cls(
            basel_confidence_level=float(os.getenv("CREDIT_BASEL_CONFIDENCE", "0.999")),
            downturn_multiplier=float(os.getenv("CREDIT_DOWNTURN_MULTIPLIER", "1.25")),
            capital_approach=os.getenv("CREDIT_CAPITAL_APPROACH", "ADVANCED_IRB"),
        )
    
    @classmethod
    def for_basel_iii(cls) -> "CreditConfig":
        """Basel III compliant configuration."""
        return cls(
            default_confidence_level=0.999,
            basel_confidence_level=0.999,
            downturn_multiplier=1.25,
            use_downturn_lgd=True,
            capital_approach="ADVANCED_IRB"
        )


@dataclass
class VaRConfig:
    """Centralized VaR configuration."""
    
    # General
    default_confidence_level: float = 0.95
    default_time_horizon: int = 1  # days
    default_method: str = "historical"  # parametric, historical, monte_carlo
    
    # Parametric
    assume_normal: bool = True
    use_ewma_volatility: bool = False
    ewma_lambda: float = 0.94  # RiskMetrics standard
    
    # Historical Simulation
    min_observations: int = 252  # 1 year of daily data
    bootstrap_enabled: bool = False
    bootstrap_iterations: int = 1000
    
    # Monte Carlo
    default_simulations: int = 10000
    max_simulations: int = 1000000
    variance_reduction: str = "antithetic"
    random_seed: Optional[int] = None
    
    # Performance
    cache_results: bool = True
    parallel_mc: bool = True
    max_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "VaRConfig":
        """Create configuration from environment variables."""
        return cls(
            default_confidence_level=float(os.getenv("VAR_CONFIDENCE", "0.95")),
            default_method=os.getenv("VAR_METHOD", "historical"),
            min_observations=int(os.getenv("VAR_MIN_OBS", "252")),
        )


@dataclass
class PortfolioConfig:
    """Centralized portfolio optimization configuration."""
    
    # Optimization
    default_risk_free_rate: float = 0.03
    periods_per_year: int = 252  # Daily returns
    optimization_method: str = "max_sharpe"  # max_sharpe, min_volatility, risk_parity
    
    # Constraints
    long_only: bool = True
    fully_invested: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    sector_limits: Dict[str, float] = field(default_factory=dict)
    
    # Efficient Frontier
    frontier_points: int = 100
    target_return_range: str = "auto"  # auto, or tuple (min, max)
    
    # Risk Parity
    risk_parity_max_iter: int = 1000
    risk_parity_tolerance: float = 1e-6
    
    # Black-Litterman
    bl_tau: float = 0.05  # Uncertainty in equilibrium
    bl_confidence_method: str = "idzorek"  # idzorek, meucci
    
    # Rebalancing
    rebalance_threshold: float = 0.05  # 5% drift
    transaction_costs: float = 0.001  # 10 bps
    min_trade_size: float = 100.0  # Minimum trade in dollars
    
    # Performance
    cache_covariance: bool = True
    use_shrinkage: bool = False
    shrinkage_target: str = "constant_correlation"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "PortfolioConfig":
        """Create configuration from environment variables."""
        return cls(
            default_risk_free_rate=float(os.getenv("PORTFOLIO_RISK_FREE_RATE", "0.03")),
            optimization_method=os.getenv("PORTFOLIO_METHOD", "max_sharpe"),
            long_only=os.getenv("PORTFOLIO_LONG_ONLY", "true").lower() == "true",
        )


@dataclass
class TimeSeriesConfig:
    """Centralized time series configuration."""
    
    # ARIMA
    arima_auto_select: bool = True
    arima_ic: str = "aic"  # aic, bic, hqic
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    arima_seasonal: bool = False
    arima_m: int = 12  # Seasonal period
    
    # GARCH
    garch_order: tuple = (1, 1)
    garch_distribution: str = "normal"  # normal, t, ged
    garch_vol_target: Optional[float] = None
    garch_rescale: bool = True
    
    # EWMA
    ewma_decay_factor: float = 0.94  # RiskMetrics
    ewma_min_periods: int = 30
    ewma_fast_span: int = 12
    ewma_slow_span: int = 26
    
    # General
    min_observations: int = 100
    confidence_level: float = 0.95
    forecast_horizon: int = 5
    
    # Performance
    cache_models: bool = True
    parallel_fitting: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['garch_order'] = list(self.garch_order)  # Convert tuple to list for JSON
        return data
    
    @classmethod
    def from_env(cls) -> "TimeSeriesConfig":
        """Create configuration from environment variables."""
        return cls(
            ewma_decay_factor=float(os.getenv("TS_EWMA_LAMBDA", "0.94")),
            forecast_horizon=int(os.getenv("TS_FORECAST_HORIZON", "5")),
        )
    
    @classmethod
    def for_intraday_trading(cls) -> "TimeSeriesConfig":
        """Configuration optimized for intraday trading."""
        return cls(
            ewma_decay_factor=0.99,  # More reactive
            ewma_min_periods=10,
            ewma_fast_span=5,
            ewma_slow_span=15,
            min_observations=50,
            forecast_horizon=1
        )
    
    @classmethod
    def for_swing_trading(cls) -> "TimeSeriesConfig":
        """Configuration for swing trading (multi-day holds)."""
        return cls(
            ewma_decay_factor=0.94,  # RiskMetrics standard
            ewma_min_periods=30,
            ewma_fast_span=12,
            ewma_slow_span=26,
            min_observations=100,
            forecast_horizon=5
        )
    
    @classmethod
    def for_position_trading(cls) -> "TimeSeriesConfig":
        """Configuration for position trading (weeks to months)."""
        return cls(
            ewma_decay_factor=0.88,  # Less reactive
            ewma_min_periods=60,
            ewma_fast_span=26,
            ewma_slow_span=52,
            min_observations=252,
            forecast_horizon=20
        )
    
    @classmethod
    def for_risk_management(cls, profile: RiskProfile = RiskProfile.MODERATE) -> "TimeSeriesConfig":
        """Configuration for risk management based on profile."""
        if profile == RiskProfile.CONSERVATIVE:
            return cls(
                ewma_decay_factor=0.88,
                confidence_level=0.99,
                forecast_horizon=10
            )
        elif profile == RiskProfile.AGGRESSIVE:
            return cls(
                ewma_decay_factor=0.99,
                confidence_level=0.90,
                forecast_horizon=3
            )
        else:  # MODERATE
            return cls(
                ewma_decay_factor=0.94,
                confidence_level=0.95,
                forecast_horizon=5
            )


@dataclass
class MicrostructureConfig:
    """Centralized market microstructure configuration."""
    
    # Order Flow
    ofi_window: int = 100  # Ticks for OFI calculation
    toxicity_threshold: float = 0.7  # VPIN threshold
    vpin_buckets: int = 50  # Number of volume buckets for VPIN
    classification_method: str = "lee_ready"  # lee_ready, tick_test, quote_rule, bvc
    
    # VWAP/TWAP
    vwap_method: str = "standard"  # standard, anchored, rolling
    twap_intervals: int = 10  # Number of intervals
    participation_rate: float = 0.10  # 10% of volume
    rolling_window: int = 100  # For rolling VWAP
    variance_bands: float = 2.0  # Standard deviations for bands
    
    # Liquidity
    illiquidity_window: int = 20  # Days for Amihud
    spread_estimator: str = "roll"  # roll, high_low, quoted
    market_cap: Optional[float] = None  # Market cap for turnover calculation
    depth_levels: int = 5  # Number of order book levels to analyze
    
    # Market Impact
    impact_model: str = "almgren_chriss"  # kyle, almgren_chriss, sqrt_law
    risk_aversion: float = 1e-6  # Almgren-Chriss risk aversion
    permanent_impact: float = 0.1  # Permanent impact coefficient
    temporary_impact: float = 0.5  # Temporary impact coefficient
    impact_coefficient: float = 1.0  # Square-root law coefficient
    
    # Spread Analysis
    spread_decomposition_method: str = "glosten_harris"  # glosten_harris, mrr, stoll
    estimation_window: int = 100  # Window for spread estimation
    opening_window_minutes: int = 30  # Opening period window
    closing_window_minutes: int = 30  # Closing period window
    
    # Price Discovery
    variance_ratio_lags: List[int] = field(default_factory=lambda: [2, 5, 10])
    
    # Performance
    tick_processing_batch_size: int = 1000
    enable_streaming: bool = False
    enable_logging: bool = True
    enable_performance_tracking: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "MicrostructureConfig":
        """Create configuration from environment variables."""
        return cls(
            ofi_window=int(os.getenv("MS_OFI_WINDOW", "100")),
            vwap_method=os.getenv("MS_VWAP_METHOD", "standard"),
            participation_rate=float(os.getenv("MS_PARTICIPATION_RATE", "0.10")),
        )
    
    @classmethod
    def for_high_frequency_trading(cls) -> "MicrostructureConfig":
        """Configuration optimized for HFT."""
        return cls(
            ofi_window=50,  # Smaller window for faster reaction
            vwap_method="rolling",
            participation_rate=0.05,  # Lower participation
            tick_processing_batch_size=500,
            enable_streaming=True
        )
    
    @classmethod
    def for_institutional_execution(cls) -> "MicrostructureConfig":
        """Configuration for institutional execution."""
        return cls(
            ofi_window=200,  # Larger window for stability
            vwap_method="standard",
            participation_rate=0.15,  # Higher participation
            impact_model="almgren_chriss",
            risk_aversion=1e-6
        )


@dataclass
class FixedIncomeConfig:
    """Centralized fixed income configuration."""
    
    # Bond pricing
    day_count_convention: str = "30/360"  # 30/360, Actual/360, Actual/365, Actual/Actual
    settlement_days: int = 2  # T+2 settlement
    compounding_frequency: int = 2  # Semi-annual
    price_precision: float = 1e-8
    
    # Yield curve
    curve_model: str = "nelson_siegel"  # nelson_siegel, svensson, cubic_spline, bootstrap
    interpolation_method: str = "cubic_spline"  # linear, cubic_spline, monotone_convex
    curve_tenors: List[float] = field(default_factory=lambda: [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    extrapolation_method: str = "flat"  # flat, linear
    
    # Yield calculations
    ytm_solver_method: str = "newton_raphson"  # newton_raphson, bisection, brent
    ytm_max_iterations: int = 100
    ytm_tolerance: float = 1e-8
    ytm_initial_guess: float = 0.05
    
    # Duration/Convexity
    shock_size_bps: float = 1.0  # Basis points for numerical derivatives
    key_rate_tenors: List[float] = field(default_factory=lambda: [1, 2, 3, 5, 7, 10, 20, 30])
    calculate_effective_duration: bool = True
    
    # Term structure models
    short_rate_model: str = "vasicek"  # vasicek, cir, hull_white, ho_lee
    calibration_method: str = "mle"  # mle, kalman, least_squares
    simulation_paths: int = 10000
    time_steps: int = 252
    
    # Spread analysis
    calculate_z_spread: bool = True
    calculate_oas: bool = False  # More computationally intensive
    oas_paths: int = 1000
    credit_curve_tenors: List[float] = field(default_factory=lambda: [1, 2, 3, 5, 7, 10])
    
    # Portfolio
    rebalancing_threshold: float = 0.1  # Duration drift tolerance
    default_rating: str = "AAA"
    rating_scale: List[str] = field(default_factory=lambda: ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"])
    concentration_limit: float = 0.10  # Max 10% in single issuer
    
    # Performance
    enable_caching: bool = True
    cache_curve_minutes: int = 5
    parallel_pricing: bool = True
    max_workers: int = 4
    enable_logging: bool = True
    enable_performance_tracking: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "FixedIncomeConfig":
        """Create configuration from environment variables."""
        return cls(
            day_count_convention=os.getenv("FI_DAY_COUNT", "30/360"),
            settlement_days=int(os.getenv("FI_SETTLEMENT_DAYS", "2")),
            curve_model=os.getenv("FI_CURVE_MODEL", "nelson_siegel"),
            ytm_solver_method=os.getenv("FI_YTM_SOLVER", "newton_raphson"),
        )
    
    @classmethod
    def for_high_performance(cls) -> "FixedIncomeConfig":
        """Configuration optimized for speed."""
        return cls(
            ytm_max_iterations=50,
            ytm_tolerance=1e-6,
            simulation_paths=5000,
            time_steps=100,
            calculate_oas=False,
            enable_caching=True,
            parallel_pricing=True
        )
    
    @classmethod
    def for_high_precision(cls) -> "FixedIncomeConfig":
        """Configuration optimized for accuracy."""
        return cls(
            price_precision=1e-12,
            ytm_tolerance=1e-10,
            shock_size_bps=0.1,
            simulation_paths=100000,
            time_steps=500,
            calculate_oas=True,
            oas_paths=10000
        )


@dataclass
class MandAConfig:
    """Centralized M&A quantitative models configuration."""
    
    # Synergy parameters
    cost_synergy_realization_years: int = 3
    revenue_synergy_realization_years: int = 5
    synergy_discount_rate: float = 0.12  # Higher risk than WACC
    integration_cost_multiple: float = 0.15  # 15% of synergies
    synergy_tax_rate: float = 0.21  # Tax rate for synergy benefits
    
    # Deal financing
    target_debt_ebitda: float = 5.0  # Maximum leverage
    min_interest_coverage: float = 2.0  # Minimum EBITDA/Interest
    optimal_cash_stock_mix: str = "npv_maximizing"  # tax_efficient, rating_neutral, npv_maximizing
    default_risk_free_rate: float = 0.04
    default_market_return: float = 0.10
    default_tax_rate: float = 0.21
    
    # Merger arbitrage
    min_deal_spread_bps: int = 100  # 1% minimum spread for trade
    max_position_size_pct: float = 0.10  # 10% of portfolio max
    hedge_ratio_method: str = "optimal"  # optimal, static, dynamic
    default_close_probability: float = 0.85
    kelly_fraction: float = 0.25  # Use 25% of Kelly for safety
    
    # LBO modeling
    target_irr: float = 0.20  # 20% target IRR for PE
    exit_multiple_method: str = "entry_multiple"  # entry_multiple, sector_median, custom
    holding_period_years: int = 5
    max_leverage_multiple: float = 6.0  # Max Debt/EBITDA
    min_equity_contribution_pct: float = 0.30  # Min 30% equity
    
    # Risk parameters
    regulatory_delay_months: int = 6
    break_fee_pct: float = 0.03  # 3% termination fee
    mac_probability: float = 0.05  # 5% MAC clause trigger probability
    financing_risk_factor: float = 0.10  # 10% risk financing falls through
    
    # Valuation
    control_premium_range: Tuple[float, float] = (0.20, 0.40)  # 20-40% typical
    dcf_terminal_growth: float = 0.02  # 2% perpetual growth
    precedent_lookback_years: int = 3  # Years of precedent transactions
    
    # Performance
    enable_monte_carlo: bool = True
    monte_carlo_scenarios: int = 10000
    enable_sensitivity_analysis: bool = True
    sensitivity_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "synergies": (0.5, 1.5),  # 50% to 150% of base case
            "discount_rate": (0.08, 0.16),
            "exit_multiple": (0.8, 1.2)
        }
    )
    
    # Logging
    enable_logging: bool = True
    enable_performance_tracking: bool = True
    cache_results: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert tuples to lists for JSON serialization
        data['control_premium_range'] = list(self.control_premium_range)
        data['sensitivity_ranges'] = {
            k: list(v) for k, v in self.sensitivity_ranges.items()
        }
        return data
    
    @classmethod
    def from_env(cls) -> "MandAConfig":
        """Create configuration from environment variables."""
        return cls(
            synergy_discount_rate=float(os.getenv("MA_SYNERGY_DISCOUNT_RATE", "0.12")),
            target_debt_ebitda=float(os.getenv("MA_TARGET_DEBT_EBITDA", "5.0")),
            target_irr=float(os.getenv("MA_TARGET_IRR", "0.20")),
            default_close_probability=float(os.getenv("MA_CLOSE_PROBABILITY", "0.85")),
        )
    
    @classmethod
    def for_conservative_approach(cls) -> "MandAConfig":
        """Configuration for conservative M&A approach."""
        return cls(
            synergy_discount_rate=0.15,  # Higher discount for risk
            integration_cost_multiple=0.20,  # Higher integration costs
            target_debt_ebitda=4.0,  # Lower leverage
            default_close_probability=0.75,  # Lower success probability
            kelly_fraction=0.15  # More conservative position sizing
        )
    
    @classmethod
    def for_aggressive_approach(cls) -> "MandAConfig":
        """Configuration for aggressive M&A approach."""
        return cls(
            synergy_discount_rate=0.10,  # Lower discount
            integration_cost_multiple=0.10,  # Lower integration costs
            target_debt_ebitda=6.0,  # Higher leverage
            default_close_probability=0.90,  # Higher success probability
            kelly_fraction=0.35  # More aggressive position sizing
        )


@dataclass
class ModelConfig:
    """Master configuration container for all models."""
    
    options: OptionsConfig = field(default_factory=OptionsConfig)
    credit: CreditConfig = field(default_factory=CreditConfig)
    var: VaRConfig = field(default_factory=VaRConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    time_series: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    fixed_income: FixedIncomeConfig = field(default_factory=FixedIncomeConfig)
    ma: MandAConfig = field(default_factory=MandAConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configurations to dictionary."""
        return {
            "options": self.options.to_dict(),
            "credit": self.credit.to_dict(),
            "var": self.var.to_dict(),
            "portfolio": self.portfolio.to_dict(),
            "time_series": self.time_series.to_dict(),
            "microstructure": self.microstructure.to_dict(),
            "fixed_income": self.fixed_income.to_dict(),
            "ma": self.ma.to_dict()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create full configuration from environment variables."""
        return cls(
            options=OptionsConfig.from_env(),
            credit=CreditConfig.from_env(),
            var=VaRConfig.from_env(),
            portfolio=PortfolioConfig.from_env(),
            time_series=TimeSeriesConfig.from_env(),
            microstructure=MicrostructureConfig.from_env(),
            fixed_income=FixedIncomeConfig.from_env(),
            ma=MandAConfig.from_env()
        )
    
    @classmethod
    def from_file(cls, filepath: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            options=OptionsConfig(**data.get("options", {})),
            credit=CreditConfig(**data.get("credit", {})),
            var=VaRConfig(**data.get("var", {})),
            portfolio=PortfolioConfig(**data.get("portfolio", {})),
            time_series=TimeSeriesConfig(**data.get("time_series", {})),
            microstructure=MicrostructureConfig(**data.get("microstructure", {})),
            fixed_income=FixedIncomeConfig(**data.get("fixed_income", {})),
            ma=MandAConfig(**data.get("ma", {}))
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def for_basel_iii_compliance(cls) -> "ModelConfig":
        """Configuration profile for Basel III compliance."""
        return cls(
            credit=CreditConfig.for_basel_iii(),
            var=VaRConfig(
                default_confidence_level=0.999,
                default_time_horizon=10,
                min_observations=252
            )
        )
    
    @classmethod
    def for_high_performance(cls) -> "ModelConfig":
        """Configuration profile optimized for speed."""
        return cls(
            options=OptionsConfig(
                monte_carlo_paths_default=5000,
                binomial_steps_default=50,
                cache_results=True
            ),
            credit=CreditConfig(
                monte_carlo_scenarios=5000,
                parallel_processing=True,
                enable_caching=True
            ),
            var=VaRConfig(
                default_simulations=5000,
                parallel_mc=True,
                cache_results=True
            )
        )
    
    @classmethod
    def for_high_precision(cls) -> "ModelConfig":
        """Configuration profile optimized for accuracy."""
        return cls(
            options=OptionsConfig(
                monte_carlo_paths_default=100000,
                binomial_steps_default=500,
                black_scholes_precision=1e-10
            ),
            credit=CreditConfig(
                monte_carlo_scenarios=100000,
                concentration_threshold=0.05
            ),
            var=VaRConfig(
                default_simulations=100000,
                min_observations=500
            )
        )


# Global configuration instance
_global_config: Optional[ModelConfig] = None


def get_config() -> ModelConfig:
    """Get global configuration instance (singleton pattern)."""
    global _global_config
    if _global_config is None:
        # Try to load from environment, fall back to defaults
        try:
            _global_config = ModelConfig.from_env()
        except Exception:
            _global_config = ModelConfig()
    return _global_config


def set_config(config: ModelConfig):
    """Set global configuration instance."""
    global _global_config
    _global_config = config


def reset_config():
    """Reset configuration to defaults."""
    global _global_config
    _global_config = ModelConfig()


__all__ = [
    "RiskProfile",
    "TradingStyle",
    "OptionsConfig",
    "CreditConfig",
    "VaRConfig",
    "PortfolioConfig",
    "TimeSeriesConfig",
    "MicrostructureConfig",
    "FixedIncomeConfig",
    "MandAConfig",
    "ModelConfig",
    "get_config",
    "set_config",
    "reset_config"
]