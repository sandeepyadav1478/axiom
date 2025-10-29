"""
Model Factory Pattern for Creating Financial Models
===================================================

Provides a centralized factory for creating any financial model with
custom configuration, implementing the Factory design pattern for
maximum flexibility and DRY principles.

Features:
- Single point of model creation
- Configuration injection
- Plugin registration
- Type-safe model instantiation
"""

from enum import Enum
from typing import Dict, Any, Optional, Type, Callable
from dataclasses import dataclass

from axiom.config.model_config import get_config, ModelConfig
from axiom.models.base.base_model import BaseFinancialModel


class ModelType(Enum):
    """Enumeration of all available financial model types."""
    
    # Options Models
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL_TREE = "binomial_tree"
    MONTE_CARLO_OPTIONS = "monte_carlo_options"
    GREEKS_CALCULATOR = "greeks_calculator"
    IMPLIED_VOLATILITY = "implied_volatility"
    
    # Credit Risk Models
    KMV_MERTON_PD = "kmv_merton_pd"
    ALTMAN_Z_SCORE = "altman_z_score"
    CREDIT_VAR = "credit_var"
    PORTFOLIO_CREDIT_RISK = "portfolio_credit_risk"
    
    # VaR Models
    PARAMETRIC_VAR = "parametric_var"
    HISTORICAL_VAR = "historical_var"
    MONTE_CARLO_VAR = "monte_carlo_var"
    RL_GARCH_VAR = "rl_garch_var"  # 2025 cutting-edge: RL + GARCH
    
    # Portfolio Models
    MARKOWITZ_OPTIMIZER = "markowitz_optimizer"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    
    # Time Series Models
    ARIMA = "arima"
    GARCH = "garch"
    EWMA = "ewma"
    
    # Market Microstructure Models
    ORDER_FLOW_ANALYZER = "order_flow_analyzer"
    VWAP_CALCULATOR = "vwap_calculator"
    TWAP_SCHEDULER = "twap_scheduler"
    EXECUTION_ANALYZER = "execution_analyzer"
    LIQUIDITY_ANALYZER = "liquidity_analyzer"
    KYLE_LAMBDA = "kyle_lambda"
    ALMGREN_CHRISS = "almgren_chriss"
    SQUARE_ROOT_LAW = "square_root_law"
    MARKET_IMPACT_ANALYZER = "market_impact_analyzer"
    SPREAD_DECOMPOSITION = "spread_decomposition"
    INTRADAY_SPREAD_ANALYZER = "intraday_spread_analyzer"
    INFORMATION_SHARE = "information_share"
    MARKET_QUALITY_ANALYZER = "market_quality_analyzer"
    
    # Fixed Income Models
    BOND_PRICING = "bond_pricing"
    YIELD_CURVE = "yield_curve"
    NELSON_SIEGEL = "nelson_siegel"
    SVENSSON = "svensson"
    BOOTSTRAPPING = "bootstrapping"
    CUBIC_SPLINE_CURVE = "cubic_spline_curve"
    DURATION_CALCULATOR = "duration_calculator"
    VASICEK = "vasicek"
    CIR = "cir"
    HULL_WHITE = "hull_white"
    HO_LEE = "ho_lee"
    TERM_STRUCTURE = "term_structure"
    SPREAD_ANALYZER = "spread_analyzer"
    BOND_PORTFOLIO = "bond_portfolio"
    BOND_PORTFOLIO_ANALYZER = "bond_portfolio_analyzer"
    
    # M&A Models
    SYNERGY_VALUATION = "synergy_valuation"
    DEAL_FINANCING = "deal_financing"
    MERGER_ARBITRAGE = "merger_arbitrage"
    LBO_MODEL = "lbo_model"
    VALUATION_INTEGRATION = "valuation_integration"
    DEAL_SCREENING = "deal_screening"


@dataclass
class ModelRegistration:
    """Registration info for a model type."""
    model_class: Type[BaseFinancialModel]
    config_key: str  # Which config section to use (options, credit, var, etc.)
    description: str


class ModelFactory:
    """
    Factory for creating financial models with dependency injection.
    
    Usage:
        # Create model with default config
        model = ModelFactory.create(ModelType.BLACK_SCHOLES)
        
        # Create model with custom config
        custom_config = OptionsConfig(monte_carlo_paths_default=50000)
        model = ModelFactory.create(
            ModelType.BLACK_SCHOLES,
            config=custom_config
        )
        
        # Register custom model
        ModelFactory.register_model(
            "custom_option_model",
            CustomOptionModel,
            config_key="options",
            description="Custom option pricing model"
        )
    """
    
    # Registry of all available models
    _registry: Dict[str, ModelRegistration] = {}
    
    @classmethod
    def register_model(
        cls,
        model_type: str,
        model_class: Type[BaseFinancialModel],
        config_key: str,
        description: str = ""
    ):
        """
        Register a new model type.
        
        Args:
            model_type: Unique identifier for the model
            model_class: The model class to instantiate
            config_key: Configuration section key
            description: Human-readable description
        """
        cls._registry[model_type] = ModelRegistration(
            model_class=model_class,
            config_key=config_key,
            description=description
        )
    
    @classmethod
    def create(
        cls,
        model_type: ModelType,
        config: Optional[Any] = None,
        **kwargs
    ) -> BaseFinancialModel:
        """
        Create a financial model instance.
        
        Args:
            model_type: Type of model to create
            config: Optional custom configuration
            **kwargs: Additional model-specific parameters
            
        Returns:
            Instantiated model
            
        Raises:
            ValueError: If model type is not registered
        """
        model_key = model_type.value
        
        if model_key not in cls._registry:
            raise ValueError(
                f"Model type '{model_key}' not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        registration = cls._registry[model_key]
        
        # Get configuration if not provided
        if config is None:
            global_config = get_config()
            config_obj = getattr(global_config, registration.config_key)
        else:
            # Pass config object directly (don't convert to dict)
            config_obj = config
        
        # Instantiate model with config object and additional kwargs
        # Models can handle config objects or will convert as needed
        if kwargs:
            # If there are additional kwargs, we need to handle them carefully
            # Most models accept config as first parameter
            model_instance = registration.model_class(config=config_obj, **kwargs)
        else:
            model_instance = registration.model_class(config=config_obj)
        
        return model_instance
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all registered models.
        
        Returns:
            Dictionary mapping model type to description
        """
        return {
            model_type: reg.description
            for model_type, reg in cls._registry.items()
        }
    
    @classmethod
    def get_model_info(cls, model_type: ModelType) -> ModelRegistration:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Model type to query
            
        Returns:
            ModelRegistration with model info
        """
        model_key = model_type.value
        if model_key not in cls._registry:
            raise ValueError(f"Model type '{model_key}' not registered")
        return cls._registry[model_key]


# Initialize registry with built-in models
def _init_builtin_models():
    """Initialize registry with Axiom's built-in models."""
    
    # Import models (lazy import to avoid circular dependencies)
    try:
        from axiom.models.risk.var_models import (
            ParametricVaR,
            HistoricalSimulationVaR,
            MonteCarloVaR
        )
        
        # Register VaR Models
        ModelFactory.register_model(
            ModelType.PARAMETRIC_VAR.value,
            ParametricVaR,
            config_key="var",
            description="Parametric VaR using variance-covariance method (normal distribution)"
        )
        
        ModelFactory.register_model(
            ModelType.HISTORICAL_VAR.value,
            HistoricalSimulationVaR,
            config_key="var",
            description="Historical simulation VaR using empirical distribution"
        )
        
        ModelFactory.register_model(
            ModelType.MONTE_CARLO_VAR.value,
            MonteCarloVaR,
            config_key="var",
            description="Monte Carlo VaR with simulated scenarios"
        )
        
        # RL-GARCH VaR (2025 research paper)
        try:
            from axiom.models.risk.rl_garch_var import RLGARCHVaR
            
            ModelFactory.register_model(
                ModelType.RL_GARCH_VAR.value,
                RLGARCHVaR,
                config_key="var",
                description="RL-GARCH VaR combining GARCH volatility with Deep RL (arXiv:2504.16635, April 2025)"
            )
        except ImportError:
            # PyTorch or arch not available
            pass
            
    except ImportError as e:
        # Models not yet available, skip registration
        pass
    
    # Time Series Models
    try:
        from axiom.models.time_series.arima import ARIMAModel
        from axiom.models.time_series.garch import GARCHModel
        from axiom.models.time_series.ewma import EWMAModel
        
        ModelFactory.register_model(
            ModelType.ARIMA.value,
            ARIMAModel,
            config_key="time_series",
            description="ARIMA(p,d,q) for price forecasting and trend prediction"
        )
        
        ModelFactory.register_model(
            ModelType.GARCH.value,
            GARCHModel,
            config_key="time_series",
            description="GARCH(p,q) for volatility forecasting and clustering"
        )
        
        ModelFactory.register_model(
            ModelType.EWMA.value,
            EWMAModel,
            config_key="time_series",
            description="Exponentially Weighted Moving Average for trend following"
        )
    except ImportError as e:
        # Models not yet available, skip registration
        pass
    
    # Portfolio Models
    try:
        from axiom.models.portfolio.optimization import (
            PortfolioOptimizer,
            MarkowitzOptimizer
        )
        from axiom.models.portfolio.allocation import (
            AssetAllocator
        )
        
        ModelFactory.register_model(
            ModelType.MARKOWITZ_OPTIMIZER.value,
            PortfolioOptimizer,
            config_key="portfolio",
            description="Markowitz mean-variance portfolio optimization with multiple objectives"
        )
        
        ModelFactory.register_model(
            "asset_allocator",
            AssetAllocator,
            config_key="portfolio",
            description="Multi-strategy asset allocation engine (risk parity, Black-Litterman, HRP, etc.)"
        )
        
        ModelFactory.register_model(
            ModelType.RISK_PARITY.value,
            AssetAllocator,
            config_key="portfolio",
            description="Risk parity allocation strategy"
        )
        
        ModelFactory.register_model(
            ModelType.BLACK_LITTERMAN.value,
            AssetAllocator,
            config_key="portfolio",
            description="Black-Litterman allocation with investor views"
        )
    except ImportError as e:
        # Portfolio models not yet available
        pass
    
    # Market Microstructure Models
    try:
        from axiom.models.microstructure.order_flow import OrderFlowAnalyzer
        from axiom.models.microstructure.execution_algos import (
            VWAPCalculator,
            TWAPScheduler,
            ExecutionAnalyzer
        )
        from axiom.models.microstructure.liquidity import LiquidityAnalyzer
        from axiom.models.microstructure.market_impact import (
            KyleLambdaModel,
            AlmgrenChrissModel,
            SquareRootLawModel,
            MarketImpactAnalyzer
        )
        from axiom.models.microstructure.spread_analysis import (
            SpreadDecompositionModel,
            IntradaySpreadAnalyzer
        )
        from axiom.models.microstructure.price_discovery import (
            InformationShareModel,
            MarketQualityAnalyzer
        )
        
        # Order Flow
        ModelFactory.register_model(
            ModelType.ORDER_FLOW_ANALYZER.value,
            OrderFlowAnalyzer,
            config_key="microstructure",
            description="Order flow analysis with OFI, VPIN, and trade classification"
        )
        
        # Execution Algorithms
        ModelFactory.register_model(
            ModelType.VWAP_CALCULATOR.value,
            VWAPCalculator,
            config_key="microstructure",
            description="VWAP calculation with variance bands and intraday tracking"
        )
        
        ModelFactory.register_model(
            ModelType.TWAP_SCHEDULER.value,
            TWAPScheduler,
            config_key="microstructure",
            description="TWAP execution scheduling with adaptive slicing"
        )
        
        ModelFactory.register_model(
            ModelType.EXECUTION_ANALYZER.value,
            ExecutionAnalyzer,
            config_key="microstructure",
            description="Execution performance analysis vs VWAP/TWAP benchmarks"
        )
        
        # Liquidity
        ModelFactory.register_model(
            ModelType.LIQUIDITY_ANALYZER.value,
            LiquidityAnalyzer,
            config_key="microstructure",
            description="Comprehensive liquidity metrics (Amihud, spreads, depth, etc.)"
        )
        
        # Market Impact
        ModelFactory.register_model(
            ModelType.KYLE_LAMBDA.value,
            KyleLambdaModel,
            config_key="microstructure",
            description="Kyle's lambda market impact estimation"
        )
        
        ModelFactory.register_model(
            ModelType.ALMGREN_CHRISS.value,
            AlmgrenChrissModel,
            config_key="microstructure",
            description="Almgren-Chriss optimal execution model"
        )
        
        ModelFactory.register_model(
            ModelType.SQUARE_ROOT_LAW.value,
            SquareRootLawModel,
            config_key="microstructure",
            description="Square-root law of market impact"
        )
        
        ModelFactory.register_model(
            ModelType.MARKET_IMPACT_ANALYZER.value,
            MarketImpactAnalyzer,
            config_key="microstructure",
            description="Comprehensive market impact analysis (all models combined)"
        )
        
        # Spread Analysis
        ModelFactory.register_model(
            ModelType.SPREAD_DECOMPOSITION.value,
            SpreadDecompositionModel,
            config_key="microstructure",
            description="Glosten-Harris/MRR spread decomposition"
        )
        
        ModelFactory.register_model(
            ModelType.INTRADAY_SPREAD_ANALYZER.value,
            IntradaySpreadAnalyzer,
            config_key="microstructure",
            description="Intraday spread pattern analysis (U-shape detection, etc.)"
        )
        
        # Price Discovery
        ModelFactory.register_model(
            ModelType.INFORMATION_SHARE.value,
            InformationShareModel,
            config_key="microstructure",
            description="Hasbrouck information share and price discovery"
        )
        
        ModelFactory.register_model(
            ModelType.MARKET_QUALITY_ANALYZER.value,
            MarketQualityAnalyzer,
            config_key="microstructure",
            description="Market quality and price efficiency analysis"
        )
        
    except ImportError as e:
        # Microstructure models not yet available
        pass
    
    # Fixed Income Models
    try:
        from axiom.models.fixed_income.bond_pricing import BondPricingModel
        from axiom.models.fixed_income.yield_curve import (
            NelsonSiegelModel,
            SvenssonModel,
            BootstrappingModel,
            CubicSplineModel
        )
        from axiom.models.fixed_income.duration import DurationCalculator
        from axiom.models.fixed_income.term_structure import (
            VasicekModel,
            CIRModel,
            HullWhiteModel,
            HoLeeModel
        )
        from axiom.models.fixed_income.spreads import SpreadAnalyzer
        from axiom.models.fixed_income.portfolio import BondPortfolioAnalyzer
        
        # Bond Pricing
        ModelFactory.register_model(
            ModelType.BOND_PRICING.value,
            BondPricingModel,
            config_key="fixed_income",
            description="Comprehensive bond pricing for all bond types"
        )
        
        # Yield Curve Models
        ModelFactory.register_model(
            ModelType.NELSON_SIEGEL.value,
            NelsonSiegelModel,
            config_key="fixed_income",
            description="Nelson-Siegel parametric yield curve model"
        )
        
        ModelFactory.register_model(
            ModelType.SVENSSON.value,
            SvenssonModel,
            config_key="fixed_income",
            description="Svensson extended yield curve model"
        )
        
        ModelFactory.register_model(
            ModelType.BOOTSTRAPPING.value,
            BootstrappingModel,
            config_key="fixed_income",
            description="Bootstrapping yield curve construction"
        )
        
        ModelFactory.register_model(
            ModelType.CUBIC_SPLINE_CURVE.value,
            CubicSplineModel,
            config_key="fixed_income",
            description="Cubic spline yield curve interpolation"
        )
        
        # Duration & Convexity
        ModelFactory.register_model(
            ModelType.DURATION_CALCULATOR.value,
            DurationCalculator,
            config_key="fixed_income",
            description="Comprehensive duration and convexity analytics"
        )
        
        # Term Structure Models
        ModelFactory.register_model(
            ModelType.VASICEK.value,
            VasicekModel,
            config_key="fixed_income",
            description="Vasicek short rate model with mean reversion"
        )
        
        ModelFactory.register_model(
            ModelType.CIR.value,
            CIRModel,
            config_key="fixed_income",
            description="Cox-Ingersoll-Ross model with non-negative rates"
        )
        
        ModelFactory.register_model(
            ModelType.HULL_WHITE.value,
            HullWhiteModel,
            config_key="fixed_income",
            description="Hull-White extended Vasicek model"
        )
        
        ModelFactory.register_model(
            ModelType.HO_LEE.value,
            HoLeeModel,
            config_key="fixed_income",
            description="Ho-Lee binomial lattice model"
        )
        
        # Spread Analysis
        ModelFactory.register_model(
            ModelType.SPREAD_ANALYZER.value,
            SpreadAnalyzer,
            config_key="fixed_income",
            description="Credit spread and relative value analysis"
        )
        
        # Portfolio Analytics
        ModelFactory.register_model(
            ModelType.BOND_PORTFOLIO_ANALYZER.value,
            BondPortfolioAnalyzer,
            config_key="fixed_income",
            description="Bond portfolio risk and performance analytics"
        )
        
    except ImportError as e:
        # Fixed income models not yet available
        pass
    
    # M&A Models
    try:
        from axiom.models.ma.synergy_valuation import SynergyValuationModel
        from axiom.models.ma.deal_financing import DealFinancingModel
        from axiom.models.ma.merger_arbitrage import MergerArbitrageModel
        from axiom.models.ma.lbo_modeling import LBOModel
        from axiom.models.ma.valuation_integration import ValuationIntegrationModel
        from axiom.models.ma.deal_screening import DealScreeningModel
        
        # Synergy Valuation
        ModelFactory.register_model(
            ModelType.SYNERGY_VALUATION.value,
            SynergyValuationModel,
            config_key="ma",
            description="Comprehensive synergy valuation with cost and revenue synergies"
        )
        
        # Deal Financing
        ModelFactory.register_model(
            ModelType.DEAL_FINANCING.value,
            DealFinancingModel,
            config_key="ma",
            description="Capital structure optimization and EPS accretion/dilution analysis"
        )
        
        # Merger Arbitrage
        ModelFactory.register_model(
            ModelType.MERGER_ARBITRAGE.value,
            MergerArbitrageModel,
            config_key="ma",
            description="Merger arbitrage spread analysis and position sizing"
        )
        
        # LBO Modeling
        ModelFactory.register_model(
            ModelType.LBO_MODEL.value,
            LBOModel,
            config_key="ma",
            description="Leveraged buyout returns and exit strategy modeling"
        )
        
        # Valuation Integration
        ModelFactory.register_model(
            ModelType.VALUATION_INTEGRATION.value,
            ValuationIntegrationModel,
            config_key="ma",
            description="Integrated DCF, comps, and precedent transaction valuation"
        )
        
        # Deal Screening
        ModelFactory.register_model(
            ModelType.DEAL_SCREENING.value,
            DealScreeningModel,
            config_key="ma",
            description="Quantitative deal screening and comparison"
        )
        
    except ImportError as e:
        # M&A models not yet available
        pass
    
    # Options Models (to be registered after refactoring)
    # Credit Risk Models (to be registered after verification)


# Initialize on module load
_init_builtin_models()


class PluginManager:
    """
    Manager for registering and loading custom model plugins.
    
    Allows users to extend Axiom with custom models without modifying core code.
    
    Usage:
        # Define custom model
        class CustomVaRModel(BaseRiskModel):
            def calculate_risk(self, **kwargs):
                # Custom implementation
                pass
        
        # Register plugin
        PluginManager.register_plugin(
            "custom_var",
            CustomVaRModel,
            config_key="var",
            description="Custom VaR methodology"
        )
        
        # Use via factory
        model = ModelFactory.create("custom_var")
    """
    
    _plugins: Dict[str, ModelRegistration] = {}
    
    @classmethod
    def register_plugin(
        cls,
        plugin_name: str,
        model_class: Type[BaseFinancialModel],
        config_key: str,
        description: str = "",
        override: bool = False
    ):
        """
        Register a plugin model.
        
        Args:
            plugin_name: Unique plugin identifier
            model_class: Model class
            config_key: Configuration section
            description: Plugin description
            override: Whether to override existing plugin
            
        Raises:
            ValueError: If plugin already exists and override=False
        """
        if plugin_name in cls._plugins and not override:
            raise ValueError(
                f"Plugin '{plugin_name}' already registered. "
                f"Use override=True to replace."
            )
        
        # Register with factory
        ModelFactory.register_model(
            plugin_name,
            model_class,
            config_key,
            description
        )
        
        # Track in plugin manager
        cls._plugins[plugin_name] = ModelRegistration(
            model_class=model_class,
            config_key=config_key,
            description=description
        )
    
    @classmethod
    def list_plugins(cls) -> Dict[str, str]:
        """List all registered plugins."""
        return {
            name: reg.description
            for name, reg in cls._plugins.items()
        }
    
    @classmethod
    def unregister_plugin(cls, plugin_name: str):
        """Unregister a plugin."""
        if plugin_name in cls._plugins:
            del cls._plugins[plugin_name]
            # Note: Can't easily remove from ModelFactory._registry
            # Consider this a feature - once registered, models stay available


__all__ = [
    "ModelType",
    "ModelFactory",
    "PluginManager",
    "ModelRegistration",
]