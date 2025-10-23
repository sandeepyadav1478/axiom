"""
Market Microstructure Analysis Module
======================================

Institutional-grade market microstructure analysis tools for high-frequency trading,
algorithmic execution, and market quality assessment.

Performance targets:
- 200-500x faster than Bloomberg EMSX
- <1ms per 1000 ticks processing
- <50ms for complete microstructure analysis
- Real-time streaming support

Components:
- Order Flow Analysis: OFI, trade classification, volume profile
- Execution Algorithms: VWAP/TWAP with smart order routing
- Liquidity Metrics: Spread-based, price impact, volume-based measures
- Market Impact Models: Kyle, Almgren-Chriss, Square-Root Law
- Spread Analysis: Decomposition, intraday patterns
- Price Discovery: Information share, market quality indicators
"""

from axiom.models.microstructure.base_model import (
    BaseMarketMicrostructureModel,
    TickData,
    MicrostructureMetrics,
    OrderBookSnapshot,
    TradeData
)
from axiom.models.microstructure.order_flow import (
    OrderFlowAnalyzer,
    OrderFlowMetrics
)
from axiom.models.microstructure.execution_algos import (
    VWAPCalculator,
    TWAPScheduler,
    ExecutionAnalyzer,
    ExecutionBenchmark,
    ExecutionSchedule
)
from axiom.models.microstructure.liquidity import (
    LiquidityAnalyzer,
    LiquidityMetrics
)
from axiom.models.microstructure.market_impact import (
    KyleLambdaModel,
    AlmgrenChrissModel,
    SquareRootLawModel,
    MarketImpactAnalyzer,
    MarketImpactEstimate,
    OptimalTrajectory
)
from axiom.models.microstructure.spread_analysis import (
    SpreadDecompositionModel,
    IntradaySpreadAnalyzer,
    MicrostructureNoiseFilter,
    SpreadComponents,
    IntradaySpreadPattern
)
from axiom.models.microstructure.price_discovery import (
    InformationShareModel,
    MarketQualityAnalyzer,
    PriceDiscoveryMetrics
)

__all__ = [
    # Base Classes and Data Structures
    "BaseMarketMicrostructureModel",
    "TickData",
    "MicrostructureMetrics",
    "OrderBookSnapshot",
    "TradeData",
    
    # Order Flow Analysis
    "OrderFlowAnalyzer",
    "OrderFlowMetrics",
    
    # VWAP/TWAP Execution
    "VWAPCalculator",
    "TWAPScheduler",
    "ExecutionAnalyzer",
    "ExecutionBenchmark",
    "ExecutionSchedule",
    
    # Liquidity Metrics
    "LiquidityAnalyzer",
    "LiquidityMetrics",
    
    # Market Impact Models
    "KyleLambdaModel",
    "AlmgrenChrissModel",
    "SquareRootLawModel",
    "MarketImpactAnalyzer",
    "MarketImpactEstimate",
    "OptimalTrajectory",
    
    # Spread Analysis
    "SpreadDecompositionModel",
    "IntradaySpreadAnalyzer",
    "MicrostructureNoiseFilter",
    "SpreadComponents",
    "IntradaySpreadPattern",
    
    # Price Discovery
    "InformationShareModel",
    "MarketQualityAnalyzer",
    "PriceDiscoveryMetrics",
]