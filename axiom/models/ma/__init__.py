"""
M&A Quantitative Models
=======================

Comprehensive M&A quantitative modeling framework providing:
- Synergy Valuation: Cost and revenue synergies with NPV analysis
- Deal Financing: Capital structure optimization and accretion/dilution
- Merger Arbitrage: Spread analysis, hedging strategies, position sizing
- Valuation Integration: DCF, comps, precedent transactions
- LBO Modeling: Leveraged buyout returns and exit modeling
- Deal Screening: Quantitative deal comparison and screening

Designed to rival Goldman Sachs and Morgan Stanley M&A models with
100-500x better performance.

Performance Targets:
- Synergy Valuation: <50ms
- Deal Financing: <30ms
- Merger Arbitrage: <20ms (spread analysis), <10ms (position sizing)
- Valuation Integration: <40ms
- LBO Modeling: <60ms
- Deal Screening: <15ms per deal
"""

# Base classes and data structures
from axiom.models.ma.base_model import (
    BaseMandAModel,
    SynergyEstimate,
    DealFinancing,
    MergerArbPosition,
    LBOAnalysis,
    DealScreeningResult
)

# Models
from axiom.models.ma.synergy_valuation import (
    SynergyValuationModel,
    CostSynergy,
    RevenueSynergy
)

from axiom.models.ma.deal_financing import (
    DealFinancingModel,
    FinancingSource
)

from axiom.models.ma.merger_arbitrage import (
    MergerArbitrageModel,
    DealEvent
)

from axiom.models.ma.lbo_modeling import (
    LBOModel,
    OperationalImprovements,
    DebtStructure
)

from axiom.models.ma.valuation_integration import (
    ValuationIntegrationModel,
    ValuationResult
)

from axiom.models.ma.deal_screening import (
    DealScreeningModel,
    DealMetrics
)

__all__ = [
    # Base classes
    "BaseMandAModel",
    
    # Data structures
    "SynergyEstimate",
    "DealFinancing",
    "MergerArbPosition",
    "LBOAnalysis",
    "DealScreeningResult",
    "ValuationResult",
    
    # Models
    "SynergyValuationModel",
    "DealFinancingModel",
    "MergerArbitrageModel",
    "LBOModel",
    "ValuationIntegrationModel",
    "DealScreeningModel",
    
    # Supporting classes
    "CostSynergy",
    "RevenueSynergy",
    "FinancingSource",
    "DealEvent",
    "OperationalImprovements",
    "DebtStructure",
    "DealMetrics",
]

__version__ = "1.0.0"