"""
M&A Deal Screening and Comparison
==================================

Quantitative deal screening and comparison framework:
- Strategic fit scoring
- Financial attractiveness analysis
- Risk assessment scoring
- Synergy potential estimation
- Integration difficulty assessment
- Overall deal ranking
- Precedent deal comparison

Performance target: <15ms for single deal scoring
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import time

from axiom.models.ma.base_model import BaseMandAModel, DealScreeningResult
from axiom.models.base.base_model import ModelResult
from axiom.config.model_config import MandAConfig


@dataclass
class DealMetrics:
    """Key metrics for deal screening."""
    ebitda_multiple: float
    revenue_multiple: float
    ebitda_margin: float
    revenue_growth: float
    market_share: float
    customer_concentration: float
    technology_score: float


class DealScreeningModel(BaseMandAModel):
    """
    Quantitative deal screening and comparison model.
    
    Provides systematic framework for evaluating and comparing M&A opportunities
    across multiple dimensions including strategic fit, financial metrics,
    risk factors, and synergy potential.
    
    Performance: <15ms per deal evaluation
    """
    
    def __init__(
        self,
        config: Optional[MandAConfig] = None,
        enable_logging: bool = True,
        enable_performance_tracking: bool = True
    ):
        """Initialize deal screening model."""
        super().__init__(
            config=config.__dict__ if config and hasattr(config, '__dict__') else (config or {}),
            enable_logging=enable_logging,
            enable_performance_tracking=enable_performance_tracking
        )
    
    def calculate(
        self,
        deal_id: str,
        deal_metrics: DealMetrics,
        strategic_factors: Dict[str, float],
        financial_factors: Dict[str, float],
        risk_factors: Dict[str, float],
        synergy_estimate: float = 0.0,
        purchase_price: float = 0.0,
        comparable_deals: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResult[DealScreeningResult]:
        """
        Score and evaluate an M&A deal.
        
        Args:
            deal_id: Unique deal identifier
            deal_metrics: Key deal metrics
            strategic_factors: Strategic fit factors (0-10 scale)
            financial_factors: Financial attractiveness factors
            risk_factors: Risk assessment factors (0-10, higher = more risk)
            synergy_estimate: Estimated synergy value
            purchase_price: Proposed purchase price
            comparable_deals: List of comparable precedent deals
            
        Returns:
            ModelResult containing DealScreeningResult
        """
        start_time = time.perf_counter()
        
        # Calculate strategic fit score (0-100)
        strategic_fit_score = self._calculate_strategic_fit(strategic_factors)
        
        # Calculate financial attractiveness (0-100)
        financial_attractiveness = self._calculate_financial_attractiveness(
            deal_metrics, financial_factors
        )
        
        # Calculate risk score (0-100, lower is better)
        risk_score = self._calculate_risk_score(risk_factors, deal_metrics)
        
        # Calculate synergy potential
        synergy_potential = synergy_estimate
        
        # Calculate integration difficulty (0-100, lower is easier)
        integration_difficulty = self._calculate_integration_difficulty(
            deal_metrics, strategic_factors
        )
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(
            strategic_fit_score,
            financial_attractiveness,
            risk_score,
            synergy_potential,
            purchase_price
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_score, risk_score, synergy_potential, purchase_price
        )
        
        # Key metrics summary
        key_metrics = {
            'ebitda_multiple': deal_metrics.ebitda_multiple,
            'revenue_growth': deal_metrics.revenue_growth,
            'ebitda_margin': deal_metrics.ebitda_margin,
            'market_share': deal_metrics.market_share,
            'synergy_as_pct_price': (synergy_estimate / purchase_price * 100) if purchase_price > 0 else 0
        }
        
        result = DealScreeningResult(
            deal_id=deal_id,
            strategic_fit_score=strategic_fit_score,
            financial_attractiveness=financial_attractiveness,
            risk_score=risk_score,
            synergy_potential=synergy_potential,
            integration_difficulty=integration_difficulty,
            overall_score=overall_score,
            recommendation=recommendation,
            key_metrics=key_metrics,
            comparable_deals=comparable_deals
        )
        
        execution_time_ms = self._track_performance("deal_screening", start_time)
        metadata = self._create_metadata(execution_time_ms)
        
        return ModelResult(value=result, metadata=metadata, success=True)
    
    def _calculate_strategic_fit(
        self,
        factors: Dict[str, float]
    ) -> float:
        """
        Calculate strategic fit score.
        
        Factors might include:
        - Market adjacency (0-10)
        - Technology alignment (0-10)
        - Customer base overlap (0-10)
        - Geographic expansion (0-10)
        - Product portfolio fit (0-10)
        """
        if not factors:
            return 50.0  # Neutral score
        
        # Weighted average of factors
        weights = {
            'market_adjacency': 0.25,
            'technology_alignment': 0.20,
            'customer_overlap': 0.20,
            'geographic_expansion': 0.20,
            'product_fit': 0.15
        }
        
        score = sum(
            factors.get(factor, 5.0) * weight
            for factor, weight in weights.items()
        )
        
        # Scale to 0-100
        return min(100, max(0, score * 10))
    
    def _calculate_financial_attractiveness(
        self,
        metrics: DealMetrics,
        factors: Dict[str, float]
    ) -> float:
        """Calculate financial attractiveness score."""
        scores = []
        
        # Valuation multiple score (lower is better, up to 12x EBITDA)
        if metrics.ebitda_multiple <= 8:
            scores.append(100)
        elif metrics.ebitda_multiple <= 12:
            scores.append(100 - (metrics.ebitda_multiple - 8) * 10)
        else:
            scores.append(max(0, 60 - (metrics.ebitda_multiple - 12) * 5))
        
        # Growth score
        if metrics.revenue_growth >= 0.20:
            scores.append(100)
        elif metrics.revenue_growth >= 0.10:
            scores.append(80)
        elif metrics.revenue_growth >= 0.05:
            scores.append(60)
        else:
            scores.append(40)
        
        # Margin score
        if metrics.ebitda_margin >= 0.25:
            scores.append(100)
        elif metrics.ebitda_margin >= 0.15:
            scores.append(80)
        elif metrics.ebitda_margin >= 0.10:
            scores.append(60)
        else:
            scores.append(40)
        
        # Additional financial factors
        for factor, value in factors.items():
            scores.append(value * 10)  # Scale 0-10 to 0-100
        
        return np.mean(scores) if scores else 50.0
    
    def _calculate_risk_score(
        self,
        factors: Dict[str, float],
        metrics: DealMetrics
    ) -> float:
        """
        Calculate risk score (0-100, higher = more risky).
        
        Risk factors might include:
        - Regulatory risk (0-10)
        - Integration complexity (0-10)
        - Customer concentration (0-10)
        - Technology risk (0-10)
        - Market risk (0-10)
        """
        scores = []
        
        # Risk factors
        for factor, value in factors.items():
            scores.append(value * 10)  # Scale to 0-100
        
        # Customer concentration risk
        if metrics.customer_concentration > 0.30:
            scores.append(80)  # High risk
        elif metrics.customer_concentration > 0.20:
            scores.append(60)
        else:
            scores.append(30)
        
        return np.mean(scores) if scores else 50.0
    
    def _calculate_integration_difficulty(
        self,
        metrics: DealMetrics,
        strategic_factors: Dict[str, float]
    ) -> float:
        """Calculate integration difficulty score (0-100, higher = harder)."""
        difficulty_factors = []
        
        # Technology complexity
        tech_score = strategic_factors.get('technology_alignment', 5.0)
        difficulty_factors.append((10 - tech_score) * 10)  # Invert
        
        # Size/scale difference
        if metrics.market_share < 0.05:
            difficulty_factors.append(30)  # Small, easier
        elif metrics.market_share < 0.15:
            difficulty_factors.append(50)
        else:
            difficulty_factors.append(70)  # Large, harder
        
        # Geographic complexity
        geo_score = strategic_factors.get('geographic_expansion', 5.0)
        if geo_score > 7:
            difficulty_factors.append(70)  # International expansion is hard
        else:
            difficulty_factors.append(40)
        
        return np.mean(difficulty_factors) if difficulty_factors else 50.0
    
    def _calculate_overall_score(
        self,
        strategic_fit: float,
        financial_attractiveness: float,
        risk_score: float,
        synergy_potential: float,
        purchase_price: float
    ) -> float:
        """Calculate weighted overall deal score."""
        # Weights
        w_strategic = 0.30
        w_financial = 0.30
        w_risk = 0.25  # Inverse (lower risk = higher score)
        w_synergy = 0.15
        
        # Normalize synergy score (as % of purchase price)
        synergy_score = min(100, (synergy_potential / purchase_price * 100)) if purchase_price > 0 else 50
        
        # Weighted average (risk is inverted)
        overall = (
            strategic_fit * w_strategic +
            financial_attractiveness * w_financial +
            (100 - risk_score) * w_risk +
            synergy_score * w_synergy
        )
        
        return overall
    
    def _generate_recommendation(
        self,
        overall_score: float,
        risk_score: float,
        synergy_potential: float,
        purchase_price: float
    ) -> str:
        """Generate deal recommendation."""
        synergy_pct = (synergy_potential / purchase_price * 100) if purchase_price > 0 else 0
        
        if overall_score >= 75 and risk_score < 50:
            return "Strong Buy"
        elif overall_score >= 65 and synergy_pct >= 15:
            return "Buy"
        elif overall_score >= 50 and risk_score < 60:
            return "Consider"
        elif overall_score >= 40:
            return "Hold"
        else:
            return "Pass"
    
    def compare_deals(
        self,
        deals: List[Tuple[str, DealMetrics, Dict, Dict, Dict, float, float]]
    ) -> List[DealScreeningResult]:
        """
        Compare and rank multiple deals.
        
        Args:
            deals: List of deal tuples (id, metrics, strategic, financial, risk, synergies, price)
            
        Returns:
            List of DealScreeningResult sorted by overall score
        """
        results = []
        
        for deal in deals:
            result = self.calculate(
                deal_id=deal[0],
                deal_metrics=deal[1],
                strategic_factors=deal[2],
                financial_factors=deal[3],
                risk_factors=deal[4],
                synergy_estimate=deal[5],
                purchase_price=deal[6]
            )
            results.append(result.value)
        
        # Sort by overall score (descending)
        results.sort(key=lambda x: x.overall_score, reverse=True)
        
        return results
    
    def calculate_value(
        self,
        deal_id: str,
        deal_metrics: DealMetrics,
        **kwargs
    ) -> float:
        """Calculate overall deal score."""
        result = self.calculate(deal_id, deal_metrics, **kwargs)
        return result.value.overall_score
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate inputs."""
        super().validate_inputs(**kwargs)
        
        if 'deal_metrics' in kwargs:
            metrics = kwargs['deal_metrics']
            if metrics.ebitda_multiple < 0:
                raise ValueError("EBITDA multiple must be non-negative")
        
        return True


__all__ = ["DealScreeningModel", "DealMetrics"]