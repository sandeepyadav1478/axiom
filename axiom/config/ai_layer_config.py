"""
AI Layer Configuration for Investment Banking Analytics
Determines which AI providers to use at different analysis layers
"""

from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel


class AnalysisLayer(Enum):
    """Different layers of investment banking analysis - M&A Priority Focus"""

    PLANNER = "planner"  # Initial query planning and decomposition
    TASK_RUNNER = "task_runner"  # Parallel research execution
    OBSERVER = "observer"  # Final synthesis and validation

    # M&A Priority Features (Phase 1 Development)
    MA_DUE_DILIGENCE = "ma_due_diligence"  # M&A-specific due diligence analysis
    MA_VALUATION = "ma_valuation"  # M&A valuation and synergy analysis
    MA_MARKET_ANALYSIS = "ma_market_analysis"  # M&A market and competitive impact
    MA_STRATEGIC_FIT = "ma_strategic_fit"  # Strategic fit and integration analysis

    # Future Investment Banking Features (Phase 2+)
    DUE_DILIGENCE = "due_diligence"  # General due diligence
    VALUATION = "valuation"  # General valuation
    MARKET_INTELLIGENCE = "market_intel"  # General market analysis


class AIProviderType(Enum):
    """Supported AI provider types"""

    OPENAI = "openai"
    CLAUDE = "claude"
    SGLANG = "sglang"
    HUGGINGFACE = "huggingface"


class LayerAIConfig(BaseModel):
    """Configuration for AI provider at specific layer"""

    layer: AnalysisLayer
    primary_provider: AIProviderType
    fallback_providers: List[AIProviderType] = []
    use_consensus: bool = False  # Whether to use multiple providers for consensus
    temperature: float = 0.1  # Conservative for financial analysis
    max_tokens: int = 2000


class AILayerMapping(BaseModel):
    """
    Complete mapping of AI providers to analysis layers
    Users can configure which AI provider handles which type of analysis
    """

    # M&A-Focused Configuration (Phase 1 Priority)
    layer_configs: Dict[AnalysisLayer, LayerAIConfig] = {
        # Core System Layers
        AnalysisLayer.PLANNER: LayerAIConfig(
            layer=AnalysisLayer.PLANNER,
            primary_provider=AIProviderType.CLAUDE,  # Best for complex M&A reasoning
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.1,
            max_tokens=1500,
        ),
        AnalysisLayer.TASK_RUNNER: LayerAIConfig(
            layer=AnalysisLayer.TASK_RUNNER,
            primary_provider=AIProviderType.OPENAI,  # Good for structured M&A tasks
            fallback_providers=[AIProviderType.CLAUDE],
            temperature=0.1,
            max_tokens=2000,
        ),
        AnalysisLayer.OBSERVER: LayerAIConfig(
            layer=AnalysisLayer.OBSERVER,
            primary_provider=AIProviderType.CLAUDE,  # Best for M&A synthesis
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.05,  # Very conservative for M&A decisions
            max_tokens=4000,
        ),
        # M&A Priority Features (Phase 1 Development)
        AnalysisLayer.MA_DUE_DILIGENCE: LayerAIConfig(
            layer=AnalysisLayer.MA_DUE_DILIGENCE,
            primary_provider=AIProviderType.CLAUDE,  # Superior reasoning for M&A DD
            fallback_providers=[AIProviderType.OPENAI],
            use_consensus=True,  # Critical M&A analysis needs consensus
            temperature=0.03,  # Extremely conservative for M&A decisions
            max_tokens=5000,
        ),
        AnalysisLayer.MA_VALUATION: LayerAIConfig(
            layer=AnalysisLayer.MA_VALUATION,
            primary_provider=AIProviderType.OPENAI,  # Excellent for structured M&A math
            fallback_providers=[AIProviderType.CLAUDE],
            use_consensus=True,  # M&A valuations need multiple perspectives
            temperature=0.05,
            max_tokens=4000,
        ),
        AnalysisLayer.MA_MARKET_ANALYSIS: LayerAIConfig(
            layer=AnalysisLayer.MA_MARKET_ANALYSIS,
            primary_provider=AIProviderType.CLAUDE,  # Complex M&A market dynamics
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.1,
            max_tokens=3500,
        ),
        AnalysisLayer.MA_STRATEGIC_FIT: LayerAIConfig(
            layer=AnalysisLayer.MA_STRATEGIC_FIT,
            primary_provider=AIProviderType.CLAUDE,  # Strategic reasoning expertise
            fallback_providers=[AIProviderType.OPENAI],
            use_consensus=True,  # Strategic decisions need multiple viewpoints
            temperature=0.05,
            max_tokens=3000,
        ),
        # Future Features (Phase 2+) - Lower priority
        AnalysisLayer.DUE_DILIGENCE: LayerAIConfig(
            layer=AnalysisLayer.DUE_DILIGENCE,
            primary_provider=AIProviderType.CLAUDE,
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.1,
            max_tokens=3000,
        ),
        AnalysisLayer.VALUATION: LayerAIConfig(
            layer=AnalysisLayer.VALUATION,
            primary_provider=AIProviderType.OPENAI,
            fallback_providers=[AIProviderType.CLAUDE],
            temperature=0.1,
            max_tokens=3000,
        ),
        AnalysisLayer.MARKET_INTELLIGENCE: LayerAIConfig(
            layer=AnalysisLayer.MARKET_INTELLIGENCE,
            primary_provider=AIProviderType.CLAUDE,
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.1,
            max_tokens=2500,
        ),
    }

    def get_layer_config(self, layer: AnalysisLayer) -> LayerAIConfig:
        """Get AI configuration for specific analysis layer"""
        return self.layer_configs.get(layer, self._get_default_config(layer))

    def _get_default_config(self, layer: AnalysisLayer) -> LayerAIConfig:
        """Get default configuration for unknown layers"""
        return LayerAIConfig(
            layer=layer,
            primary_provider=AIProviderType.CLAUDE,  # Claude as safe default
            fallback_providers=[AIProviderType.OPENAI],
        )

    def get_required_providers(self) -> List[AIProviderType]:
        """Get list of all AI providers required by current configuration"""
        providers = set()

        for config in self.layer_configs.values():
            providers.add(config.primary_provider)
            providers.update(config.fallback_providers)

        return list(providers)

    def override_layer_provider(
        self,
        layer: AnalysisLayer,
        provider: AIProviderType,
        fallbacks: Optional[List[AIProviderType]] = None,
    ) -> None:
        """
        Override AI provider for specific layer
        Allows users to customize which AI handles which analysis type
        """
        config = self.layer_configs.get(layer, self._get_default_config(layer))
        config.primary_provider = provider
        if fallbacks:
            config.fallback_providers = fallbacks
        self.layer_configs[layer] = config


# Global AI layer mapping instance
ai_layer_mapping = AILayerMapping()
