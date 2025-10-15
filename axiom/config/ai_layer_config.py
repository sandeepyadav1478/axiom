"""
AI Layer Configuration for Investment Banking Analytics
Determines which AI providers to use at different analysis layers
"""

from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel


class AnalysisLayer(Enum):
    """Different layers of investment banking analysis"""

    PLANNER = "planner"  # Initial query planning and decomposition
    TASK_RUNNER = "task_runner"  # Parallel research execution
    OBSERVER = "observer"  # Final synthesis and validation
    DUE_DILIGENCE = "due_diligence"  # Deep company analysis
    VALUATION = "valuation"  # Financial modeling and valuation
    MARKET_INTELLIGENCE = "market_intel"  # Market and competitive analysis


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

    # Default configuration - users can override
    layer_configs: Dict[AnalysisLayer, LayerAIConfig] = {
        AnalysisLayer.PLANNER: LayerAIConfig(
            layer=AnalysisLayer.PLANNER,
            primary_provider=AIProviderType.CLAUDE,  # Best for complex reasoning
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.1,
            max_tokens=1500,
        ),
        AnalysisLayer.TASK_RUNNER: LayerAIConfig(
            layer=AnalysisLayer.TASK_RUNNER,
            primary_provider=AIProviderType.OPENAI,  # Good for structured tasks
            fallback_providers=[AIProviderType.CLAUDE],
            temperature=0.1,
            max_tokens=2000,
        ),
        AnalysisLayer.OBSERVER: LayerAIConfig(
            layer=AnalysisLayer.OBSERVER,
            primary_provider=AIProviderType.CLAUDE,  # Best for synthesis
            fallback_providers=[AIProviderType.OPENAI],
            temperature=0.05,  # Very conservative for final analysis
            max_tokens=3000,
        ),
        AnalysisLayer.DUE_DILIGENCE: LayerAIConfig(
            layer=AnalysisLayer.DUE_DILIGENCE,
            primary_provider=AIProviderType.CLAUDE,  # Superior reasoning for DD
            fallback_providers=[AIProviderType.OPENAI],
            use_consensus=True,  # Critical analysis needs consensus
            temperature=0.05,
            max_tokens=4000,
        ),
        AnalysisLayer.VALUATION: LayerAIConfig(
            layer=AnalysisLayer.VALUATION,
            primary_provider=AIProviderType.OPENAI,  # Good for structured math
            fallback_providers=[AIProviderType.CLAUDE],
            temperature=0.1,
            max_tokens=3000,
        ),
        AnalysisLayer.MARKET_INTELLIGENCE: LayerAIConfig(
            layer=AnalysisLayer.MARKET_INTELLIGENCE,
            primary_provider=AIProviderType.CLAUDE,  # Complex market analysis
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
