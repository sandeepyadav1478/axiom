"""AI Provider Factory for Investment Banking Analytics with API Key Failover."""

from axiom.config.ai_layer_config import AnalysisLayer, ai_layer_mapping
from axiom.config.settings import settings
from axiom.core.api_management.failover_key_manager import failover_manager

from .base_ai_provider import AIProviderError, BaseAIProvider
from .claude_provider import ClaudeProvider
from .openai_provider import OpenAIProvider
from .sglang_provider import SGLangProvider


class AIProviderFactory:
    """Factory for creating and managing AI provider instances."""

    # Registry of available provider classes
    PROVIDER_CLASSES: dict[str, type[BaseAIProvider]] = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "sglang": SGLangProvider,
    }

    def __init__(self):
        self._providers: dict[str, BaseAIProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize all configured AI providers with failover support."""
        configured_providers = settings.get_configured_providers()

        for provider_name in configured_providers:
            try:
                # Configure failover keys for this provider
                provider_keys = settings.get_provider_keys(provider_name)
                if provider_keys:
                    failover_manager.configure_provider(
                        provider_name,
                        provider_keys,
                        enable_rotation=settings.is_rotation_enabled(provider_name)
                    )

                config = settings.get_provider_config(provider_name)
                provider_class = self.PROVIDER_CLASSES.get(provider_name)

                if provider_class:
                    # Use current active key from failover manager
                    current_key = failover_manager.get_active_key(provider_name) or config["api_key"]
                    
                    provider = provider_class(
                        api_key=current_key,
                        base_url=config.get("base_url"),
                        model_name=config["model_name"],
                    )
                    self._providers[provider_name] = provider
                    
                    key_count = len(provider_keys) if provider_keys else 1
                    rotation_status = "with failover" if settings.is_rotation_enabled(provider_name) else "single key"
                    print(f"✅ Initialized {provider_name} provider ({key_count} keys, {rotation_status})")
                else:
                    print(f"⚠️  Unknown provider type: {provider_name}")

            except Exception as e:
                print(f"❌ Failed to initialize {provider_name}: {str(e)}")

    def get_provider(self, provider_name: str) -> BaseAIProvider | None:
        """Get a specific AI provider by name."""
        return self._providers.get(provider_name.lower())

    def get_available_providers(self) -> list[str]:
        """Get list of available AI providers."""
        return list(self._providers.keys())

    def get_provider_for_layer(self, layer: AnalysisLayer) -> BaseAIProvider | None:
        """Get the optimal AI provider for a specific analysis layer."""
        layer_config = ai_layer_mapping.get_layer_config(layer)

        # Try primary provider first
        primary_provider = self.get_provider(layer_config.primary_provider.value)
        if primary_provider and primary_provider.is_available():
            return primary_provider

        # Fall back to available fallback providers
        for fallback in layer_config.fallback_providers:
            provider = self.get_provider(fallback.value)
            if provider and provider.is_available():
                print(f"🔄 Using fallback provider {fallback.value} for {layer.value}")
                return provider

        # Last resort: any available provider
        for provider in self._providers.values():
            if provider.is_available():
                print(
                    f"⚠️  Using last resort provider {provider.provider_name} for {layer.value}"
                )
                return provider

        raise AIProviderError(
            "Factory", f"No available providers for layer {layer.value}"
        )

    def test_all_providers(self) -> dict[str, bool]:
        """Test availability of all configured providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = provider.is_available()
            except Exception:
                results[name] = False
        return results

    def get_provider_info(self) -> dict[str, dict]:
        """Get information about all providers."""
        return {
            name: provider.get_provider_info()
            for name, provider in self._providers.items()
        }

    def add_provider(self, provider_name: str, provider: BaseAIProvider):
        """Add a custom provider instance."""
        self._providers[provider_name.lower()] = provider

    def remove_provider(self, provider_name: str):
        """Remove a provider."""
        self._providers.pop(provider_name.lower(), None)

    def get_consensus_providers(self, layer: AnalysisLayer) -> list[BaseAIProvider]:
        """Get providers for consensus analysis (if enabled for layer)."""
        layer_config = ai_layer_mapping.get_layer_config(layer)

        if not layer_config.use_consensus:
            # Return single provider if consensus not required
            provider = self.get_provider_for_layer(layer)
            return [provider] if provider else []

        # Get multiple providers for consensus
        providers = []

        # Primary provider
        primary = self.get_provider(layer_config.primary_provider.value)
        if primary and primary.is_available():
            providers.append(primary)

        # Add fallback providers for consensus
        for fallback in layer_config.fallback_providers:
            provider = self.get_provider(fallback.value)
            if provider and provider.is_available() and provider not in providers:
                providers.append(provider)
                if len(providers) >= 2:  # Limit consensus to 2-3 providers
                    break

        return providers


# Global provider factory instance
provider_factory = AIProviderFactory()


def get_ai_provider(provider_name: str) -> BaseAIProvider | None:
    """Convenience function to get an AI provider."""
    return provider_factory.get_provider(provider_name)


def get_layer_provider(layer: AnalysisLayer) -> BaseAIProvider | None:
    """Convenience function to get optimal provider for analysis layer."""
    return provider_factory.get_provider_for_layer(layer)


def test_providers() -> dict[str, bool]:
    """Convenience function to test all providers."""
    return provider_factory.test_all_providers()
