"""Tests for AI provider classes and factory."""

from unittest.mock import Mock, patch

import pytest

from axiom.integrations.ai_providers import (
    AIMessage,
    AIProviderError,
    AIResponse,
    BaseAIProvider,
    ClaudeProvider,
    OpenAIProvider,
    SGLangProvider,
    provider_factory,
)


class TestBaseAIProvider:
    """Test base AI provider functionality."""

    def test_base_provider_initialization(self):
        """Test base provider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAIProvider("test-key")

    def test_ai_message_creation(self):
        """Test AIMessage creation."""
        message = AIMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"

    def test_ai_response_creation(self):
        """Test AIResponse creation."""
        response = AIResponse(
            content="Test response",
            provider="TestProvider",
            model="test-model",
            usage_tokens=100,
            confidence=0.95,
        )
        assert response.content == "Test response"
        assert response.provider == "TestProvider"
        assert response.model == "test-model"
        assert response.usage_tokens == 100
        assert response.confidence == 0.95


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch(
            "axiom.integrations.ai_providers.openai_provider.OpenAI"
        ) as mock_client:
            yield mock_client

    @pytest.fixture
    def openai_provider(self, mock_openai_client):
        """Create OpenAI provider instance."""
        return OpenAIProvider(
            api_key="sk-test123",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
        )

    def test_openai_provider_initialization(self, openai_provider):
        """Test OpenAI provider initialization."""
        assert openai_provider.api_key == "sk-test123"
        assert openai_provider.base_url == "https://api.openai.com/v1"
        assert openai_provider.model_name == "gpt-4o-mini"
        assert openai_provider.provider_name == "OpenAI"

    def test_openai_generate_response(self, openai_provider, mock_openai_client):
        """Test OpenAI response generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.total_tokens = 150

        mock_openai_client.return_value.chat.completions.create.return_value = (
            mock_response
        )

        messages = [AIMessage(role="user", content="Test query")]
        response = openai_provider.generate_response(messages)

        assert isinstance(response, AIResponse)
        assert response.content == "Test response"
        assert response.provider == "OpenAI"
        assert response.usage_tokens == 150

    def test_openai_provider_error_handling(self, openai_provider, mock_openai_client):
        """Test OpenAI error handling."""
        mock_openai_client.return_value.chat.completions.create.side_effect = Exception(
            "API Error"
        )

        messages = [AIMessage(role="user", content="Test query")]

        with pytest.raises(AIProviderError):
            openai_provider.generate_response(messages)

    def test_openai_investment_banking_config(self, openai_provider):
        """Test OpenAI investment banking configuration."""
        config = openai_provider.get_investment_banking_config()

        assert config["temperature"] == 0.05
        assert config["max_tokens"] == 4000
        assert "top_p" in config


class TestClaudeProvider:
    """Test Claude provider implementation."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        with patch(
            "axiom.integrations.ai_providers.claude_provider.Anthropic"
        ) as mock_client:
            yield mock_client

    @pytest.fixture
    def claude_provider(self, mock_anthropic_client):
        """Create Claude provider instance."""
        return ClaudeProvider(
            api_key="sk-ant-test123", model_name="claude-3-sonnet-20240229"
        )

    def test_claude_provider_initialization(self, claude_provider):
        """Test Claude provider initialization."""
        assert claude_provider.api_key == "sk-ant-test123"
        assert claude_provider.model_name == "claude-3-sonnet-20240229"
        assert claude_provider.provider_name == "Claude"

    def test_claude_generate_response(self, claude_provider, mock_anthropic_client):
        """Test Claude response generation."""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Claude test response"
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 100

        mock_anthropic_client.return_value.messages.create.return_value = mock_response

        messages = [
            AIMessage(role="system", content="You are a financial analyst"),
            AIMessage(role="user", content="Analyze this company"),
        ]
        response = claude_provider.generate_response(messages)

        assert isinstance(response, AIResponse)
        assert response.content == "Claude test response"
        assert response.provider == "Claude"
        assert response.usage_tokens == 150  # input + output
        assert response.confidence == 0.90  # Claude default


class TestSGLangProvider:
    """Test SGLang provider implementation."""

    @pytest.fixture
    def mock_sglang_client(self):
        """Mock SGLang OpenAI-compatible client."""
        with patch(
            "axiom.integrations.ai_providers.sglang_provider.OpenAI"
        ) as mock_client:
            yield mock_client

    @pytest.fixture
    def sglang_provider(self, mock_sglang_client):
        """Create SGLang provider instance."""
        return SGLangProvider(
            api_key="local-inference",
            base_url="http://localhost:30000/v1",
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        )

    def test_sglang_provider_initialization(self, sglang_provider):
        """Test SGLang provider initialization."""
        assert sglang_provider.api_key == "local-inference"
        assert sglang_provider.base_url == "http://localhost:30000/v1"
        assert sglang_provider.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"
        assert sglang_provider.provider_name == "SGLang"

    def test_sglang_financial_analysis_prompt(self, sglang_provider):
        """Test SGLang financial analysis prompt generation."""
        company_info = {"name": "Tesla Inc"}
        messages = sglang_provider.financial_analysis_prompt(
            "ma_due_diligence", company_info, "M&A analysis context"
        )

        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert "Tesla Inc" in messages[1].content
        assert "M&A analysis context" in messages[1].content

    def test_sglang_check_server_status(self, sglang_provider, mock_sglang_client):
        """Test SGLang server status check."""
        # Mock models list
        mock_models = Mock()
        mock_models.data = [Mock(), Mock()]
        mock_models.data[0].id = "model1"
        mock_models.data[1].id = "model2"

        mock_sglang_client.return_value.models.list.return_value = mock_models

        status = sglang_provider.check_server_status()

        assert status["status"] == "available"
        assert status["base_url"] == "http://localhost:30000/v1"
        assert len(status["models"]) == 2


class TestProviderFactory:
    """Test AI provider factory."""

    def test_provider_factory_initialization(self):
        """Test provider factory initializes with available providers."""
        # This test may need mocking depending on actual environment
        available_providers = provider_factory.get_available_providers()
        assert isinstance(available_providers, list)

    def test_provider_factory_get_provider(self):
        """Test getting specific provider from factory."""
        # Mock a provider in the factory
        mock_provider = Mock(spec=BaseAIProvider)
        provider_factory._providers["test_provider"] = mock_provider

        retrieved_provider = provider_factory.get_provider("test_provider")
        assert retrieved_provider == mock_provider

        # Clean up
        del provider_factory._providers["test_provider"]

    def test_provider_factory_test_all_providers(self):
        """Test provider availability testing."""
        # Mock some providers
        mock_provider1 = Mock(spec=BaseAIProvider)
        mock_provider1.is_available.return_value = True
        mock_provider2 = Mock(spec=BaseAIProvider)
        mock_provider2.is_available.return_value = False

        provider_factory._providers["working_provider"] = mock_provider1
        provider_factory._providers["broken_provider"] = mock_provider2

        results = provider_factory.test_all_providers()

        assert results["working_provider"]
        assert not results["broken_provider"]

        # Clean up
        del provider_factory._providers["working_provider"]
        del provider_factory._providers["broken_provider"]

    def test_provider_factory_info(self):
        """Test getting provider information."""
        # Mock a provider
        mock_provider = Mock(spec=BaseAIProvider)
        mock_provider.get_provider_info.return_value = {
            "name": "TestProvider",
            "model": "test-model",
            "available": True,
        }

        provider_factory._providers["test_provider"] = mock_provider

        info = provider_factory.get_provider_info()
        assert "test_provider" in info
        assert info["test_provider"]["name"] == "TestProvider"

        # Clean up
        del provider_factory._providers["test_provider"]


class TestInvestmentBankingPrompts:
    """Test investment banking specific prompt generation."""

    def test_financial_analysis_prompt_due_diligence(self):
        """Test due diligence prompt generation."""
        provider = OpenAIProvider("test-key")
        company_info = {"name": "Microsoft Corporation"}

        messages = provider.financial_analysis_prompt(
            "due_diligence", company_info, "Acquisition target analysis"
        )

        assert len(messages) == 2
        system_prompt = messages[0].content
        user_prompt = messages[1].content

        # Check system prompt has investment banking context
        assert "investment banking" in system_prompt.lower()
        assert "due diligence" in system_prompt.lower()

        # Check user prompt has specific analysis request
        assert "Microsoft Corporation" in user_prompt
        assert "due diligence" in user_prompt.lower()
        assert "Acquisition target analysis" in user_prompt

    def test_financial_analysis_prompt_valuation(self):
        """Test valuation prompt generation."""
        provider = OpenAIProvider("test-key")
        company_info = {"name": "Tesla Inc"}

        messages = provider.financial_analysis_prompt("valuation", company_info)

        user_prompt = messages[1].content
        assert "Tesla Inc" in user_prompt
        assert "valuation" in user_prompt.lower()
        assert "DCF" in user_prompt or "comparable" in user_prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__])
