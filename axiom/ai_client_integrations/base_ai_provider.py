"""
Base AI Provider Abstract Class for Investment Banking Analytics
Multi-provider AI integration supporting OpenAI, Claude, SGLang, and others
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class AIMessage(BaseModel):
    """Standardized message format for all AI providers"""

    role: str  # "user", "assistant", "system"
    content: str


class AIResponse(BaseModel):
    """Standardized response format from all AI providers"""

    content: str
    provider: str
    model: str
    usage_tokens: Optional[int] = None
    confidence: Optional[float] = None


class BaseAIProvider(ABC):
    """
    Abstract base class for all AI provider integrations

    Supports: OpenAI, Claude, SGLang, Hugging Face, local models, etc.

    Design Principles:
    - Simple: Just API key + endpoint configuration
    - Standardized: Same interface regardless of provider
    - Investment Banking Focused: Optimized for financial analysis
    - Flexible: Easy to add new providers
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "default",
        **kwargs,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.provider_name = self.__class__.__name__.replace("Provider", "")
        self.config = kwargs

    @abstractmethod
    def generate_response(
        self,
        messages: List[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,  # Conservative for financial analysis
        **kwargs,
    ) -> AIResponse:
        """
        Generate response from AI provider

        Args:
            messages: List of conversation messages
            max_tokens: Maximum response length
            temperature: Sampling temperature (low for finance)

        Returns:
            Standardized AI response
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and properly configured"""
        pass

    def financial_analysis_prompt(
        self, analysis_type: str, company_info: Dict[str, Any], context: str = ""
    ) -> List[AIMessage]:
        """
        Create optimized prompts for investment banking analysis

        Args:
            analysis_type: "due_diligence", "valuation", "market_analysis", etc.
            company_info: Company data and context
            context: Additional research context

        Returns:
            Optimized message list for financial analysis
        """

        # Investment banking system prompt
        system_prompt = """You are a senior investment banking analyst with expertise in:
- M&A due diligence and valuation
- Financial statement analysis  
- Market intelligence and competitive analysis
- Risk assessment and regulatory compliance

Provide detailed, structured analysis with:
1. Clear methodology and assumptions
2. Quantitative analysis where possible
3. Risk factors and mitigation strategies
4. Confidence levels for key findings
5. Sources and citations for all claims

Be precise, professional, and conservative in assessments."""

        # Analysis-specific user prompt
        analysis_prompts = {
            "due_diligence": f"Perform comprehensive due diligence analysis for {company_info.get('name', 'target company')}. Include financial health, operational risks, market position, regulatory compliance, and strategic value assessment.",
            "valuation": f"Conduct detailed valuation analysis for {company_info.get('name', 'target company')} using DCF, comparable company analysis, and precedent transactions. Provide fair value range with confidence intervals.",
            "market_analysis": f"Analyze the market dynamics and competitive landscape for {company_info.get('name', 'target company')}'s industry sector. Include market size, growth trends, key players, and strategic implications.",
            "risk_assessment": f"Perform comprehensive risk analysis for {company_info.get('name', 'target company')} covering operational, financial, regulatory, and market risks with mitigation strategies.",
        }

        user_prompt = analysis_prompts.get(
            analysis_type,
            f"Analyze {company_info.get('name', 'target company')} for investment banking purposes.",
        )

        if context:
            user_prompt += f"\n\nAdditional Context: {context}"

        return [
            AIMessage(role="system", content=system_prompt),
            AIMessage(role="user", content=user_prompt),
        ]

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information and configuration"""
        return {
            "name": self.provider_name,
            "model": self.model_name,
            "base_url": self.base_url,
            "available": self.is_available(),
            "config": {k: v for k, v in self.config.items() if "key" not in k.lower()},
        }


class AIProviderError(Exception):
    """Custom exception for AI provider errors"""

    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")
