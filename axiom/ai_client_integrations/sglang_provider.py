"""SGLang provider implementation for local inference in Investment Banking Analytics."""

import asyncio
from typing import List, Optional, Dict, Any
import openai
from openai import OpenAI, AsyncOpenAI

from .base_ai_provider import BaseAIProvider, AIMessage, AIResponse, AIProviderError


class SGLangProvider(BaseAIProvider):
    """SGLang provider implementation for local inference with investment banking optimization."""

    def __init__(
        self,
        api_key: str = "local-inference",
        base_url: str = "http://localhost:30000/v1",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        **kwargs,
    ):
        super().__init__(api_key, base_url, model_name, **kwargs)

        # Initialize OpenAI-compatible client for SGLang
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_response(
        self,
        messages: List[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response using SGLang local inference."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Call SGLang API (OpenAI-compatible)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract response content
            content = response.choices[0].message.content
            usage_tokens = response.usage.total_tokens if response.usage else None

            return AIResponse(
                content=content,
                provider="SGLang",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.80,  # Local models may have slightly lower confidence
            )

        except Exception as e:
            raise AIProviderError("SGLang", f"Failed to generate response: {str(e)}", e)

    async def generate_response_async(
        self,
        messages: List[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response asynchronously using SGLang."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Call SGLang API asynchronously
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Extract response content
            content = response.choices[0].message.content
            usage_tokens = response.usage.total_tokens if response.usage else None

            return AIResponse(
                content=content,
                provider="SGLang",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.80,
            )

        except Exception as e:
            raise AIProviderError(
                "SGLang", f"Failed to generate async response: {str(e)}", e
            )

    def is_available(self) -> bool:
        """Check if SGLang server is running and available."""
        try:
            # Test connection with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    def get_investment_banking_config(self) -> Dict[str, Any]:
        """Get investment banking optimized configuration for SGLang."""
        return {
            "temperature": 0.1,  # Conservative for financial analysis
            "max_tokens": 3000,  # Reasonable length for local inference
            "top_p": 0.9,  # Focused responses
            "frequency_penalty": 0.2,  # Reduce repetition
            "presence_penalty": 0.1,  # Encourage varied analysis
        }

    def financial_analysis_prompt(
        self, analysis_type: str, company_info: Dict[str, Any], context: str = ""
    ) -> List[AIMessage]:
        """
        Override base class with SGLang-optimized prompts.
        Local models benefit from clearer, more structured prompts.
        """

        # System prompt optimized for local models
        system_prompt = """You are an expert investment banking analyst specializing in financial analysis.

EXPERTISE AREAS:
- M&A due diligence and valuation
- Financial modeling and risk assessment  
- Market analysis and competitive intelligence
- Regulatory compliance and strategic evaluation

ANALYSIS REQUIREMENTS:
1. Provide structured, methodical analysis
2. Include specific financial metrics and ratios
3. Identify key risks and mitigation strategies
4. Rate confidence levels for major findings
5. Cite sources and data points clearly

RESPONSE FORMAT:
- Use clear headings and bullet points
- Provide executive summary first
- Include quantitative analysis with calculations
- End with clear recommendations

Be precise, professional, and conservative in assessments."""

        # Simplified but comprehensive prompts for local models
        analysis_prompts = {
            "ma_due_diligence": f"Perform M&A due diligence analysis for {company_info.get('name', 'target company')}. Include financial health, operational risks, strategic value, and integration complexity. Provide go/no-go recommendation.",
            "ma_valuation": f"Conduct comprehensive valuation of {company_info.get('name', 'target company')} for M&A purposes. Use DCF, comparable company analysis, and precedent transactions. Include synergy analysis and fair value range.",
            "ma_market_analysis": f"Analyze market dynamics and competitive landscape for {company_info.get('name', 'target company')} acquisition. Include market size, growth trends, competitive positioning, and strategic implications.",
        }

        user_prompt = analysis_prompts.get(
            analysis_type,
            f"Analyze {company_info.get('name', 'target company')} for {analysis_type} investment banking purposes.",
        )

        if context:
            user_prompt += f"\n\nAdditional Context: {context}"

        return [
            AIMessage(role="system", content=system_prompt),
            AIMessage(role="user", content=user_prompt),
        ]

    def check_server_status(self) -> Dict[str, Any]:
        """Check SGLang server status and model information."""
        try:
            # Try to get model information
            models = self.client.models.list()
            return {
                "status": "available",
                "base_url": self.base_url,
                "models": [model.id for model in models.data] if models else [],
                "current_model": self.model_name,
            }
        except Exception as e:
            return {"status": "unavailable", "base_url": self.base_url, "error": str(e)}
