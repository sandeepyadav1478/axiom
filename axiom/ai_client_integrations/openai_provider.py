"""OpenAI provider implementation for Investment Banking Analytics."""

import asyncio
from typing import List, Optional, Dict, Any
import openai
from openai import OpenAI, AsyncOpenAI

from .base_ai_provider import BaseAIProvider, AIMessage, AIResponse, AIProviderError


class OpenAIProvider(BaseAIProvider):
    """OpenAI provider implementation with investment banking optimization."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        **kwargs,
    ):
        super().__init__(api_key, base_url, model_name, **kwargs)

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url or "https://api.openai.com/v1"
        )

        self.async_client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url or "https://api.openai.com/v1"
        )

    def generate_response(
        self,
        messages: List[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Call OpenAI API
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
                provider="OpenAI",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.85,  # Default confidence for OpenAI
            )

        except Exception as e:
            raise AIProviderError("OpenAI", f"Failed to generate response: {str(e)}", e)

    async def generate_response_async(
        self,
        messages: List[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response asynchronously using OpenAI API."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

            # Call OpenAI API asynchronously
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
                provider="OpenAI",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.85,
            )

        except Exception as e:
            raise AIProviderError(
                "OpenAI", f"Failed to generate async response: {str(e)}", e
            )

    def is_available(self) -> bool:
        """Check if OpenAI provider is available."""
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
        """Get investment banking optimized configuration."""
        return {
            "temperature": 0.05,  # Very conservative for financial analysis
            "max_tokens": 4000,  # Longer responses for detailed analysis
            "top_p": 0.9,  # High quality responses
            "frequency_penalty": 0.1,  # Reduce repetition
            "presence_penalty": 0.1,  # Encourage diverse analysis
        }
