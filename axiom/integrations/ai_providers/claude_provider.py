"""Claude provider implementation for Axiom Analytics Platform."""

from typing import Any

from anthropic import Anthropic, AsyncAnthropic

from .base_ai_provider import AIMessage, AIProviderError, AIResponse, BaseAIProvider


class ClaudeProvider(BaseAIProvider):
    """Claude provider implementation with investment banking optimization."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model_name: str = "claude-sonnet-4-20250514",
        **kwargs,
    ):
        super().__init__(api_key, base_url, model_name, **kwargs)

        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key, base_url=self.base_url)

        self.async_client = AsyncAnthropic(api_key=self.api_key, base_url=self.base_url)

    def generate_response(
        self,
        messages: list[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response using Claude API."""
        try:
            # Separate system message from conversation messages
            system_msg = ""
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_msg = msg.content
                else:
                    conversation_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )

            # Call Claude API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg if system_msg else None,
                messages=conversation_messages,
                **kwargs,
            )

            # Extract response content
            content = response.content[0].text if response.content else ""
            usage_tokens = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else None
            )

            return AIResponse(
                content=content,
                provider="Claude",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.90,  # Claude typically has high confidence for reasoning
            )

        except Exception as e:
            raise AIProviderError("Claude", f"Failed to generate response: {str(e)}", e)

    async def generate_response_async(
        self,
        messages: list[AIMessage],
        max_tokens: int = 2000,
        temperature: float = 0.1,
        **kwargs,
    ) -> AIResponse:
        """Generate response asynchronously using Claude API."""
        try:
            # Separate system message from conversation messages
            system_msg = ""
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_msg = msg.content
                else:
                    conversation_messages.append(
                        {"role": msg.role, "content": msg.content}
                    )

            # Call Claude API asynchronously
            response = await self.async_client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg if system_msg else None,
                messages=conversation_messages,
                **kwargs,
            )

            # Extract response content
            content = response.content[0].text if response.content else ""
            usage_tokens = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else None
            )

            return AIResponse(
                content=content,
                provider="Claude",
                model=self.model_name,
                usage_tokens=usage_tokens,
                confidence=0.90,
            )

        except Exception as e:
            raise AIProviderError(
                "Claude", f"Failed to generate async response: {str(e)}", e
            )

    def is_available(self) -> bool:
        """Check if Claude provider is available."""
        try:
            # Test connection with a simple request
            self.client.messages.create(
                model=self.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception:
            return False

    def get_investment_banking_config(self) -> dict[str, Any]:
        """Get investment banking optimized configuration for Claude."""
        return {
            "temperature": 0.03,  # Extremely conservative for M&A analysis
            "max_tokens": 5000,  # Long detailed responses for complex analysis
            "top_p": 0.95,  # High quality, focused responses
        }

    def financial_analysis_prompt(
        self, analysis_type: str, company_info: dict[str, Any], context: str = ""
    ) -> list[AIMessage]:
        """
        Override base class with Claude-optimized prompts for investment banking.
        Claude excels at complex reasoning and structured analysis.
        """

        # Investment banking system prompt optimized for Claude
        system_prompt = """You are a distinguished senior investment banking analyst with deep expertise in:

**Core Competencies:**
- M&A due diligence, valuation, and strategic assessment
- Financial modeling (DCF, LBO, comparable analysis, precedent transactions)
- Market intelligence and competitive landscape analysis
- Risk assessment, regulatory compliance, and ESG factors

**Analysis Framework:**
1. **Methodology**: State clear analytical framework and assumptions
2. **Quantitative Analysis**: Provide detailed financial metrics and calculations
3. **Qualitative Assessment**: Strategic positioning, competitive advantages, risks
4. **Confidence Scoring**: Rate confidence levels for each major finding (0-100%)
5. **Citations**: Reference all sources and provide specific data points

**Investment Banking Standards:**
- Conservative assumptions and sensitivity analysis
- Multiple valuation methodologies where applicable
- Comprehensive risk factor identification
- Regulatory and compliance considerations
- Clear investment recommendations with supporting rationale

Be methodical, precise, and professional. Structure responses for investment committee consumption."""

        # Claude-specific analysis prompts (more detailed than base class)
        analysis_prompts = {
            "ma_due_diligence": f"""Conduct comprehensive M&A due diligence for {company_info.get('name', 'target company')}:

**Financial Due Diligence:**
- Historical performance analysis (5-year trend)
- Quality of earnings assessment
- Working capital analysis and normalization
- Debt capacity and covenant analysis

**Strategic Due Diligence:**
- Market position and competitive advantages
- Strategic rationale for acquisition
- Synergy identification and quantification
- Integration complexity assessment

**Risk Assessment:**
- Key business risks and mitigation strategies
- Regulatory/compliance risks
- Technology and operational risks
- Management and cultural fit assessment

Provide executive summary with clear go/no-go recommendation.""",
            "ma_valuation": f"""Perform comprehensive M&A valuation for {company_info.get('name', 'target company')}:

**Valuation Methodologies:**
1. Discounted Cash Flow (DCF) - Standalone and with synergies
2. Comparable Company Analysis (Trading multiples)
3. Precedent Transaction Analysis (Transaction multiples)
4. Sum-of-the-Parts (if applicable)

**Synergy Analysis:**
- Revenue synergies (cross-selling, market expansion)
- Cost synergies (operational efficiencies, overhead reduction)
- Tax synergies and one-time costs

**Valuation Range:**
- Fair value range with confidence intervals
- Sensitivity analysis on key assumptions
- Accretion/dilution analysis
- Premium analysis vs. market comparables

Deliver investment committee-ready valuation summary.""",
            "ma_market_analysis": f"""Analyze market dynamics for {company_info.get('name', 'target company')} acquisition:

**Market Assessment:**
- Total addressable market (TAM) and growth projections
- Market share analysis and competitive positioning
- Industry consolidation trends and M&A activity
- Regulatory environment and potential changes

**Strategic Implications:**
- Market leadership opportunities post-acquisition
- Defensive positioning against competitors
- Market expansion and diversification benefits
- Technology/digital transformation implications

**Competitive Response:**
- Expected competitor reactions
- Potential counter-bidders
- Market consolidation acceleration
- Customer/supplier relationship impacts

Conclude with strategic market rationale for the transaction.""",
        }

        user_prompt = analysis_prompts.get(
            analysis_type,
            f"Provide comprehensive investment banking analysis of {company_info.get('name', 'target company')} for {analysis_type} purposes.",
        )

        if context:
            user_prompt += f"\n\n**Additional Context:**\n{context}"

        return [
            AIMessage(role="system", content=system_prompt),
            AIMessage(role="user", content=user_prompt),
        ]
