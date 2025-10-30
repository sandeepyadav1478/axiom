"""
Large Language Model Credit Scoring System

Based on: U.O. Ogbuonyalu, K. Abiodun, S. Dzamefe (2025)
"Beyond the credit score: The untapped power of LLMS in banking risk models"
Finance & Accounting, 2025

This implementation uses Large Language Models to assess credit risk by analyzing:
- Social media sentiment
- Financial news
- Earnings call transcripts
- Alternative data sources
- Qualitative factors beyond traditional credit scores

Provides forward-looking risk indicators and standardized credit assessments.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio
import re

try:
    from axiom.integrations.ai_providers import AIMessage, get_provider
    from axiom.config.ai_layer_config import AIProvider
    AI_PROVIDERS_AVAILABLE = True
except ImportError:
    AI_PROVIDERS_AVAILABLE = False


class SentimentSource(Enum):
    """Sources of sentiment data"""
    SOCIAL_MEDIA = "social_media"
    FINANCIAL_NEWS = "financial_news"
    EARNINGS_CALLS = "earnings_calls"
    ANALYST_REPORTS = "analyst_reports"
    CUSTOMER_REVIEWS = "customer_reviews"


class CreditRiskLevel(Enum):
    """Credit risk levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result from a specific source"""
    source_type: SentimentSource
    sentiment_score: float  # -1.0 (very negative) to +1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    key_themes: List[str]
    risk_indicators: List[str]
    positive_factors: List[str]
    negative_factors: List[str]
    sample_text: str = ""


@dataclass
class LLMCreditAssessment:
    """Complete LLM-based credit assessment"""
    entity_name: str
    assessment_date: datetime
    
    # Overall Assessment
    credit_risk_level: CreditRiskLevel
    default_probability: float  # 0.0 to 1.0
    credit_score: float  # 300-850 scale
    recommendation: str  # approve/review/decline
    
    # Sentiment Analysis
    overall_sentiment: float  # -1.0 to +1.0
    sentiment_by_source: Dict[str, SentimentAnalysis] = None
    
    # Risk Factors
    identified_risks: List[str] = None
    risk_severity: Dict[str, str] = None  # risk -> severity level
    early_warning_signals: List[str] = None
    
    # Positive Factors
    credit_strengths: List[str] = None
    mitigating_factors: List[str] = None
    
    # Traditional Metrics Enhancement
    traditional_score: Optional[float] = None
    llm_adjustment: float = 0.0  # Points added/subtracted by LLM
    
    # Confidence and Quality
    analysis_confidence: float = 0.0
    data_quality_score: float = 0.0
    sources_analyzed: int = 0
    
    # Narrative Summary
    executive_summary: str = ""
    detailed_rationale: str = ""
    
    def __post_init__(self):
        if self.sentiment_by_source is None:
            self.sentiment_by_source = {}
        if self.identified_risks is None:
            self.identified_risks = []
        if self.risk_severity is None:
            self.risk_severity = {}
        if self.early_warning_signals is None:
            self.early_warning_signals = []
        if self.credit_strengths is None:
            self.credit_strengths = []
        if self.mitigating_factors is None:
            self.mitigating_factors = []


@dataclass
class LLMScoringConfig:
    """Configuration for LLM Credit Scoring"""
    # LLM parameters
    provider: str = "claude"  # claude, openai, or both
    model_name: str = "claude-3-sonnet-20240229"
    temperature: float = 0.1  # Low for consistent credit decisions
    max_tokens: int = 2000
    use_consensus: bool = True  # Use multiple providers for validation
    
    # Analysis parameters
    sentiment_sources: List[SentimentSource] = None
    lookback_days: int = 90  # Analyze last 90 days of data
    min_confidence_threshold: float = 0.75
    
    # Scoring parameters
    sentiment_weight: float = 0.30  # Weight of sentiment in final score
    traditional_weight: float = 0.70  # Weight of traditional metrics
    
    def __post_init__(self):
        if self.sentiment_sources is None:
            self.sentiment_sources = [
                SentimentSource.FINANCIAL_NEWS,
                SentimentSource.SOCIAL_MEDIA,
                SentimentSource.EARNINGS_CALLS
            ]


class LLMCreditScoring:
    """
    LLM-Based Credit Scoring System
    
    Uses large language models to analyze unstructured data and provide
    comprehensive credit risk assessments beyond traditional credit scores.
    """
    
    def __init__(self, config: Optional[LLMScoringConfig] = None):
        if not AI_PROVIDERS_AVAILABLE:
            raise ImportError("AI providers required for LLMCreditScoring")
        
        self.config = config or LLMScoringConfig()
        
        # Get LLM provider(s)
        self.primary_provider = get_provider(self.config.provider)
        
        if self.config.use_consensus:
            # Use both Claude and OpenAI for validation
            self.secondary_provider = get_provider("openai" if self.config.provider == "claude" else "claude")
        else:
            self.secondary_provider = None
    
    async def assess_credit_risk(
        self,
        entity_name: str,
        traditional_credit_score: Optional[float] = None,
        financial_data: Optional[Dict] = None,
        alternative_data: Optional[Dict] = None
    ) -> LLMCreditAssessment:
        """
        Comprehensive credit risk assessment using LLM
        
        Args:
            entity_name: Company or individual name
            traditional_credit_score: Traditional credit score (if available)
            financial_data: Financial metrics and statements
            alternative_data: Social media, news, etc.
            
        Returns:
            Complete LLM-based credit assessment
        """
        # Gather multi-source data
        sentiment_analyses = await self._analyze_multi_source_sentiment(
            entity_name, alternative_data
        )
        
        # Perform LLM credit analysis
        llm_analysis = await self._perform_llm_credit_analysis(
            entity_name,
            sentiment_analyses,
            financial_data,
            traditional_credit_score
        )
        
        # Calculate final assessment
        assessment = self._synthesize_credit_assessment(
            entity_name,
            llm_analysis,
            sentiment_analyses,
            traditional_credit_score
        )
        
        return assessment
    
    async def _analyze_multi_source_sentiment(
        self,
        entity_name: str,
        alternative_data: Optional[Dict] = None
    ) -> Dict[str, SentimentAnalysis]:
        """Analyze sentiment from multiple sources using LLM"""
        
        sentiment_results = {}
        
        for source_type in self.config.sentiment_sources:
            # Get data for this source
            source_data = self._get_source_data(entity_name, source_type, alternative_data)
            
            if not source_data:
                continue
            
            # Analyze sentiment with LLM
            sentiment = await self._analyze_sentiment_with_llm(
                entity_name,
                source_type,
                source_data
            )
            
            sentiment_results[source_type.value] = sentiment
        
        return sentiment_results
    
    async def _analyze_sentiment_with_llm(
        self,
        entity_name: str,
        source_type: SentimentSource,
        source_data: str
    ) -> SentimentAnalysis:
        """Analyze sentiment for specific source using LLM"""
        
        messages = [
            AIMessage(
                role="system",
                content=f"""You are a credit risk analyst analyzing {source_type.value} sentiment for credit assessment.
                
                Analyze the provided text and extract:
                1. Overall sentiment (-1.0 to +1.0 scale)
                2. Credit-relevant themes
                3. Risk indicators (bankruptcy, defaults, lawsuits, etc.)
                4. Positive factors (growth, profitability, innovation)
                5. Negative factors (losses, layoffs, scandals, etc.)
                
                Focus on credit risk implications."""
            ),
            AIMessage(
                role="user",
                content=f"""Analyze {source_type.value} data for {entity_name}:
                
{source_data[:2000]}

Provide structured analysis:
1. Sentiment Score (-1.0 to +1.0): Overall sentiment
2. Confidence (0.0-1.0): Assessment confidence
3. Key Themes: Main topics discussed
4. Risk Indicators: Credit risk red flags
5. Positive Factors: Credit-positive signals
6. Negative Factors: Credit-negative signals

Be conservative and highlight any credit concerns."""
            )
        ]
        
        try:
            if self.primary_provider:
                response = await self.primary_provider.generate_response_async(
                    messages,
                    max_tokens=1000,
                    temperature=self.config.temperature
                )
                
                # Parse response
                parsed = self._parse_sentiment_response(response.content)
                
                return SentimentAnalysis(
                    source_type=source_type,
                    sentiment_score=parsed.get('sentiment', 0.0),
                    confidence=parsed.get('confidence', 0.5),
                    key_themes=parsed.get('themes', []),
                    risk_indicators=parsed.get('risks', []),
                    positive_factors=parsed.get('positives', []),
                    negative_factors=parsed.get('negatives', []),
                    sample_text=source_data[:200]
                )
        except Exception as e:
            # Return neutral sentiment on error
            return SentimentAnalysis(
                source_type=source_type,
                sentiment_score=0.0,
                confidence=0.3,
                key_themes=[],
                risk_indicators=[f"Analysis error: {str(e)}"],
                positive_factors=[],
                negative_factors=[]
            )
    
    async def _perform_llm_credit_analysis(
        self,
        entity_name: str,
        sentiment_analyses: Dict[str, SentimentAnalysis],
        financial_data: Optional[Dict],
        traditional_score: Optional[float]
    ) -> Dict[str, Any]:
        """Comprehensive LLM credit analysis"""
        
        # Compile all data
        sentiment_summary = self._compile_sentiment_summary(sentiment_analyses)
        financial_summary = self._compile_financial_summary(financial_data) if financial_data else "Limited financial data available"
        
        messages = [
            AIMessage(
                role="system",
                content="""You are a senior credit risk analyst providing comprehensive credit assessment.
                
                Use ALL available information:
                - Traditional financial metrics
                - Multi-source sentiment analysis
                - News and social media indicators
                - Qualitative risk factors
                
                Provide structured, conservative credit assessment with clear reasoning."""
            ),
            AIMessage(
                role="user",
                content=f"""Comprehensive credit assessment for {entity_name}:

TRADITIONAL CREDIT DATA:
{f"Credit Score: {traditional_score}" if traditional_score else "No traditional score available"}
{financial_summary}

SENTIMENT ANALYSIS:
{sentiment_summary}

Provide detailed credit assessment covering:

1. CREDIT RISK LEVEL: very_low/low/medium/high/very_high
2. DEFAULT PROBABILITY: Estimate (0.0-1.0)
3. RECOMMENDED SCORE: 300-850 scale
4. LLM ADJUSTMENT: Points to add/subtract from traditional score
5. RECOMMENDATION: approve/review/decline

6. IDENTIFIED RISKS:
   - List specific credit risks found in data
   - Severity of each risk (low/medium/high/critical)
   - Early warning signals

7. CREDIT STRENGTHS:
   - Positive credit factors
   - Mitigating factors

8. EXECUTIVE SUMMARY: 2-3 sentence credit assessment
9. DETAILED RATIONALE: Explain your assessment

Be conservative and highlight any concerns that could impact creditworthiness."""
            )
        ]
        
        try:
            response = await self.primary_provider.generate_response_async(
                messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Get consensus if enabled
            if self.config.use_consensus and self.secondary_provider:
                secondary_response = await self.secondary_provider.generate_response_async(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                # Combine both analyses
                combined_analysis = self._combine_llm_responses(
                    response.content,
                    secondary_response.content
                )
                return combined_analysis
            
            return self._parse_llm_credit_analysis(response.content)
            
        except Exception as e:
            # Return conservative default assessment
            return {
                'risk_level': CreditRiskLevel.MEDIUM,
                'default_prob': 0.15,
                'credit_score': 650,
                'llm_adjustment': 0,
                'recommendation': 'review',
                'risks': [f"LLM analysis error: {str(e)}"],
                'strengths': [],
                'executive_summary': f"Unable to complete full LLM analysis for {entity_name}",
                'confidence': 0.3
            }
    
    def _synthesize_credit_assessment(
        self,
        entity_name: str,
        llm_analysis: Dict[str, Any],
        sentiment_analyses: Dict[str, SentimentAnalysis],
        traditional_score: Optional[float]
    ) -> LLMCreditAssessment:
        """Synthesize final credit assessment"""
        
        # Calculate overall sentiment
        if sentiment_analyses:
            sentiments = [s.sentiment_score for s in sentiment_analyses.values() if s.confidence > 0.5]
            overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        else:
            overall_sentiment = 0.0
        
        # Calculate final credit score
        if traditional_score:
            # Start with traditional score
            final_score = traditional_score
            
            # Apply LLM adjustment
            final_score += llm_analysis.get('llm_adjustment', 0)
            
            # Apply sentiment influence
            sentiment_adjustment = overall_sentiment * 50  # +/- 50 points max
            final_score += sentiment_adjustment * self.config.sentiment_weight
            
            # Clamp to valid range
            final_score = max(300, min(850, final_score))
        else:
            # Use LLM score directly
            final_score = llm_analysis.get('credit_score', 650)
        
        # Create assessment
        return LLMCreditAssessment(
            entity_name=entity_name,
            assessment_date=datetime.now(),
            credit_risk_level=llm_analysis.get('risk_level', CreditRiskLevel.MEDIUM),
            default_probability=llm_analysis.get('default_prob', 0.15),
            credit_score=final_score,
            recommendation=llm_analysis.get('recommendation', 'review'),
            overall_sentiment=overall_sentiment,
            sentiment_by_source=sentiment_analyses,
            identified_risks=llm_analysis.get('risks', []),
            risk_severity=llm_analysis.get('risk_severity', {}),
            early_warning_signals=llm_analysis.get('warnings', []),
            credit_strengths=llm_analysis.get('strengths', []),
            mitigating_factors=llm_analysis.get('mitigating', []),
            traditional_score=traditional_score,
            llm_adjustment=llm_analysis.get('llm_adjustment', 0),
            analysis_confidence=llm_analysis.get('confidence', 0.7),
            data_quality_score=self._assess_data_quality(sentiment_analyses, financial_data=None),
            sources_analyzed=len(sentiment_analyses),
            executive_summary=llm_analysis.get('executive_summary', ''),
            detailed_rationale=llm_analysis.get('detailed_rationale', '')
        )
    
    def _get_source_data(
        self,
        entity_name: str,
        source_type: SentimentSource,
        alternative_data: Optional[Dict]
    ) -> str:
        """Get data for specific source"""
        
        if alternative_data and source_type.value in alternative_data:
            return alternative_data[source_type.value]
        
        # Return sample/placeholder data
        samples = {
            SentimentSource.FINANCIAL_NEWS: f"Recent news about {entity_name}: Company reported quarterly results...",
            SentimentSource.SOCIAL_MEDIA: f"Social media discussions about {entity_name}: Customers discussing products...",
            SentimentSource.EARNINGS_CALLS: f"Earnings call transcript for {entity_name}: Management discussed outlook...",
            SentimentSource.ANALYST_REPORTS: f"Analyst reports on {entity_name}: Analysts recommend...",
            SentimentSource.CUSTOMER_REVIEWS: f"Customer reviews for {entity_name}: Users report satisfaction..."
        }
        
        return samples.get(source_type, "")
    
    def _compile_sentiment_summary(self, analyses: Dict[str, SentimentAnalysis]) -> str:
        """Compile sentiment summary from all sources"""
        
        if not analyses:
            return "No sentiment data available"
        
        summary_parts = []
        
        for source_name, analysis in analyses.items():
            summary_parts.append(f"""
{source_name.upper()}:
- Sentiment: {analysis.sentiment_score:.2f} ({self._sentiment_label(analysis.sentiment_score)})
- Confidence: {analysis.confidence:.1%}
- Key Themes: {', '.join(analysis.key_themes[:3])}
- Risk Indicators: {', '.join(analysis.risk_indicators[:2]) if analysis.risk_indicators else 'None'}
- Positive Factors: {', '.join(analysis.positive_factors[:2]) if analysis.positive_factors else 'None'}
""")
        
        return '\n'.join(summary_parts)
    
    def _compile_financial_summary(self, financial_data: Dict) -> str:
        """Compile financial data summary"""
        
        summary = []
        
        if 'revenue' in financial_data:
            summary.append(f"Revenue: ${financial_data['revenue']:,.0f}")
        if 'ebitda' in financial_data:
            summary.append(f"EBITDA: ${financial_data['ebitda']:,.0f}")
        if 'debt' in financial_data:
            summary.append(f"Total Debt: ${financial_data['debt']:,.0f}")
        if 'cash' in financial_data:
            summary.append(f"Cash: ${financial_data['cash']:,.0f}")
        
        return '\n'.join(summary) if summary else "Limited financial data"
    
    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.5:
            return "Very Positive"
        elif score > 0.2:
            return "Positive"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def _parse_sentiment_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM sentiment analysis response"""
        
        parsed = {}
        
        # Extract sentiment score
        sentiment_match = re.search(r"sentiment score:?\s*(-?[0-9.]+)", content.lower())
        parsed['sentiment'] = float(sentiment_match.group(1)) if sentiment_match else 0.0
        
        # Extract confidence
        conf_match = re.search(r"confidence:?\s*([0-9.]+)", content.lower())
        parsed['confidence'] = float(conf_match.group(1)) if conf_match else 0.5
        
        # Extract lists (simplified)
        parsed['themes'] = self._extract_list_items(content, "key themes")
        parsed['risks'] = self._extract_list_items(content, "risk indicators")
        parsed['positives'] = self._extract_list_items(content, "positive factors")
        parsed['negatives'] = self._extract_list_items(content, "negative factors")
        
        return parsed
    
    def _parse_llm_credit_analysis(self, content: str) -> Dict[str, Any]:
        """Parse comprehensive LLM credit analysis"""
        
        parsed = {}
        
        # Extract risk level
        content_lower = content.lower()
        if "very_high" in content_lower or "critical" in content_lower:
            parsed['risk_level'] = CreditRiskLevel.VERY_HIGH
        elif "high" in content_lower:
            parsed['risk_level'] = CreditRiskLevel.HIGH
        elif "low" in content_lower and "very" not in content_lower:
            parsed['risk_level'] = CreditRiskLevel.LOW
        else:
            parsed['risk_level'] = CreditRiskLevel.MEDIUM
        
        # Extract default probability
        prob_match = re.search(r"default probability:?\s*([0-9.]+)", content_lower)
        parsed['default_prob'] = float(prob_match.group(1)) if prob_match else 0.15
        
        # Extract recommended score
        score_match = re.search(r"recommended score:?\s*([0-9]+)", content_lower)
        parsed['credit_score'] = float(score_match.group(1)) if score_match else 650
        
        # Extract LLM adjustment
        adj_match = re.search(r"llm adjustment:?\s*([+-]?[0-9]+)", content_lower)
        parsed['llm_adjustment'] = float(adj_match.group(1)) if adj_match else 0
        
        # Extract recommendation
        if "approve" in content_lower and "decline" not in content_lower:
            parsed['recommendation'] = "approve"
        elif "decline" in content_lower:
            parsed['recommendation'] = "decline"
        else:
            parsed['recommendation'] = "review"
        
        # Extract lists
        parsed['risks'] = self._extract_list_items(content, "identified risks")
        parsed['strengths'] = self._extract_list_items(content, "credit strengths")
        parsed['warnings'] = self._extract_list_items(content, "early warning")
        
        # Extract summaries
        exec_match = re.search(r"executive summary:?\s*(.+?)(?:\n\n|detailed rationale)", content, re.IGNORECASE | re.DOTALL)
        parsed['executive_summary'] = exec_match.group(1).strip() if exec_match else ""
        
        return parsed
    
    def _extract_list_items(self, content: str, section_name: str) -> List[str]:
        """Extract list items from a section"""
        
        # Simple extraction - look for bullet points or numbered items
        items = []
        
        section_match = re.search(
            f"{section_name}:?(.+?)(?:\n\n|[A-Z][A-Z ]+:|\Z)",
            content,
            re.IGNORECASE | re.DOTALL
        )
        
        if section_match:
            section_text = section_match.group(1)
            # Find items (bullets or numbers)
            pattern = r"[-•*]\s*(.+?)(?:\n|$)|[0-9]+\.\s*(.+?)(?:\n|$)"
            matches = re.findall(pattern, section_text)
            items = [m[0] or m[1] for m in matches if m[0] or m[1]]
        
        return items[:10]  # Limit to 10 items
    
    def _combine_llm_responses(self, primary: str, secondary: str) -> Dict[str, Any]:
        """Combine responses from two LLM providers for consensus"""
        
        primary_parsed = self._parse_llm_credit_analysis(primary)
        secondary_parsed = self._parse_llm_credit_analysis(secondary)
        
        # Average numerical values
        combined = {
            'default_prob': (primary_parsed['default_prob'] + secondary_parsed['default_prob']) / 2,
            'credit_score': (primary_parsed['credit_score'] + secondary_parsed['credit_score']) / 2,
            'llm_adjustment': (primary_parsed['llm_adjustment'] + secondary_parsed['llm_adjustment']) / 2,
        }
        
        # Use more conservative risk level
        risk_levels = [CreditRiskLevel.VERY_LOW, CreditRiskLevel.LOW, CreditRiskLevel.MEDIUM, 
                      CreditRiskLevel.HIGH, CreditRiskLevel.VERY_HIGH]
        primary_idx = risk_levels.index(primary_parsed['risk_level'])
        secondary_idx = risk_levels.index(secondary_parsed['risk_level'])
        combined['risk_level'] = risk_levels[max(primary_idx, secondary_idx)]
        
        # Combine lists (union)
        combined['risks'] = list(set(primary_parsed['risks'] + secondary_parsed['risks']))
        combined['strengths'] = list(set(primary_parsed['strengths'] + secondary_parsed['strengths']))
        
        # Use more conservative recommendation
        if 'decline' in [primary_parsed['recommendation'], secondary_parsed['recommendation']]:
            combined['recommendation'] = 'decline'
        elif 'review' in [primary_parsed['recommendation'], secondary_parsed['recommendation']]:
            combined['recommendation'] = 'review'
        else:
            combined['recommendation'] = 'approve'
        
        # Higher confidence with consensus
        combined['confidence'] = 0.85
        
        return combined
    
    def _assess_data_quality(
        self,
        sentiment_analyses: Dict,
        financial_data: Optional[Dict]
    ) -> float:
        """Assess quality of input data"""
        
        score = 0.0
        
        # Quality from number of sources
        if sentiment_analyses:
            score += min(0.5, len(sentiment_analyses) * 0.15)
        
        # Quality from sentiment confidence
        if sentiment_analyses:
            avg_confidence = sum(s.confidence for s in sentiment_analyses.values()) / len(sentiment_analyses)
            score += avg_confidence * 0.3
        
        # Quality from financial data
        if financial_data:
            score += 0.2
        
        return min(1.0, score)


# Example usage
if __name__ == "__main__":
    print("LLM Credit Scoring - Example Usage")
    print("=" * 60)
    
    if not AI_PROVIDERS_AVAILABLE:
        print("ERROR: AI providers required")
        print("Configure: OPENAI_API_KEY or ANTHROPIC_API_KEY")
    else:
        # Note: This demo shows structure but won't run without API keys
        print("\n1. Configuration")
        config = LLMScoringConfig(
            provider="claude",
            use_consensus=True,
            sentiment_sources=[
                SentimentSource.FINANCIAL_NEWS,
                SentimentSource.SOCIAL_MEDIA
            ]
        )
        print(f"   Provider: {config.provider}")
        print(f"   Consensus: {config.use_consensus}")
        print(f"   Sources: {len(config.sentiment_sources)}")
        
        print("\n2. Sample Data Structure")
        sample_alternative_data = {
            'financial_news': "Company reported strong Q3 results, beat estimates...",
            'social_media': "Customers praising new product launch, positive reviews..."
        }
        
        sample_financial_data = {
            'revenue': 500_000_000,
            'ebitda': 100_000_000,
            'debt': 200_000_000,
            'cash': 150_000_000
        }
        
        print("   Alternative data sources: 2")
        print("   Financial metrics: 4")
        
        print("\n3. LLM Credit Scoring System")
        print("   ✓ Multi-source sentiment analysis")
        print("   ✓ Financial data integration")
        print("   ✓ Consensus validation (Claude + OpenAI)")
        print("   ✓ Beyond traditional credit scores")
        
        print("\n4. Expected Output Structure:")
        print("   • Credit Risk Level: MEDIUM")
        print("   • Default Probability: 12.5%")
        print("   • Credit Score: 720 (traditional) + LLM adjustment")
        print("   • Recommendation: APPROVE")
        print("   • Identified Risks: [list]")
        print("   • Credit Strengths: [list]")
        print("   • Overall Sentiment: +0.35 (positive)")
        print("   • Analysis Confidence: 85%")
        
        print("\n" + "=" * 60)
        print("Model structure complete!")
        print("\nBased on: Ogbuonyalu et al. (2025)")
        print("Innovation: LLMs for alternative data credit assessment")
        print("\nNote: Requires API keys to run actual analysis")