"""
Earnings Call Sentiment Analysis for M&A Signals

Analyzes management tone and language in earnings calls to detect:
- Strategic review discussions
- M&A appetite/readiness
- Financial distress indicators
- Growth strategy changes

Often precedes M&A activity by 1-2 quarters.
"""

from typing import Dict, List
from dataclasses import dataclass
import re


@dataclass
class EarningsCallAnalysis:
    """Analysis of earnings call"""
    company: str
    quarter: str
    overall_sentiment: float  # -1 to +1
    ma_indicators: List[str]
    strategic_themes: List[str]
    management_confidence: float  # 0-1


class EarningsCallSentiment:
    """Analyze earnings calls for M&A signals"""
    
    def __init__(self):
        # M&A indicator phrases
        self.ma_phrases = [
            'strategic alternatives',
            'reviewing options',
            'M&A opportunities',
            'inorganic growth',
            'acquisition pipeline',
            'strategic partnerships',
            'business transformation'
        ]
        
        # Distress indicators
        self.distress_phrases = [
            'challenging environment',
            'headwinds',
            'restructuring',
            'cost reduction',
            'strategic review'
        ]
    
    def analyze_transcript(self, transcript: str) -> EarningsCallAnalysis:
        """Analyze earnings call transcript"""
        
        transcript_lower = transcript.lower()
        
        # Detect M&A indicators
        ma_indicators = [
            phrase for phrase in self.ma_phrases
            if phrase in transcript_lower
        ]
        
        # Detect distress
        distress_count = sum(
            1 for phrase in self.distress_phrases
            if phrase in transcript_lower
        )
        
        # Sentiment scoring
        positive_words = ['growth', 'opportunity', 'strong', 'positive', 'increased']
        negative_words = ['decline', 'weak', 'challenge', 'risk', 'decreased']
        
        pos_count = sum(word in transcript_lower for word in positive_words)
        neg_count = sum(word in transcript_lower for word in negative_words)
        
        if pos_count + neg_count > 0:
            sentiment = (pos_count - neg_count) / (pos_count + neg_count)
        else:
            sentiment = 0.0
        
        # Management confidence (based on language certainty)
        certain_words = ['will', 'confident', 'expect', 'committed']
        uncertain_words = ['may', 'might', 'possibly', 'uncertain']
        
        certain_count = sum(word in transcript_lower for word in certain_words)
        uncertain_count = sum(word in transcript_lower for word in uncertain_words)
        
        confidence = certain_count / (certain_count + uncertain_count + 1)
        
        # Strategic themes (simplified)
        themes = []
        if 'acquisition' in transcript_lower or 'm&a' in transcript_lower:
            themes.append('M&A Strategy')
        if distress_count > 2:
            themes.append('Financial Distress')
        if 'partnership' in transcript_lower:
            themes.append('Strategic Partnerships')
        
        return EarningsCallAnalysis(
            company='Target Company',
            quarter='Q3 2024',
            overall_sentiment=sentiment,
            ma_indicators=ma_indicators,
            strategic_themes=themes,
            management_confidence=confidence
        )


if __name__ == "__main__":
    print("Earnings Call Sentiment Analysis")
    
    analyzer = EarningsCallSentiment()
    
    sample_transcript = """
    We are reviewing strategic alternatives and opportunities for growth.
    The M&A pipeline remains active. We are committed to creating shareholder value
    through both organic and inorganic growth strategies.
    """
    
    analysis = analyzer.analyze_transcript(sample_transcript)
    
    print(f"Sentiment: {analysis.overall_sentiment:.2f}")
    print(f"M&A indicators: {analysis.ma_indicators}")
    print(f"Themes: {analysis.strategic_themes}")
    print(f"Confidence: {analysis.management_confidence:.2f}")
    print("✓ Earnings call intelligence")