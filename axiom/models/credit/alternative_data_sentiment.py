"""
Alternative Data & Sentiment Analysis for Credit

Based on: K.B. Hansen, C. Borch (2022)
"Alternative data and sentiment analysis: Prospecting non-standard data in machine learning-driven finance"
Big Data & Society, 2022

Integrates alternative data sources for credit assessment:
- Social media sentiment
- News sentiment  
- Web scraping data
- Transaction data
- Mobile data
"""

from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class AltDataConfig:
    """Config for alternative data credit model"""
    use_social_media: bool = True
    use_news: bool = True
    use_transaction: bool = True


class AlternativeDataCreditModel:
    """Credit model using alternative data sources"""
    
    def __init__(self, config: AltDataConfig):
        self.config = config
    
    def assess_credit(self, traditional_score: float, alt_data: Dict) -> float:
        """Adjust traditional score with alternative data"""
        
        score = traditional_score
        
        # Social media sentiment adjustment
        if self.config.use_social_media and 'social_sentiment' in alt_data:
            sentiment = alt_data['social_sentiment']  # -1 to +1
            score += sentiment * 20  # +/- 20 points
        
        # News sentiment
        if self.config.use_news and 'news_sentiment' in alt_data:
            news_sent = alt_data['news_sentiment']
            score += news_sent * 15
        
        # Transaction patterns
        if self.config.use_transaction and 'transaction_stability' in alt_data:
            stability = alt_data['transaction_stability']  # 0-1
            if stability > 0.8:
                score += 10
            elif stability < 0.5:
                score -= 15
        
        return max(300, min(850, score))


if __name__ == "__main__":
    print("Alternative Data Credit - 2022")
    model = AlternativeDataCreditModel(AltDataConfig())
    
    adj_score = model.assess_credit(
        traditional_score=700,
        alt_data={
            'social_sentiment': 0.3,
            'news_sentiment': 0.2,
            'transaction_stability': 0.85
        }
    )
    
    print(f"Adjusted score: {adj_score:.0f}")
    print("✓ Alternative data integrated")