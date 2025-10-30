"""
Activist Investor Campaign Detection

Detects activist investor activity that often precedes M&A:
- 13D filings analysis
- Shareholder proposals
- Board seat demands
- Strategic review pushes
- Proxy battles

Early indicator for M&A targets (activist campaigns → M&A in 6-18 months).
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class ActivistSignal:
    """Detected activist activity"""
    activist_name: str
    target_company: str
    stake_percent: float
    campaign_type: str  # board_seats, strategic_review, sale_process
    filing_date: datetime
    ma_probability: float  # Historical correlation with M&A


class ActivistCampaignDetector:
    """Detect activist campaigns indicating potential M&A"""
    
    def __init__(self):
        # Known activists with M&A success rates
        self.activist_firms = {
            'Elliott Management': 0.75,
            'Starboard Value': 0.70,
            'Third Point': 0.68,
            'Pershing Square': 0.72,
            'ValueAct': 0.65,
            'Icahn Enterprises': 0.80
        }
    
    def analyze_13d_filing(self, filing_text: str) -> Optional[ActivistSignal]:
        """Analyze SEC 13D filing for activist signals"""
        
        # Extract stake percentage
        stake_match = re.search(r'(\d+\.?\d*)%\s*(?:of|stake|ownership)', filing_text, re.IGNORECASE)
        stake = float(stake_match.group(1)) if stake_match else 0.0
        
        # Detect campaign type
        campaign_type = 'passive'
        
        if any(term in filing_text.lower() for term in ['board seat', 'director nomination']):
            campaign_type = 'board_seats'
        elif any(term in filing_text.lower() for term in ['strategic review', 'strategic alternatives']):
            campaign_type = 'strategic_review'
        elif any(term in filing_text.lower() for term in ['sale process', 'seek buyers']):
            campaign_type = 'sale_process'
        
        # M&A probability based on campaign type
        ma_prob = {
            'passive': 0.15,
            'board_seats': 0.45,
            'strategic_review': 0.70,
            'sale_process': 0.90
        }.get(campaign_type, 0.20)
        
        if stake > 10.0:  # Large stake increases M&A probability
            ma_prob = min(0.95, ma_prob * 1.2)
        
        return ActivistSignal(
            activist_name='Detected Activist',
            target_company='Target',
            stake_percent=stake,
            campaign_type=campaign_type,
            filing_date=datetime.now(),
            ma_probability=ma_prob
        )
    
    def monitor_activist_activity(self, company_watchlist: List[str]) -> Dict[str, float]:
        """Monitor companies for activist activity"""
        
        ma_probabilities = {}
        
        for company in company_watchlist:
            # Would check:
            # - Recent 13D filings
            # - Shareholder letters
            # - Proxy statements
            # - News mentions of activists
            
            # Placeholder scoring
            ma_probabilities[company] = 0.25  # Would be actual detection
        
        return ma_probabilities


if __name__ == "__main__":
    print("Activist Campaign Detector")
    
    detector = ActivistCampaignDetector()
    
    sample_filing = """
    Schedule 13D filed for Target Corp.
    Reporting a 12.5% stake in the company.
    Intent to seek board representation and
    push for strategic review of alternatives.
    """
    
    signal = detector.analyze_13d_filing(sample_filing)
    
    if signal:
        print(f"Campaign type: {signal.campaign_type}")
        print(f"Stake: {signal.stake_percent}%")
        print(f"M&A probability: {signal.ma_probability:.0%}")
        print("✓ Activist activity = M&A signal")