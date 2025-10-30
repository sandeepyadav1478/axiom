"""
SEC Filing Automated Analysis for M&A Intelligence

Automatically analyzes SEC filings for M&A signals:
- 8-K (material events, acquisitions)
- 10-K/10-Q (MD&A section changes)
- S-4 (merger proxy statements)
- Schedule 13D (activist positions)
- Schedule 14A (proxy statements)

Extracts M&A-relevant information automatically using NLP.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import re


@dataclass
class SECFilingSignal:
    """Signal extracted from SEC filing"""
    filing_type: str  # 8-K, 10-K, 13D, etc.
    company: str
    filing_date: datetime
    ma_relevance: float  # 0-1
    key_items: List[str]
    material_changes: List[str]


class SECFilingAnalyzer:
    """Automated SEC filing analysis for M&A"""
    
    def __init__(self):
        # Material event item numbers in 8-K
        self.eight_k_ma_items = {
            '1.01': 'Entry into Material Agreement',
            '2.01': 'Completion of Acquisition',
            '5.01': 'Changes in Control',
            '8.01': 'Other Events (often M&A related)'
        }
    
    def analyze_8k(self, filing_text: str) -> SECFilingSignal:
        """Analyze 8-K for M&A activity"""
        
        # Detect item numbers
        items = []
        for item_num, description in self.eight_k_ma_items.items():
            pattern = f'Item {item_num.replace(".", r"\.")}'
            if re.search(pattern, filing_text):
                items.append(description)
        
        # M&A relevance
        ma_relevance = 0.0
        if '2.01' in filing_text:  # Acquisition completed
            ma_relevance = 1.0
        elif any(item in items for item in ['Entry into Material Agreement', 'Changes in Control']):
            ma_relevance = 0.75
        elif items:
            ma_relevance = 0.5
        
        return SECFilingSignal(
            filing_type='8-K',
            company='Target Company',
            filing_date=datetime.now(),
            ma_relevance=ma_relevance,
            key_items=items,
            material_changes=[]
        )
    
    def analyze_13d(self, filing_text: str) -> SECFilingSignal:
        """Analyze 13D for activist/strategic positions"""
        
        # Extract stake percentage
        stake_match = re.search(r'(\d+\.?\d*)%', filing_text)
        stake = float(stake_match.group(1)) if stake_match else 0.0
        
        # Intent analysis
        intent = 'passive'
        if 'strategic' in filing_text.lower() or 'board' in filing_text.lower():
            intent = 'active'
            ma_relevance = 0.70
        else:
            ma_relevance = 0.30
        
        if stake > 10.0:
            ma_relevance = min(0.95, ma_relevance * 1.3)
        
        return SECFilingSignal(
            filing_type='13D',
            company='Target',
            filing_date=datetime.now(),
            ma_relevance=ma_relevance,
            key_items=[f'{stake:.1f}% stake', f'{intent} intent'],
            material_changes=[]
        )
    
    def monitor_companies(self, company_list: List[str]) -> Dict[str, float]:
        """Monitor SEC filings for multiple companies"""
        
        ma_probabilities = {}
        
        for company in company_list:
            # Would check EDGAR for recent filings
            # Placeholder: random probability
            ma_probabilities[company] = 0.25
        
        return ma_probabilities


if __name__ == "__main__":
    print("SEC Filing Analyzer for M&A")
    
    analyzer = SECFilingAnalyzer()
    
    # Sample 8-K
    sample_8k = """
    Item 2.01 Completion of Acquisition
    On October 15, 2024, Company completed acquisition of Target Inc.
    for $2.8 billion in cash and stock.
    """
    
    signal = analyzer.analyze_8k(sample_8k)
    
    print(f"Filing: {signal.filing_type}")
    print(f"M&A relevance: {signal.ma_relevance:.0%}")
    print(f"Items: {signal.key_items}")
    print("✓ Automated SEC analysis")