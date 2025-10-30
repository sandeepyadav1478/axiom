"""Peer Comparison Credit Model"""
import numpy as np

class PeerComparisonCredit:
    def assess_vs_peers(self, company_metrics, industry_metrics):
        score = 50
        
        if company_metrics['leverage'] < industry_metrics['avg_leverage']:
            score += 10
        if company_metrics['coverage'] > industry_metrics['avg_coverage']:
            score += 10
        if company_metrics['growth'] > industry_metrics['avg_growth']:
            score += 5
        
        return min(100, max(0, score))

if __name__ == "__main__":
    model = PeerComparisonCredit()
    s = model.assess_vs_peers(
        {'leverage': 2.5, 'coverage': 4.0, 'growth': 0.15},
        {'avg_leverage': 3.0, 'avg_coverage': 3.5, 'avg_growth': 0.10}
    )
    print(f"Score: {s}")