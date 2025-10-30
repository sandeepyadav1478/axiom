"""Patent Analysis for M&A Technology Assessment"""
class PatentAnalysisMA:
    def analyze_ip_portfolio(self, target_patents):
        score = 0.0
        if target_patents.get('count', 0) > 50:
            score += 0.3
        if target_patents.get('citations', 0) > 100:
            score += 0.25
        if target_patents.get('recent_filings', 0) > 10:
            score += 0.20
        return min(1.0, score)

if __name__ == "__main__":
    analyzer = PatentAnalysisMA()
    s = analyzer.analyze_ip_portfolio({'count': 75, 'citations': 150, 'recent_filings': 15})
    print(f"IP score: {s:.0%}")