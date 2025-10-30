"""Competitive Intelligence Monitor for M&A"""
class CompetitiveIntelligenceMonitor:
    def monitor_competitor_activity(self, competitor, signals):
        ma_score = 0.0
        if signals.get('acquisitions_last_year', 0) > 2:
            ma_score += 0.3
        if signals.get('hiring_ma_bankers'):
            ma_score += 0.25
        if signals.get('credit_facility_expansion'):
            ma_score += 0.20
        return min(1.0, ma_score)

if __name__ == "__main__":
    monitor = CompetitiveIntelligenceMonitor()
    score = monitor.monitor_competitor_activity('Competitor', {'acquisitions_last_year': 3, 'hiring_ma_bankers': True})
    print(f"M&A activity score: {score:.0%}")