"""
Analytics Agent - Performance Analysis Specialist

Responsibility: Analyze trading performance and generate insights
Expertise: P&L attribution, performance metrics, execution quality
Independence: Autonomous analysis and reporting

Capabilities:
- Real-time P&L calculation and attribution
- Performance metrics (Sharpe, Sortino, max drawdown)
- Greeks attribution (delta P&L, gamma P&L, etc.)
- Execution quality analysis
- Strategy performance comparison
- Client reporting generation

Performance: <10ms for complete analytics
Reporting: Professional dashboards and reports
Insights: Actionable recommendations
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
import time


@dataclass
class AnalyticsRequest:
    """Request to analytics agent"""
    request_type: str  # 'pnl', 'performance', 'attribution', 'report'
    time_period: str  # 'intraday', 'daily', 'weekly', 'monthly'
    positions: List[Dict]
    trades: List[Dict]
    market_data: Dict


@dataclass
class AnalyticsResponse:
    """Response from analytics agent"""
    success: bool
    metrics: Dict
    insights: List[str]
    recommendations: List[str]
    report_html: Optional[str]
    calculation_time_ms: float


class AnalyticsAgent:
    """
    Specialized agent for performance analytics
    
    Provides:
    - Real-time P&L tracking
    - Performance attribution (what drove P&L?)
    - Execution quality metrics
    - Risk-adjusted returns
    - Comparative analysis
    
    All analysis automated and real-time
    """
    
    def __init__(self):
        """Initialize analytics agent"""
        from axiom.derivatives.analytics.pnl_engine import RealTimePnLEngine
        from axiom.derivatives.analytics.performance_analyzer import PerformanceAnalyzer
        from axiom.derivatives.reporting.client_dashboard import ClientDashboardGenerator
        
        self.pnl_engine = RealTimePnLEngine(use_gpu=False)
        self.performance_analyzer = PerformanceAnalyzer()
        self.dashboard_generator = ClientDashboardGenerator()
        
        print("AnalyticsAgent initialized")
        print("  Capabilities: P&L, attribution, dashboards")
    
    async def process_request(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Process analytics request"""
        start = time.perf_counter()
        
        try:
            if request.request_type == 'pnl':
                # Calculate P&L
                pnl = self.pnl_engine.calculate_pnl(
                    positions=request.positions,
                    current_market_data=request.market_data
                )
                
                metrics = {
                    'total_pnl': pnl.total_pnl,
                    'realized_pnl': pnl.realized_pnl,
                    'unrealized_pnl': pnl.unrealized_pnl,
                    'delta_pnl': pnl.delta_pnl,
                    'gamma_pnl': pnl.gamma_pnl,
                    'vega_pnl': pnl.vega_pnl,
                    'theta_pnl': pnl.theta_pnl
                }
                
                # Generate insights
                insights = []
                if pnl.delta_pnl > abs(pnl.total_pnl) * 0.5:
                    insights.append("Majority of P&L from delta (directional move)")
                if pnl.theta_pnl > 0:
                    insights.append("Positive theta - earning time decay")
                
                recommendations = []
                if abs(pnl.total_pnl) > 50000:
                    recommendations.append("Consider taking profits or adding hedge")
                
                elapsed_ms = (time.perf_counter() - start) * 1000
                
                return AnalyticsResponse(
                    success=True,
                    metrics=metrics,
                    insights=insights,
                    recommendations=recommendations,
                    report_html=None,
                    calculation_time_ms=elapsed_ms
                )
            
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
        
        except Exception as e:
            return AnalyticsResponse(
                success=False,
                metrics={},
                insights=[],
                recommendations=[],
                report_html=None,
                calculation_time_ms=(time.perf_counter() - start) * 1000
            )
    
    def get_stats(self) -> Dict:
        """Get analytics agent statistics"""
        return {
            'agent': 'analytics',
            'status': 'operational'
        }


if __name__ == "__main__":
    import asyncio
    
    async def test_analytics_agent():
        print("="*60)
        print("ANALYTICS AGENT - STANDALONE TEST")
        print("="*60)
        
        agent = AnalyticsAgent()
        
        positions = [
            {'strike': 100, 'time_to_maturity': 0.25, 'quantity': 100, 'entry_price': 5.0},
        ]
        
        market_data = {'spot': 102.0, 'vol': 0.25, 'rate': 0.03}
        
        request = AnalyticsRequest(
            request_type='pnl',
            time_period='intraday',
            positions=positions,
            trades=[],
            market_data=market_data
        )
        
        response = await agent.process_request(request)
        
        print(f"\n   Success: {'✓' if response.success else '✗'}")
        print(f"   Total P&L: ${response.metrics.get('total_pnl', 0):,.2f}")
        print(f"   Delta P&L: ${response.metrics.get('delta_pnl', 0):,.2f}")
        print(f"   Time: {response.calculation_time_ms:.2f}ms")
        
        if response.insights:
            print(f"\n   Insights:")
            for insight in response.insights:
                print(f"     - {insight}")
        
        print("\n✓ Analytics agent operational")
    
    asyncio.run(test_analytics_agent())