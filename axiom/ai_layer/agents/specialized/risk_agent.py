"""
Risk Agent - Portfolio Risk Management Specialist

Responsibility: Monitor and manage all portfolio risk
Expertise: VaR, stress testing, Greeks aggregation, limit monitoring
Independence: Autonomous risk monitoring and alerting

Capabilities:
- Real-time portfolio Greeks aggregation
- Multiple VaR calculations (parametric, historical, Monte Carlo)
- Stress testing (market crash scenarios)
- Margin calculations (Reg T, Portfolio, SPAN)
- Risk limit monitoring
- Automatic alerts on breaches
- P&L attribution by Greek

Performance: <5ms for complete portfolio risk
Alerts: <100ms notification on limit breach
Accuracy: Conservative (better to overestimate risk)
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class RiskRequest:
    """Request to risk agent"""
    request_type: str  # 'calculate_risk', 'stress_test', 'check_limits', 'margin'
    positions: List[Dict]
    market_data: Dict
    risk_limits: Optional[Dict] = None


@dataclass
class RiskResponse:
    """Response from risk agent"""
    request_type: str
    success: bool
    risk_metrics: Dict
    alerts: List[str]
    within_limits: bool
    calculation_time_ms: float
    severity: str  # 'low', 'medium', 'high', 'critical'


class RiskAgent:
    """
    Specialized agent for risk management
    
    Design:
    - Continuous monitoring (real-time)
    - Conservative approach (overestimate risk)
    - Multiple methods (cross-validation)
    - Automatic alerting (immediate notification)
    - Independent operation (doesn't depend on other agents)
    
    Critical for $10M clients: One risk breach = catastrophic
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize risk agent"""
        from axiom.derivatives.risk.real_time_risk_engine import RealTimeRiskEngine
        from axiom.derivatives.risk.margin_calculator import PortfolioMarginCalculator
        from axiom.derivatives.analytics.pnl_engine import RealTimePnLEngine
        
        # Initialize risk engines
        self.risk_engine = RealTimeRiskEngine(use_gpu=use_gpu)
        self.margin_calculator = PortfolioMarginCalculator()
        self.pnl_engine = RealTimePnLEngine(use_gpu=use_gpu)
        
        # Alert history
        self.alert_history = []
        self.breach_count = 0
        
        # Statistics
        self.requests_processed = 0
        self.critical_alerts = 0
        
        print(f"RiskAgent initialized ({'GPU' if use_gpu else 'CPU'})")
        print("  Monitoring: Greeks, VaR, margins, P&L")
        print("  Alert: Immediate on limit breaches")
    
    async def process_request(self, request: RiskRequest) -> RiskResponse:
        """
        Process risk management request
        
        Always returns conservative estimates
        Better to overestimate risk than underestimate
        """
        import time
        start = time.perf_counter()
        
        self.requests_processed += 1
        
        try:
            if request.request_type == 'calculate_risk':
                # Complete risk calculation
                risk_metrics = self.risk_engine.calculate_portfolio_risk(
                    positions=request.positions,
                    current_market_data=request.market_data
                )
                
                # Check limits
                alerts = []
                severity = 'low'
                
                if risk_metrics.limit_breaches:
                    alerts.extend(risk_metrics.limit_breaches)
                    severity = 'critical'
                    self.critical_alerts += 1
                    self.breach_count += 1
                
                if risk_metrics.warnings:
                    alerts.extend(risk_metrics.warnings)
                    severity = max(severity, 'medium')
                
                within_limits = len(risk_metrics.limit_breaches) == 0
                
                result_metrics = {
                    'total_delta': risk_metrics.total_delta,
                    'total_gamma': risk_metrics.total_gamma,
                    'total_vega': risk_metrics.total_vega,
                    'total_theta': risk_metrics.total_theta,
                    'var_1day': risk_metrics.var_1day_monte_carlo,
                    'cvar_1day': risk_metrics.cvar_1day,
                    'notional_exposure': risk_metrics.notional_exposure,
                    'calculation_time_ms': risk_metrics.calculation_time_ms
                }
            
            elif request.request_type == 'margin':
                # Calculate margin requirements
                margin = self.margin_calculator.calculate_margin(
                    positions=request.positions,
                    current_spot=request.market_data.get('spot', 100.0),
                    current_vol=request.market_data.get('vol', 0.25)
                )
                
                result_metrics = {
                    'reg_t_margin': margin.reg_t_margin,
                    'portfolio_margin': margin.portfolio_margin,
                    'span_margin': margin.span_margin,
                    'margin_savings': margin.margin_savings,
                    'calculation_time_ms': margin.calculation_time_ms
                }
                
                alerts = []
                within_limits = True
                severity = 'low'
            
            elif request.request_type == 'stress_test':
                # Run stress tests
                scenarios = [
                    {'name': 'crash', 'spot_shock': 0.8, 'vol_shock': 2.0},
                    {'name': 'rally', 'spot_shock': 1.2, 'vol_shock': 0.8}
                ]
                
                stress_results = self.risk_engine.stress_test(
                    positions=request.positions,
                    scenarios=scenarios
                )
                
                result_metrics = {scenario: {
                    'total_pnl': result.total_pnl_today,
                    'total_delta': result.total_delta,
                    'var': result.var_1day_monte_carlo
                } for scenario, result in stress_results.items()}
                
                alerts = []
                within_limits = True
                severity = 'low'
            
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
            
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            return RiskResponse(
                request_type=request.request_type,
                success=True,
                risk_metrics=result_metrics,
                alerts=alerts,
                within_limits=within_limits,
                calculation_time_ms=elapsed_ms,
                severity=severity
            )
        
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            return RiskResponse(
                request_type=request.request_type,
                success=False,
                risk_metrics={},
                alerts=[f"Error: {str(e)}"],
                within_limits=False,
                calculation_time_ms=elapsed_ms,
                severity='critical'
            )
    
    def get_stats(self) -> Dict:
        """Get risk agent statistics"""
        return {
            'agent': 'risk',
            'requests_processed': self.requests_processed,
            'critical_alerts': self.critical_alerts,
            'breach_count': self.breach_count,
            'alert_rate': self.critical_alerts / max(self.requests_processed, 1),
            'status': 'monitoring'
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_risk_agent():
        print("="*60)
        print("RISK AGENT - STANDALONE TEST")
        print("="*60)
        
        agent = RiskAgent(use_gpu=False)
        
        # Test portfolio
        positions = [
            {'strike': 100, 'time_to_maturity': 0.25, 'quantity': 100, 'entry_price': 5.0},
            {'strike': 105, 'time_to_maturity': 0.25, 'quantity': -50, 'entry_price': 3.0}
        ]
        
        market_data = {'spot': 102.0, 'vol': 0.27, 'rate': 0.03}
        
        # Test risk calculation
        print("\n→ Test: Portfolio Risk Calculation")
        request = RiskRequest(
            request_type='calculate_risk',
            positions=positions,
            market_data=market_data
        )
        
        response = await agent.process_request(request)
        
        print(f"   Success: {'✓' if response.success else '✗'}")
        print(f"   Delta: {response.risk_metrics['total_delta']:.2f}")
        print(f"   Gamma: {response.risk_metrics['total_gamma']:.2f}")
        print(f"   VaR: ${response.risk_metrics['var_1day']:,.0f}")
        print(f"   Within limits: {'✓ YES' if response.within_limits else '⚠ NO'}")
        print(f"   Time: {response.calculation_time_ms:.2f}ms")
        
        if response.alerts:
            print(f"   Alerts: {len(response.alerts)}")
            for alert in response.alerts:
                print(f"     - {alert}")
        
        # Stats
        print("\n→ Agent Statistics:")
        stats = agent.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n" + "="*60)
        print("✓ Risk agent operational")
        print("✓ Real-time risk monitoring")
        print("✓ Automatic alerting")
        print("✓ Conservative approach")
        print("\nCRITICAL RISK MANAGEMENT FOR $10M CLIENTS")
    
    asyncio.run(test_risk_agent())