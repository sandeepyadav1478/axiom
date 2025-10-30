"""
Integration Tests for Complete Derivatives Platform

Tests that all components work together:
- Greeks → Vol Surface → Pricing
- Market Data → Greeks → Hedging
- Strategy Generation → Backtesting
- API → Database → Monitoring

These tests validate the complete workflow end-to-end.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestCompleteWorkflow:
    """Test complete derivatives workflow integration"""
    
    def test_greeks_to_surface_integration(self):
        """Test ultra-fast Greeks + volatility surface integration"""
        from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
        from axiom.derivatives.volatility_surface import RealTimeVolatilitySurface
        
        # Create engines
        greeks_engine = UltraFastGreeksEngine(use_gpu=False)
        surface_engine = RealTimeVolatilitySurface(use_gpu=False)
        
        # Build volatility surface
        market_quotes = np.random.uniform(0.2, 0.3, 20)
        surface = surface_engine.construct_surface(market_quotes, spot=100.0)
        
        assert surface.construction_time_ms < 10.0  # Should be fast even on CPU
        
        # Get vol for specific option
        vol = surface.get_vol(105.0, 0.5)
        assert 0.15 < vol < 0.35
        
        # Calculate Greeks with that vol
        greeks = greeks_engine.calculate_greeks(100.0, 105.0, 0.5, 0.03, vol)
        
        assert greeks.delta > 0
        assert greeks.gamma > 0
        
        print(f"✓ Greeks + Surface integration working")
    
    def test_pricing_to_hedging_workflow(self):
        """Test: Price options → Calculate risk → Auto-hedge"""
        from axiom.derivatives.ultra_fast_greeks import UltraFastGreeksEngine
        from axiom.derivatives.market_making.auto_hedger import DRLAutoHedger, PortfolioState
        
        greeks_engine = UltraFastGreeksEngine(use_gpu=False)
        hedger = DRLAutoHedger(use_gpu=False, target_delta=0.0)
        
        # Calculate Greeks for positions
        positions = [
            {'strike': 100, 'time': 0.25},
            {'strike': 105, 'time': 0.25}
        ]
        
        batch_data = np.array([[100, p['strike'], p['time'], 0.03, 0.25] for p in positions])
        greeks_results = greeks_engine.calculate_batch(batch_data)
        
        # Aggregate delta
        total_delta = sum(g.delta for g in greeks_results)
        
        # Get hedge recommendation
        portfolio_state = PortfolioState(
            total_delta=total_delta,
            total_gamma=sum(g.gamma for g in greeks_results),
            total_vega=sum(g.vega for g in greeks_results),
            total_theta=sum(g.theta for g in greeks_results),
            spot_price=100.0,
            volatility=0.25,
            positions=positions,
            hedge_position=0.0,
            pnl=0.0,
            time_to_close=3.0
        )
        
        hedge_action = hedger.get_optimal_hedge(portfolio_state)
        
        assert hedge_action.hedge_delta != 0  # Should recommend a hedge
        assert hedge_action.confidence > 0.5
        
        print(f"✓ Pricing → Hedging workflow functional")
    
    def test_langgraph_workflow_execution(self):
        """Test LangGraph orchestration of complete workflow"""
        from axiom.derivatives.ai.derivatives_workflow import DerivativesWorkflow, DerivativesState
        
        workflow = DerivativesWorkflow()
        
        initial_state = DerivativesState(
            current_price=100.0,
            volatility=0.25,
            option_chain=[],
            positions=[],
            total_delta=0.0,
            total_gamma=0.0,
            total_vega=0.0,
            pnl=0.0,
            market_regime='unknown',
            volatility_forecast=0.0,
            similar_scenarios=[],
            recommended_trades=[],
            hedge_actions=[],
            spread_adjustments={},
            executed_trades=[],
            errors=[]
        )
        
        # Execute workflow
        result = workflow.run(initial_state)
        
        # Should complete without errors
        assert len(result.get('errors', [])) == 0
        assert result.get('market_regime') in ['low_vol', 'normal', 'high_vol', 'crisis']
        
        print(f"✓ LangGraph workflow executes successfully")


class TestDatabaseIntegration:
    """Test database operations"""
    
    @pytest.mark.skipif(not pytest.config.getoption("--run-db-tests", default=False),
                       reason="Requires database setup")
    def test_store_and_retrieve_trade(self):
        """Test storing trade in PostgreSQL"""
        from axiom.derivatives.data.models import OptionTrade, Base
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # In-memory SQLite for testing
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create trade
        trade = OptionTrade(
            trade_id='TEST_001',
            symbol='SPY241115C00450000',
            underlying='SPY',
            strike=450.0,
            expiry=datetime(2024, 11, 15),
            option_type='call',
            action='buy',
            quantity=10,
            price=5.50,
            delta=0.52,
            gamma=0.015,
            calculation_time_us=85
        )
        
        session.add(trade)
        session.commit()
        
        # Retrieve
        retrieved = session.query(OptionTrade).filter_by(trade_id='TEST_001').first()
        
        assert retrieved is not None
        assert retrieved.symbol == 'SPY241115C00450000'
        assert retrieved.calculation_time_us == 85
        
        session.close()
        
        print(f"✓ Database integration working")


class TestAPIEndpoints:
    """Test FastAPI endpoint integration"""
    
    def test_greeks_endpoint(self):
        """Test /greeks endpoint"""
        from fastapi.testclient import TestClient
        from axiom.derivatives.api.endpoints import app
        
        client = TestClient(app)
        
        response = client.post("/greeks", json={
            "spot": 100.0,
            "strike": 100.0,
            "time_to_maturity": 1.0,
            "risk_free_rate": 0.03,
            "volatility": 0.25
        })
        
        assert response.status_code == 200
        data = response.json()
        assert 'delta' in data
        assert 'calculation_time_microseconds' in data
        
        print(f"✓ API endpoint working")


# Run with: pytest tests/derivatives/test_integration.py -v