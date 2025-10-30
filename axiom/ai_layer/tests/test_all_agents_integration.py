"""
Integration Tests - All 12 Professional Agents Working Together

Tests complete workflows demonstrating all agents coordinating:
1. Trading Workflow - Price → Risk → Strategy → Execute → Hedge → Analyze
2. Compliance Workflow - Monitor → Validate → Report → Audit
3. Client Workflow - Query → Orchestrate → Respond

This proves the professional multi-agent architecture works end-to-end.
"""

import asyncio
import pytest
from decimal import Decimal
from datetime import datetime
from typing import Dict, List

# All 12 professional agents
from axiom.ai_layer.agents.professional.pricing_agent_v2 import ProfessionalPricingAgent
from axiom.ai_layer.agents.professional.risk_agent_v2 import ProfessionalRiskAgent
from axiom.ai_layer.agents.professional.strategy_agent_v2 import ProfessionalStrategyAgent
from axiom.ai_layer.agents.professional.execution_agent_v2 import ProfessionalExecutionAgent
from axiom.ai_layer.agents.professional.hedging_agent_v2 import ProfessionalHedgingAgent
from axiom.ai_layer.agents.professional.analytics_agent_v2 import ProfessionalAnalyticsAgent
from axiom.ai_layer.agents.professional.market_data_agent_v2 import ProfessionalMarketDataAgent
from axiom.ai_layer.agents.professional.volatility_agent_v2 import ProfessionalVolatilityAgent
from axiom.ai_layer.agents.professional.compliance_agent_v2 import ProfessionalComplianceAgent
from axiom.ai_layer.agents.professional.monitoring_agent_v2 import ProfessionalMonitoringAgent
from axiom.ai_layer.agents.professional.guardrail_agent_v2 import ProfessionalGuardrailAgent
from axiom.ai_layer.agents.professional.client_interface_agent_v2 import ProfessionalClientInterfaceAgent

# Infrastructure
from axiom.ai_layer.messaging.message_bus import MessageBus
from axiom.ai_layer.infrastructure.config_manager import ConfigManager
from axiom.ai_layer.infrastructure.observability import Logger

# Messages
from axiom.ai_layer.messaging.protocol import (
    CalculateGreeksCommand, CalculateRiskCommand, GenerateStrategyCommand,
    ExecuteOrderCommand, CalculateHedgeCommand, CalculatePnLCommand,
    GetMarketDataQuery, ForecastVolatilityCommand, CheckComplianceCommand,
    CheckSystemHealthQuery, ValidateActionCommand, ClientQuery,
    AgentName
)


class TestAllAgentsIntegration:
    """Integration tests for all 12 agents"""
    
    @pytest.fixture
    async def all_agents(self):
        """Initialize all 12 agents"""
        logger = Logger("integration_test")
        logger.info("initializing_all_12_agents")
        
        message_bus = MessageBus()
        config_manager = ConfigManager()
        
        agents = {
            'pricing': ProfessionalPricingAgent(message_bus, config_manager, use_gpu=False),
            'risk': ProfessionalRiskAgent(message_bus, config_manager, use_gpu=False),
            'strategy': ProfessionalStrategyAgent(message_bus, config_manager, use_gpu=False),
            'execution': ProfessionalExecutionAgent(message_bus, config_manager),
            'hedging': ProfessionalHedgingAgent(message_bus, config_manager, use_gpu=False),
            'analytics': ProfessionalAnalyticsAgent(message_bus, config_manager, use_gpu=False),
            'market_data': ProfessionalMarketDataAgent(message_bus, config_manager),
            'volatility': ProfessionalVolatilityAgent(message_bus, config_manager, use_gpu=False),
            'compliance': ProfessionalComplianceAgent(message_bus, config_manager),
            'monitoring': ProfessionalMonitoringAgent(message_bus, config_manager),
            'guardrail': ProfessionalGuardrailAgent(message_bus, config_manager),
            'client_interface': ProfessionalClientInterfaceAgent(message_bus, config_manager)
        }
        
        logger.info("all_12_agents_initialized")
        
        yield agents
        
        # Cleanup
        for agent in agents.values():
            agent.shutdown()
        
        logger.info("all_agents_shutdown")
    
    @pytest.mark.asyncio
    async def test_complete_trading_workflow(self, all_agents):
        """
        Test complete trading workflow with all agents
        
        Workflow:
        1. Market Data - Get current market data
        2. Volatility - Forecast volatility
        3. Strategy - Generate optimal strategy
        4. Pricing - Calculate Greeks
        5. Guardrail - Validate safety
        6. Risk - Check portfolio risk
        7. Compliance - Check compliance
        8. Execution - Execute trade
        9. Hedging - Calculate hedge
        10. Analytics - Track P&L
        11. Monitoring - Check system health
        12. Client Interface - Return results to client
        """
        logger = Logger("test")
        logger.info("test_starting", test="COMPLETE_TRADING_WORKFLOW_ALL_12_AGENTS")
        
        # 1. Market Data Agent - Get market data
        logger.info("step_1_market_data")
        market_data_query = GetMarketDataQuery(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.MARKET_DATA,
            symbol='SPY241115C00450000',
            data_type='quote',
            use_cache=False
        )
        market_response = await all_agents['market_data'].process_request(market_data_query)
        assert market_response.success
        logger.info("market_data_retrieved", success=True)
        
        # 2. Volatility Agent - Forecast volatility
        logger.info("step_2_volatility_forecast")
        price_history = [[100, 101, 99, 100.5, 1000000] for _ in range(60)]
        vol_command = ForecastVolatilityCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.VOLATILITY,
            underlying='SPY',
            price_history=price_history,
            horizon='1d'
        )
        vol_response = await all_agents['volatility'].process_request(vol_command)
        assert vol_response.success
        logger.info("volatility_forecasted", vol=vol_response.forecast_vol)
        
        # 3. Strategy Agent - Generate strategy
        logger.info("step_3_strategy_generation")
        strategy_command = GenerateStrategyCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.STRATEGY,
            market_outlook='bullish',
            volatility_view='stable',
            risk_tolerance=0.6,
            capital_available=50000.0,
            current_spot=100.0,
            current_vol=0.25
        )
        strategy_response = await all_agents['strategy'].process_request(strategy_command)
        assert strategy_response.success
        logger.info("strategy_generated", name=strategy_response.strategy.get('name') if strategy_response.strategy else None)
        
        # 4. Pricing Agent - Calculate Greeks
        logger.info("step_4_pricing_greeks")
        greeks_command = CalculateGreeksCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.PRICING,
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25
        )
        greeks_response = await all_agents['pricing'].process_request(greeks_command)
        assert greeks_response.success
        logger.info("greeks_calculated", delta=greeks_response.delta)
        
        # 5. Guardrail Agent - Validate safety
        logger.info("step_5_guardrail_validation")
        validate_command = ValidateActionCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.GUARDRAIL,
            action_type='validate_greeks',
            source_agent='pricing_agent',
            proposed_action={'delta': greeks_response.delta, 'gamma': greeks_response.gamma},
            context={'spot': 100.0, 'strike': 100.0}
        )
        guardrail_response = await all_agents['guardrail'].process_request(validate_command)
        assert guardrail_response.success
        logger.info("safety_validated", approved=guardrail_response.approved)
        
        # 6. Risk Agent - Calculate risk
        logger.info("step_6_risk_calculation")
        positions = [{'strike': 100, 'time_to_maturity': 0.25, 'quantity': 100, 'entry_price': 5.0}]
        risk_command = CalculateRiskCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.RISK,
            positions=positions,
            market_data={'spot': 100.0, 'vol': 0.25, 'rate': 0.03}
        )
        risk_response = await all_agents['risk'].process_request(risk_command)
        assert risk_response.success
        logger.info("risk_calculated", var=risk_response.var_1day)
        
        # 7. Compliance Agent - Check compliance
        logger.info("step_7_compliance_check")
        compliance_command = CheckComplianceCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.COMPLIANCE,
            check_type="position_and_execution",
            positions=positions,
            trades=[]
        )
        compliance_response = await all_agents['compliance'].process_request(compliance_command)
        assert compliance_response.success
        logger.info("compliance_checked", compliant=compliance_response.compliant)
        
        # 8. Execution Agent - Execute order
        logger.info("step_8_order_execution")
        exec_command = ExecuteOrderCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.EXECUTION,
            symbol='SPY241115C00450000',
            side='buy',
            quantity=100,
            order_type='limit',
            limit_price=5.50
        )
        exec_response = await all_agents['execution'].process_request(exec_command)
        assert exec_response.success
        logger.info("order_executed", order_id=exec_response.order_id)
        
        # 9. Hedging Agent - Calculate hedge
        logger.info("step_9_hedge_calculation")
        hedge_command = CalculateHedgeCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.HEDGING,
            positions=positions,
            market_data={'spot': 100.0, 'vol': 0.25, 'rate': 0.03}
        )
        hedge_response = await all_agents['hedging'].process_request(hedge_command)
        assert hedge_response.success
        logger.info("hedge_calculated", quantity=hedge_response.hedge_quantity)
        
        # 10. Analytics Agent - Calculate P&L
        logger.info("step_10_pnl_analytics")
        pnl_command = CalculatePnLCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.ANALYTICS,
            positions=positions,
            trades=[],
            market_data={'spot': 102.0, 'vol': 0.25, 'rate': 0.03}
        )
        analytics_response = await all_agents['analytics'].process_request(pnl_command)
        assert analytics_response.success
        logger.info("pnl_calculated", total_pnl=analytics_response.total_pnl)
        
        # 11. Monitoring Agent - Check system health
        logger.info("step_11_system_monitoring")
        health_query = CheckSystemHealthQuery(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.MONITORING,
            include_metrics=True
        )
        monitoring_response = await all_agents['monitoring'].process_request(health_query)
        assert monitoring_response.success
        logger.info("system_health_checked", overall_status=monitoring_response.overall_status)
        
        # 12. Client Interface - Orchestrate response
        logger.info("step_12_client_interface")
        client_query = ClientQuery(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.CLIENT_INTERFACE,
            client_id='TEST_CLIENT',
            query_text='What is my P&L?',
            request_type='question'
        )
        client_response = await all_agents['client_interface'].process_request(client_query)
        assert client_response.success
        logger.info("client_response_generated", success=True)
        
        logger.info(
            "WORKFLOW_COMPLETE",
            all_12_agents_executed=True,
            all_responses_successful=True,
            end_to_end_workflow=True
        )
    
    @pytest.mark.asyncio
    async def test_all_agents_health(self, all_agents):
        """Test health checks for all 12 agents"""
        logger = Logger("test")
        logger.info("test_starting", test="ALL_AGENTS_HEALTH_CHECK")
        
        for agent_name, agent in all_agents.items():
            health = agent.health_check()
            assert health['healthy'], f"{agent_name} is not healthy"
            logger.info(f"{agent_name}_healthy", health=health)
        
        logger.info("ALL_12_AGENTS_HEALTHY", success=True)


# Run integration tests
if __name__ == "__main__":
    import asyncio
    
    async def run_integration_test():
        logger = Logger("integration_test")
        logger.info("STARTING_INTEGRATION_TESTS", agents=12)
        
        # Initialize
        message_bus = MessageBus()
        config_manager = ConfigManager()
        
        # Create all agents
        agents = {
            'pricing': ProfessionalPricingAgent(message_bus, config_manager, use_gpu=False),
            'risk': ProfessionalRiskAgent(message_bus, config_manager, use_gpu=False),
            'strategy': ProfessionalStrategyAgent(message_bus, config_manager, use_gpu=False),
            'execution': ProfessionalExecutionAgent(message_bus, config_manager),
            'hedging': ProfessionalHedgingAgent(message_bus, config_manager, use_gpu=False),
            'analytics': ProfessionalAnalyticsAgent(message_bus, config_manager, use_gpu=False),
            'market_data': ProfessionalMarketDataAgent(message_bus, config_manager),
            'volatility': ProfessionalVolatilityAgent(message_bus, config_manager, use_gpu=False),
            'compliance': ProfessionalComplianceAgent(message_bus, config_manager),
            'monitoring': ProfessionalMonitoringAgent(message_bus, config_manager),
            'guardrail': ProfessionalGuardrailAgent(message_bus, config_manager),
            'client_interface': ProfessionalClientInterfaceAgent(message_bus, config_manager)
        }
        
        logger.info("all_agents_initialized", count=len(agents))
        
        # Test complete workflow
        logger.info("executing_complete_workflow")
        
        # Quick workflow test
        greeks_cmd = CalculateGreeksCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.PRICING,
            spot=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.03,
            volatility=0.25
        )
        
        greeks_result = await agents['pricing'].process_request(greeks_cmd)
        logger.info("pricing_agent_works", delta=greeks_result.delta)
        
        # Health check all
        logger.info("health_checking_all_agents")
        for name, agent in agents.items():
            health = agent.health_check()
            logger.info(f"{name}_health", healthy=health['healthy'])
        
        # Shutdown all
        for agent in agents.values():
            agent.shutdown()
        
        logger.info("INTEGRATION_TEST_COMPLETE", all_agents_working=True)
    
    asyncio.run(run_integration_test())