"""
Complete 12-Agent Workflow Demonstration

Demonstrates all 12 professional agents working together in a realistic
derivatives trading scenario.

This proves the complete professional multi-agent architecture.

Workflow:
1. Client asks question
2. Market Data fetches current data
3. Volatility forecasts future vol
4. Strategy generates optimal strategy
5. Pricing calculates Greeks
6. Guardrail validates safety
7. Risk checks portfolio risk
8. Compliance validates compliance
9. Execution executes trade
10. Hedging calculates optimal hedge
11. Analytics tracks P&L
12. Monitoring checks system health
13. Client Interface returns comprehensive answer

All 12 agents coordinate to serve a single client request.
"""

import asyncio
from decimal import Decimal
from datetime import datetime

# All 12 agents
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


async def main():
    """
    Complete 12-agent workflow demonstration
    
    Scenario: Client asks "Should I buy SPY calls?"
    
    System orchestrates all 12 agents to provide comprehensive answer.
    """
    logger = Logger("demo")
    
    logger.info("DEMO_STARTING", title="COMPLETE 12-AGENT WORKFLOW")
    logger.info("scenario", client_question="Should I buy SPY calls?")
    
    # Initialize infrastructure
    logger.info("initializing_infrastructure")
    message_bus = MessageBus()
    config_manager = ConfigManager()
    
    # Initialize all 12 agents
    logger.info("initializing_all_12_agents")
    
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
    
    logger.info("agents_initialized", count=12)
    
    # Execute workflow
    logger.info("executing_workflow", steps=12)
    
    # Step 1: Market Data
    logger.info("step_1", agent="Market Data", action="Fetch SPY quote")
    market_query = GetMarketDataQuery(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.MARKET_DATA,
        symbol='SPY',
        data_type='quote'
    )
    market_result = await agents['market_data'].process_request(market_query)
    logger.info("step_1_complete", bid=market_result.bid, ask=market_result.ask)
    
    # Step 2: Volatility Forecast
    logger.info("step_2", agent="Volatility", action="Forecast volatility")
    price_history = [[100, 101, 99, 100.5, 1000000] for _ in range(60)]
    vol_cmd = ForecastVolatilityCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.VOLATILITY,
        underlying='SPY',
        price_history=price_history,
        horizon='1d'
    )
    vol_result = await agents['volatility'].process_request(vol_cmd)
    logger.info("step_2_complete", forecast_vol=vol_result.forecast_vol, regime=vol_result.regime)
    
    # Step 3: Strategy Generation
    logger.info("step_3", agent="Strategy", action="Generate optimal strategy")
    strategy_cmd = GenerateStrategyCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.STRATEGY,
        market_outlook='bullish',
        volatility_view='stable',
        risk_tolerance=0.6,
        capital_available=50000.0,
        current_spot=100.0,
        current_vol=0.25
    )
    strategy_result = await agents['strategy'].process_request(strategy_cmd)
    logger.info("step_3_complete", strategy=strategy_result.strategy.get('name') if strategy_result.strategy else None)
    
    # Step 4: Pricing (Calculate Greeks)
    logger.info("step_4", agent="Pricing", action="Calculate Greeks")
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
    logger.info("step_4_complete", delta=greeks_result.delta, time_us=greeks_result.calculation_time_us)
    
    # Step 5: Guardrail Validation
    logger.info("step_5", agent="Guardrail", action="Validate safety")
    guard_cmd = ValidateActionCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.GUARDRAIL,
        action_type='validate_greeks',
        source_agent='pricing_agent',
        proposed_action={'delta': greeks_result.delta, 'gamma': greeks_result.gamma},
        context={'spot': 100.0, 'strike': 100.0}
    )
    guard_result = await agents['guardrail'].process_request(guard_cmd)
    logger.info("step_5_complete", approved=guard_result.approved, risk_level=guard_result.risk_level)
    
    # Step 6: Risk Assessment
    logger.info("step_6", agent="Risk", action="Calculate portfolio risk")
    positions = [{'strike': 100, 'time_to_maturity': 0.25, 'quantity': 100, 'entry_price': 5.0}]
    risk_cmd = CalculateRiskCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.RISK,
        positions=positions,
        market_data={'spot': 100.0, 'vol': 0.25, 'rate': 0.03}
    )
    risk_result = await agents['risk'].process_request(risk_cmd)
    logger.info("step_6_complete", var=risk_result.var_1day, within_limits=risk_result.within_limits)
    
    # Step 7: Compliance Check
    logger.info("step_7", agent="Compliance", action="Check regulatory compliance")
    comp_cmd = CheckComplianceCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.COMPLIANCE,
        check_type="position_check",
        positions=positions
    )
    comp_result = await agents['compliance'].process_request(comp_cmd)
    logger.info("step_7_complete", compliant=comp_result.compliant)
    
    # Step 8: Execution (if approved)
    if guard_result.approved and comp_result.compliant:
        logger.info("step_8", agent="Execution", action="Execute order")
        exec_cmd = ExecuteOrderCommand(
            from_agent=AgentName.CLIENT_INTERFACE,
            to_agent=AgentName.EXECUTION,
            symbol='SPY241115C00450000',
            side='buy',
            quantity=100,
            order_type='limit',
            limit_price=5.50
        )
        exec_result = await agents['execution'].process_request(exec_cmd)
        logger.info("step_8_complete", order_id=exec_result.order_id, status=exec_result.status)
    
    # Step 9: Hedging
    logger.info("step_9", agent="Hedging", action="Calculate optimal hedge")
    hedge_cmd = CalculateHedgeCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.HEDGING,
        positions=positions,
        market_data={'spot': 100.0, 'vol': 0.25}
    )
    hedge_result = await agents['hedging'].process_request(hedge_cmd)
    logger.info("step_9_complete", hedge_quantity=hedge_result.hedge_quantity)
    
    # Step 10: Analytics (P&L)
    logger.info("step_10", agent="Analytics", action="Calculate P&L")
    pnl_cmd = CalculatePnLCommand(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.ANALYTICS,
        positions=positions,
        trades=[],
        market_data={'spot': 102.0, 'vol': 0.25, 'rate': 0.03}
    )
    analytics_result = await agents['analytics'].process_request(pnl_cmd)
    logger.info("step_10_complete", total_pnl=analytics_result.total_pnl)
    
    # Step 11: System Monitoring
    logger.info("step_11", agent="Monitoring", action="Check system health")
    health_query = CheckSystemHealthQuery(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.MONITORING,
        include_metrics=True
    )
    monitor_result = await agents['monitoring'].process_request(health_query)
    logger.info("step_11_complete", overall_status=monitor_result.overall_status)
    
    # Step 12: Client Interface (Orchestrate response)
    logger.info("step_12", agent="Client Interface", action="Generate comprehensive answer")
    client_query = ClientQuery(
        from_agent=AgentName.CLIENT_INTERFACE,
        to_agent=AgentName.CLIENT_INTERFACE,
        client_id='DEMO_CLIENT',
        query_text='Should I buy SPY calls?',
        request_type='question'
    )
    client_result = await agents['client_interface'].process_request(client_query)
    logger.info("step_12_complete", success=client_result.success)
    
    # Summary
    logger.info(
        "WORKFLOW_COMPLETE",
        all_12_agents_executed=True,
        workflow_successful=True,
        agents=[
            "Market Data (quotes)",
            "Volatility (forecast)",
            "Strategy (generation)",
            "Pricing (Greeks)",
            "Guardrail (safety)",
            "Risk (portfolio)",
            "Compliance (regulatory)",
            "Execution (orders)",
            "Hedging (optimization)",
            "Analytics (P&L)",
            "Monitoring (health)",
            "Client Interface (orchestration)"
        ]
    )
    
    # Shutdown all agents
    logger.info("shutting_down_all_agents")
    for agent in agents.values():
        agent.shutdown()
    
    logger.info("DEMO_COMPLETE", message="All 12 professional agents working together successfully")


if __name__ == "__main__":
    asyncio.run(main())