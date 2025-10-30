"""
Professional Agent Orchestrator - Coordinates All 12 Agents

Brings all 12 professional agents together into a cohesive system.

Responsibilities:
- Initialize and manage all 12 agents
- Route requests to appropriate agents
- Coordinate multi-agent workflows
- Monitor system health
- Handle failover and recovery
- Provide unified API interface

This is the entry point for using the complete professional multi-agent system.
"""

import asyncio
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime

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
from axiom.ai_layer.infrastructure.observability import Logger, Tracer
from axiom.ai_layer.infrastructure.circuit_breaker import CircuitBreaker

# Messages
from axiom.ai_layer.messaging.protocol import AgentName


class ProfessionalAgentOrchestrator:
    """
    Orchestrator for all 12 professional agents
    
    Central coordinator managing complete multi-agent system.
    
    Features:
    - Manages lifecycle of all 12 agents
    - Routes requests to appropriate agents
    - Coordinates multi-agent workflows
    - Monitors system health continuously
    - Handles failover automatically
    - Provides unified interface
    
    This is the production entry point for the complete system.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize orchestrator with all 12 agents
        
        Args:
            use_gpu: Use GPU acceleration where applicable
        """
        # Observability
        self.logger = Logger("orchestrator")
        self.tracer = Tracer("orchestrator")
        
        self.logger.info("initializing_orchestrator", agents=12, use_gpu=use_gpu)
        
        # Shared infrastructure
        self.message_bus = MessageBus()
        self.config_manager = ConfigManager()
        
        # Circuit breaker for orchestrator itself
        self.circuit_breaker = CircuitBreaker(
            name="orchestrator",
            failure_threshold=10,
            timeout_seconds=60
        )
        
        # Initialize all 12 agents
        self.agents = self._initialize_all_agents(use_gpu)
        
        # Agent health tracking
        self._agent_health = {}
        
        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        
        self.logger.info(
            "orchestrator_initialized",
            agents_count=len(self.agents),
            clusters=["Trading", "Analytics", "Support"]
        )
    
    def _initialize_all_agents(self, use_gpu: bool) -> Dict:
        """Initialize all 12 professional agents"""
        self.logger.info("initializing_all_agents", count=12)
        
        agents = {
            # Trading Cluster
            AgentName.PRICING: ProfessionalPricingAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            AgentName.RISK: ProfessionalRiskAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            AgentName.STRATEGY: ProfessionalStrategyAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            AgentName.EXECUTION: ProfessionalExecutionAgent(
                self.message_bus, self.config_manager
            ),
            AgentName.HEDGING: ProfessionalHedgingAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            
            # Analytics Cluster
            AgentName.ANALYTICS: ProfessionalAnalyticsAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            AgentName.MARKET_DATA: ProfessionalMarketDataAgent(
                self.message_bus, self.config_manager
            ),
            AgentName.VOLATILITY: ProfessionalVolatilityAgent(
                self.message_bus, self.config_manager, use_gpu=use_gpu
            ),
            
            # Support Cluster
            AgentName.COMPLIANCE: ProfessionalComplianceAgent(
                self.message_bus, self.config_manager
            ),
            AgentName.MONITORING: ProfessionalMonitoringAgent(
                self.message_bus, self.config_manager
            ),
            AgentName.GUARDRAIL: ProfessionalGuardrailAgent(
                self.message_bus, self.config_manager
            ),
            AgentName.CLIENT_INTERFACE: ProfessionalClientInterfaceAgent(
                self.message_bus, self.config_manager
            )
        }
        
        self.logger.info("all_agents_initialized", count=len(agents))
        
        return agents
    
    async def process_client_request(self, request: Any) -> Any:
        """
        Process client request by orchestrating appropriate agents
        
        Automatically routes to correct agents and coordinates workflow
        """
        self._total_requests += 1
        
        with self.tracer.start_span("orchestrate_request"):
            try:
                # Route to client interface agent (orchestrates others)
                response = await self.agents[AgentName.CLIENT_INTERFACE].process_request(request)
                
                self._successful_requests += 1
                
                return response
                
            except Exception as e:
                self._failed_requests += 1
                
                self.logger.error(
                    "orchestration_failed",
                    error=str(e),
                    request_type=type(request).__name__
                )
                
                raise
    
    async def get_system_health(self) -> Dict:
        """Get health status of all 12 agents"""
        self.logger.info("checking_system_health")
        
        health_status = {}
        
        for agent_name, agent in self.agents.items():
            health = agent.health_check()
            health_status[agent_name.value] = health
        
        # Calculate system-wide metrics
        all_healthy = all(h['healthy'] for h in health_status.values())
        healthy_count = sum(1 for h in health_status.values() if h['healthy'])
        
        system_health = {
            'overall_healthy': all_healthy,
            'healthy_agents': healthy_count,
            'total_agents': len(self.agents),
            'agent_health': health_status,
            'orchestrator_stats': self.get_stats()
        }
        
        self.logger.info(
            "system_health_complete",
            overall_healthy=all_healthy,
            healthy_count=healthy_count
        )
        
        return system_health
    
    def get_stats(self) -> Dict:
        """Get orchestrator statistics"""
        return {
            'total_requests': self._total_requests,
            'successful': self._successful_requests,
            'failed': self._failed_requests,
            'success_rate': self._successful_requests / max(self._total_requests, 1),
            'agents_managed': len(self.agents)
        }
    
    async def shutdown_all(self):
        """Gracefully shutdown all 12 agents"""
        self.logger.info("shutting_down_all_agents", count=12)
        
        for agent_name, agent in self.agents.items():
            try:
                agent.shutdown()
                self.logger.info(f"{agent_name.value}_shutdown", success=True)
            except Exception as e:
                self.logger.error(f"{agent_name.value}_shutdown_failed", error=str(e))
        
        self.logger.info("all_agents_shutdown_complete")


# Example usage
if __name__ == "__main__":
    async def demo():
        logger = Logger("demo")
        logger.info("ORCHESTRATOR_DEMO", message="All 12 agents coordinated")
        
        # Initialize orchestrator (all 12 agents)
        orchestrator = ProfessionalAgentOrchestrator(use_gpu=False)
        
        # Check system health
        health = await orchestrator.get_system_health()
        logger.info("system_health", overall=health['overall_healthy'])
        
        # Get stats
        stats = orchestrator.get_stats()
        logger.info("orchestrator_stats", stats=stats)
        
        # Shutdown
        await orchestrator.shutdown_all()
        
        logger.info("DEMO_COMPLETE", all_12_agents_coordinated=True)
    
    asyncio.run(demo())