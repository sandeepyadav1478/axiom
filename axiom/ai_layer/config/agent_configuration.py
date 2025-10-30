"""
Dynamic Agent Configuration System

Allows flexible agent configuration based on requirements:
- Customizable roles and responsibilities
- Configurable capabilities and duties
- Dynamic behavior modification
- Plugin-based extensions
- Environment-specific configurations

This makes the multi-agent system adaptable for ANY use case.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Flexible agent roles"""
    PRICING = "pricing"
    RISK_MANAGEMENT = "risk_management"
    STRATEGY_GENERATION = "strategy_generation"
    ORDER_EXECUTION = "order_execution"
    PORTFOLIO_HEDGING = "portfolio_hedging"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    MARKET_DATA_PROVIDER = "market_data_provider"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    SYSTEM_MONITORING = "system_monitoring"
    SAFETY_VALIDATION = "safety_validation"
    CLIENT_INTERFACE = "client_interface"
    CUSTOM = "custom"


class AgentCapability(str, Enum):
    """Agent capabilities (can mix and match)"""
    CALCULATE_GREEKS = "calculate_greeks"
    CALCULATE_RISK = "calculate_risk"
    GENERATE_STRATEGY = "generate_strategy"
    EXECUTE_ORDERS = "execute_orders"
    HEDGE_PORTFOLIO = "hedge_portfolio"
    ANALYZE_PERFORMANCE = "analyze_performance"
    FETCH_MARKET_DATA = "fetch_market_data"
    FORECAST_VOLATILITY = "forecast_volatility"
    CHECK_COMPLIANCE = "check_compliance"
    MONITOR_HEALTH = "monitor_health"
    VALIDATE_SAFETY = "validate_safety"
    INTERACT_WITH_CLIENT = "interact_with_client"


class AgentBehavior(str, Enum):
    """Agent behavior modes"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class AgentConfiguration(BaseModel):
    """Flexible agent configuration"""
    agent_id: str
    agent_name: str
    agent_role: AgentRole
    capabilities: List[AgentCapability] = Field(default_factory=list)
    behavior_mode: AgentBehavior = AgentBehavior.BALANCED
    target_latency_ms: Decimal = Decimal('100')
    target_accuracy_pct: Decimal = Decimal('95')
    target_uptime_pct: Decimal('99.999')
    max_memory_mb: int = 1024
    max_cpu_pct: int = 50
    use_gpu: bool = False
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    enable_retry: bool = True
    max_retry_attempts: int = 3
    enable_guardrails: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_tracing: bool = True
    enable_metrics: bool = True
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    plugins: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


# Example
if __name__ == "__main__":
    from axiom.ai_layer.infrastructure.observability import Logger
    
    logger = Logger("test")
    logger.info("FLEXIBLE_CONFIGURATION_SYSTEM", message="Agents can be customized for any requirement")