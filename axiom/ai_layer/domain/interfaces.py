"""
Domain Interfaces - Proper Contracts

Abstract base classes defining contracts for all components.
This is how you ensure consistency and enable testing/mocking.

Principle: Program to interfaces, not implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal
from axiom.ai_layer.domain.value_objects import Greeks, OptionType
from axiom.ai_layer.domain.exceptions import ModelError, ValidationError


class IPricingModel(ABC):
    """
    Interface for all pricing models
    
    Contract that all pricing implementations must follow.
    Enables swapping models, testing, validation.
    """
    
    @abstractmethod
    def calculate_greeks(
        self,
        spot: Decimal,
        strike: Decimal,
        time_to_maturity: Decimal,
        risk_free_rate: Decimal,
        volatility: Decimal,
        option_type: OptionType
    ) -> Greeks:
        """
        Calculate option Greeks
        
        Returns: Validated Greeks object
        Raises: ValidationError if inputs invalid, ModelError if calculation fails
        
        Contract guarantees:
        - Always returns valid Greeks (or raises)
        - Greeks validated against Black-Scholes
        - Calculation time tracked
        - Errors properly categorized
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict:
        """
        Get model metadata
        
        Returns: Dict with version, accuracy, performance metrics
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if model is healthy and ready
        
        Returns: True if ready, False if needs attention
        """
        pass


class IAgent(ABC):
    """
    Interface for all agents
    
    Every agent must implement these methods.
    Ensures consistent agent behavior.
    """
    
    @abstractmethod
    async def process_request(self, request: Any) -> Any:
        """
        Process request (agent-specific)
        
        Must handle:
        - Input validation
        - Processing
        - Output validation
        - Error handling
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get agent statistics
        
        Must include:
        - Requests processed
        - Error rate
        - Average latency
        - Current state
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict:
        """
        Detailed health check
        
        Returns:
        - Overall health (healthy/degraded/down)
        - Component status
        - Resource utilization
        - Recent errors
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """
        Graceful shutdown
        
        Must:
        - Finish current requests
        - Release resources
        - Persist state if needed
        - Clean up connections
        """
        pass


class IMessageBroker(ABC):
    """
    Interface for message broker
    
    Abstracts message passing implementation.
    Could be Ray, Redis, RabbitMQ, etc.
    """
    
    @abstractmethod
    async def send_message(
        self,
        to_agent: str,
        message: Dict,
        priority: str = 'normal'
    ) -> str:
        """
        Send message to agent
        
        Returns: Message ID
        Raises: NetworkError if send fails
        """
        pass
    
    @abstractmethod
    async def receive_message(
        self,
        agent_name: str,
        timeout_ms: int = 5000
    ) -> Optional[Dict]:
        """
        Receive message for agent
        
        Returns: Message dict or None if timeout
        """
        pass
    
    @abstractmethod
    def ack_message(self, message_id: str):
        """
        Acknowledge message processed
        
        Ensures at-least-once delivery
        """
        pass


class IStateStore(ABC):
    """
    Interface for state persistence
    
    Abstracts state storage implementation.
    Could be Redis, PostgreSQL, etc.
    """
    
    @abstractmethod
    async def save_state(
        self,
        key: str,
        state: Dict,
        ttl_seconds: Optional[int] = None
    ):
        """
        Save state
        
        Must be atomic
        """
        pass
    
    @abstractmethod
    async def load_state(
        self,
        key: str
    ) -> Optional[Dict]:
        """
        Load state
        
        Returns: State dict or None if not found
        """
        pass
    
    @abstractmethod
    async def delete_state(self, key: str):
        """Delete state"""
        pass


class ICircuitBreaker(ABC):
    """
    Interface for circuit breaker pattern
    
    Prevents cascading failures
    """
    
    @abstractmethod
    def call(self, func: callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        States: Closed (normal), Open (blocking), Half-Open (testing)
        """
        pass
    
    @abstractmethod
    def get_state(self) -> str:
        """Get current circuit breaker state"""
        pass
    
    @abstractmethod
    def reset(self):
        """Manually reset circuit breaker"""
        pass


class IRetryPolicy(ABC):
    """
    Interface for retry policies
    
    Defines how to retry failed operations
    """
    
    @abstractmethod
    def should_retry(
        self,
        attempt: int,
        error: Exception
    ) -> bool:
        """
        Determine if should retry
        
        Args:
            attempt: Current attempt number (0-indexed)
            error: Exception that occurred
        
        Returns: True if should retry, False otherwise
        """
        pass
    
    @abstractmethod
    def get_delay_seconds(self, attempt: int) -> float:
        """
        Get delay before retry
        
        Typically exponential backoff
        """
        pass


class IHealthCheck(ABC):
    """
    Interface for health checks
    
    All components should be health-checkable
    """
    
    @abstractmethod
    def check_health(self) -> Dict:
        """
        Perform health check
        
        Returns: Dict with status, details, dependencies
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Quick health check
        
        Returns: Boolean
        """
        pass


# Example usage showing how interfaces enable proper architecture
if __name__ == "__main__":
    print("="*60)
    print("DOMAIN INTERFACES - PROPER CONTRACTS")
    print("="*60)
    
    print("\n✓ IPricingModel - Contract for all pricing models")
    print("✓ IAgent - Contract for all agents")
    print("✓ IMessageBroker - Abstraction for messaging")
    print("✓ IStateStore - Abstraction for persistence")
    print("✓ ICircuitBreaker - Circuit breaker pattern")
    print("✓ IRetryPolicy - Retry logic abstraction")
    print("✓ IHealthCheck - Health check contract")
    
    print("\n" + "="*60)
    print("BENEFITS OF PROPER INTERFACES:")
    print("="*60)
    print("✓ Easy to test (mock implementations)")
    print("✓ Easy to swap (different implementations)")
    print("✓ Clear contracts (what must be implemented)")
    print("✓ Type safety (mypy validates)")
    print("✓ Documentation (interfaces document requirements)")
    
    print("\nTHIS IS PROFESSIONAL ARCHITECTURE")