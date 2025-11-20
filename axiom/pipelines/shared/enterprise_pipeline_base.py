"""
Enterprise Base LangGraph Pipeline Framework
Production-grade pipeline with:
- Metrics and monitoring  
- Circuit breakers and resilience
- Health checks and observability
- Structured logging
- Rate limiting
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, TypedDict
from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from .metrics import PipelineMetrics, StructuredLogger
from .resilience import (
    CircuitBreaker, 
    RetryStrategy, 
    RateLimiter, 
    CircuitBreakerConfig,
    CircuitBreakerOpenError
)
from .health_server import HealthCheckServer

logger = logging.getLogger(__name__)


class EnterpriseBasePipeline(ABC):
    """
    Enterprise-grade base class for all LangGraph-powered pipelines.
    
    Features:
    - Claude AI integration with circuit breaker
    - LangGraph workflow management
    - Prometheus metrics
    - Health check HTTP server
    - Structured JSON logging
    - Retry logic with exponential backoff
    - Rate limiting
    - Error tracking and alerting
    """
    
    def __init__(
        self, 
        pipeline_name: str,
        health_port: int = 8080,
        claude_rate_limit: float = 10.0  # requests per second
    ):
        """Initialize enterprise pipeline."""
        self.pipeline_name = pipeline_name
        
        # Structured logging
        self.struct_logger = StructuredLogger(pipeline_name)
        
        # Metrics
        self.metrics = PipelineMetrics()
        
        # Circuit breakers
        self.claude_circuit = CircuitBreaker(
            name=f"{pipeline_name}-claude",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0
            )
        )
        
        self.neo4j_circuit = CircuitBreaker(
            name=f"{pipeline_name}-neo4j",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0
            )
        )
        
        # Retry strategy
        self.retry_strategy = RetryStrategy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Rate limiting for Claude API
        self.claude_rate_limiter = RateLimiter(
            rate=claude_rate_limit,
            capacity=int(claude_rate_limit * 2)
        )
        
        # Initialize Claude client
        self.claude = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_tokens=4096
        )
        
        # Pipeline configuration
        self.interval = int(os.getenv('PIPELINE_INTERVAL', '60'))
        
        # Health check server
        self.health_server = HealthCheckServer(
            port=health_port,
            metrics_collector=self.metrics
        )
        
        self.struct_logger.info(
            f"Pipeline initialized",
            health_port=health_port,
            interval=self.interval,
            claude_rate_limit=claude_rate_limit
        )
        
        # Build workflow
        self.workflow = self.build_workflow()
        self.app = self.workflow.compile()
    
    @abstractmethod
    def build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    async def process_item(self, item: Any) -> Dict[str, Any]:
        """
        Process a single item through the workflow.
        Must be implemented by subclasses.
        """
        pass
    
    async def call_claude_with_protection(
        self,
        prompt: str,
        **kwargs
    ) -> Any:
        """
        Call Claude API with circuit breaker and rate limiting
        """
        # Rate limiting
        await self.claude_rate_limiter.acquire()
        
        # Circuit breaker + retry
        async def claude_call():
            try:
                start_time = time.time()
                response = await self.claude.ainvoke(prompt, **kwargs)
                
                # Record success metrics
                tokens = len(response.content) // 4  # Rough estimate
                cost = tokens * 0.00003  # Rough cost estimate
                self.metrics.record_claude_request(
                    tokens=tokens,
                    cost=cost,
                    error=False
                )
                
                return response
                
            except Exception as e:
                self.metrics.record_claude_request(error=True)
                self.metrics.record_error(
                    error_type='claude_api_error',
                    error_msg=str(e)
                )
                raise
        
        try:
            # Execute with circuit breaker and retry
            result = await self.retry_strategy.execute(
                lambda: self.claude_circuit.execute(claude_call),
                retry_on_exceptions=(Exception,)
            )
            return result
            
        except CircuitBreakerOpenError as e:
            self.struct_logger.error(
                "Claude circuit breaker open",
                error=str(e)
            )
            raise
        except Exception as e:
            self.struct_logger.error(
                "Claude API call failed after retries",
                error=str(e)
            )
            raise
    
    async def run_continuous(self, items: List[Any]):
        """
        Run pipeline in continuous mode with enterprise features
        
        Args:
            items: List of items to process (symbols, events, etc.)
        """
        # Start health check server
        await self.health_server.start()
        
        self.struct_logger.info(
            "Starting continuous pipeline execution",
            items_count=len(items),
            interval_seconds=self.interval
        )
        
        cycle_count = 0
        
        try:
            while True:
                cycle_count += 1
                self.metrics.start_cycle()
                
                self.struct_logger.info(
                    "Cycle started",
                    cycle=cycle_count,
                    items_to_process=len(items)
                )
                
                cycle_start_time = time.time()
                results = []
                
                for item in items:
                    item_start_time = time.time()
                    
                    try:
                        result = await self.process_item(item)
                        item_time = time.time() - item_start_time
                        
                        self.metrics.record_item_processed(
                            success=result.get('success', False),
                            execution_time=item_time
                        )
                        results.append(result)
                        
                    except Exception as e:
                        item_time = time.time() - item_start_time
                        
                        self.metrics.record_item_processed(
                            success=False,
                            execution_time=item_time
                        )
                        
                        self.metrics.record_error(
                            error_type=type(e).__name__,
                            error_msg=str(e)
                        )
                        
                        self.struct_logger.error(
                            "Item processing failed",
                            item=str(item),
                            error=str(e),
                            error_type=type(e).__name__
                        )
                
                # Cycle complete
                cycle_time = time.time() - cycle_start_time
                successful = len([r for r in results if r.get('success', False)])
                
                self.metrics.end_cycle(success=successful > 0)
                
                self.struct_logger.info(
                    "Cycle completed",
                    cycle=cycle_count,
                    successful=successful,
                    total=len(items),
                    cycle_time_seconds=round(cycle_time, 2),
                    success_rate=round(successful / len(items) * 100, 1) if items else 0
                )
                
                # Log circuit breaker statuses
                self.struct_logger.debug(
                    "Circuit breaker status",
                    claude_circuit=self.claude_circuit.get_status(),
                    neo4j_circuit=self.neo4j_circuit.get_status()
                )
                
                # Wait for next cycle
                await asyncio.sleep(self.interval)
                
        except KeyboardInterrupt:
            self.struct_logger.info("Pipeline interrupted by user")
        except Exception as e:
            self.struct_logger.error(
                "Pipeline fatal error",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
        finally:
            # Cleanup
            await self.health_server.stop()
            self.struct_logger.info(
                "Pipeline shutdown complete",
                total_cycles=cycle_count,
                metrics=self.metrics.get_health_status()
            )