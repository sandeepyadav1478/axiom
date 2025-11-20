"""
Enterprise Metrics and Monitoring for Pipelines
Prometheus-compatible metrics with structured logging
"""
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Track pipeline execution metrics"""
    
    # Counters
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    items_processed: int = 0
    items_failed: int = 0
    
    # Timings (seconds)
    total_execution_time: float = 0.0
    avg_cycle_time: float = 0.0
    avg_item_time: float = 0.0
    
    # Claude API
    claude_requests: int = 0
    claude_errors: int = 0
    claude_total_tokens: int = 0
    claude_total_cost: float = 0.0
    
    # Neo4j
    neo4j_writes: int = 0
    neo4j_errors: int = 0
    
    # Current state
    last_cycle_start: Optional[datetime] = None
    last_cycle_end: Optional[datetime] = None
    current_cycle_items: int = 0
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_errors: list = field(default_factory=list)
    
    def start_cycle(self):
        """Mark cycle start"""
        self.total_cycles += 1
        self.last_cycle_start = datetime.utcnow()
        self.current_cycle_items = 0
        
    def end_cycle(self, success: bool = True):
        """Mark cycle end"""
        self.last_cycle_end = datetime.utcnow()
        if success:
            self.successful_cycles += 1
        else:
            self.failed_cycles += 1
            
        # Calculate cycle time
        if self.last_cycle_start:
            cycle_time = (self.last_cycle_end - self.last_cycle_start).total_seconds()
            self.total_execution_time += cycle_time
            self.avg_cycle_time = self.total_execution_time / self.total_cycles
            
    def record_item_processed(self, success: bool = True, execution_time: float = 0.0):
        """Record item processing"""
        self.current_cycle_items += 1
        if success:
            self.items_processed += 1
        else:
            self.items_failed += 1
            
        if execution_time > 0 and self.items_processed > 0:
            self.avg_item_time = (
                (self.avg_item_time * (self.items_processed - 1) + execution_time) 
                / self.items_processed
            )
            
    def record_claude_request(self, tokens: int = 0, cost: float = 0.0, error: bool = False):
        """Record Claude API usage"""
        self.claude_requests += 1
        if error:
            self.claude_errors += 1
        else:
            self.claude_total_tokens += tokens
            self.claude_total_cost += cost
            
    def record_neo4j_write(self, count: int = 1, error: bool = False):
        """Record Neo4j operations"""
        if error:
            self.neo4j_errors += 1
        else:
            self.neo4j_writes += count
            
    def record_error(self, error_type: str, error_msg: str):
        """Track error occurrences"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_errors.append({
            'type': error_type,
            'message': error_msg,
            'timestamp': datetime.utcnow().isoformat()
        })
        # Keep only last 10 errors
        self.last_errors = self.last_errors[-10:]
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        success_rate = (
            (self.successful_cycles / self.total_cycles * 100)
            if self.total_cycles > 0 else 100.0
        )
        
        item_success_rate = (
            (self.items_processed / (self.items_processed + self.items_failed) * 100)
            if (self.items_processed + self.items_failed) > 0 else 100.0
        )
        
        status = "healthy"
        if success_rate < 80:
            status = "degraded"
        if success_rate < 50:
            status = "unhealthy"
            
        return {
            'status': status,
            'metrics': {
                'cycles': {
                    'total': self.total_cycles,
                    'successful': self.successful_cycles,
                    'failed': self.failed_cycles,
                    'success_rate': round(success_rate, 2)
                },
                'items': {
                    'processed': self.items_processed,
                    'failed': self.items_failed,
                    'success_rate': round(item_success_rate, 2),
                    'avg_processing_time': round(self.avg_item_time, 3)
                },
                'claude': {
                    'requests': self.claude_requests,
                    'errors': self.claude_errors,
                    'total_tokens': self.claude_total_tokens,
                    'total_cost': round(self.claude_total_cost, 4)
                },
                'neo4j': {
                    'writes': self.neo4j_writes,
                    'errors': self.neo4j_errors
                },
                'performance': {
                    'avg_cycle_time': round(self.avg_cycle_time, 2),
                    'total_execution_time': round(self.total_execution_time, 2)
                }
            },
            'errors': {
                'counts': self.error_counts,
                'recent': self.last_errors[-5:]  # Last 5 errors
            },
            'last_cycle': {
                'start': self.last_cycle_start.isoformat() if self.last_cycle_start else None,
                'end': self.last_cycle_end.isoformat() if self.last_cycle_end else None,
                'items_processed': self.current_cycle_items
            }
        }
        
    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        pipeline_name = "pipeline"  # Will be set by pipeline
        metrics = []
        
        # Counters
        metrics.append(f'pipeline_cycles_total{{pipeline="{pipeline_name}"}} {self.total_cycles}')
        metrics.append(f'pipeline_cycles_successful{{pipeline="{pipeline_name}"}} {self.successful_cycles}')
        metrics.append(f'pipeline_cycles_failed{{pipeline="{pipeline_name}"}} {self.failed_cycles}')
        metrics.append(f'pipeline_items_processed{{pipeline="{pipeline_name}"}} {self.items_processed}')
        metrics.append(f'pipeline_items_failed{{pipeline="{pipeline_name}"}} {self.items_failed}')
        
        # Claude metrics
        metrics.append(f'pipeline_claude_requests{{pipeline="{pipeline_name}"}} {self.claude_requests}')
        metrics.append(f'pipeline_claude_errors{{pipeline="{pipeline_name}"}} {self.claude_errors}')
        metrics.append(f'pipeline_claude_tokens{{pipeline="{pipeline_name}"}} {self.claude_total_tokens}')
        metrics.append(f'pipeline_claude_cost{{pipeline="{pipeline_name}"}} {self.claude_total_cost}')
        
        # Neo4j metrics
        metrics.append(f'pipeline_neo4j_writes{{pipeline="{pipeline_name}"}} {self.neo4j_writes}')
        metrics.append(f'pipeline_neo4j_errors{{pipeline="{pipeline_name}"}} {self.neo4j_errors}')
        
        # Gauges
        metrics.append(f'pipeline_avg_cycle_time_seconds{{pipeline="{pipeline_name}"}} {self.avg_cycle_time}')
        metrics.append(f'pipeline_avg_item_time_seconds{{pipeline="{pipeline_name}"}} {self.avg_item_time}')
        
        return '\n'.join(metrics)


class StructuredLogger:
    """Structured JSON logging for pipelines"""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.logger = logging.getLogger(pipeline_name)
        
    def _log(self, level: str, message: str, **kwargs):
        """Log structured message"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'pipeline': self.pipeline_name,
            'level': level,
            'message': message,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
        
    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)