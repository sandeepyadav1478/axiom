"""
Axiom Custom Logger System
Custom logging wrapper with debug mode control
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AxiomLogger:
    """Custom logger for Axiom platform with debug mode control."""
    
    def __init__(self, name: str = "axiom", debug_enabled: bool = False):
        self.name = name
        self.debug_enabled = debug_enabled
        self._logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        level = logging.DEBUG if self.debug_enabled else logging.INFO
        self._logger.setLevel(level)
        self._logger.handlers.clear()
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def debug(self, message: str, **context):
        if self.debug_enabled:
            self._log_with_context(logging.DEBUG, message, **context)
    
    def info(self, message: str, **context):
        self._log_with_context(logging.INFO, message, **context)
    
    def warning(self, message: str, **context):
        self._log_with_context(logging.WARNING, message, **context)
    
    def error(self, message: str, **context):
        self._log_with_context(logging.ERROR, message, **context)
    
    def _log_with_context(self, level: int, message: str, **context):
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            full_message = f"{message} | {context_str}"
        else:
            full_message = message
        self._logger.log(level, full_message)
    
    def provider_status(self, provider_name: str, status: str, **details):
        self.info(f"Provider {provider_name}: {status}", **details)


def get_logger(name: str = "axiom") -> AxiomLogger:
    return AxiomLogger(name)


# Pre-configured logger instances for institutional-grade logging
axiom_logger = get_logger("axiom")
provider_logger = get_logger("axiom.providers")
workflow_logger = get_logger("axiom.workflows")
validation_logger = get_logger("axiom.validation")
var_logger = get_logger("axiom.models.var")
portfolio_logger = get_logger("axiom.models.portfolio") 
timeseries_logger = get_logger("axiom.models.timeseries")
database_logger = get_logger("axiom.database")
vector_logger = get_logger("axiom.database.vector")
ma_valuation_logger = get_logger("axiom.ma.valuation")
ma_dd_logger = get_logger("axiom.ma.due_diligence")
ma_risk_logger = get_logger("axiom.ma.risk_assessment")
financial_data_logger = get_logger("axiom.data.financial")
aggregator_logger = get_logger("axiom.data.aggregator")
ai_logger = get_logger("axiom.ai")
integration_logger = get_logger("axiom.integrations")
