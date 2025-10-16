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
