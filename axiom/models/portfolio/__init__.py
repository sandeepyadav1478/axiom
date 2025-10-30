"""
Portfolio Models Module

This module contains portfolio optimization and management models.
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid unnecessary dependencies
if TYPE_CHECKING:
    from .rl_portfolio_manager import (
        RLPortfolioManager,
        PortfolioConfig,
        CNNFeatureExtractor,
        PortfolioActorCritic,
        PortfolioEnvironment,
    )

__all__ = [
    'RLPortfolioManager',
    'PortfolioConfig',
    'CNNFeatureExtractor',
    'PortfolioActorCritic',
    'PortfolioEnvironment',
]


def get_rl_portfolio_manager():
    """Lazy import of RLPortfolioManager"""
    from .rl_portfolio_manager import RLPortfolioManager
    return RLPortfolioManager


def get_portfolio_config():
    """Lazy import of PortfolioConfig"""
    from .rl_portfolio_manager import PortfolioConfig
    return PortfolioConfig