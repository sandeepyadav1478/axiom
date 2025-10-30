"""
Options Pricing Models Module

This module contains advanced options pricing models including
deep learning approaches for exotic options and volatility surfaces.
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid unnecessary dependencies
if TYPE_CHECKING:
    from .vae_option_pricer import (
        VAEMLPOptionPricer,
        VAEConfig,
        OptionType,
        VolatilitySurfaceVAE,
        MLPOptionPricer,
    )

__all__ = [
    'VAEMLPOptionPricer',
    'VAEConfig',
    'OptionType',
    'VolatilitySurfaceVAE',
    'MLPOptionPricer',
]


def get_vae_option_pricer():
    """Lazy import of VAEMLPOptionPricer"""
    from .vae_option_pricer import VAEMLPOptionPricer
    return VAEMLPOptionPricer


def get_vae_config():
    """Lazy import of VAEConfig"""
    from .vae_option_pricer import VAEConfig
    return VAEConfig


def get_option_type():
    """Lazy import of OptionType"""
    from .vae_option_pricer import OptionType
    return OptionType