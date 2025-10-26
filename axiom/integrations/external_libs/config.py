"""
Configuration for External Library Integrations

This module manages configuration settings and availability checking for external libraries.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class LibraryConfig:
    """Configuration for external library usage."""
    
    # Library enable/disable flags
    use_quantlib: bool = True
    use_pypfopt: bool = True
    use_talib: bool = True
    use_pandas_ta: bool = True
    use_statsmodels: bool = True
    use_arch: bool = True
    use_openbb: bool = True
    
    # Preference settings
    prefer_external: bool = True  # Prefer external libs over custom implementations
    fallback_to_custom: bool = True  # Fallback to custom if library unavailable
    
    # Performance settings
    cache_results: bool = True
    parallel_processing: bool = True
    
    # Logging settings
    log_library_usage: bool = True
    log_fallbacks: bool = True
    
    # Library-specific settings
    quantlib_settings: Dict = field(default_factory=dict)
    pypfopt_settings: Dict = field(default_factory=dict)
    talib_settings: Dict = field(default_factory=dict)


class LibraryAvailability:
    """Check and cache availability of external libraries."""
    
    _availability_cache: Dict[str, bool] = {}
    
    @classmethod
    def check_library(cls, library_name: str) -> bool:
        """Check if a library is available for import.
        
        Args:
            library_name: Name of the library to check
            
        Returns:
            True if library is available, False otherwise
        """
        if library_name in cls._availability_cache:
            return cls._availability_cache[library_name]
        
        try:
            __import__(library_name)
            cls._availability_cache[library_name] = True
            logger.debug(f"Library '{library_name}' is available")
            return True
        except ImportError as e:
            cls._availability_cache[library_name] = False
            logger.warning(f"Library '{library_name}' is not available: {e}")
            return False
    
    @classmethod
    def get_all_availability(cls) -> Dict[str, bool]:
        """Get availability status of all supported libraries.
        
        Returns:
            Dictionary mapping library names to availability status
        """
        libraries = {
            'QuantLib': 'QuantLib',
            'PyPortfolioOpt': 'pypfopt',
            'TA-Lib': 'talib',
            'pandas-ta': 'pandas_ta',
            'statsmodels': 'statsmodels',
            'arch': 'arch',
            'openbb': 'openbb',
        }
        
        return {
            name: cls.check_library(module)
            for name, module in libraries.items()
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear the availability cache."""
        cls._availability_cache.clear()


def get_library_availability() -> Dict[str, bool]:
    """Get availability status of all external libraries.
    
    Returns:
        Dictionary with library names and their availability status
    """
    return LibraryAvailability.get_all_availability()


def log_library_status():
    """Log the availability status of all libraries."""
    availability = get_library_availability()
    
    available = [name for name, status in availability.items() if status]
    unavailable = [name for name, status in availability.items() if not status]
    
    if available:
        logger.info(f"Available libraries: {', '.join(available)}")
    
    if unavailable:
        logger.warning(f"Unavailable libraries: {', '.join(unavailable)}")
        logger.warning("Install missing libraries with: pip install -r requirements.txt")


# Global configuration instance
_global_config: Optional[LibraryConfig] = None


def get_config() -> LibraryConfig:
    """Get the global library configuration.
    
    Returns:
        The global LibraryConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = LibraryConfig()
    return _global_config


def set_config(config: LibraryConfig):
    """Set the global library configuration.
    
    Args:
        config: New configuration to use
    """
    global _global_config
    _global_config = config


def reset_config():
    """Reset the global configuration to defaults."""
    global _global_config
    _global_config = LibraryConfig()