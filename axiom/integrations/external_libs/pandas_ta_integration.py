"""
Pandas-TA Integration for Technical Analysis

This module provides integration with pandas-ta, a pure Python technical analysis library
that works seamlessly with pandas DataFrames. It offers 130+ indicators and is easier to
install than TA-Lib (no C dependencies).

Features:
- 130+ technical indicators
- Pure Python (easy installation)
- Pandas DataFrame native
- Custom indicator creation
- Strategy development
- Trend, momentum, volatility, volume, and more

Categories:
- Candles: Candlestick patterns
- Cycles: Cycle indicators
- Momentum: RSI, MACD, Stochastic, etc.
- Overlap: Moving averages, Bollinger Bands
- Performance: Returns, log returns
- Statistics: Correlation, variance, etc.
- Trend: ADX, Aroon, PSAR
- Volatility: ATR, Bollinger Bands Width
- Volume: OBV, AD, CMF
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import get_config, LibraryAvailability

logger = logging.getLogger(__name__)

# Check if pandas-ta is available
PANDAS_TA_AVAILABLE = LibraryAvailability.check_library('pandas_ta')

if PANDAS_TA_AVAILABLE:
    import pandas_ta as ta


@dataclass
class StrategyResult:
    """Result from strategy application."""
    df: pd.DataFrame
    indicators_applied: List[str]
    metadata: Dict


class PandasTAIntegration:
    """Integration wrapper for pandas-ta technical analysis library.
    
    This class provides a convenient interface to pandas-ta's indicators
    with added error handling and logging capabilities.
    
    Example:
        >>> pta = PandasTAIntegration()
        >>> df = pd.DataFrame({
        ...     'open': [...],
        ...     'high': [...],
        ...     'low': [...],
        ...     'close': [...],
        ...     'volume': [...]
        ... })
        >>> # Add all common indicators
        >>> df = pta.add_all_ta_indicators(df)
        >>> # Or add specific indicators
        >>> df = pta.add_rsi(df)
        >>> df = pta.add_macd(df)
    """
    
    def __init__(self):
        """Initialize the pandas-ta integration."""
        if not PANDAS_TA_AVAILABLE:
            raise ImportError(
                "pandas-ta is not available. Install it with: pip install pandas-ta"
            )
        
        self.config = get_config()
        logger.info("pandas-ta integration initialized")
    
    # ============================================================================
    # Momentum Indicators
    # ============================================================================
    
    def add_rsi(
        self,
        df: pd.DataFrame,
        length: int = 14,
        close_col: str = 'close',
        append: bool = True
    ) -> pd.DataFrame:
        """Add Relative Strength Index.
        
        Args:
            df: DataFrame with price data
            length: RSI period
            close_col: Name of close price column
            append: If True, append to df; if False, return indicator only
            
        Returns:
            DataFrame with RSI added
        """
        result = df.ta.rsi(length=length, close=close_col, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added RSI({length}) indicator")
        return df if append else result
    
    def add_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        close_col: str = 'close',
        append: bool = True
    ) -> pd.DataFrame:
        """Add MACD indicator.
        
        Args:
            df: DataFrame with price data
            fast: Fast period
            slow: Slow period
            signal: Signal period
            close_col: Name of close price column
            append: If True, append to df
            
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        result = df.ta.macd(
            fast=fast,
            slow=slow,
            signal=signal,
            close=close_col,
            append=append
        )
        if self.config.log_library_usage:
            logger.debug(f"Added MACD({fast},{slow},{signal}) indicator")
        return df if append else result
    
    def add_stoch(
        self,
        df: pd.DataFrame,
        k: int = 14,
        d: int = 3,
        smooth_k: int = 3,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLC data
            k: %K period
            d: %D period
            smooth_k: Smoothing period for %K
            append: If True, append to df
            
        Returns:
            DataFrame with Stochastic
        """
        result = df.ta.stoch(k=k, d=d, smooth_k=smooth_k, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added Stochastic({k},{d},{smooth_k}) indicator")
        return df if append else result
    
    def add_cci(
        self,
        df: pd.DataFrame,
        length: int = 14,
        c: float = 0.015,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Commodity Channel Index.
        
        Args:
            df: DataFrame with HLC data
            length: CCI period
            c: Constant
            append: If True, append to df
            
        Returns:
            DataFrame with CCI
        """
        result = df.ta.cci(length=length, c=c, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added CCI({length}) indicator")
        return df if append else result
    
    # ============================================================================
    # Overlap Studies
    # ============================================================================
    
    def add_sma(
        self,
        df: pd.DataFrame,
        length: int = 20,
        close_col: str = 'close',
        append: bool = True
    ) -> pd.DataFrame:
        """Add Simple Moving Average.
        
        Args:
            df: DataFrame with price data
            length: SMA period
            close_col: Name of close price column
            append: If True, append to df
            
        Returns:
            DataFrame with SMA
        """
        result = df.ta.sma(length=length, close=close_col, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added SMA({length}) indicator")
        return df if append else result
    
    def add_ema(
        self,
        df: pd.DataFrame,
        length: int = 20,
        close_col: str = 'close',
        append: bool = True
    ) -> pd.DataFrame:
        """Add Exponential Moving Average.
        
        Args:
            df: DataFrame with price data
            length: EMA period
            close_col: Name of close price column
            append: If True, append to df
            
        Returns:
            DataFrame with EMA
        """
        result = df.ta.ema(length=length, close=close_col, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added EMA({length}) indicator")
        return df if append else result
    
    def add_bbands(
        self,
        df: pd.DataFrame,
        length: int = 20,
        std: float = 2,
        close_col: str = 'close',
        append: bool = True
    ) -> pd.DataFrame:
        """Add Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            length: Period
            std: Number of standard deviations
            close_col: Name of close price column
            append: If True, append to df
            
        Returns:
            DataFrame with Bollinger Bands (lower, mid, upper, bandwidth, percent)
        """
        result = df.ta.bbands(length=length, std=std, close=close_col, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added BBands({length},{std}) indicator")
        return df if append else result
    
    # ============================================================================
    # Volatility Indicators
    # ============================================================================
    
    def add_atr(
        self,
        df: pd.DataFrame,
        length: int = 14,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Average True Range.
        
        Args:
            df: DataFrame with HLC data
            length: ATR period
            append: If True, append to df
            
        Returns:
            DataFrame with ATR
        """
        result = df.ta.atr(length=length, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added ATR({length}) indicator")
        return df if append else result
    
    def add_kc(
        self,
        df: pd.DataFrame,
        length: int = 20,
        scalar: float = 2,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Keltner Channels.
        
        Args:
            df: DataFrame with OHLC data
            length: Period
            scalar: Multiplier for ATR
            append: If True, append to df
            
        Returns:
            DataFrame with Keltner Channels
        """
        result = df.ta.kc(length=length, scalar=scalar, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added Keltner Channels({length},{scalar})")
        return df if append else result
    
    # ============================================================================
    # Volume Indicators
    # ============================================================================
    
    def add_obv(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
        volume_col: str = 'volume',
        append: bool = True
    ) -> pd.DataFrame:
        """Add On Balance Volume.
        
        Args:
            df: DataFrame with close and volume data
            close_col: Name of close price column
            volume_col: Name of volume column
            append: If True, append to df
            
        Returns:
            DataFrame with OBV
        """
        result = df.ta.obv(close=close_col, volume=volume_col, append=append)
        if self.config.log_library_usage:
            logger.debug("Added OBV indicator")
        return df if append else result
    
    def add_ad(
        self,
        df: pd.DataFrame,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Accumulation/Distribution.
        
        Args:
            df: DataFrame with OHLCV data
            append: If True, append to df
            
        Returns:
            DataFrame with A/D
        """
        result = df.ta.ad(append=append)
        if self.config.log_library_usage:
            logger.debug("Added A/D indicator")
        return df if append else result
    
    def add_cmf(
        self,
        df: pd.DataFrame,
        length: int = 20,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Chaikin Money Flow.
        
        Args:
            df: DataFrame with OHLCV data
            length: CMF period
            append: If True, append to df
            
        Returns:
            DataFrame with CMF
        """
        result = df.ta.cmf(length=length, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added CMF({length}) indicator")
        return df if append else result
    
    # ============================================================================
    # Trend Indicators
    # ============================================================================
    
    def add_adx(
        self,
        df: pd.DataFrame,
        length: int = 14,
        append: bool = True
    ) -> pd.DataFrame:
        """Add Average Directional Index.
        
        Args:
            df: DataFrame with HLC data
            length: ADX period
            append: If True, append to df
            
        Returns:
            DataFrame with ADX, DMP, DMN
        """
        result = df.ta.adx(length=length, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added ADX({length}) indicator")
        return df if append else result
    
    def add_supertrend(
        self,
        df: pd.DataFrame,
        length: int = 7,
        multiplier: float = 3.0,
        append: bool = True
    ) -> pd.DataFrame:
        """Add SuperTrend indicator.
        
        Args:
            df: DataFrame with HLC data
            length: ATR period
            multiplier: ATR multiplier
            append: If True, append to df
            
        Returns:
            DataFrame with SuperTrend
        """
        result = df.ta.supertrend(length=length, multiplier=multiplier, append=append)
        if self.config.log_library_usage:
            logger.debug(f"Added SuperTrend({length},{multiplier})")
        return df if append else result
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def add_all_ta_indicators(
        self,
        df: pd.DataFrame,
        strategy: Optional[str] = None
    ) -> pd.DataFrame:
        """Add all common technical indicators to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy name ('all', 'common', 'momentum', 'volatility', etc.)
                     If None, uses 'common' strategy
            
        Returns:
            DataFrame with all indicators added
        """
        if strategy is None:
            strategy = 'common'
        
        # Apply pandas-ta strategy
        df.ta.strategy(strategy)
        
        if self.config.log_library_usage:
            logger.info(f"Applied '{strategy}' strategy with pandas-ta")
        
        return df
    
    def create_custom_strategy(
        self,
        df: pd.DataFrame,
        indicators: List[Dict]
    ) -> pd.DataFrame:
        """Create and apply a custom strategy.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicator dictionaries with 'name' and parameters
                       Example: [
                           {'name': 'rsi', 'length': 14},
                           {'name': 'macd', 'fast': 12, 'slow': 26, 'signal': 9},
                           {'name': 'bbands', 'length': 20, 'std': 2}
                       ]
            
        Returns:
            DataFrame with custom indicators added
        """
        for ind in indicators:
            ind_name = ind.pop('name')
            try:
                # Get the indicator function
                ind_func = getattr(df.ta, ind_name)
                # Apply with parameters
                ind_func(append=True, **ind)
            except AttributeError:
                logger.warning(f"Indicator '{ind_name}' not found in pandas-ta")
            except Exception as e:
                logger.error(f"Error applying indicator '{ind_name}': {e}")
        
        if self.config.log_library_usage:
            logger.info(f"Applied custom strategy with {len(indicators)} indicators")
        
        return df
    
    def get_available_indicators(self) -> List[str]:
        """Get list of all available pandas-ta indicators.
        
        Returns:
            List of indicator names
        """
        # Get all indicator categories
        categories = [
            'candles',
            'cycles',
            'momentum',
            'overlap',
            'performance',
            'statistics',
            'trend',
            'volatility',
            'volume'
        ]
        
        indicators = []
        for category in categories:
            try:
                cat_indicators = getattr(ta, category)
                if hasattr(cat_indicators, '__all__'):
                    indicators.extend(cat_indicators.__all__)
            except AttributeError:
                pass
        
        return sorted(set(indicators))
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        close_col: str = 'close',
        cumulative: bool = False,
        log_returns: bool = False,
        append: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        """Calculate returns.
        
        Args:
            df: DataFrame with price data
            close_col: Name of close price column
            cumulative: If True, calculate cumulative returns
            log_returns: If True, calculate log returns
            append: If True, append to df
            
        Returns:
            DataFrame with returns or Series if not appending
        """
        if log_returns:
            result = df.ta.log_return(close=close_col, cumulative=cumulative, append=append)
        else:
            result = df.ta.percent_return(close=close_col, cumulative=cumulative, append=append)
        
        if self.config.log_library_usage:
            ret_type = "log" if log_returns else "percent"
            cum_str = "cumulative " if cumulative else ""
            logger.debug(f"Calculated {cum_str}{ret_type} returns")
        
        return df if append else result


def check_pandas_ta_availability() -> bool:
    """Check if pandas-ta is available.
    
    Returns:
        True if pandas-ta is available, False otherwise
    """
    return PANDAS_TA_AVAILABLE


def get_pandas_ta_version() -> Optional[str]:
    """Get pandas-ta version.
    
    Returns:
        Version string or None if not available
    """
    if not PANDAS_TA_AVAILABLE:
        return None
    return ta.version