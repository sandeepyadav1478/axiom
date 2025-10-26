"""
TA-Lib Indicators Wrapper

This module provides a wrapper around TA-Lib for technical analysis indicators.
TA-Lib is a C library with 150+ indicators used by Bloomberg, Reuters, and major trading platforms.

Features:
- 150+ technical indicators
- Extremely fast (C implementation)
- Industry-standard calculations
- Pattern recognition
- Statistical functions

Indicator Categories:
- Overlap Studies: Moving averages, Bollinger Bands, etc.
- Momentum Indicators: RSI, MACD, Stochastic, etc.
- Volume Indicators: OBV, AD, MFI, etc.
- Volatility Indicators: ATR, NATR, etc.
- Price Transform: AVGPRICE, MEDPRICE, TYPPRICE, etc.
- Cycle Indicators: HT_DCPERIOD, HT_DCPHASE, etc.
- Pattern Recognition: Candlestick patterns
- Statistical Functions: Linear regression, correlation, etc.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import get_config, LibraryAvailability

logger = logging.getLogger(__name__)

# Check if TA-Lib is available
TALIB_AVAILABLE = LibraryAvailability.check_library('talib')

if TALIB_AVAILABLE:
    import talib


class MAType(Enum):
    """Moving average types."""
    SMA = 0  # Simple Moving Average
    EMA = 1  # Exponential Moving Average
    WMA = 2  # Weighted Moving Average
    DEMA = 3  # Double Exponential Moving Average
    TEMA = 4  # Triple Exponential Moving Average
    TRIMA = 5  # Triangular Moving Average
    KAMA = 6  # Kaufman Adaptive Moving Average
    MAMA = 7  # MESA Adaptive Moving Average
    T3 = 8  # Triple Exponential Moving Average (T3)


@dataclass
class IndicatorResult:
    """Result from indicator calculation."""
    name: str
    values: Union[np.ndarray, pd.Series, Dict[str, np.ndarray]]
    parameters: Dict[str, any]


class TALibIndicators:
    """Wrapper for TA-Lib technical indicators.
    
    This class provides easy access to TA-Lib's 150+ technical indicators
    with pandas DataFrame support and proper error handling.
    
    Example:
        >>> indicators = TALibIndicators()
        >>> df = pd.DataFrame({'close': [...]})
        >>> rsi = indicators.rsi(df['close'], timeperiod=14)
        >>> macd = indicators.macd(df['close'])
        >>> bb = indicators.bollinger_bands(df['close'])
    """
    
    def __init__(self):
        """Initialize the TA-Lib indicators wrapper."""
        if not TALIB_AVAILABLE:
            raise ImportError(
                "TA-Lib is not available. Install it with: pip install TA-Lib\n"
                "Note: TA-Lib requires system dependencies. "
                "On macOS: brew install ta-lib"
            )
        
        self.config = get_config()
        logger.info("TA-Lib indicators initialized")
    
    # ============================================================================
    # Overlap Studies (Moving Averages, Bands, etc.)
    # ============================================================================
    
    def sma(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            SMA values
        """
        return talib.SMA(np.array(data), timeperiod=timeperiod)
    
    def ema(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Exponential Moving Average.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            EMA values
        """
        return talib.EMA(np.array(data), timeperiod=timeperiod)
    
    def wma(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Weighted Moving Average.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            WMA values
        """
        return talib.WMA(np.array(data), timeperiod=timeperiod)
    
    def dema(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Double Exponential Moving Average.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            DEMA values
        """
        return talib.DEMA(np.array(data), timeperiod=timeperiod)
    
    def tema(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Triple Exponential Moving Average.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            TEMA values
        """
        return talib.TEMA(np.array(data), timeperiod=timeperiod)
    
    def bollinger_bands(
        self,
        data: Union[pd.Series, np.ndarray],
        timeperiod: int = 20,
        nbdevup: float = 2,
        nbdevdn: float = 2,
        matype: MAType = MAType.SMA
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            nbdevup: Number of standard deviations for upper band
            nbdevdn: Number of standard deviations for lower band
            matype: Moving average type
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        upper, middle, lower = talib.BBANDS(
            np.array(data),
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype.value
        )
        return upper, middle, lower
    
    # ============================================================================
    # Momentum Indicators
    # ============================================================================
    
    def rsi(self, data: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> np.ndarray:
        """Relative Strength Index.
        
        Args:
            data: Price data
            timeperiod: Number of periods
            
        Returns:
            RSI values (0-100)
        """
        return talib.RSI(np.array(data), timeperiod=timeperiod)
    
    def macd(
        self,
        data: Union[pd.Series, np.ndarray],
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moving Average Convergence/Divergence.
        
        Args:
            data: Price data
            fastperiod: Fast EMA period
            slowperiod: Slow EMA period
            signalperiod: Signal line period
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        macd, signal, hist = talib.MACD(
            np.array(data),
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return macd, signal, hist
    
    def stochastic(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        fastk_period: int = 5,
        slowk_period: int = 3,
        slowk_matype: MAType = MAType.SMA,
        slowd_period: int = 3,
        slowd_matype: MAType = MAType.SMA
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            fastk_period: Fast %K period
            slowk_period: Slow %K period
            slowk_matype: Slow %K MA type
            slowd_period: Slow %D period
            slowd_matype: Slow %D MA type
            
        Returns:
            Tuple of (slowk, slowd)
        """
        slowk, slowd = talib.STOCH(
            np.array(high),
            np.array(low),
            np.array(close),
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=slowk_matype.value,
            slowd_period=slowd_period,
            slowd_matype=slowd_matype.value
        )
        return slowk, slowd
    
    def cci(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14
    ) -> np.ndarray:
        """Commodity Channel Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Number of periods
            
        Returns:
            CCI values
        """
        return talib.CCI(
            np.array(high),
            np.array(low),
            np.array(close),
            timeperiod=timeperiod
        )
    
    def adx(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14
    ) -> np.ndarray:
        """Average Directional Movement Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Number of periods
            
        Returns:
            ADX values
        """
        return talib.ADX(
            np.array(high),
            np.array(low),
            np.array(close),
            timeperiod=timeperiod
        )
    
    # ============================================================================
    # Volume Indicators
    # ============================================================================
    
    def obv(
        self,
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """On Balance Volume.
        
        Args:
            close: Close prices
            volume: Volume data
            
        Returns:
            OBV values
        """
        return talib.OBV(np.array(close), np.array(volume))
    
    def ad(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Chaikin A/D Line.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            
        Returns:
            A/D Line values
        """
        return talib.AD(
            np.array(high),
            np.array(low),
            np.array(close),
            np.array(volume)
        )
    
    def mfi(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray],
        timeperiod: int = 14
    ) -> np.ndarray:
        """Money Flow Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
            timeperiod: Number of periods
            
        Returns:
            MFI values (0-100)
        """
        return talib.MFI(
            np.array(high),
            np.array(low),
            np.array(close),
            np.array(volume),
            timeperiod=timeperiod
        )
    
    # ============================================================================
    # Volatility Indicators
    # ============================================================================
    
    def atr(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14
    ) -> np.ndarray:
        """Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Number of periods
            
        Returns:
            ATR values
        """
        return talib.ATR(
            np.array(high),
            np.array(low),
            np.array(close),
            timeperiod=timeperiod
        )
    
    def natr(
        self,
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14
    ) -> np.ndarray:
        """Normalized Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Number of periods
            
        Returns:
            NATR values (percentage)
        """
        return talib.NATR(
            np.array(high),
            np.array(low),
            np.array(close),
            timeperiod=timeperiod
        )
    
    # ============================================================================
    # Pattern Recognition
    # ============================================================================
    
    def cdl_doji(
        self,
        open_: Union[pd.Series, np.ndarray],
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Doji candlestick pattern.
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Pattern values (100=bullish, -100=bearish, 0=none)
        """
        return talib.CDLDOJI(
            np.array(open_),
            np.array(high),
            np.array(low),
            np.array(close)
        )
    
    def cdl_hammer(
        self,
        open_: Union[pd.Series, np.ndarray],
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Hammer candlestick pattern.
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Pattern values (100=bullish, -100=bearish, 0=none)
        """
        return talib.CDLHAMMER(
            np.array(open_),
            np.array(high),
            np.array(low),
            np.array(close)
        )
    
    def cdl_engulfing(
        self,
        open_: Union[pd.Series, np.ndarray],
        high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """Engulfing candlestick pattern.
        
        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            
        Returns:
            Pattern values (100=bullish, -100=bearish, 0=none)
        """
        return talib.CDLENGULFING(
            np.array(open_),
            np.array(high),
            np.array(low),
            np.array(close)
        )
    
    # ============================================================================
    # Utility Methods
    # ============================================================================
    
    def get_all_indicators(self) -> List[str]:
        """Get list of all available TA-Lib indicators.
        
        Returns:
            List of indicator names
        """
        return talib.get_functions()
    
    def get_indicator_info(self, indicator_name: str) -> Dict:
        """Get information about a specific indicator.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Dictionary with indicator information
        """
        func = getattr(talib, indicator_name.upper())
        return talib.abstract.Function(indicator_name.upper()).info
    
    def calculate_multiple_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str],
        **kwargs
    ) -> pd.DataFrame:
        """Calculate multiple indicators and add them to DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicator names to calculate
            **kwargs: Parameters for indicators
            
        Returns:
            DataFrame with indicators added as columns
        """
        result_df = df.copy()
        
        for indicator in indicators:
            try:
                if indicator.lower() == 'rsi':
                    result_df['RSI'] = self.rsi(df['close'])
                elif indicator.lower() == 'macd':
                    macd, signal, hist = self.macd(df['close'])
                    result_df['MACD'] = macd
                    result_df['MACD_Signal'] = signal
                    result_df['MACD_Hist'] = hist
                elif indicator.lower() == 'bbands':
                    upper, middle, lower = self.bollinger_bands(df['close'])
                    result_df['BB_Upper'] = upper
                    result_df['BB_Middle'] = middle
                    result_df['BB_Lower'] = lower
                elif indicator.lower() == 'atr':
                    result_df['ATR'] = self.atr(df['high'], df['low'], df['close'])
                elif indicator.lower() == 'obv':
                    result_df['OBV'] = self.obv(df['close'], df['volume'])
                else:
                    logger.warning(f"Indicator {indicator} not implemented in batch calculation")
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
        
        return result_df


def check_talib_availability() -> bool:
    """Check if TA-Lib is available.
    
    Returns:
        True if TA-Lib is available, False otherwise
    """
    return TALIB_AVAILABLE


def get_available_indicators() -> List[str]:
    """Get list of all available TA-Lib indicators.
    
    Returns:
        List of indicator function names
    """
    if not TALIB_AVAILABLE:
        return []
    return talib.get_functions()