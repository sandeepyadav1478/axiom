"""
Demo: External Libraries Integration

This demo showcases the production-grade external library integrations for
quantitative finance in Axiom. It demonstrates:

1. QuantLib for fixed income pricing
2. PyPortfolioOpt for portfolio optimization
3. TA-Lib for technical indicators
4. pandas-ta for technical analysis
5. arch for GARCH volatility modeling

Run with: python demos/demo_external_libraries.py
"""

import logging
from datetime import date, timedelta
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_library_availability():
    """Check and display which external libraries are available."""
    print("\n" + "="*70)
    print("EXTERNAL LIBRARY AVAILABILITY CHECK")
    print("="*70)
    
    from axiom.integrations.external_libs import get_library_availability
    
    availability = get_library_availability()
    
    for lib_name, is_available in availability.items():
        status = "✓ Available" if is_available else "✗ Not Available"
        print(f"{lib_name:20s}: {status}")
    
    return availability


def demo_quantlib_bond_pricing():
    """Demonstrate QuantLib bond pricing."""
    print("\n" + "="*70)
    print("QUANTLIB: BOND PRICING")
    print("="*70)
    
    try:
        from axiom.integrations.external_libs import (
            QuantLibBondPricer,
            BondSpecification,
            DayCountConvention,
            Frequency
        )
        
        # Create bond pricer
        pricer = QuantLibBondPricer()
        
        # Define a 10-year corporate bond
        bond = BondSpecification(
            face_value=1000,
            coupon_rate=0.05,  # 5% annual coupon
            issue_date=date(2020, 1, 15),
            maturity_date=date(2030, 1, 15),
            payment_frequency=Frequency.SEMIANNUAL,
            day_count=DayCountConvention.ACTUAL_ACTUAL
        )
        
        # Price the bond
        settlement_date = date(2024, 1, 15)
        yield_rate = 0.04  # 4% yield
        
        result = pricer.price_bond(bond, settlement_date, yield_rate)
        
        print(f"\nBond Details:")
        print(f"  Face Value: ${bond.face_value:,.2f}")
        print(f"  Coupon Rate: {bond.coupon_rate:.2%}")
        print(f"  Maturity: {bond.maturity_date}")
        
        print(f"\nPricing Results (YTM={yield_rate:.2%}):")
        print(f"  Clean Price: ${result.clean_price:,.2f}")
        print(f"  Dirty Price: ${result.dirty_price:,.2f}")
        print(f"  Accrued Interest: ${result.accrued_interest:,.2f}")
        print(f"  Duration: {result.duration:.4f} years")
        print(f"  Convexity: {result.convexity:.4f}")
        
        # Price at different yields
        print(f"\nPrice Sensitivity:")
        for ytm in [0.03, 0.04, 0.05, 0.06]:
            r = pricer.price_bond(bond, settlement_date, ytm)
            print(f"  YTM {ytm:.2%}: ${r.clean_price:,.2f}")
        
    except ImportError as e:
        print(f"\n⚠ QuantLib not available: {e}")
        print("Install with: pip install QuantLib-Python")
    except Exception as e:
        logger.error(f"Error in QuantLib demo: {e}")


def demo_pypfopt_optimization():
    """Demonstrate PyPortfolioOpt portfolio optimization."""
    print("\n" + "="*70)
    print("PYPORTFOLIOOPT: PORTFOLIO OPTIMIZATION")
    print("="*70)
    
    try:
        from axiom.integrations.external_libs import (
            PyPortfolioOptAdapter,
            OptimizationObjective
        )
        
        # Generate sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        n_assets = 5
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        # Simulate prices
        returns = np.random.randn(len(dates), n_assets) * 0.02 + 0.0005
        prices = pd.DataFrame(
            100 * np.exp(returns.cumsum(axis=0)),
            index=dates,
            columns=assets
        )
        
        # Create optimizer
        optimizer = PyPortfolioOptAdapter()
        
        # Optimize for maximum Sharpe ratio
        result = optimizer.optimize_portfolio(
            prices,
            objective=OptimizationObjective.MAX_SHARPE,
            risk_free_rate=0.02
        )
        
        print(f"\nOptimization Results (Max Sharpe):")
        print(f"  Expected Annual Return: {result.expected_return:.2%}")
        print(f"  Annual Volatility: {result.volatility:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        
        print(f"\nOptimal Weights:")
        for asset, weight in result.weights.items():
            if weight > 0.01:  # Only show significant weights
                print(f"  {asset}: {weight:.2%}")
        
        # Minimum volatility portfolio
        result_minvol = optimizer.optimize_portfolio(
            prices,
            objective=OptimizationObjective.MIN_VOLATILITY
        )
        
        print(f"\nMinimum Volatility Portfolio:")
        print(f"  Expected Annual Return: {result_minvol.expected_return:.2%}")
        print(f"  Annual Volatility: {result_minvol.volatility:.2%}")
        print(f"  Sharpe Ratio: {result_minvol.sharpe_ratio:.2f}")
        
        # Discrete allocation
        latest_prices = prices.iloc[-1]
        allocation_result = optimizer.discrete_allocation(
            result.weights,
            latest_prices,
            total_portfolio_value=100000
        )
        
        print(f"\nDiscrete Allocation ($100,000):")
        for asset, shares in allocation_result.discrete_allocation.items():
            value = shares * latest_prices[asset]
            print(f"  {asset}: {shares} shares (${value:,.2f})")
        print(f"  Leftover Cash: ${allocation_result.leftover_cash:,.2f}")
        
    except ImportError as e:
        print(f"\n⚠ PyPortfolioOpt not available: {e}")
        print("Install with: pip install PyPortfolioOpt")
    except Exception as e:
        logger.error(f"Error in PyPortfolioOpt demo: {e}")


def demo_talib_indicators():
    """Demonstrate TA-Lib technical indicators."""
    print("\n" + "="*70)
    print("TA-LIB: TECHNICAL INDICATORS")
    print("="*70)
    
    try:
        from axiom.integrations.external_libs import TALibIndicators
        
        # Generate sample OHLCV data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        base_price = 100
        returns = np.random.randn(len(dates)) * 0.02
        close = base_price * np.exp(returns.cumsum())
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(len(dates)) * 0.01),
            'high': close * (1 + abs(np.random.randn(len(dates))) * 0.02),
            'low': close * (1 - abs(np.random.randn(len(dates))) * 0.02),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Create indicators calculator
        indicators = TALibIndicators()
        
        # Calculate various indicators
        rsi = indicators.rsi(df['close'], timeperiod=14)
        macd, signal, hist = indicators.macd(df['close'])
        upper, middle, lower = indicators.bollinger_bands(df['close'])
        atr = indicators.atr(df['high'], df['low'], df['close'])
        obv = indicators.obv(df['close'], df['volume'])
        
        # Display recent values
        print(f"\nTechnical Indicators (last 5 days):")
        
        recent_df = pd.DataFrame({
            'Close': df['close'].iloc[-5:],
            'RSI': rsi[-5:],
            'MACD': macd[-5:],
            'BB_Upper': upper[-5:],
            'BB_Lower': lower[-5:],
            'ATR': atr[-5:]
        })
        
        print(recent_df.to_string())
        
        # Current signals
        current_rsi = rsi[-1]
        current_macd = macd[-1]
        current_signal = signal[-1]
        
        print(f"\nCurrent Signals:")
        print(f"  RSI: {current_rsi:.2f} ", end="")
        if current_rsi > 70:
            print("(Overbought)")
        elif current_rsi < 30:
            print("(Oversold)")
        else:
            print("(Neutral)")
        
        print(f"  MACD: {current_macd:.2f}, Signal: {current_signal:.2f} ", end="")
        if current_macd > current_signal:
            print("(Bullish)")
        else:
            print("(Bearish)")
        
    except ImportError as e:
        print(f"\n⚠ TA-Lib not available: {e}")
        print("Install with: brew install ta-lib && pip install TA-Lib")
    except Exception as e:
        logger.error(f"Error in TA-Lib demo: {e}")


def demo_pandas_ta():
    """Demonstrate pandas-ta technical analysis."""
    print("\n" + "="*70)
    print("PANDAS-TA: TECHNICAL ANALYSIS")
    print("="*70)
    
    try:
        from axiom.integrations.external_libs import PandasTAIntegration
        
        # Generate sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        
        base_price = 100
        returns = np.random.randn(len(dates)) * 0.02
        close = base_price * np.exp(returns.cumsum())
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(len(dates)) * 0.01),
            'high': close * (1 + abs(np.random.randn(len(dates))) * 0.02),
            'low': close * (1 - abs(np.random.randn(len(dates))) * 0.02),
            'close': close,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Create pandas-ta integration
        pta = PandasTAIntegration()
        
        # Add individual indicators
        df = pta.add_rsi(df, length=14)
        df = pta.add_macd(df)
        df = pta.add_bbands(df)
        df = pta.add_atr(df)
        
        # Show recent data with indicators
        print(f"\nDataFrame with Indicators (last 3 days):")
        columns = ['close', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBU_20_2.0', 'ATRr_14']
        available_cols = [col for col in columns if col in df.columns]
        print(df[available_cols].tail(3).to_string())
        
        # Calculate returns
        df = pta.calculate_returns(df, cumulative=False)
        
        print(f"\nReturns Statistics:")
        if 'PCTRET_1' in df.columns:
            returns = df['PCTRET_1'].dropna()
            print(f"  Mean Daily Return: {returns.mean():.4%}")
            print(f"  Volatility: {returns.std():.4%}")
            print(f"  Sharpe (annualized): {(returns.mean() / returns.std()) * np.sqrt(252):.2f}")
        
    except ImportError as e:
        print(f"\n⚠ pandas-ta not available: {e}")
        print("Install with: pip install pandas-ta")
    except Exception as e:
        logger.error(f"Error in pandas-ta demo: {e}")


def demo_arch_garch():
    """Demonstrate arch GARCH volatility modeling."""
    print("\n" + "="*70)
    print("ARCH: GARCH VOLATILITY MODELING")
    print("="*70)
    
    try:
        from axiom.integrations.external_libs import (
            ArchGARCH,
            VolatilityModel,
            Distribution
        )
        
        # Generate sample returns data
        np.random.seed(42)
        n = 1000
        returns = np.random.randn(n) * 0.01  # Daily returns
        
        # Add some volatility clustering
        for i in range(1, n):
            if abs(returns[i-1]) > 0.02:
                returns[i] *= 1.5
        
        returns = pd.Series(returns, index=pd.date_range('2020-01-01', periods=n, freq='D'))
        
        # Create GARCH model
        garch = ArchGARCH()
        
        # Fit GARCH(1,1) with Student-t distribution
        result = garch.fit_garch(
            returns,
            p=1, q=1,
            model_type=VolatilityModel.GARCH,
            dist=Distribution.STUDENTS_T
        )
        
        print(f"\nGARCH(1,1) Model Results:")
        print(f"  Log-Likelihood: {result.log_likelihood:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        
        print(f"\nModel Parameters:")
        for param, value in result.parameters.items():
            print(f"  {param}: {value:.6f}")
        
        # Forecast volatility
        forecast = garch.forecast_volatility(result, horizon=10)
        
        print(f"\nVolatility Forecast (next 10 days):")
        for i, vol in enumerate(forecast.variance[:10], 1):
            print(f"  Day {i}: {np.sqrt(vol):.4%}")
        
        # Compare models
        print(f"\nModel Comparison:")
        models_to_compare = [
            (VolatilityModel.GARCH, 1, 1),
            (VolatilityModel.GARCH, 1, 2),
            (VolatilityModel.GARCH, 2, 1),
        ]
        
        comparison = garch.compare_models(
            returns,
            models_to_compare,
            dist=Distribution.STUDENTS_T
        )
        
        print(comparison.to_string(index=False))
        
        # Calculate VaR
        var_95 = garch.calculate_var(result, confidence_level=0.95, portfolio_value=1000000)
        
        print(f"\nValue at Risk (95% confidence):")
        print(f"  Current VaR: ${var_95[-1]:,.2f}")
        print(f"  Average VaR (last 30 days): ${var_95[-30:].mean():,.2f}")
        
    except ImportError as e:
        print(f"\n⚠ arch not available: {e}")
        print("Install with: pip install arch")
    except Exception as e:
        logger.error(f"Error in arch demo: {e}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("AXIOM: EXTERNAL LIBRARIES INTEGRATION DEMO")
    print("="*70)
    
    # Check availability
    availability = demo_library_availability()
    
    # Run demos for available libraries
    if availability.get('QuantLib'):
        demo_quantlib_bond_pricing()
    
    if availability.get('PyPortfolioOpt'):
        demo_pypfopt_optimization()
    
    if availability.get('TA-Lib'):
        demo_talib_indicators()
    
    if availability.get('pandas-ta'):
        demo_pandas_ta()
    
    if availability.get('arch'):
        demo_arch_garch()
    
    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFor more information, see:")
    print("  - axiom/integrations/external_libs/README.md")
    print("  - Individual module docstrings")
    print("\nTo install missing libraries:")
    print("  pip install -r requirements.txt")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()