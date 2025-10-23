"""
Comprehensive Demo of Time Series Models for Algorithmic Trading

Demonstrates ARIMA, GARCH, and EWMA models with real market data from Yahoo Finance.
Includes integration with VaR and Portfolio Optimization models.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import time series models
from axiom.models.time_series import (
    ARIMAModel,
    GARCHModel,
    EWMAModel,
    TimeSeriesConfig,
    prepare_returns,
    check_stationarity,
    calculate_forecast_accuracy,
    detect_seasonality
)

# Import VaR and Portfolio models for integration
from axiom.models.risk.var_models import (
    VaRCalculator,
    VaRMethod
)
from axiom.models.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationMethod
)


def fetch_market_data(symbols, start_date, end_date):
    """Fetch historical data from Yahoo Finance."""
    print(f"\n{'='*80}")
    print("FETCHING MARKET DATA")
    print(f"{'='*80}")
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            data[symbol] = hist['Close'].values
            print(f"✓ {symbol}: {len(hist)} days of data")
        except Exception as e:
            print(f"✗ {symbol}: Error - {str(e)}")
    
    return data


def demo_arima_price_forecasting(prices, symbol):
    """Demonstrate ARIMA for price forecasting."""
    print(f"\n{'='*80}")
    print(f"ARIMA MODEL - PRICE FORECASTING FOR {symbol}")
    print(f"{'='*80}")
    
    # Check stationarity
    stationarity = check_stationarity(prices)
    print(f"\nStationarity Check:")
    print(f"  Is Stationary: {stationarity['is_stationary']}")
    print(f"  Recommendation: {stationarity['recommendation']}")
    
    # Fit ARIMA with auto parameter selection
    print(f"\n1. Fitting ARIMA model with auto parameter selection...")
    arima = ARIMAModel()
    arima.fit(prices)
    
    print(f"   Selected Order: {arima.order}")
    print(f"   AIC: {arima.diagnostics.aic:.2f}")
    print(f"   BIC: {arima.diagnostics.bic:.2f}")
    print(f"   RMSE: {arima.diagnostics.rmse:.4f}")
    
    # Generate forecasts
    print(f"\n2. Generating 5-day ahead forecast...")
    forecast_result = arima.forecast(horizon=5, confidence_level=0.95)
    
    print(f"\n   Price Forecasts:")
    for i, (pred, lower, upper) in enumerate(zip(
        forecast_result.forecast,
        forecast_result.confidence_intervals[0],
        forecast_result.confidence_intervals[1]
    ), 1):
        print(f"   Day {i}: ${pred:.2f} (95% CI: ${lower:.2f} - ${upper:.2f})")
    
    return arima, forecast_result


def demo_garch_volatility_forecasting(returns, symbol):
    """Demonstrate GARCH for volatility forecasting."""
    print(f"\n{'='*80}")
    print(f"GARCH MODEL - VOLATILITY FORECASTING FOR {symbol}")
    print(f"{'='*80}")
    
    # Fit GARCH(1,1) model
    print(f"\n1. Fitting GARCH(1,1) model...")
    garch = GARCHModel(order=(1, 1))
    garch.fit(returns, use_returns=True)
    
    print(f"   Model Parameters:")
    print(f"   ω (omega): {garch.omega:.6f}")
    print(f"   α (alpha): {garch.alpha[0]:.6f}")
    print(f"   β (beta): {garch.beta[0]:.6f}")
    print(f"   Persistence: {garch.model_params['persistence']:.6f}")
    
    # Detect volatility clustering
    print(f"\n2. Detecting volatility clustering...")
    clustering = garch.detect_volatility_clustering()
    print(f"   Clustering Detected: {clustering['volatility_clustering_detected']}")
    print(f"   Ljung-Box Statistic: {clustering['ljung_box_statistic']:.2f}")
    print(f"   Volatility Half-life: {clustering['half_life']:.1f} periods")
    
    # Forecast volatility
    print(f"\n3. Forecasting volatility (10 days ahead)...")
    vol_forecast = garch.forecast(horizon=10, method='analytic')
    annualized_vol = vol_forecast.get_annualized_volatility()
    
    print(f"\n   Volatility Forecasts (Annualized):")
    for i, vol in enumerate(annualized_vol[:5], 1):
        print(f"   Day {i}: {vol*100:.2f}%")
    
    return garch, vol_forecast


def demo_ewma_trend_analysis(prices, returns, symbol):
    """Demonstrate EWMA for trend analysis."""
    print(f"\n{'='*80}")
    print(f"EWMA MODEL - TREND ANALYSIS FOR {symbol}")
    print(f"{'='*80}")
    
    # Fit EWMA for prices (trend)
    print(f"\n1. Fitting EWMA for trend detection (RiskMetrics λ=0.94)...")
    ewma_trend = EWMAModel(decay_factor=0.94)
    ewma_trend.fit(prices, use_returns=False, calculate_volatility=False)
    
    # Detect trend
    trend_info = ewma_trend.detect_trend(lookback=30)
    print(f"   Current Trend: {trend_info['trend'].upper()}")
    print(f"   Trend Strength: {trend_info['strength']:.4f}")
    print(f"   Current EWMA: ${trend_info['current_ewma']:.2f}")
    
    # Fit EWMA for volatility
    print(f"\n2. Calculating EWMA volatility...")
    ewma_vol = EWMAModel(decay_factor=0.94)
    ewma_vol.fit(returns, use_returns=True, calculate_volatility=True)
    
    current_vol = ewma_vol.ewma_volatility[-1]
    annualized_vol = current_vol * np.sqrt(252)
    print(f"   Current Volatility (Daily): {current_vol*100:.2f}%")
    print(f"   Current Volatility (Annualized): {annualized_vol*100:.2f}%")
    
    # Generate trading signals
    print(f"\n3. Generating trading signals (EWMA crossover)...")
    signals = ewma_trend.get_trading_signals(prices, fast_span=12, slow_span=26)
    print(f"   Current Position: {signals['current_position'].upper()}")
    print(f"   Recent Crossovers:")
    if len(signals['buy_crossovers']) > 0:
        print(f"   - Last Buy Signal: Day {signals['buy_crossovers'][-1]}")
    if len(signals['sell_crossovers']) > 0:
        print(f"   - Last Sell Signal: Day {signals['sell_crossovers'][-1]}")
    
    return ewma_trend, ewma_vol


def demo_integrated_risk_management(returns_dict, symbols):
    """Demonstrate integration with VaR and Portfolio models."""
    print(f"\n{'='*80}")
    print("INTEGRATED RISK MANAGEMENT")
    print(f"{'='*80}")
    
    # Portfolio setup
    portfolio_value = 1000000  # $1M portfolio
    equal_weights = {symbol: 1.0/len(symbols) for symbol in symbols}
    
    print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
    print(f"Equal Weight Allocation: {100/len(symbols):.1f}% per asset")
    
    # 1. GARCH-based VaR
    print(f"\n1. GARCH-Enhanced VaR Calculation:")
    for symbol in symbols:
        returns = returns_dict[symbol]
        
        # Fit GARCH
        garch = GARCHModel(order=(1, 1))
        garch.fit(returns, use_returns=True)
        
        # Forecast volatility
        vol_forecast = garch.forecast(horizon=1)
        next_day_vol = vol_forecast.volatility[0]
        
        # Calculate VaR using GARCH volatility
        position_value = portfolio_value * equal_weights[symbol]
        var_amount = position_value * 1.65 * next_day_vol  # 95% VaR
        
        print(f"   {symbol}: VaR = ${var_amount:,.0f} (Vol: {next_day_vol*100:.2f}%)")
    
    # 2. Portfolio VaR
    print(f"\n2. Portfolio-Level VaR (Multiple Methods):")
    
    # Aggregate portfolio returns
    portfolio_returns = np.zeros(len(returns_dict[symbols[0]]))
    for symbol in symbols:
        portfolio_returns += returns_dict[symbol] * equal_weights[symbol]
    
    var_calc = VaRCalculator(default_confidence=0.95)
    
    # Historical VaR
    hist_var = var_calc.calculate_var(
        portfolio_value,
        portfolio_returns,
        method=VaRMethod.HISTORICAL,
        time_horizon_days=1
    )
    
    # Monte Carlo VaR
    mc_var = var_calc.calculate_var(
        portfolio_value,
        portfolio_returns,
        method=VaRMethod.MONTE_CARLO,
        time_horizon_days=1,
        num_simulations=10000
    )
    
    print(f"   Historical VaR: ${hist_var.var_amount:,.0f}")
    print(f"   Monte Carlo VaR: ${mc_var.var_amount:,.0f}")
    print(f"   Expected Shortfall (CVaR): ${hist_var.expected_shortfall:,.0f}")
    
    # 3. Portfolio Optimization
    print(f"\n3. Portfolio Optimization with Volatility Forecasts:")
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(returns_dict)
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    result = optimizer.optimize(
        returns_df,
        method=OptimizationMethod.MAX_SHARPE
    )
    
    print(f"   Optimization Result:")
    print(f"   Expected Return: {result.metrics.expected_return*100:.2f}%")
    print(f"   Volatility: {result.metrics.volatility*100:.2f}%")
    print(f"   Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
    
    print(f"\n   Optimal Weights:")
    for asset, weight in result.get_weights_dict().items():
        if weight > 0.01:  # Show only significant weights
            print(f"   {asset}: {weight*100:.1f}%")
    
    return var_calc, optimizer, result


def demo_forecast_comparison(prices, returns, symbol):
    """Compare forecasting accuracy across models."""
    print(f"\n{'='*80}")
    print(f"FORECAST ACCURACY COMPARISON FOR {symbol}")
    print(f"{'='*80}")
    
    # Split data
    train_size = int(len(prices) * 0.8)
    train_prices = prices[:train_size]
    test_prices = prices[train_size:]
    train_returns = returns[:train_size]
    
    print(f"\nTrain Size: {train_size} days")
    print(f"Test Size: {len(test_prices)} days")
    
    # ARIMA forecast
    print(f"\n1. ARIMA Price Forecast:")
    arima = ARIMAModel()
    arima.fit(train_prices)
    arima_forecast = arima.forecast(horizon=len(test_prices))
    arima_accuracy = calculate_forecast_accuracy(test_prices, arima_forecast.forecast)
    
    print(f"   RMSE: {arima_accuracy['rmse']:.4f}")
    print(f"   MAE: {arima_accuracy['mae']:.4f}")
    print(f"   MAPE: {arima_accuracy['mape']:.2f}%")
    
    # EWMA trend forecast
    print(f"\n2. EWMA Trend Forecast:")
    ewma = EWMAModel(span=20)
    ewma.fit(train_prices, use_returns=False)
    ewma_forecast = ewma.forecast(horizon=len(test_prices))
    ewma_accuracy = calculate_forecast_accuracy(test_prices, ewma_forecast.forecast)
    
    print(f"   RMSE: {ewma_accuracy['rmse']:.4f}")
    print(f"   MAE: {ewma_accuracy['mae']:.4f}")
    print(f"   MAPE: {ewma_accuracy['mape']:.2f}%")
    
    # Volatility forecast comparison
    print(f"\n3. Volatility Forecast (GARCH vs EWMA):")
    test_returns = returns[train_size:]
    actual_vol = np.abs(test_returns)
    
    # GARCH
    garch = GARCHModel(order=(1, 1))
    garch.fit(train_returns, use_returns=True)
    garch_vol_forecast = garch.forecast(horizon=len(test_returns))
    garch_vol_accuracy = calculate_forecast_accuracy(actual_vol, garch_vol_forecast.volatility)
    
    # EWMA
    ewma_vol = EWMAModel(decay_factor=0.94)
    ewma_vol.fit(train_returns, use_returns=True, calculate_volatility=True)
    ewma_vol_forecast = ewma_vol.get_volatility_forecast(horizon=len(test_returns), annualize=False)
    ewma_vol_accuracy = calculate_forecast_accuracy(actual_vol, ewma_vol_forecast)
    
    print(f"   GARCH RMSE: {garch_vol_accuracy['rmse']:.6f}")
    print(f"   EWMA RMSE: {ewma_vol_accuracy['rmse']:.6f}")
    
    winner = "GARCH" if garch_vol_accuracy['rmse'] < ewma_vol_accuracy['rmse'] else "EWMA"
    print(f"   Winner: {winner}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("TIME SERIES MODELS FOR ALGORITHMIC TRADING - COMPREHENSIVE DEMO")
    print("="*80)
    print("\nThis demo demonstrates:")
    print("  1. ARIMA for price forecasting")
    print("  2. GARCH for volatility forecasting")
    print("  3. EWMA for trend analysis and signals")
    print("  4. Integration with VaR and Portfolio Optimization")
    print("  5. Forecast accuracy comparison")
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Fetch data
    price_data = fetch_market_data(symbols, start_date, end_date)
    
    if len(price_data) == 0:
        print("\n✗ Error: Could not fetch market data. Please check your internet connection.")
        return
    
    # Calculate returns
    returns_data = {}
    for symbol, prices in price_data.items():
        returns_data[symbol] = prepare_returns(prices, log_returns=True)
    
    # Demo 1: ARIMA Price Forecasting
    arima_model, arima_forecast = demo_arima_price_forecasting(
        price_data[symbols[0]], symbols[0]
    )
    
    # Demo 2: GARCH Volatility Forecasting
    garch_model, garch_forecast = demo_garch_volatility_forecasting(
        returns_data[symbols[0]], symbols[0]
    )
    
    # Demo 3: EWMA Trend Analysis
    ewma_trend, ewma_vol = demo_ewma_trend_analysis(
        price_data[symbols[0]], returns_data[symbols[0]], symbols[0]
    )
    
    # Demo 4: Integrated Risk Management
    var_calc, optimizer, opt_result = demo_integrated_risk_management(
        returns_data, symbols
    )
    
    # Demo 5: Forecast Comparison
    demo_forecast_comparison(
        price_data[symbols[0]], returns_data[symbols[0]], symbols[0]
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\n✓ All time series models successfully demonstrated")
    print("✓ Models integrated with VaR and Portfolio Optimization")
    print("✓ Real market data from Yahoo Finance")
    print("\nKey Takeaways:")
    print("  • ARIMA: Best for price trend forecasting")
    print("  • GARCH: Superior for volatility modeling and clustering")
    print("  • EWMA: Fast, efficient for real-time trading signals")
    print("  • Integration: Combined models provide comprehensive risk management")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()