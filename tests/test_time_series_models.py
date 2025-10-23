"""
Comprehensive Tests for Time Series Models

Tests ARIMA, GARCH, and EWMA models with synthetic and real data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import time series models
from axiom.models.time_series import (
    ARIMAModel,
    ARIMAOrder,
    GARCHModel,
    GARCHOrder,
    EWMAModel,
    TimeSeriesData,
    ModelType,
    calculate_ewma,
    calculate_ewma_volatility,
    fit_garch,
    prepare_returns,
    check_stationarity,
    calculate_acf,
    detect_seasonality,
    split_train_test,
    calculate_forecast_accuracy
)


class TestTimeSeriesData:
    """Test TimeSeriesData container."""
    
    def test_creation_from_array(self):
        """Test creation from numpy array."""
        data = np.random.randn(100)
        ts_data = TimeSeriesData(values=data)
        
        assert len(ts_data.values) == 100
        assert ts_data.dates is not None
        assert len(ts_data.dates) == 100
    
    def test_creation_from_series(self):
        """Test creation from pandas Series."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        series = pd.Series(np.random.randn(100), index=dates)
        
        ts_data = TimeSeriesData(values=series)
        
        assert len(ts_data.values) == 100
        assert isinstance(ts_data.dates, pd.DatetimeIndex)
    
    def test_get_returns(self):
        """Test return calculation."""
        prices = np.array([100, 101, 102, 101, 103])
        ts_data = TimeSeriesData(values=prices)
        
        # Log returns
        log_returns = ts_data.get_returns(log_returns=True)
        assert len(log_returns) == 4
        
        # Simple returns
        simple_returns = ts_data.get_returns(log_returns=False)
        assert len(simple_returns) == 4
        assert np.allclose(simple_returns[0], 0.01)  # (101-100)/100


class TestARIMAModel:
    """Test ARIMA model."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        # Generate AR(1) process
        n = 200
        phi = 0.7
        self.data = np.zeros(n)
        self.data[0] = np.random.randn()
        for t in range(1, n):
            self.data[t] = phi * self.data[t-1] + np.random.randn()
    
    def test_model_creation(self):
        """Test ARIMA model creation."""
        model = ARIMAModel(order=(1, 0, 0))
        assert model.order.p == 1
        assert model.order.d == 0
        assert model.order.q == 0
        assert not model.is_fitted
    
    def test_model_fitting(self):
        """Test ARIMA model fitting."""
        model = ARIMAModel(order=(1, 0, 1))
        model.fit(self.data)
        
        assert model.is_fitted
        assert model.ar_params is not None
        assert model.fitted_values is not None
        assert model.residuals is not None
    
    def test_auto_arima(self):
        """Test auto-ARIMA parameter selection."""
        model = ARIMAModel()  # No order specified
        model.fit(self.data)
        
        assert model.is_fitted
        assert model.order is not None
        assert model.order.p >= 0
        assert model.order.d >= 0
        assert model.order.q >= 0
    
    def test_forecasting(self):
        """Test ARIMA forecasting."""
        model = ARIMAModel(order=(1, 0, 1))
        model.fit(self.data)
        
        forecast_result = model.forecast(horizon=5)
        
        assert len(forecast_result.forecast) == 5
        assert forecast_result.confidence_intervals is not None
        assert forecast_result.model_type == ModelType.ARIMA
        
        # Check confidence intervals
        lower, upper = forecast_result.confidence_intervals
        assert len(lower) == 5
        assert len(upper) == 5
        assert np.all(lower <= forecast_result.forecast)
        assert np.all(upper >= forecast_result.forecast)
    
    def test_diagnostics(self):
        """Test model diagnostics."""
        model = ARIMAModel(order=(2, 0, 2))
        model.fit(self.data)
        
        assert model.diagnostics is not None
        assert model.diagnostics.aic is not None
        assert model.diagnostics.bic is not None
        assert model.diagnostics.mse is not None
        assert model.diagnostics.rmse is not None


class TestGARCHModel:
    """Test GARCH model."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        # Generate GARCH(1,1) process
        n = 500
        omega = 0.01
        alpha = 0.1
        beta = 0.85
        
        self.returns = np.zeros(n)
        variance = np.zeros(n)
        variance[0] = omega / (1 - alpha - beta)
        
        for t in range(1, n):
            variance[t] = omega + alpha * self.returns[t-1]**2 + beta * variance[t-1]
            self.returns[t] = np.sqrt(variance[t]) * np.random.randn()
    
    def test_model_creation(self):
        """Test GARCH model creation."""
        model = GARCHModel(order=(1, 1))
        assert model.order.p == 1
        assert model.order.q == 1
        assert not model.is_fitted
    
    def test_model_fitting(self):
        """Test GARCH model fitting."""
        model = GARCHModel(order=(1, 1))
        model.fit(self.returns, use_returns=True)
        
        assert model.is_fitted
        assert model.omega > 0
        assert len(model.alpha) == 1
        assert len(model.beta) == 1
        assert model.conditional_volatility is not None
    
    def test_volatility_forecasting(self):
        """Test volatility forecasting."""
        model = GARCHModel(order=(1, 1))
        model.fit(self.returns, use_returns=True)
        
        vol_forecast = model.forecast(horizon=10)
        
        assert len(vol_forecast.volatility) == 10
        assert len(vol_forecast.variance) == 10
        assert np.all(vol_forecast.volatility > 0)
        assert np.all(vol_forecast.variance > 0)
    
    def test_return_forecasting(self):
        """Test return forecasting with GARCH volatility."""
        model = GARCHModel(order=(1, 1))
        model.fit(self.returns, use_returns=True)
        
        forecast_result = model.forecast_returns(horizon=5)
        
        assert len(forecast_result.forecast) == 5
        assert forecast_result.confidence_intervals is not None
        assert forecast_result.model_type == ModelType.GARCH
    
    def test_persistence_calculation(self):
        """Test volatility persistence calculation."""
        model = GARCHModel(order=(1, 1))
        model.fit(self.returns, use_returns=True)
        
        persistence = model._calculate_persistence()
        assert 0 <= persistence < 1  # Should be stationary
    
    def test_volatility_clustering_detection(self):
        """Test volatility clustering detection."""
        model = GARCHModel(order=(1, 1))
        model.fit(self.returns, use_returns=True)
        
        clustering = model.detect_volatility_clustering()
        
        assert "volatility_clustering_detected" in clustering
        assert "ljung_box_statistic" in clustering
        assert "persistence" in clustering
        assert "half_life" in clustering
    
    def test_fit_garch_convenience(self):
        """Test convenience function."""
        model = fit_garch(self.returns, order=(1, 1))
        
        assert model.is_fitted
        assert model.conditional_volatility is not None


class TestEWMAModel:
    """Test EWMA model."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        # Generate time series with trend
        n = 200
        self.data = np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * 0.1
    
    def test_model_creation(self):
        """Test EWMA model creation."""
        model = EWMAModel(decay_factor=0.94)
        assert model.decay_factor == 0.94
        assert not model.is_fitted
    
    def test_model_creation_from_span(self):
        """Test EWMA creation from span."""
        model = EWMAModel(span=20)
        assert model.span == pytest.approx(20, rel=1e-2)
        assert 0 < model.decay_factor < 1
    
    def test_model_fitting(self):
        """Test EWMA model fitting."""
        model = EWMAModel(decay_factor=0.9)
        model.fit(self.data, use_returns=True)
        
        assert model.is_fitted
        assert model.ewma_mean is not None
        assert len(model.ewma_mean) == len(self.data)
    
    def test_volatility_calculation(self):
        """Test EWMA volatility calculation."""
        returns = np.random.randn(200) * 0.02
        model = EWMAModel(decay_factor=0.94)
        model.fit(returns, use_returns=True, calculate_volatility=True)
        
        assert model.ewma_volatility is not None
        assert len(model.ewma_volatility) == len(returns)
        assert np.all(model.ewma_volatility > 0)
    
    def test_forecasting(self):
        """Test EWMA forecasting."""
        model = EWMAModel(span=20)
        model.fit(self.data, use_returns=True, calculate_volatility=True)
        
        forecast_result = model.forecast(horizon=5)
        
        assert len(forecast_result.forecast) == 5
        assert forecast_result.confidence_intervals is not None
        assert forecast_result.model_type == ModelType.EWMA
    
    def test_volatility_forecast(self):
        """Test volatility forecasting."""
        returns = np.random.randn(200) * 0.02
        model = EWMAModel(decay_factor=0.94)
        model.fit(returns, use_returns=True, calculate_volatility=True)
        
        vol_forecast = model.get_volatility_forecast(horizon=10, annualize=False)
        
        assert len(vol_forecast) == 10
        assert np.all(vol_forecast > 0)
    
    def test_real_time_update(self):
        """Test real-time EWMA update."""
        model = EWMAModel(decay_factor=0.9)
        model.fit(self.data[:100], use_returns=True, calculate_volatility=True)
        
        new_value = 0.5
        new_mean, new_vol = model.update(new_value, update_volatility=True)
        
        assert isinstance(new_mean, (int, float))
        assert new_vol is not None
        assert new_vol > 0
    
    def test_trend_detection(self):
        """Test trend detection."""
        model = EWMAModel(span=20)
        model.fit(self.data, use_returns=False)
        
        trend_info = model.detect_trend(lookback=50)
        
        assert "trend" in trend_info
        assert trend_info["trend"] in ["uptrend", "downtrend", "sideways"]
        assert "slope" in trend_info
        assert "strength" in trend_info
    
    def test_trading_signals(self):
        """Test trading signal generation."""
        prices = np.cumsum(np.random.randn(200))
        model = EWMAModel(span=20)
        
        signals = model.get_trading_signals(prices, fast_span=12, slow_span=26)
        
        assert "fast_ewma" in signals
        assert "slow_ewma" in signals
        assert "signal" in signals
        assert "current_position" in signals
        assert signals["current_position"] in ["long", "short", "neutral"]
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        data = np.random.randn(100)
        
        # Test calculate_ewma
        ewma_values = calculate_ewma(data, decay_factor=0.94)
        assert len(ewma_values) == len(data)
        
        # Test calculate_ewma_volatility
        vol_values = calculate_ewma_volatility(data, decay_factor=0.94, annualize=False)
        assert len(vol_values) == len(data)
        assert np.all(vol_values > 0)


class TestUtilities:
    """Test utility functions."""
    
    def test_prepare_returns(self):
        """Test return preparation."""
        prices = np.array([100, 102, 101, 103, 105])
        
        # Log returns
        log_ret = prepare_returns(prices, log_returns=True)
        assert len(log_ret) == 4
        
        # Simple returns
        simple_ret = prepare_returns(prices, log_returns=False)
        assert len(simple_ret) == 4
    
    def test_check_stationarity(self):
        """Test stationarity check."""
        # Stationary series (white noise)
        stationary = np.random.randn(200)
        result = check_stationarity(stationary)
        assert "is_stationary" in result
        
        # Non-stationary series (random walk)
        non_stationary = np.cumsum(np.random.randn(200))
        result = check_stationarity(non_stationary)
        assert "is_stationary" in result
    
    def test_acf_calculation(self):
        """Test ACF calculation."""
        data = np.random.randn(100)
        acf = calculate_acf(data, nlags=20)
        
        assert len(acf) == 21  # 0 to 20 lags
        assert acf[0] == pytest.approx(1.0)  # ACF at lag 0 is 1
    
    def test_seasonality_detection(self):
        """Test seasonality detection."""
        # Create seasonal data
        t = np.arange(200)
        seasonal = np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 0.1
        
        result = detect_seasonality(seasonal, max_period=50)
        
        assert "has_seasonality" in result
        assert "primary_period" in result
    
    def test_train_test_split(self):
        """Test train/test split."""
        data = np.arange(100)
        
        # Test with proportion
        train, test = split_train_test(data, test_size=0.2)
        assert len(train) == 80
        assert len(test) == 20
        
        # Test with absolute size
        train, test = split_train_test(data, test_size=25)
        assert len(train) == 75
        assert len(test) == 25
    
    def test_forecast_accuracy(self):
        """Test forecast accuracy metrics."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        forecast = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        metrics = calculate_forecast_accuracy(actual, forecast)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0


class TestIntegration:
    """Integration tests with multiple models."""
    
    def test_arima_garch_combination(self):
        """Test combining ARIMA for mean and GARCH for variance."""
        np.random.seed(42)
        
        # Generate data
        n = 300
        data = np.cumsum(np.random.randn(n) * 0.5)
        returns = np.diff(data) / data[:-1]
        
        # Fit ARIMA to returns
        arima = ARIMAModel(order=(1, 0, 1))
        arima.fit(returns)
        mean_forecast = arima.forecast(horizon=5)
        
        # Fit GARCH to returns
        garch = GARCHModel(order=(1, 1))
        garch.fit(returns, use_returns=True)
        vol_forecast = garch.forecast(horizon=5)
        
        # Both should produce valid forecasts
        assert len(mean_forecast.forecast) == 5
        assert len(vol_forecast.volatility) == 5
    
    def test_ewma_garch_volatility_comparison(self):
        """Compare EWMA and GARCH volatility estimates."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        
        # EWMA volatility
        ewma = EWMAModel(decay_factor=0.94)
        ewma.fit(returns, use_returns=True, calculate_volatility=True)
        
        # GARCH volatility
        garch = GARCHModel(order=(1, 1))
        garch.fit(returns, use_returns=True)
        
        # Both should produce positive volatility estimates
        assert np.all(ewma.ewma_volatility > 0)
        assert np.all(garch.conditional_volatility > 0)
        
        # Correlation should be positive
        correlation = np.corrcoef(
            ewma.ewma_volatility,
            garch.conditional_volatility
        )[0, 1]
        assert correlation > 0.5  # Should be reasonably correlated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])