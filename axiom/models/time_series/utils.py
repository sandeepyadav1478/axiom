"""
Utility Functions for Time Series Models

Helper functions for data preprocessing, model selection, and diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


def prepare_returns(
    prices: Union[np.ndarray, pd.Series],
    log_returns: bool = True,
    dropna: bool = True
) -> np.ndarray:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        log_returns: If True, calculate log returns, else simple returns
        dropna: Whether to drop NaN values
    
    Returns:
        Returns array
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if log_returns:
        returns = np.log(prices[1:] / prices[:-1])
    else:
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
    
    if dropna:
        returns = returns[~np.isnan(returns)]
    
    return returns


def check_stationarity(
    data: Union[np.ndarray, pd.Series],
    max_lags: int = 20
) -> Dict:
    """
    Check if time series is stationary using multiple tests.
    
    Args:
        data: Time series data
        max_lags: Maximum lags for autocorrelation
    
    Returns:
        Dictionary with stationarity test results
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calculate autocorrelation
    acf = calculate_acf(data, nlags=max_lags)
    
    # Simple stationarity checks
    # 1. Check if ACF decays quickly
    significant_lags = np.sum(np.abs(acf[1:]) > 0.5)
    
    # 2. Check variance stability (split into halves)
    mid = len(data) // 2
    var_first_half = np.var(data[:mid])
    var_second_half = np.var(data[mid:])
    var_ratio = max(var_first_half, var_second_half) / min(var_first_half, var_second_half)
    
    # 3. Check mean stability
    mean_first_half = np.mean(data[:mid])
    mean_second_half = np.mean(data[mid:])
    mean_diff = abs(mean_first_half - mean_second_half) / np.std(data)
    
    is_stationary = (
        significant_lags < max_lags * 0.2 and
        var_ratio < 2.0 and
        mean_diff < 0.5
    )
    
    return {
        "is_stationary": is_stationary,
        "significant_acf_lags": int(significant_lags),
        "variance_ratio": float(var_ratio),
        "mean_difference": float(mean_diff),
        "recommendation": "Differencing not needed" if is_stationary else "Consider differencing"
    }


def calculate_acf(
    data: Union[np.ndarray, pd.Series],
    nlags: int = 40,
    adjusted: bool = False
) -> np.ndarray:
    """
    Calculate autocorrelation function.
    
    Args:
        data: Time series data
        nlags: Number of lags
        adjusted: Whether to use adjusted ACF
    
    Returns:
        ACF values
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    data = data - np.mean(data)
    c0 = np.sum(data ** 2) / len(data)
    
    acf = [1.0]
    for k in range(1, min(nlags + 1, len(data))):
        if adjusted:
            ck = np.sum(data[:-k] * data[k:]) / (len(data) - k)
        else:
            ck = np.sum(data[:-k] * data[k:]) / len(data)
        acf.append(ck / c0 if c0 > 0 else 0)
    
    return np.array(acf)


def calculate_pacf(
    data: Union[np.ndarray, pd.Series],
    nlags: int = 40
) -> np.ndarray:
    """
    Calculate partial autocorrelation function using Yule-Walker equations.
    
    Args:
        data: Time series data
        nlags: Number of lags
    
    Returns:
        PACF values
    """
    acf_values = calculate_acf(data, nlags=nlags)
    
    pacf = [1.0]
    for k in range(1, min(nlags + 1, len(acf_values))):
        # Solve Yule-Walker equations
        if k == 1:
            pacf.append(acf_values[1])
        else:
            # Build autocorrelation matrix
            R = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    R[i, j] = acf_values[abs(i - j)]
            
            r = acf_values[1:k+1]
            
            try:
                phi = np.linalg.solve(R, r)
                pacf.append(phi[-1])
            except:
                pacf.append(0.0)
    
    return np.array(pacf)


def detect_seasonality(
    data: Union[np.ndarray, pd.Series],
    max_period: int = 365
) -> Dict:
    """
    Detect seasonality in time series.
    
    Args:
        data: Time series data
        max_period: Maximum period to test
    
    Returns:
        Dictionary with seasonality information
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calculate ACF
    acf = calculate_acf(data, nlags=min(max_period, len(data) // 2))
    
    # Find peaks in ACF
    peaks = []
    for i in range(2, len(acf) - 1):
        if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.3:
            peaks.append((i, acf[i]))
    
    # Sort by ACF value
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    has_seasonality = len(peaks) > 0
    primary_period = peaks[0][0] if peaks else None
    
    return {
        "has_seasonality": has_seasonality,
        "primary_period": primary_period,
        "secondary_periods": [p[0] for p in peaks[1:3]] if len(peaks) > 1 else [],
        "seasonal_strength": float(peaks[0][1]) if peaks else 0.0
    }


def calculate_rolling_statistics(
    data: Union[np.ndarray, pd.Series],
    window: int = 20
) -> Dict[str, np.ndarray]:
    """
    Calculate rolling mean and volatility.
    
    Args:
        data: Time series data
        window: Rolling window size
    
    Returns:
        Dictionary with rolling statistics
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    n = len(data)
    rolling_mean = np.full(n, np.nan)
    rolling_std = np.full(n, np.nan)
    
    for i in range(window, n):
        rolling_mean[i] = np.mean(data[i-window:i])
        rolling_std[i] = np.std(data[i-window:i])
    
    return {
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_cv": rolling_std / np.abs(rolling_mean)  # Coefficient of variation
    }


def detect_outliers(
    data: Union[np.ndarray, pd.Series],
    method: str = 'iqr',
    threshold: float = 3.0
) -> Dict:
    """
    Detect outliers in time series.
    
    Args:
        data: Time series data
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: Threshold for outlier detection
    
    Returns:
        Dictionary with outlier information
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
    
    elif method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        outliers = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    outlier_indices = np.where(outliers)[0]
    
    return {
        "has_outliers": np.any(outliers),
        "n_outliers": int(np.sum(outliers)),
        "outlier_percentage": float(np.mean(outliers) * 100),
        "outlier_indices": outlier_indices.tolist(),
        "outlier_values": data[outliers].tolist()
    }


def calculate_information_criteria(
    residuals: np.ndarray,
    n_params: int,
    criterion: str = 'aic'
) -> float:
    """
    Calculate information criteria for model selection.
    
    Args:
        residuals: Model residuals
        n_params: Number of model parameters
        criterion: 'aic', 'bic', or 'hqic'
    
    Returns:
        Information criterion value
    """
    n = len(residuals)
    sigma2 = np.var(residuals)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    
    if criterion == 'aic':
        return 2 * n_params - 2 * log_likelihood
    elif criterion == 'bic':
        return n_params * np.log(n) - 2 * log_likelihood
    elif criterion == 'hqic':
        return 2 * n_params * np.log(np.log(n)) - 2 * log_likelihood
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def split_train_test(
    data: Union[np.ndarray, pd.Series],
    test_size: Union[int, float] = 0.2,
    shuffle: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series into train and test sets.
    
    Args:
        data: Time series data
        test_size: Size of test set (int for absolute size, float for proportion)
        shuffle: Whether to shuffle (usually False for time series)
    
    Returns:
        Tuple of (train, test)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    n = len(data)
    
    if isinstance(test_size, float):
        test_size = int(n * test_size)
    
    if shuffle:
        indices = np.random.permutation(n)
        train_idx = indices[:-test_size]
        test_idx = indices[-test_size:]
        return data[train_idx], data[test_idx]
    else:
        # Time series: use last observations for test
        return data[:-test_size], data[-test_size:]


def calculate_forecast_accuracy(
    actual: np.ndarray,
    forecast: np.ndarray
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual: Actual values
        forecast: Forecasted values
    
    Returns:
        Dictionary with accuracy metrics
    """
    errors = actual - forecast
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Avoid division by zero
    actual_nonzero = np.where(actual != 0, actual, 1)
    pct_errors = abs_errors / np.abs(actual_nonzero) * 100
    
    return {
        "mae": float(np.mean(abs_errors)),  # Mean Absolute Error
        "rmse": float(np.sqrt(np.mean(squared_errors))),  # Root Mean Squared Error
        "mape": float(np.mean(pct_errors)),  # Mean Absolute Percentage Error
        "mse": float(np.mean(squared_errors)),  # Mean Squared Error
        "r2": float(1 - np.sum(squared_errors) / np.sum((actual - np.mean(actual)) ** 2)),  # R²
        "max_error": float(np.max(abs_errors)),
        "median_error": float(np.median(abs_errors))
    }


def detrend(
    data: Union[np.ndarray, pd.Series],
    method: str = 'linear'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove trend from time series.
    
    Args:
        data: Time series data
        method: 'linear', 'mean', or 'diff'
    
    Returns:
        Tuple of (detrended_data, trend)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if method == 'linear':
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        trend = np.polyval(coeffs, x)
        detrended = data - trend
    
    elif method == 'mean':
        trend = np.full(len(data), np.mean(data))
        detrended = data - trend
    
    elif method == 'diff':
        detrended = np.diff(data)
        trend = data[:-1]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return detrended, trend


def normalize_series(
    data: Union[np.ndarray, pd.Series],
    method: str = 'zscore'
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize time series.
    
    Args:
        data: Time series data
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if isinstance(data, pd.Series):
        data = data.values
    
    if method == 'zscore':
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / std if std > 0 else data
        params = {"mean": mean, "std": std}
    
    elif method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val) if max_val > min_val else data
        params = {"min": min_val, "max": max_val}
    
    elif method == 'robust':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        normalized = (data - median) / mad if mad > 0 else data
        params = {"median": median, "mad": mad}
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized, params


# Export
__all__ = [
    "prepare_returns",
    "check_stationarity",
    "calculate_acf",
    "calculate_pacf",
    "detect_seasonality",
    "calculate_rolling_statistics",
    "detect_outliers",
    "calculate_information_criteria",
    "split_train_test",
    "calculate_forecast_accuracy",
    "detrend",
    "normalize_series"
]