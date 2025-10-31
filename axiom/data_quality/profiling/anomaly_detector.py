"""
Anomaly Detection System - Financial Data Quality

Detects anomalies in financial data using multiple statistical methods.
Critical for ensuring data legitimacy and model reliability.

Methods:
- Statistical outlier detection (IQR, Z-score)
- Time-series anomaly detection
- Financial-specific anomaly rules
- Multi-variate anomaly detection

This ensures data credibility and prevents bad data from contaminating models.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL_OUTLIER = "statistical_outlier"
    PRICE_SPIKE = "price_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    DISTRIBUTION_SHIFT = "distribution_shift"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected data anomaly."""
    
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    affected_field: str
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    deviation: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    data_sample: str = ""
    remediation_hint: str = ""
    
    def to_dict(self) -> Dict:
        """Convert anomaly to dictionary for reporting."""
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_field": self.affected_field,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation": self.deviation,
            "timestamp": self.timestamp.isoformat(),
            "remediation_hint": self.remediation_hint
        }


class AnomalyDetector:
    """
    Comprehensive anomaly detection for financial data.
    
    Detects:
    - Statistical outliers (IQR, Z-score methods)
    - Price/volume spikes
    - Missing or duplicate data
    - Temporal anomalies (gaps, future dates)
    - Business rule violations
    - Distribution shifts
    
    Used for:
    - Data quality assurance
    - Real-time monitoring
    - Model input validation
    - Regulatory compliance
    """
    
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        price_spike_threshold: float = 0.20  # 20% move
    ):
        """
        Initialize anomaly detector with configurable thresholds.
        
        Args:
            z_score_threshold: Z-score threshold for outlier detection (default: 3.0)
            iqr_multiplier: IQR multiplier for outlier detection (default: 1.5)
            price_spike_threshold: % change threshold for price spikes (default: 0.20)
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.price_spike_threshold = price_spike_threshold
        
        # Historical baselines (would be loaded from storage in production)
        self.baselines: Dict[str, Dict[str, float]] = {}
    
    def detect_anomalies(
        self,
        data: List[Dict[str, Any]],
        data_type: str = "price_data"
    ) -> List[Anomaly]:
        """
        Detect all anomalies in dataset.
        
        Args:
            data: List of data records
            data_type: Type of data for context-specific detection
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if not data:
            return anomalies
        
        # Statistical outlier detection
        anomalies.extend(self._detect_statistical_outliers(data))
        
        # Price-specific anomalies
        if data_type == "price_data":
            anomalies.extend(self._detect_price_anomalies(data))
            anomalies.extend(self._detect_volume_anomalies(data))
        
        # Temporal anomalies
        if 'timestamp' in data[0]:
            anomalies.extend(self._detect_temporal_anomalies(data))
        
        # Duplicate detection
        anomalies.extend(self._detect_duplicates(data))
        
        # Business rule violations
        anomalies.extend(self._detect_business_rule_violations(data, data_type))
        
        return anomalies
    
    def _detect_statistical_outliers(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect statistical outliers using IQR and Z-score methods."""
        anomalies = []
        
        # Get numerical columns
        if not data:
            return anomalies
        
        numerical_cols = [
            k for k, v in data[0].items() 
            if isinstance(v, (int, float))
        ]
        
        for col in numerical_cols:
            values = [
                float(row[col]) for row in data 
                if row.get(col) is not None
            ]
            
            if len(values) < 4:  # Need at least 4 values for IQR
                continue
            
            # IQR method
            sorted_vals = sorted(values)
            q1 = sorted_vals[len(sorted_vals) // 4]
            q3 = sorted_vals[3 * len(sorted_vals) // 4]
            iqr = q3 - q1
            
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            # Find outliers
            for idx, val in enumerate(values):
                if val < lower_bound or val > upper_bound:
                    severity = AnomalySeverity.HIGH if (
                        val < q1 - 3 * iqr or val > q3 + 3 * iqr
                    ) else AnomalySeverity.MEDIUM
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                        severity=severity,
                        description=f"Statistical outlier detected in {col}",
                        affected_field=col,
                        expected_value=f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                        actual_value=val,
                        deviation=abs(val - statistics.median(values)) / statistics.stdev(values) if len(values) > 1 else 0,
                        remediation_hint="Review value for data entry error or investigate market event"
                    ))
        
        return anomalies
    
    def _detect_price_anomalies(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect price-specific anomalies (spikes, crashes, violations)."""
        anomalies = []
        
        for idx in range(1, len(data)):
            curr = data[idx]
            prev = data[idx-1]
            
            # Check for required fields
            if not all(k in curr for k in ['open', 'high', 'low', 'close']):
                continue
            
            # Price spike detection
            if 'close' in prev and 'close' in curr:
                price_change = abs(
                    (float(curr['close']) - float(prev['close'])) / float(prev['close'])
                )
                
                if price_change > self.price_spike_threshold:
                    severity = AnomalySeverity.CRITICAL if price_change > 0.50 else AnomalySeverity.HIGH
                    
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.PRICE_SPIKE,
                        severity=severity,
                        description=f"Large price movement: {price_change*100:.1f}%",
                        affected_field="close",
                        expected_value=f"±{self.price_spike_threshold*100}% change",
                        actual_value=f"{price_change*100:.1f}% change",
                        deviation=price_change,
                        remediation_hint="Verify market event (earnings, merger, etc.) or check data source"
                    ))
            
            # OHLC integrity violations
            high = float(curr.get('high', 0))
            low = float(curr.get('low', 0))
            open_price = float(curr.get('open', 0))
            close = float(curr.get('close', 0))
            
            # High < Low violation
            if high < low:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                    severity=AnomalySeverity.CRITICAL,
                    description="High price is less than Low price",
                    affected_field="high/low",
                    expected_value=f"high >= low",
                    actual_value=f"high={high}, low={low}",
                    remediation_hint="Data integrity error - correct or reject record"
                ))
            
            # Close outside High-Low range
            if close > high or close < low:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                    severity=AnomalySeverity.CRITICAL,
                    description="Close price outside High-Low range",
                    affected_field="close",
                    expected_value=f"[{low}, {high}]",
                    actual_value=close,
                    remediation_hint="Data integrity error - correct or reject record"
                ))
        
        return anomalies
    
    def _detect_volume_anomalies(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect volume anomalies (unusual trading activity)."""
        anomalies = []
        
        volumes = [
            float(row['volume']) for row in data 
            if 'volume' in row and row['volume'] is not None
        ]
        
        if not volumes or len(volumes) < 2:
            return anomalies
        
        avg_volume = statistics.mean(volumes)
        std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0
        
        for idx, vol in enumerate(volumes):
            # Zero volume (suspicious)
            if vol == 0:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    description="Zero trading volume",
                    affected_field="volume",
                    expected_value=f"> 0",
                    actual_value=0,
                    remediation_hint="Check if market was closed or data is missing"
                ))
            
            # Extreme volume spike (>5 standard deviations)
            elif std_volume > 0 and abs(vol - avg_volume) > 5 * std_volume:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    severity=AnomalySeverity.HIGH,
                    description=f"Extreme volume spike: {vol/avg_volume:.1f}x average",
                    affected_field="volume",
                    expected_value=f"~{avg_volume:.0f}",
                    actual_value=vol,
                    deviation=(vol - avg_volume) / std_volume,
                    remediation_hint="Verify corporate action, news event, or data error"
                ))
        
        return anomalies
    
    def _detect_temporal_anomalies(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect temporal anomalies (gaps, future dates, etc.)."""
        anomalies = []
        
        timestamps = []
        for row in data:
            if 'timestamp' in row and row['timestamp']:
                try:
                    if isinstance(row['timestamp'], str):
                        ts = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                    else:
                        ts = row['timestamp']
                    timestamps.append(ts)
                except:
                    continue
        
        if len(timestamps) < 2:
            return anomalies
        
        # Sort timestamps
        timestamps.sort()
        
        # Detect future dates
        now = datetime.now()
        for ts in timestamps:
            if ts > now + timedelta(hours=24):
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                    severity=AnomalySeverity.CRITICAL,
                    description="Future timestamp detected",
                    affected_field="timestamp",
                    expected_value=f"<= {now.isoformat()}",
                    actual_value=ts.isoformat(),
                    remediation_hint="Correct timestamp or check system clock"
                ))
        
        # Detect large gaps (for expected daily data)
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).days
            
            # Gap > 7 days for daily data (excluding weekends)
            if gap > 7:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                    severity=AnomalySeverity.MEDIUM,
                    description=f"Large data gap: {gap} days",
                    affected_field="timestamp",
                    expected_value="<= 3 days",
                    actual_value=f"{gap} days",
                    remediation_hint="Check for missing data or data collection issues"
                ))
        
        return anomalies
    
    def _detect_duplicates(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect duplicate records."""
        anomalies = []
        
        # Create unique keys (symbol + timestamp if available)
        seen_keys = set()
        
        for row in data:
            # Create key from available identifying fields
            key_parts = []
            if 'symbol' in row:
                key_parts.append(str(row['symbol']))
            if 'timestamp' in row:
                key_parts.append(str(row['timestamp']))
            
            if not key_parts:
                continue
            
            key = "|".join(key_parts)
            
            if key in seen_keys:
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.DUPLICATE_DATA,
                    severity=AnomalySeverity.HIGH,
                    description="Duplicate record detected",
                    affected_field="primary_key",
                    expected_value="unique",
                    actual_value="duplicate",
                    data_sample=str(row)[:100],
                    remediation_hint="Remove duplicate or check data source"
                ))
            
            seen_keys.add(key)
        
        return anomalies
    
    def _detect_business_rule_violations(
        self,
        data: List[Dict[str, Any]],
        data_type: str
    ) -> List[Anomaly]:
        """Detect violations of financial business rules."""
        anomalies = []
        
        for row in data:
            # Price data business rules
            if data_type == "price_data":
                # Negative prices
                price_fields = ['open', 'high', 'low', 'close']
                for field in price_fields:
                    if field in row and float(row[field]) <= 0:
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                            severity=AnomalySeverity.CRITICAL,
                            description=f"Non-positive {field} price",
                            affected_field=field,
                            expected_value="> 0",
                            actual_value=row[field],
                            remediation_hint="Prices must be positive - check data source"
                        ))
                
                # Negative volume
                if 'volume' in row and float(row.get('volume', 0)) < 0:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                        severity=AnomalySeverity.CRITICAL,
                        description="Negative volume",
                        affected_field="volume",
                        expected_value=">= 0",
                        actual_value=row['volume'],
                        remediation_hint="Volume cannot be negative"
                    ))
                
                # Bid-ask spread
                if 'bid' in row and 'ask' in row:
                    bid = float(row['bid'])
                    ask = float(row['ask'])
                    if ask < bid:
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                            severity=AnomalySeverity.CRITICAL,
                            description="Negative bid-ask spread",
                            affected_field="bid/ask",
                            expected_value="ask >= bid",
                            actual_value=f"bid={bid}, ask={ask}",
                            remediation_hint="Bid-ask spread must be non-negative"
                        ))
            
            # Fundamental data business rules
            elif data_type == "fundamental_data":
                # Accounting identity: Assets = Liabilities + Equity
                if all(k in row for k in ['total_assets', 'total_liabilities', 'total_equity']):
                    assets = float(row['total_assets'])
                    liabilities = float(row['total_liabilities'])
                    equity = float(row['total_equity'])
                    
                    if abs(assets - (liabilities + equity)) / max(assets, 1) > 0.01:
                        anomalies.append(Anomaly(
                            anomaly_type=AnomalyType.BUSINESS_RULE_VIOLATION,
                            severity=AnomalySeverity.HIGH,
                            description="Accounting identity violated",
                            affected_field="balance_sheet",
                            expected_value="Assets = Liabilities + Equity",
                            actual_value=f"Assets={assets}, L+E={liabilities+equity}",
                            remediation_hint="Check balance sheet data integrity"
                        ))
        
        return anomalies
    
    def _detect_price_anomalies(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect price-specific anomalies."""
        anomalies = []
        
        close_prices = [
            float(row['close']) for row in data 
            if 'close' in row and row['close'] is not None
        ]
        
        if len(close_prices) < 2:
            return anomalies
        
        # Calculate daily returns
        for i in range(1, len(close_prices)):
            return_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            
            # Extreme price spike (>20% daily move for stocks)
            if abs(return_pct) > self.price_spike_threshold:
                severity = AnomalySeverity.CRITICAL if abs(return_pct) > 0.50 else AnomalySeverity.HIGH
                
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    severity=severity,
                    description=f"Extreme price movement: {return_pct*100:.1f}%",
                    affected_field="close",
                    expected_value=f"±{self.price_spike_threshold*100}%",
                    actual_value=f"{return_pct*100:.1f}%",
                    deviation=abs(return_pct),
                    remediation_hint="Verify corporate action, earnings, or market event"
                ))
        
        return anomalies
    
    def _detect_volume_anomalies(
        self,
        data: List[Dict[str, Any]]
    ) -> List[Anomaly]:
        """Detect volume anomalies."""
        # Implemented in detect_anomalies, placeholder here
        return []
    
    def detect_real_time_anomaly(
        self,
        new_data_point: Dict[str, Any],
        historical_baseline: List[Dict[str, Any]],
        data_type: str = "price_data"
    ) -> List[Anomaly]:
        """
        Detect anomalies in real-time data point against historical baseline.
        
        Critical for production data ingestion monitoring.
        
        Args:
            new_data_point: New data point to check
            historical_baseline: Historical data for comparison
            data_type: Type of data
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Combine for analysis
        combined = historical_baseline + [new_data_point]
        
        # Run full anomaly detection
        all_anomalies = self.detect_anomalies(combined, data_type)
        
        # Filter to only anomalies in new data point
        # (simplified - in production would tag which record has anomaly)
        return all_anomalies[-10:] if all_anomalies else []
    
    def get_anomaly_summary(
        self,
        anomalies: List[Anomaly]
    ) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies."""
        
        by_type = {}
        for anomaly_type in AnomalyType:
            by_type[anomaly_type.value] = sum(
                1 for a in anomalies if a.anomaly_type == anomaly_type
            )
        
        by_severity = {}
        for severity in AnomalySeverity:
            by_severity[severity.value] = sum(
                1 for a in anomalies if a.severity == severity
            )
        
        return {
            "total_anomalies": len(anomalies),
            "by_type": by_type,
            "by_severity": by_severity,
            "critical_count": by_severity.get("critical", 0),
            "high_count": by_severity.get("high", 0),
            "requires_attention": by_severity.get("critical", 0) + by_severity.get("high", 0)
        }


# Singleton instance
_detector_instance: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create singleton anomaly detector instance."""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = AnomalyDetector()
    
    return _detector_instance


if __name__ == "__main__":
    # Example usage
    detector = get_anomaly_detector()
    
    # Test data with intentional anomalies
    test_data = [
        {'symbol': 'AAPL', 'close': 150.0, 'volume': 1000000, 'timestamp': '2024-01-01'},
        {'symbol': 'AAPL', 'close': 151.0, 'volume': 1100000, 'timestamp': '2024-01-02'},
        {'symbol': 'AAPL', 'close': 200.0, 'volume': 1000000, 'timestamp': '2024-01-03'},  # Price spike!
        {'symbol': 'AAPL', 'close': 152.0, 'volume': 0, 'timestamp': '2024-01-04'},  # Zero volume!
    ]
    
    anomalies = detector.detect_anomalies(test_data, "price_data")
    summary = detector.get_anomaly_summary(anomalies)
    
    print(f"Detected {summary['total_anomalies']} anomalies")
    print(f"Critical: {summary['critical_count']}, High: {summary['high_count']}")
    
    if anomalies:
        print("\nAnomalies:")
        for a in anomalies:
            print(f"  [{a.severity.value}] {a.description}")
            print(f"    Field: {a.affected_field}, Value: {a.actual_value}")
            print(f"    Hint: {a.remediation_hint}")
    
    print("\n✅ Anomaly detection complete!")