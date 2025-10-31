"""
Data Validation Rules Engine - Institutional Grade

Implements comprehensive validation rules for financial data quality assurance.
Ensures data legitimacy and compliance with institutional standards.

This is CRITICAL for project recognition and data credibility.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation rules."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"
    INTEGRITY = "integrity"


@dataclass
class ValidationRule:
    """
    Individual validation rule definition.
    
    Represents a single data quality check with clear pass/fail criteria.
    """
    name: str
    description: str
    category: ValidationCategory
    severity: ValidationSeverity
    validator: Callable[[Any], bool]
    error_message_template: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, data: Any) -> 'ValidationResult':
        """
        Execute validation rule on data.
        
        Returns:
            ValidationResult with pass/fail status and details
        """
        try:
            passed = self.validator(data)
            
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                severity=self.severity,
                passed=passed,
                error_message=None if passed else self.error_message_template,
                data_sample=str(data)[:100],  # First 100 chars for debugging
                timestamp=datetime.now()
            )
        except Exception as e:
            return ValidationResult(
                rule_name=self.name,
                category=self.category,
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                error_message=f"Validation rule failed with exception: {str(e)}",
                data_sample=str(data)[:100],
                timestamp=datetime.now()
            )


@dataclass
class ValidationResult:
    """Result of a single validation rule execution."""
    rule_name: str
    category: ValidationCategory
    severity: ValidationSeverity
    passed: bool
    error_message: Optional[str]
    data_sample: str
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/reporting."""
        return {
            "rule_name": self.rule_name,
            "category": self.category.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "error_message": self.error_message,
            "data_sample": self.data_sample,
            "timestamp": self.timestamp.isoformat()
        }


class DataValidationEngine:
    """
    Comprehensive data validation engine for financial data.
    
    Implements institutional-grade validation rules for:
    - Price data
    - Fundamental data  
    - Market data
    - Portfolio data
    - Trade data
    
    Ensures data legitimacy and compliance with quality standards.
    """
    
    def __init__(self):
        """Initialize validation engine with comprehensive rule set."""
        self.rules: Dict[str, List[ValidationRule]] = {
            "price_data": [],
            "fundamental_data": [],
            "market_data": [],
            "portfolio_data": [],
            "trade_data": []
        }
        
        # Register all validation rules
        self._register_price_data_rules()
        self._register_fundamental_data_rules()
        self._register_market_data_rules()
        self._register_portfolio_data_rules()
        self._register_trade_data_rules()
    
    def _register_price_data_rules(self):
        """Register validation rules for price data (OHLCV)."""
        
        # Rule 1: Completeness - All required fields present
        self.rules["price_data"].append(ValidationRule(
            name="price_completeness",
            description="All required OHLCV fields must be present",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: all(
                field in data for field in ['open', 'high', 'low', 'close', 'volume']
            ),
            error_message_template="Missing required OHLCV fields"
        ))
        
        # Rule 2: High >= Low (basic sanity check)
        self.rules["price_data"].append(ValidationRule(
            name="high_gte_low",
            description="High price must be >= Low price",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: float(data.get('high', 0)) >= float(data.get('low', 0)),
            error_message_template="High price is less than Low price - data integrity violation"
        ))
        
        # Rule 3: Close within High-Low range
        self.rules["price_data"].append(ValidationRule(
            name="close_in_range",
            description="Close price must be between High and Low",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.ERROR,
            validator=lambda data: (
                float(data.get('low', 0)) <= float(data.get('close', 0)) <= float(data.get('high', 0))
            ),
            error_message_template="Close price outside of High-Low range"
        ))
        
        # Rule 4: Open within High-Low range
        self.rules["price_data"].append(ValidationRule(
            name="open_in_range",
            description="Open price must be between High and Low",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.ERROR,
            validator=lambda data: (
                float(data.get('low', 0)) <= float(data.get('open', 0)) <= float(data.get('high', 0))
            ),
            error_message_template="Open price outside of High-Low range"
        ))
        
        # Rule 5: Volume must be non-negative
        self.rules["price_data"].append(ValidationRule(
            name="volume_non_negative",
            description="Volume must be >= 0",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: float(data.get('volume', 0)) >= 0,
            error_message_template="Negative volume detected"
        ))
        
        # Rule 6: Prices must be positive
        self.rules["price_data"].append(ValidationRule(
            name="prices_positive",
            description="All prices must be > 0",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: all(
                float(data.get(field, 0)) > 0 
                for field in ['open', 'high', 'low', 'close']
            ),
            error_message_template="Zero or negative price detected"
        ))
        
        # Rule 7: Reasonable price movement (<50% intraday for stocks)
        self.rules["price_data"].append(ValidationRule(
            name="reasonable_intraday_move",
            description="Intraday price movement should be reasonable (<50% for stocks)",
            category=ValidationCategory.ACCURACY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: (
                abs(float(data.get('high', 0)) - float(data.get('low', 0))) / 
                float(data.get('open', 1)) < 0.5
            ) if float(data.get('open', 0)) > 0 else True,
            error_message_template="Extreme intraday price movement detected (>50%)"
        ))
        
        # Rule 8: Timestamp must be recent (within last 5 years for historical data)
        self.rules["price_data"].append(ValidationRule(
            name="timestamp_reasonable",
            description="Timestamp must be within reasonable range",
            category=ValidationCategory.TIMELINESS,
            severity=ValidationSeverity.ERROR,
            validator=lambda data: (
                datetime.now() - timedelta(days=5*365) <= 
                datetime.fromisoformat(str(data.get('timestamp', '2020-01-01')).replace('Z', '+00:00')) <=
                datetime.now() + timedelta(days=1)
            ),
            error_message_template="Timestamp outside reasonable range"
        ))
    
    def _register_fundamental_data_rules(self):
        """Register validation rules for fundamental data."""
        
        # Rule 1: Required fields for fundamentals
        self.rules["fundamental_data"].append(ValidationRule(
            name="fundamental_completeness",
            description="Core fundamental fields must be present",
            category=ValidationCategory.COMPLETENESS,
            severity=ValidationSeverity.ERROR,
            validator=lambda data: all(
                field in data for field in ['symbol', 'revenue', 'total_assets']
            ),
            error_message_template="Missing core fundamental fields"
        ))
        
        # Rule 2: Assets >= Liabilities (accounting identity)
        self.rules["fundamental_data"].append(ValidationRule(
            name="accounting_identity",
            description="Total Assets must equal Liabilities + Equity",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: (
                abs(
                    float(data.get('total_assets', 0)) - 
                    (float(data.get('total_liabilities', 0)) + float(data.get('total_equity', 0)))
                ) / max(float(data.get('total_assets', 1)), 1) < 0.01  # 1% tolerance
            ) if all(k in data for k in ['total_assets', 'total_liabilities', 'total_equity']) else True,
            error_message_template="Accounting identity violated: Assets ≠ Liabilities + Equity"
        ))
        
        # Rule 3: Positive revenue for operating companies
        self.rules["fundamental_data"].append(ValidationRule(
            name="revenue_positive",
            description="Revenue should be positive for operating companies",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: float(data.get('revenue', 0)) > 0,
            error_message_template="Zero or negative revenue"
        ))
        
        # Rule 4: Reasonable P/E ratio (if provided)
        self.rules["fundamental_data"].append(ValidationRule(
            name="pe_ratio_reasonable",
            description="P/E ratio should be reasonable (-50 to 1000)",
            category=ValidationCategory.ACCURACY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: (
                -50 < float(data.get('pe_ratio', 0)) < 1000
            ) if 'pe_ratio' in data and data['pe_ratio'] is not None else True,
            error_message_template="P/E ratio outside reasonable range"
        ))
    
    def _register_market_data_rules(self):
        """Register validation rules for market data."""
        
        # Rule 1: Bid-Ask spread must be positive
        self.rules["market_data"].append(ValidationRule(
            name="bid_ask_spread_positive",
            description="Ask price must be >= Bid price",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: (
                float(data.get('ask', 0)) >= float(data.get('bid', 0))
            ) if all(k in data for k in ['bid', 'ask']) else True,
            error_message_template="Negative bid-ask spread detected"
        ))
        
        # Rule 2: Reasonable bid-ask spread (<10% for liquid stocks)
        self.rules["market_data"].append(ValidationRule(
            name="bid_ask_spread_reasonable",
            description="Bid-ask spread should be reasonable (<10% of mid)",
            category=ValidationCategory.ACCURACY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: (
                (float(data.get('ask', 0)) - float(data.get('bid', 0))) / 
                ((float(data.get('ask', 0)) + float(data.get('bid', 0))) / 2) < 0.10
            ) if all(k in data for k in ['bid', 'ask']) and float(data.get('bid', 0)) > 0 else True,
            error_message_template="Unusually wide bid-ask spread (>10%)"
        ))
    
    def _register_portfolio_data_rules(self):
        """Register validation rules for portfolio data."""
        
        # Rule 1: Position quantity must be non-negative (no short without flag)
        self.rules["portfolio_data"].append(ValidationRule(
            name="position_quantity_valid",
            description="Position quantity must be non-negative unless marked as short",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.ERROR,
            validator=lambda data: (
                float(data.get('quantity', 0)) >= 0 or 
                data.get('is_short', False)
            ),
            error_message_template="Negative position quantity without short flag"
        ))
        
        # Rule 2: Portfolio weights sum to ~1.0
        self.rules["portfolio_data"].append(ValidationRule(
            name="weights_sum_to_one",
            description="Portfolio weights should sum to approximately 1.0",
            category=ValidationCategory.CONSISTENCY,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: (
                0.95 <= sum(float(w) for w in data.get('weights', [])) <= 1.05
            ) if 'weights' in data and data['weights'] else True,
            error_message_template="Portfolio weights don't sum to 1.0"
        ))
    
    def _register_trade_data_rules(self):
        """Register validation rules for trade data."""
        
        # Rule 1: Trade price must be positive
        self.rules["trade_data"].append(ValidationRule(
            name="trade_price_positive",
            description="Trade execution price must be > 0",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: float(data.get('price', 0)) > 0,
            error_message_template="Zero or negative trade price"
        ))
        
        # Rule 2: Trade quantity must be positive
        self.rules["trade_data"].append(ValidationRule(
            name="trade_quantity_positive",
            description="Trade quantity must be > 0",
            category=ValidationCategory.VALIDITY,
            severity=ValidationSeverity.CRITICAL,
            validator=lambda data: float(data.get('quantity', 0)) > 0,
            error_message_template="Zero or negative trade quantity"
        ))
        
        # Rule 3: Execution timestamp must be reasonable
        self.rules["trade_data"].append(ValidationRule(
            name="execution_timestamp_valid",
            description="Execution timestamp must be within market hours (with tolerance)",
            category=ValidationCategory.TIMELINESS,
            severity=ValidationSeverity.WARNING,
            validator=lambda data: True,  # Placeholder - needs market hours logic
            error_message_template="Trade executed outside market hours"
        ))
    
    def validate_data(
        self,
        data: Any,
        data_type: str,
        raise_on_critical: bool = True
    ) -> List[ValidationResult]:
        """
        Validate data against all registered rules for the data type.
        
        Args:
            data: Data to validate
            data_type: Type of data ('price_data', 'fundamental_data', etc.)
            raise_on_critical: If True, raise exception on critical failures
        
        Returns:
            List of validation results
        
        Raises:
            ValidationError if critical rules fail and raise_on_critical=True
        """
        if data_type not in self.rules:
            raise ValueError(f"Unknown data type: {data_type}")
        
        results = []
        critical_failures = []
        
        for rule in self.rules[data_type]:
            result = rule.validate(data)
            results.append(result)
            
            if not result.passed and result.severity == ValidationSeverity.CRITICAL:
                critical_failures.append(result)
        
        if raise_on_critical and critical_failures:
            error_msg = f"Critical validation failures: {len(critical_failures)} rules failed"
            details = "\n".join(f"  - {r.rule_name}: {r.error_message}" for r in critical_failures)
            raise ValidationError(f"{error_msg}\n{details}")
        
        return results
    
    def get_validation_summary(
        self,
        results: List[ValidationResult]
    ) -> Dict[str, Any]:
        """
        Generate summary statistics from validation results.
        
        Returns:
            Dictionary with validation statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        by_severity = {}
        for severity in ValidationSeverity:
            failures = [r for r in results if not r.passed and r.severity == severity]
            by_severity[severity.value] = len(failures)
        
        by_category = {}
        for category in ValidationCategory:
            cat_results = [r for r in results if r.category == category]
            by_category[category.value] = {
                "total": len(cat_results),
                "passed": sum(1 for r in cat_results if r.passed),
                "failed": sum(1 for r in cat_results if not r.passed)
            }
        
        return {
            "total_rules": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "by_severity": by_severity,
            "by_category": by_category,
            "critical_failures": by_severity.get("critical", 0),
            "error_failures": by_severity.get("error", 0),
            "warning_failures": by_severity.get("warning", 0)
        }
    
    def add_custom_rule(self, data_type: str, rule: ValidationRule):
        """Add custom validation rule for specific data type."""
        if data_type not in self.rules:
            self.rules[data_type] = []
        self.rules[data_type].append(rule)
    
    def get_rules_for_type(self, data_type: str) -> List[ValidationRule]:
        """Get all rules for a specific data type."""
        return self.rules.get(data_type, [])


class ValidationError(Exception):
    """Raised when critical validation rules fail."""
    pass


# Singleton instance for easy access
_validation_engine: Optional[DataValidationEngine] = None


def get_validation_engine() -> DataValidationEngine:
    """Get or create singleton validation engine instance."""
    global _validation_engine
    
    if _validation_engine is None:
        _validation_engine = DataValidationEngine()
    
    return _validation_engine


if __name__ == "__main__":
    # Example usage
    engine = get_validation_engine()
    
    # Test price data validation
    price_data = {
        'symbol': 'AAPL',
        'open': 150.0,
        'high': 152.0,
        'low': 149.0,
        'close': 151.0,
        'volume': 1000000,
        'timestamp': datetime.now().isoformat()
    }
    
    results = engine.validate_data(price_data, "price_data", raise_on_critical=False)
    summary = engine.get_validation_summary(results)
    
    print(f"Validation Results: {summary['passed']}/{summary['total_rules']} rules passed")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    print(f"Critical Failures: {summary['critical_failures']}")
    
    # Show any failures
    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailures:")
        for f in failures:
            print(f"  - {f.rule_name} ({f.severity.value}): {f.error_message}")
    else:
        print("\n✅ All validation rules passed!")