"""Tests for validation and error handling utilities."""

import pytest
from datetime import datetime
from axiom.utils import (
    FinancialValidator,
    ComplianceValidator,
    DataQualityValidator,
    AxiomError,
    AIProviderError,
    FinancialDataError,
    ComplianceError,
    ErrorSeverity,
    ErrorCategory,
    validate_investment_banking_workflow,
    raise_validation_errors,
)


class TestFinancialValidator:
    """Test financial data validation."""

    def test_validate_financial_metrics_valid_data(self):
        """Test validation with valid financial metrics."""
        valid_metrics = {
            "revenue": 1000000,
            "ebitda": 250000,
            "debt": 500000,
            "cash": 100000,
            "pe_ratio": 25.5,
            "confidence": 0.85,
        }

        errors = FinancialValidator.validate_financial_metrics(valid_metrics)
        assert len(errors) == 0

    def test_validate_financial_metrics_invalid_data(self):
        """Test validation with invalid financial metrics."""
        invalid_metrics = {
            "revenue": -1000,  # Negative revenue
            "debt": -500,  # Negative debt
            "confidence": 1.5,  # Confidence > 1
        }

        errors = FinancialValidator.validate_financial_metrics(invalid_metrics)
        assert len(errors) > 0
        assert any("revenue" in error for error in errors)
        assert any("debt" in error for error in errors)
        assert any("confidence" in error for error in errors)

    def test_validate_company_data_complete(self):
        """Test company data validation with complete data."""
        complete_data = {
            "name": "Apple Inc",
            "ticker": "AAPL",
            "sector": "Technology",
            "market_cap": 3000000000000,  # $3T
        }

        errors = FinancialValidator.validate_company_data(complete_data)
        assert len(errors) == 0

    def test_validate_company_data_incomplete(self):
        """Test company data validation with incomplete data."""
        incomplete_data = {
            "name": "Apple Inc",
            "ticker": "INVALID_TICKER_123",  # Invalid format
            # Missing sector and market_cap
        }

        errors = FinancialValidator.validate_company_data(incomplete_data)
        assert len(errors) > 0
        assert any("ticker" in error for error in errors)
        assert any("sector" in error for error in errors)

    def test_validate_ma_transaction_valid(self):
        """Test M&A transaction validation with valid data."""
        valid_transaction = {
            "target_company": "OpenAI",
            "acquirer": "Microsoft",
            "transaction_value": 10000000000,  # $10B
            "announcement_date": "2024-01-15",
        }

        errors = FinancialValidator.validate_ma_transaction(valid_transaction)
        assert len(errors) == 0

    def test_validate_ma_transaction_invalid(self):
        """Test M&A transaction validation with invalid data."""
        invalid_transaction = {
            "target_company": "OpenAI",
            # Missing acquirer
            "transaction_value": -1000000,  # Negative value
            "announcement_date": "invalid-date-format",
        }

        errors = FinancialValidator.validate_ma_transaction(invalid_transaction)
        assert len(errors) > 0
        assert any("acquirer" in error for error in errors)
        assert any("transaction_value" in error for error in errors)
        assert any("date" in error for error in errors)


class TestComplianceValidator:
    """Test compliance validation."""

    def test_validate_confidence_levels_sufficient(self):
        """Test confidence level validation with sufficient confidence."""
        analysis = {
            "confidence": 0.85,
            "evidence": [
                {"source_url": "https://sec.gov/filing1", "content": "Financial data"},
                {
                    "source_url": "https://bloomberg.com/article1",
                    "content": "Market analysis",
                },
                {
                    "source_url": "https://reuters.com/news1",
                    "content": "Industry trends",
                },
            ],
        }

        errors = ComplianceValidator.validate_confidence_levels(
            analysis, "due_diligence"
        )
        assert len(errors) == 0

    def test_validate_confidence_levels_insufficient(self):
        """Test confidence level validation with insufficient confidence."""
        analysis = {
            "confidence": 0.6,  # Below 0.8 threshold for due diligence
            "evidence": [
                {"source_url": "https://example.com", "content": "Low quality source"}
            ],
        }

        errors = ComplianceValidator.validate_confidence_levels(
            analysis, "due_diligence"
        )
        assert len(errors) > 0
        assert any("confidence" in error.lower() for error in errors)
        assert any("evidence" in error.lower() for error in errors)
        assert any("authoritative" in error.lower() for error in errors)

    def test_validate_regulatory_compliance(self):
        """Test regulatory compliance validation."""
        compliant_analysis = {
            "disclaimers": "This analysis is for informational purposes only",
            "audit_trail": "Complete analysis workflow documented",
            "citations": [
                {"source": "SEC Filing", "url": "https://sec.gov/filing1"},
                {"source": "Bloomberg", "url": "https://bloomberg.com/article1"},
            ],
        }

        errors = ComplianceValidator.validate_regulatory_compliance(compliant_analysis)
        assert len(errors) == 0

        # Test non-compliant analysis
        non_compliant_analysis = {}
        errors = ComplianceValidator.validate_regulatory_compliance(
            non_compliant_analysis
        )
        assert len(errors) > 0


class TestDataQualityValidator:
    """Test data quality validation."""

    def test_validate_search_results_quality(self):
        """Test search results validation."""
        valid_results = [
            {
                "title": "Company Financial Report",
                "url": "https://investor.company.com/financial-report",
                "snippet": "This is a comprehensive financial analysis of the company's Q3 performance showing strong revenue growth and improved margins.",
            },
            {
                "title": "Market Analysis",
                "url": "https://bloomberg.com/market-analysis",
                "snippet": "Industry trends indicate continued growth in the technology sector with increasing M&A activity.",
            },
        ]

        validated_results, errors = DataQualityValidator.validate_search_results(
            valid_results
        )
        assert len(validated_results) == 2
        assert len(errors) == 0

    def test_validate_search_results_poor_quality(self):
        """Test search results validation with poor quality data."""
        poor_results = [
            {
                "title": "",  # Empty title
                "url": "invalid-url",  # Invalid URL
                "snippet": "Short",  # Too short
            },
            {
                "title": "Good Title",
                # Missing URL
                "snippet": "This is a sufficiently long snippet that provides meaningful content for analysis purposes.",
            },
        ]

        validated_results, errors = DataQualityValidator.validate_search_results(
            poor_results
        )
        assert len(validated_results) == 0  # No results should pass validation
        assert len(errors) > 0

    def test_validate_evidence_quality_good(self):
        """Test evidence quality validation with good evidence."""
        good_evidence = [
            {
                "content": "Financial performance shows strong revenue growth and market expansion opportunities.",
                "source_url": "https://sec.gov/filing1",
                "confidence": 0.85,
            },
            {
                "content": "Market analysis indicates competitive advantages and strategic positioning benefits.",
                "source_url": "https://bloomberg.com/analysis1",
                "confidence": 0.80,
            },
            {
                "content": "Valuation analysis using multiple methodologies supports fair value assessment.",
                "source_url": "https://reuters.com/valuation1",
                "confidence": 0.90,
            },
        ]

        errors = DataQualityValidator.validate_evidence_quality(good_evidence)
        assert len(errors) == 0

    def test_validate_evidence_quality_poor(self):
        """Test evidence quality validation with poor evidence."""
        poor_evidence = [
            {
                "content": "Some basic information without financial context.",
                "source_url": "https://example.com/blog1",
                "confidence": 0.4,  # Low confidence
            }
        ]

        errors = DataQualityValidator.validate_evidence_quality(poor_evidence)
        assert len(errors) > 0
        assert any("source diversity" in error for error in errors)
        assert any("low-confidence" in error for error in errors)
        assert any("financial relevance" in error for error in errors)


class TestErrorHandling:
    """Test error handling classes and functions."""

    def test_axiom_error_creation(self):
        """Test AxiomError creation."""
        error = AxiomError(
            "Test error message",
            category=ErrorCategory.FINANCIAL_VALIDATION,
            severity=ErrorSeverity.HIGH,
            context={"test_field": "test_value"},
        )

        assert error.message == "Test error message"
        assert error.category == ErrorCategory.FINANCIAL_VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["test_field"] == "test_value"
        assert isinstance(error.timestamp, datetime)

    def test_ai_provider_error(self):
        """Test AIProviderError specialized error."""
        error = AIProviderError(
            "Provider failed", provider="TestProvider", model="test-model"
        )

        assert error.category == ErrorCategory.AI_PROVIDER
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["provider"] == "TestProvider"
        assert error.context["model"] == "test-model"

    def test_financial_data_error(self):
        """Test FinancialDataError specialized error."""
        error = FinancialDataError(
            "Invalid financial metrics", data_source="test_source"
        )

        assert error.category == ErrorCategory.FINANCIAL_VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["data_source"] == "test_source"

    def test_compliance_error(self):
        """Test ComplianceError specialized error."""
        error = ComplianceError(
            "Compliance validation failed", compliance_rule="confidence_threshold"
        )

        assert error.category == ErrorCategory.COMPLIANCE
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["compliance_rule"] == "confidence_threshold"

    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = AxiomError(
            "Test error",
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            context={"key": "value"},
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "AxiomError"
        assert error_dict["message"] == "Test error"
        assert error_dict["category"] == "processing"
        assert error_dict["severity"] == "medium"
        assert error_dict["context"]["key"] == "value"
        assert "timestamp" in error_dict


class TestWorkflowValidation:
    """Test end-to-end workflow validation."""

    def test_validate_investment_banking_workflow_complete(self):
        """Test workflow validation with complete data."""
        complete_workflow = {
            "financial_metrics": {
                "revenue": 1000000,
                "ebitda": 250000,
                "pe_ratio": 25.0,
                "confidence": 0.85,
            },
            "company_data": {
                "name": "Test Company",
                "ticker": "TEST",
                "sector": "Technology",
                "market_cap": 5000000000,
            },
            "analysis": {
                "confidence": 0.85,
                "evidence": [
                    {
                        "source_url": "https://sec.gov/filing",
                        "content": "Financial analysis",
                    },
                    {
                        "source_url": "https://bloomberg.com/news",
                        "content": "Market analysis",
                    },
                    {
                        "source_url": "https://reuters.com/report",
                        "content": "Industry analysis",
                    },
                ],
                "citations": [
                    {"source": "SEC", "url": "https://sec.gov/filing"},
                    {"source": "Bloomberg", "url": "https://bloomberg.com/news"},
                ],
                "disclaimers": "Investment analysis disclaimer",
                "audit_trail": "Complete workflow documentation",
            },
            "analysis_type": "due_diligence",
            "evidence": [
                {
                    "content": "Financial performance data",
                    "source_url": "https://sec.gov/filing",
                    "confidence": 0.8,
                },
                {
                    "content": "Market analysis data",
                    "source_url": "https://bloomberg.com/news",
                    "confidence": 0.9,
                },
            ],
        }

        results = validate_investment_banking_workflow(complete_workflow)

        # Check that validation passes (minimal errors)
        total_errors = sum(
            len(errors) for errors in results.values() if isinstance(errors, list)
        )
        assert total_errors <= 1  # Allow for minor validation issues

    def test_validate_investment_banking_workflow_incomplete(self):
        """Test workflow validation with incomplete data."""
        incomplete_workflow = {
            "financial_metrics": {
                "revenue": -1000,  # Invalid negative revenue
                "confidence": 1.5,  # Invalid confidence > 1
            },
            "company_data": {
                "name": "Test Company"
                # Missing required fields
            },
            "analysis": {
                "confidence": 0.5,  # Below threshold
                "evidence": [],  # No evidence
            },
            "analysis_type": "due_diligence",
        }

        results = validate_investment_banking_workflow(incomplete_workflow)

        # Should have multiple validation errors
        total_errors = sum(
            len(errors) for errors in results.values() if isinstance(errors, list)
        )
        assert total_errors > 0

    def test_raise_validation_errors_compliance(self):
        """Test raising compliance validation errors."""
        validation_results = {
            "compliance": ["Missing regulatory disclaimers", "Insufficient evidence"],
            "financial_metrics": [],
            "data_quality": [],
            "overall_errors": [],
        }

        with pytest.raises(ComplianceError) as exc_info:
            raise_validation_errors(validation_results)

        assert "compliance validation failed" in str(exc_info.value)

    def test_raise_validation_errors_financial(self):
        """Test raising financial validation errors."""
        validation_results = {
            "compliance": [],
            "financial_metrics": ["Invalid revenue value", "PE ratio out of range"],
            "data_quality": [],
            "overall_errors": [],
        }

        with pytest.raises(FinancialDataError) as exc_info:
            raise_validation_errors(validation_results)

        assert "financial metrics validation failed" in str(exc_info.value)


class TestDataQualityValidation:
    """Test data quality validation functions."""

    def test_search_results_validation_mixed_quality(self):
        """Test search results validation with mixed quality data."""
        mixed_results = [
            # Good result
            {
                "title": "Company Financial Report Q3 2024",
                "url": "https://investor.company.com/reports/q3-2024",
                "snippet": "Comprehensive quarterly financial analysis showing revenue growth of 15% and margin expansion across all business segments.",
            },
            # Poor result - missing URL
            {"title": "Brief news", "snippet": "Short snippet"},
            # Good result
            {
                "title": "Industry Market Analysis",
                "url": "https://bloomberg.com/industry-analysis",
                "snippet": "In-depth market analysis covering competitive dynamics, growth trends, and investment opportunities in the technology sector.",
            },
        ]

        validated_results, errors = DataQualityValidator.validate_search_results(
            mixed_results
        )

        # Should have 2 good results, 1 error
        assert len(validated_results) == 2
        assert len(errors) > 0
        assert "Missing url" in errors[0]

    def test_evidence_quality_assessment(self):
        """Test evidence quality assessment."""
        high_quality_evidence = [
            {
                "content": "Financial analysis shows strong revenue growth and profitability trends indicating solid market position.",
                "source_url": "https://sec.gov/filing123",
                "confidence": 0.9,
            },
            {
                "content": "Market valuation analysis using DCF and comparable company methodologies supports current trading range.",
                "source_url": "https://bloomberg.com/valuation-analysis",
                "confidence": 0.85,
            },
            {
                "content": "Strategic assessment reveals competitive advantages in core market segments with expansion opportunities.",
                "source_url": "https://reuters.com/strategic-review",
                "confidence": 0.8,
            },
        ]

        errors = DataQualityValidator.validate_evidence_quality(high_quality_evidence)
        assert len(errors) == 0

        # Test low quality evidence
        low_quality_evidence = [
            {
                "content": "Basic company information without financial context.",
                "source_url": "https://example.com/blog",
                "confidence": 0.3,
            }
        ]

        errors = DataQualityValidator.validate_evidence_quality(low_quality_evidence)
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])
