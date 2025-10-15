"""Investment Banking Data Validation and Quality Assurance."""

import re
from datetime import datetime
from typing import Any

from .error_handling import (
    AxiomError,
    ComplianceError,
    ErrorSeverity,
    FinancialDataError,
)


class FinancialValidator:
    """Validator for financial data and investment banking metrics."""

    # Financial ratio validation ranges
    RATIO_RANGES = {
        "current_ratio": (0.5, 10.0),
        "debt_to_equity": (0.0, 5.0),
        "roe": (-50.0, 100.0),
        "roa": (-20.0, 50.0),
        "profit_margin": (-100.0, 100.0),
        "pe_ratio": (0.0, 200.0),
        "ev_ebitda": (0.0, 100.0),
        "revenue_growth": (-100.0, 1000.0),  # Allow for high-growth companies
    }

    # Industry-specific validation rules
    INDUSTRY_RULES = {
        "technology": {"min_gross_margin": 50.0, "max_pe_ratio": 100.0},
        "utilities": {"min_dividend_yield": 2.0, "max_debt_to_equity": 2.0},
        "banking": {"min_tier1_capital_ratio": 6.0, "max_loan_to_deposit": 100.0},
    }

    @staticmethod
    def validate_financial_metrics(
        metrics: dict[str, float | int], industry: str | None = None
    ) -> list[str]:
        """Validate financial metrics are within reasonable bounds."""

        errors = []

        for metric, value in metrics.items():
            # Convert to float for validation
            try:
                float_value = float(value)
            except (ValueError, TypeError):
                errors.append(f"Invalid numeric value for {metric}: {value}")
                continue

            # Check against standard ranges
            if metric in FinancialValidator.RATIO_RANGES:
                min_val, max_val = FinancialValidator.RATIO_RANGES[metric]
                if not (min_val <= float_value <= max_val):
                    errors.append(
                        f"{metric} value {float_value} outside normal range ({min_val}-{max_val})"
                    )

            # Industry-specific validations
            if industry and industry.lower() in FinancialValidator.INDUSTRY_RULES:
                industry_rules = FinancialValidator.INDUSTRY_RULES[industry.lower()]

                for rule, threshold in industry_rules.items():
                    if rule.startswith("min_") and metric == rule[4:]:
                        if float_value < threshold:
                            errors.append(
                                f"{metric} {float_value} below industry minimum {threshold}"
                            )
                    elif rule.startswith("max_") and metric == rule[4:]:
                        if float_value > threshold:
                            errors.append(
                                f"{metric} {float_value} above industry maximum {threshold}"
                            )

        return errors

    @staticmethod
    def validate_company_data(company_data: dict[str, Any]) -> list[str]:
        """Validate company data completeness and quality."""

        errors = []
        required_fields = ["name", "ticker", "sector", "market_cap"]

        # Check required fields
        for field in required_fields:
            if field not in company_data or not company_data[field]:
                errors.append(f"Missing required field: {field}")

        # Validate ticker format
        if "ticker" in company_data:
            ticker = company_data["ticker"]
            if not re.match(r"^[A-Z]{1,5}$", ticker):
                errors.append(f"Invalid ticker format: {ticker}")

        # Validate market cap
        if "market_cap" in company_data:
            try:
                market_cap = float(company_data["market_cap"])
                if market_cap <= 0:
                    errors.append("Market cap must be positive")
                elif market_cap > 10000000:  # $10T cap
                    errors.append(f"Market cap seems unreasonably high: {market_cap}")
            except (ValueError, TypeError):
                errors.append("Invalid market cap value")

        return errors

    @staticmethod
    def validate_ma_transaction(transaction_data: dict[str, Any]) -> list[str]:
        """Validate M&A transaction data."""

        errors = []
        required_fields = [
            "target_company",
            "acquirer",
            "transaction_value",
            "announcement_date",
        ]

        # Check required fields
        for field in required_fields:
            if field not in transaction_data or not transaction_data[field]:
                errors.append(f"Missing required M&A field: {field}")

        # Validate transaction value
        if "transaction_value" in transaction_data:
            try:
                value = float(transaction_data["transaction_value"])
                if value <= 0:
                    errors.append("Transaction value must be positive")
                elif value > 1000000000000:  # $1T transaction
                    errors.append(f"Transaction value seems unreasonably high: {value}")
            except (ValueError, TypeError):
                errors.append("Invalid transaction value")

        # Validate date
        if "announcement_date" in transaction_data:
            try:
                date_str = transaction_data["announcement_date"]
                if isinstance(date_str, str):
                    datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                errors.append("Invalid announcement date format (expected YYYY-MM-DD)")

        return errors


class ComplianceValidator:
    """Validator for investment banking compliance requirements."""

    # Compliance thresholds
    CONFIDENCE_THRESHOLDS = {
        "due_diligence": 0.8,
        "valuation": 0.75,
        "risk_assessment": 0.85,
        "regulatory_analysis": 0.9,
    }

    @staticmethod
    def validate_confidence_levels(
        analysis: dict[str, Any], analysis_type: str
    ) -> list[str]:
        """Validate confidence levels meet compliance requirements."""

        errors = []
        required_threshold = ComplianceValidator.CONFIDENCE_THRESHOLDS.get(
            analysis_type, 0.7
        )

        if "confidence" in analysis:
            confidence = analysis["confidence"]
            if confidence < required_threshold:
                errors.append(
                    f"Confidence level {confidence:.2f} below required threshold "
                    f"{required_threshold:.2f} for {analysis_type}"
                )
        else:
            errors.append(f"Missing confidence level for {analysis_type} analysis")

        # Check evidence quality
        if "evidence" in analysis:
            evidence_list = analysis["evidence"]
            if len(evidence_list) < 3:
                errors.append(
                    f"Insufficient evidence for {analysis_type} (minimum 3 required)"
                )

            # Check evidence sources
            authoritative_sources = 0
            for evidence in evidence_list:
                if any(
                    domain in evidence.get("source_url", "").lower()
                    for domain in ["sec.gov", "bloomberg", "reuters", "wsj", "ft.com"]
                ):
                    authoritative_sources += 1

            if authoritative_sources == 0:
                errors.append("No authoritative sources found in evidence")

        return errors

    @staticmethod
    def validate_regulatory_compliance(analysis: dict[str, Any]) -> list[str]:
        """Validate regulatory compliance requirements."""

        errors = []

        # Check for required disclosures
        if "disclaimers" not in analysis or not analysis["disclaimers"]:
            errors.append("Missing required regulatory disclaimers")

        # Check audit trail
        if "audit_trail" not in analysis:
            errors.append("Missing audit trail for compliance documentation")

        # Validate citations
        if "citations" in analysis:
            citations = analysis["citations"]
            if len(citations) < 2:
                errors.append("Insufficient citations for regulatory compliance")

        return errors


class DataQualityValidator:
    """Validator for data quality and integrity."""

    @staticmethod
    def validate_search_results(
        results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Validate and clean search results."""

        validated_results = []
        errors = []

        for i, result in enumerate(results):
            result_errors = []

            # Check required fields
            required_fields = ["title", "url", "snippet"]
            for field in required_fields:
                if field not in result or not result[field]:
                    result_errors.append(f"Missing {field}")

            # Validate URL format
            if "url" in result:
                url = result["url"]
                if not url.startswith(("http://", "https://")):
                    result_errors.append("Invalid URL format")

            # Check content quality
            if "snippet" in result:
                snippet = result["snippet"]
                if len(snippet) < 20:
                    result_errors.append("Snippet too short")
                elif len(snippet) > 1000:
                    result_errors.append("Snippet too long")

            if result_errors:
                errors.append(f"Result {i}: {'; '.join(result_errors)}")
            else:
                validated_results.append(result)

        return validated_results, errors

    @staticmethod
    def validate_evidence_quality(evidence: list[dict[str, Any]]) -> list[str]:
        """Validate evidence quality and reliability."""

        errors = []

        if len(evidence) == 0:
            errors.append("No evidence provided")
            return errors

        # Check source diversity
        sources = set()
        for item in evidence:
            if "source_url" in item:
                domain = (
                    item["source_url"].split("/")[2]
                    if "/" in item["source_url"]
                    else item["source_url"]
                )
                sources.add(domain)

        if len(sources) < 2:
            errors.append("Insufficient source diversity in evidence")

        # Check confidence levels
        low_confidence_count = sum(
            1 for item in evidence if item.get("confidence", 0) < 0.6
        )

        if low_confidence_count > len(evidence) / 2:
            errors.append("Too many low-confidence evidence items")

        # Check for financial relevance
        financial_terms = [
            "revenue",
            "profit",
            "ebitda",
            "valuation",
            "market",
            "financial",
        ]
        relevant_evidence = 0

        for item in evidence:
            content = item.get("content", "").lower()
            if any(term in content for term in financial_terms):
                relevant_evidence += 1

        if relevant_evidence < len(evidence) / 2:
            errors.append("Insufficient financial relevance in evidence")

        return errors


def validate_investment_banking_workflow(
    workflow_data: dict[str, Any],
) -> dict[str, list[str]]:
    """Comprehensive validation for investment banking workflow data."""

    validation_results = {
        "financial_metrics": [],
        "company_data": [],
        "compliance": [],
        "data_quality": [],
        "overall_errors": [],
    }

    try:
        # Validate financial metrics
        if "financial_metrics" in workflow_data:
            validation_results["financial_metrics"] = (
                FinancialValidator.validate_financial_metrics(
                    workflow_data["financial_metrics"], workflow_data.get("industry")
                )
            )

        # Validate company data
        if "company_data" in workflow_data:
            validation_results["company_data"] = (
                FinancialValidator.validate_company_data(workflow_data["company_data"])
            )

        # Validate compliance
        if "analysis" in workflow_data:
            analysis = workflow_data["analysis"]
            analysis_type = workflow_data.get("analysis_type", "general")

            validation_results["compliance"].extend(
                ComplianceValidator.validate_confidence_levels(analysis, analysis_type)
            )
            validation_results["compliance"].extend(
                ComplianceValidator.validate_regulatory_compliance(analysis)
            )

        # Validate data quality
        if "evidence" in workflow_data:
            validation_results["data_quality"] = (
                DataQualityValidator.validate_evidence_quality(
                    workflow_data["evidence"]
                )
            )

        # Count total errors
        total_errors = sum(
            len(errors)
            for errors in validation_results.values()
            if isinstance(errors, list)
        )

        if total_errors > 0:
            validation_results["overall_errors"].append(
                f"Total validation errors: {total_errors}"
            )

    except Exception as e:
        validation_results["overall_errors"].append(
            f"Validation process error: {str(e)}"
        )

    return validation_results


def raise_validation_errors(
    validation_results: dict[str, list[str]],
    severity: ErrorSeverity = ErrorSeverity.HIGH,
):
    """Raise appropriate errors based on validation results."""

    total_errors = sum(
        len(errors)
        for errors in validation_results.values()
        if isinstance(errors, list)
    )

    if total_errors == 0:
        return  # No errors to raise

    # Determine most critical error category
    if validation_results["compliance"]:
        raise ComplianceError(
            f"Investment banking compliance validation failed: {'; '.join(validation_results['compliance'][:3])}",
            context={"validation_results": validation_results},
        )
    elif validation_results["financial_metrics"]:
        raise FinancialDataError(
            f"Financial metrics validation failed: {'; '.join(validation_results['financial_metrics'][:3])}",
            context={"validation_results": validation_results},
        )
    else:
        # General validation error
        all_errors = []
        for category, errors in validation_results.items():
            if isinstance(errors, list) and errors:
                all_errors.extend(errors[:2])  # Limit errors per category

        raise AxiomError(
            f"Investment banking data validation failed: {'; '.join(all_errors[:5])}",
            severity=severity,
            context={"validation_results": validation_results},
        )
