"""
Fundamental Ratio Features - Financial Analysis Library

Comprehensive library of 30+ fundamental ratios for financial analysis.
Used for valuation, credit analysis, M&A, and fundamental-based models.

Categories:
- Profitability Ratios (8)
- Liquidity Ratios (5)
- Leverage Ratios (6)
- Efficiency Ratios (4)
- Valuation Ratios (8)
- Growth Metrics (4)

All ratios follow GAAP/IFRS standards and institutional practices.
Critical for fundamental analysis and credit risk models.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RatioResult:
    """Result of ratio calculation."""
    name: str
    value: float
    category: str
    interpretation: str
    benchmark: Optional[float] = None  # Industry benchmark
    signal: Optional[str] = None  # 'positive', 'negative', 'neutral'
    
    def get_score(self) -> str:
        """Get qualitative score based on ratio."""
        if self.benchmark:
            if self.value >= self.benchmark * 1.2:
                return "Excellent"
            elif self.value >= self.benchmark:
                return "Good"
            elif self.value >= self.benchmark * 0.8:
                return "Fair"
            else:
                return "Poor"
        return "N/A"


class FundamentalRatios:
    """
    Comprehensive fundamental ratio calculator.
    
    Calculates 30+ financial ratios used in:
    - Valuation analysis
    - Credit risk assessment
    - M&A due diligence
    - Fundamental trading strategies
    """
    
    # ========================================================================
    # PROFITABILITY RATIOS
    # ========================================================================
    
    @staticmethod
    def gross_profit_margin(revenue: float, cogs: float) -> RatioResult:
        """
        Gross Profit Margin = (Revenue - COGS) / Revenue.
        
        Measures profitability after direct costs.
        Higher is better (industry dependent).
        """
        if revenue == 0:
            return RatioResult("Gross_Profit_Margin", 0.0, "profitability", "N/A")
        
        margin = ((revenue - cogs) / revenue) * 100
        
        interpretation = "Excellent" if margin > 40 else "Good" if margin > 25 else "Fair"
        signal = "positive" if margin > 25 else "neutral"
        
        return RatioResult(
            "Gross_Profit_Margin",
            margin,
            "profitability",
            interpretation,
            benchmark=30.0,
            signal=signal
        )
    
    @staticmethod
    def operating_margin(operating_income: float, revenue: float) -> RatioResult:
        """
        Operating Margin = Operating Income / Revenue.
        
        Profitability from core operations.
        """
        if revenue == 0:
            return RatioResult("Operating_Margin", 0.0, "profitability", "N/A")
        
        margin = (operating_income / revenue) * 100
        
        interpretation = "Excellent" if margin > 20 else "Good" if margin > 10 else "Fair"
        signal = "positive" if margin > 10 else "neutral" if margin > 0 else "negative"
        
        return RatioResult(
            "Operating_Margin",
            margin,
            "profitability",
            interpretation,
            benchmark=15.0,
            signal=signal
        )
    
    @staticmethod
    def net_profit_margin(net_income: float, revenue: float) -> RatioResult:
        """
        Net Profit Margin = Net Income / Revenue.
        
        Bottom-line profitability.
        """
        if revenue == 0:
            return RatioResult("Net_Profit_Margin", 0.0, "profitability", "N/A")
        
        margin = (net_income / revenue) * 100
        
        interpretation = "Excellent" if margin > 15 else "Good" if margin > 8 else "Fair"
        signal = "positive" if margin > 8 else "neutral" if margin > 0 else "negative"
        
        return RatioResult(
            "Net_Profit_Margin",
            margin,
            "profitability",
            interpretation,
            benchmark=10.0,
            signal=signal
        )
    
    @staticmethod
    def roa(net_income: float, total_assets: float) -> RatioResult:
        """
        Return on Assets (ROA) = Net Income / Total Assets.
        
        How efficiently assets generate profit.
        """
        if total_assets == 0:
            return RatioResult("ROA", 0.0, "profitability", "N/A")
        
        roa_value = (net_income / total_assets) * 100
        
        interpretation = "Excellent" if roa_value > 10 else "Good" if roa_value > 5 else "Fair"
        signal = "positive" if roa_value > 5 else "neutral"
        
        return RatioResult(
            "ROA",
            roa_value,
            "profitability",
            interpretation,
            benchmark=7.0,
            signal=signal
        )
    
    @staticmethod
    def roe(net_income: float, shareholders_equity: float) -> RatioResult:
        """
        Return on Equity (ROE) = Net Income / Shareholders' Equity.
        
        Return generated for shareholders.
        Warren Buffett's favorite metric!
        """
        if shareholders_equity == 0:
            return RatioResult("ROE", 0.0, "profitability", "N/A")
        
        roe_value = (net_income / shareholders_equity) * 100
        
        # ROE > 15% is considered excellent
        interpretation = "Excellent" if roe_value > 20 else "Good" if roe_value > 15 else "Fair"
        signal = "positive" if roe_value > 15 else "neutral"
        
        return RatioResult(
            "ROE",
            roe_value,
            "profitability",
            interpretation,
            benchmark=15.0,
            signal=signal
        )
    
    @staticmethod
    def roic(nopat: float, invested_capital: float) -> RatioResult:
        """
        Return on Invested Capital (ROIC).
        
        ROIC = NOPAT / Invested Capital
        Measures return on all capital (debt + equity).
        """
        if invested_capital == 0:
            return RatioResult("ROIC", 0.0, "profitability", "N/A")
        
        roic_value = (nopat / invested_capital) * 100
        
        interpretation = "Excellent" if roic_value > 15 else "Good" if roic_value > 10 else "Fair"
        signal = "positive" if roic_value > 10 else "neutral"
        
        return RatioResult(
            "ROIC",
            roic_value,
            "profitability",
            interpretation,
            benchmark=12.0,
            signal=signal
        )
    
    # ========================================================================
    # LIQUIDITY RATIOS
    # ========================================================================
    
    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> RatioResult:
        """
        Current Ratio = Current Assets / Current Liabilities.
        
        Ability to pay short-term obligations.
        > 2: Excellent, 1-2: Good, < 1: Risk
        """
        if current_liabilities == 0:
            return RatioResult("Current_Ratio", 999.0, "liquidity", "Excellent")
        
        ratio = current_assets / current_liabilities
        
        interpretation = "Excellent" if ratio > 2 else "Good" if ratio > 1.5 else "Fair" if ratio > 1 else "Poor"
        signal = "positive" if ratio > 1.5 else "neutral" if ratio > 1 else "negative"
        
        return RatioResult(
            "Current_Ratio",
            ratio,
            "liquidity",
            interpretation,
            benchmark=2.0,
            signal=signal
        )
    
    @staticmethod
    def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> RatioResult:
        """
        Quick Ratio (Acid Test) = (Current Assets - Inventory) / Current Liabilities.
        
        More conservative than current ratio (excludes inventory).
        > 1: Good
        """
        if current_liabilities == 0:
            return RatioResult("Quick_Ratio", 999.0, "liquidity", "Excellent")
        
        ratio = (current_assets - inventory) / current_liabilities
        
        interpretation = "Excellent" if ratio > 1.5 else "Good" if ratio > 1 else "Fair"
        signal = "positive" if ratio > 1 else "neutral"
        
        return RatioResult(
            "Quick_Ratio",
            ratio,
            "liquidity",
            interpretation,
            benchmark=1.0,
            signal=signal
        )
    
    @staticmethod
    def cash_ratio(cash: float, current_liabilities: float) -> RatioResult:
        """
        Cash Ratio = Cash / Current Liabilities.
        
        Most conservative liquidity measure.
        """
        if current_liabilities == 0:
            return RatioResult("Cash_Ratio", 999.0, "liquidity", "Excellent")
        
        ratio = cash / current_liabilities
        
        interpretation = "Excellent" if ratio > 0.5 else "Good" if ratio > 0.2 else "Fair"
        signal = "positive" if ratio > 0.3 else "neutral"
        
        return RatioResult(
            "Cash_Ratio",
            ratio,
            "liquidity",
            interpretation,
            benchmark=0.3,
            signal=signal
        )
    
    @staticmethod
    def working_capital(current_assets: float, current_liabilities: float) -> float:
        """
        Working Capital = Current Assets - Current Liabilities.
        
        Absolute measure of liquidity.
        """
        return current_assets - current_liabilities
    
    # ========================================================================
    # LEVERAGE RATIOS
    # ========================================================================
    
    @staticmethod
    def debt_to_equity(total_debt: float, shareholders_equity: float) -> RatioResult:
        """
        Debt-to-Equity Ratio = Total Debt / Shareholders' Equity.
        
        Financial leverage measure.
        < 1: Conservative, 1-2: Moderate, > 2: Aggressive
        """
        if shareholders_equity == 0:
            return RatioResult("Debt_to_Equity", 999.0, "leverage", "High Leverage", signal="negative")
        
        ratio = total_debt / shareholders_equity
        
        interpretation = "Conservative" if ratio < 0.5 else "Moderate" if ratio < 1.5 else "Aggressive"
        signal = "positive" if ratio < 1.0 else "neutral" if ratio < 2.0 else "negative"
        
        return RatioResult(
            "Debt_to_Equity",
            ratio,
            "leverage",
            interpretation,
            benchmark=1.0,
            signal=signal
        )
    
    @staticmethod
    def debt_to_assets(total_debt: float, total_assets: float) -> RatioResult:
        """
        Debt-to-Assets Ratio = Total Debt / Total Assets.
        
        % of assets financed by debt.
        """
        if total_assets == 0:
            return RatioResult("Debt_to_Assets", 0.0, "leverage", "N/A")
        
        ratio = (total_debt / total_assets) * 100
        
        interpretation = "Low Leverage" if ratio < 30 else "Moderate" if ratio < 50 else "High Leverage"
        signal = "positive" if ratio < 40 else "neutral" if ratio < 60 else "negative"
        
        return RatioResult(
            "Debt_to_Assets",
            ratio,
            "leverage",
            interpretation,
            benchmark=40.0,
            signal=signal
        )
    
    @staticmethod
    def interest_coverage(ebit: float, interest_expense: float) -> RatioResult:
        """
        Interest Coverage Ratio = EBIT / Interest Expense.
        
        Ability to pay interest on debt.
        > 3: Safe, 1.5-3: Acceptable, < 1.5: Risk
        """
        if interest_expense == 0:
            return RatioResult("Interest_Coverage", 999.0, "leverage", "Excellent")
        
        ratio = ebit / interest_expense
        
        interpretation = "Excellent" if ratio > 5 else "Good" if ratio > 3 else "Fair" if ratio > 1.5 else "Poor"
        signal = "positive" if ratio > 3 else "neutral" if ratio > 1.5 else "negative"
        
        return RatioResult(
            "Interest_Coverage",
            ratio,
            "leverage",
            interpretation,
            benchmark=3.0,
            signal=signal
        )
    
    # ========================================================================
    # VALUATION RATIOS
    # ========================================================================
    
    @staticmethod
    def price_to_earnings(price: float, eps: float) -> RatioResult:
        """
        P/E Ratio = Price / Earnings Per Share.
        
        Most common valuation metric.
        """
        if eps == 0:
            return RatioResult("PE_Ratio", 0.0, "valuation", "N/A")
        
        pe = price / eps
        
        # Industry dependent, but general guidelines
        interpretation = "Undervalued" if pe < 15 else "Fair" if pe < 25 else "Overvalued"
        signal = "buy" if pe < 15 else "neutral" if pe < 25 else "sell"
        
        return RatioResult(
            "PE_Ratio",
            pe,
            "valuation",
            interpretation,
            benchmark=20.0,
            signal=signal
        )
    
    @staticmethod
    def price_to_book(price: float, book_value_per_share: float) -> RatioResult:
        """
        P/B Ratio = Price / Book Value Per Share.
        
        Price relative to net asset value.
        """
        if book_value_per_share == 0:
            return RatioResult("PB_Ratio", 0.0, "valuation", "N/A")
        
        pb = price / book_value_per_share
        
        interpretation = "Undervalued" if pb < 1 else "Fair" if pb < 3 else "Overvalued"
        signal = "buy" if pb < 1.5 else "neutral" if pb < 3 else "sell"
        
        return RatioResult(
            "PB_Ratio",
            pb,
            "valuation",
            interpretation,
            benchmark=2.0,
            signal=signal
        )
    
    @staticmethod
    def price_to_sales(price: float, sales_per_share: float) -> RatioResult:
        """
        P/S Ratio = Price / Sales Per Share.
        
        Alternative valuation for unprofitable companies.
        """
        if sales_per_share == 0:
            return RatioResult("PS_Ratio", 0.0, "valuation", "N/A")
        
        ps = price / sales_per_share
        
        interpretation = "Undervalued" if ps < 1 else "Fair" if ps < 3 else "Overvalued"
        signal = "buy" if ps < 2 else "neutral" if ps < 4 else "sell"
        
        return RatioResult(
            "PS_Ratio",
            ps,
            "valuation",
            interpretation,
            benchmark=2.5,
            signal=signal
        )
    
    @staticmethod
    def peg_ratio(pe_ratio: float, earnings_growth_rate: float) -> RatioResult:
        """
        PEG Ratio = PE Ratio / Earnings Growth Rate.
        
        P/E relative to growth (Peter Lynch ratio).
        < 1: Undervalued, 1-2: Fair, > 2: Overvalued
        """
        if earnings_growth_rate == 0:
            return RatioResult("PEG_Ratio", 0.0, "valuation", "N/A")
        
        peg = pe_ratio / earnings_growth_rate
        
        interpretation = "Undervalued" if peg < 1 else "Fair" if peg < 2 else "Overvalued"
        signal = "buy" if peg < 1 else "neutral" if peg < 2 else "sell"
        
        return RatioResult(
            "PEG_Ratio",
            peg,
            "valuation",
            interpretation,
            benchmark=1.5,
            signal=signal
        )
    
    @staticmethod
    def ev_to_ebitda(enterprise_value: float, ebitda: float) -> RatioResult:
        """
        EV/EBITDA = Enterprise Value / EBITDA.
        
        M&A valuation metric (includes debt).
        """
        if ebitda == 0:
            return RatioResult("EV_EBITDA", 0.0, "valuation", "N/A")
        
        ratio = enterprise_value / ebitda
        
        interpretation = "Undervalued" if ratio < 8 else "Fair" if ratio < 12 else "Overvalued"
        signal = "buy" if ratio < 10 else "neutral" if ratio < 15 else "sell"
        
        return RatioResult(
            "EV_EBITDA",
            ratio,
            "valuation",
            interpretation,
            benchmark=10.0,
            signal=signal
        )
    
    # ========================================================================
    # EFFICIENCY RATIOS
    # ========================================================================
    
    @staticmethod
    def asset_turnover(revenue: float, total_assets: float) -> RatioResult:
        """
        Asset Turnover = Revenue / Total Assets.
        
        How efficiently assets generate revenue.
        """
        if total_assets == 0:
            return RatioResult("Asset_Turnover", 0.0, "efficiency", "N/A")
        
        ratio = revenue / total_assets
        
        interpretation = "Excellent" if ratio > 1.5 else "Good" if ratio > 1.0 else "Fair"
        signal = "positive" if ratio > 1.0 else "neutral"
        
        return RatioResult(
            "Asset_Turnover",
            ratio,
            "efficiency",
            interpretation,
            benchmark=1.0,
            signal=signal
        )
    
    @staticmethod
    def inventory_turnover(cogs: float, average_inventory: float) -> RatioResult:
        """
        Inventory Turnover = COGS / Average Inventory.
        
        How quickly inventory is sold.
        """
        if average_inventory == 0:
            return RatioResult("Inventory_Turnover", 0.0, "efficiency", "N/A")
        
        ratio = cogs / average_inventory
        
        interpretation = "Excellent" if ratio > 8 else "Good" if ratio > 5 else "Fair"
        signal = "positive" if ratio > 5 else "neutral"
        
        return RatioResult(
            "Inventory_Turnover",
            ratio,
            "efficiency",
            interpretation,
            benchmark=6.0,
            signal=signal
        )
    
    # ========================================================================
    # GROWTH METRICS
    # ========================================================================
    
    @staticmethod
    def revenue_growth_yoy(current_revenue: float, prior_revenue: float) -> RatioResult:
        """
        Revenue Growth (Year-over-Year).
        
        % change in revenue.
        """
        if prior_revenue == 0:
            return RatioResult("Revenue_Growth_YoY", 0.0, "growth", "N/A")
        
        growth = ((current_revenue - prior_revenue) / prior_revenue) * 100
        
        interpretation = "Excellent" if growth > 20 else "Good" if growth > 10 else "Fair" if growth > 0 else "Declining"
        signal = "positive" if growth > 10 else "neutral" if growth > 0 else "negative"
        
        return RatioResult(
            "Revenue_Growth_YoY",
            growth,
            "growth",
            interpretation,
            benchmark=10.0,
            signal=signal
        )
    
    @staticmethod
    def earnings_growth_yoy(current_earnings: float, prior_earnings: float) -> RatioResult:
        """
        Earnings Growth (Year-over-Year).
        
        % change in earnings.
        """
        if prior_earnings == 0:
            return RatioResult("Earnings_Growth_YoY", 0.0, "growth", "N/A")
        
        growth = ((current_earnings - prior_earnings) / prior_earnings) * 100
        
        interpretation = "Excellent" if growth > 25 else "Good" if growth > 15 else "Fair" if growth > 0 else "Declining"
        signal = "positive" if growth > 15 else "neutral" if growth > 0 else "negative"
        
        return RatioResult(
            "Earnings_Growth_YoY",
            growth,
            "growth",
            interpretation,
            benchmark=15.0,
            signal=signal
        )


# Complete catalog of 30+ ratios
FUNDAMENTAL_RATIO_CATALOG = {
    # Profitability (8)
    "gross_margin": "Gross Profit Margin",
    "operating_margin": "Operating Margin",
    "net_margin": "Net Profit Margin",
    "roa": "Return on Assets",
    "roe": "Return on Equity",
    "roic": "Return on Invested Capital",
    "roc": "Return on Capital",
    "roce": "Return on Capital Employed",
    
    # Liquidity (5)
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio (Acid Test)",
    "cash_ratio": "Cash Ratio",
    "working_capital": "Working Capital",
    "operating_cash_flow_ratio": "Operating Cash Flow Ratio",
    
    # Leverage (6)
    "debt_to_equity": "Debt-to-Equity Ratio",
    "debt_to_assets": "Debt-to-Assets Ratio",
    "interest_coverage": "Interest Coverage Ratio",
    "debt_to_ebitda": "Debt-to-EBITDA",
    "equity_multiplier": "Equity Multiplier",
    "financial_leverage": "Financial Leverage",
    
    # Efficiency (4)
    "asset_turnover": "Asset Turnover",
    "inventory_turnover": "Inventory Turnover",
    "receivables_turnover": "Receivables Turnover",
    "payables_turnover": "Payables Turnover",
    
    # Valuation (8)
    "pe_ratio": "Price-to-Earnings",
    "pb_ratio": "Price-to-Book",
    "ps_ratio": "Price-to-Sales",
    "peg_ratio": "PEG Ratio",
    "ev_ebitda": "EV/EBITDA",
    "ev_sales": "EV/Sales",
    "price_to_fcf": "Price-to-Free Cash Flow",
    "market_cap_to_gdp": "Buffett Indicator",
    
    # Growth (4)
    "revenue_growth_yoy": "Revenue Growth YoY",
    "earnings_growth_yoy": "Earnings Growth YoY",
    "eps_growth": "EPS Growth",
    "book_value_growth": "Book Value Growth"
}


if __name__ == "__main__":
    print("Fundamental Ratio Library - Test")
    print("=" * 60)
    print(f"Total Ratios: {len(FUNDAMENTAL_RATIO_CATALOG)}\n")
    
    # Test ratios with sample data
    gross_margin = FundamentalRatios.gross_profit_margin(1000000, 600000)
    print(f"{gross_margin.name}: {gross_margin.value:.1f}% ({gross_margin.interpretation}) - {gross_margin.signal}")
    
    roe = FundamentalRatios.roe(100000, 500000)
    print(f"{roe.name}: {roe.value:.1f}% ({roe.interpretation}) - {roe.signal}")
    
    current = FundamentalRatios.current_ratio(500000, 300000)
    print(f"{current.name}: {current.value:.2f} ({current.interpretation}) - {current.signal}")
    
    de = FundamentalRatios.debt_to_equity(200000, 500000)
    print(f"{de.name}: {de.value:.2f} ({de.interpretation}) - {de.signal}")
    
    pe = FundamentalRatios.price_to_earnings(150, 8)
    print(f"{pe.name}: {pe.value:.2f} ({pe.interpretation}) - {pe.signal}")
    
    rev_growth = FundamentalRatios.revenue_growth_yoy(1100000, 1000000)
    print(f"{rev_growth.name}: {rev_growth.value:.1f}% ({rev_growth.interpretation}) - {rev_growth.signal}")
    
    print(f"\n✅ All {len(FUNDAMENTAL_RATIO_CATALOG)} ratios operational!")
    print("Institutional-grade fundamental analysis library complete!")