"""
PostgreSQL Database Models for Institutional-Grade Financial Data Management.

Implements comprehensive schema for:
- Price data (OHLCV with volume)
- Portfolio positions and trades
- Company fundamentals
- VaR calculation history
- Performance metrics
- Portfolio optimization results

Designed for high-performance queries and institutional compliance.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Numeric,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class TimeFrame(enum.Enum):
    """Time frame for price data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class TradeType(enum.Enum):
    """Trade type classification."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderType(enum.Enum):
    """Order type classification."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class VaRMethodType(enum.Enum):
    """VaR calculation method."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


class OptimizationMethodType(enum.Enum):
    """Portfolio optimization method."""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    EFFICIENT_RETURN = "efficient_return"
    EFFICIENT_RISK = "efficient_risk"
    RISK_PARITY = "risk_parity"
    MIN_CVAR = "min_cvar"


class PriceData(Base):
    """
    OHLCV price data with volume.
    
    Optimized for:
    - Time-series queries
    - Technical analysis
    - Backtesting
    - Real-time data ingestion
    """
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    timeframe = Column(SQLEnum(TimeFrame), nullable=False, default=TimeFrame.DAY_1)
    
    # OHLCV data
    open = Column(Numeric(precision=20, scale=8), nullable=False)
    high = Column(Numeric(precision=20, scale=8), nullable=False)
    low = Column(Numeric(precision=20, scale=8), nullable=False)
    close = Column(Numeric(precision=20, scale=8), nullable=False)
    volume = Column(Numeric(precision=20, scale=2), nullable=False)
    
    # Additional metrics
    adj_close = Column(Numeric(precision=20, scale=8))  # Adjusted close for splits/dividends
    vwap = Column(Numeric(precision=20, scale=8))  # Volume-weighted average price
    num_trades = Column(Integer)  # Number of trades in period
    
    # Metadata
    source = Column(String(50))  # Data source (e.g., 'yahoo', 'polygon', 'finnhub')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', 'timeframe', name='uq_price_symbol_timestamp_timeframe'),
        CheckConstraint('high >= low', name='check_high_gte_low'),
        CheckConstraint('high >= open', name='check_high_gte_open'),
        CheckConstraint('high >= close', name='check_high_gte_close'),
        CheckConstraint('low <= open', name='check_low_lte_open'),
        CheckConstraint('low <= close', name='check_low_lte_close'),
        CheckConstraint('volume >= 0', name='check_volume_non_negative'),
        Index('idx_price_symbol_timestamp', 'symbol', 'timestamp'),
        Index('idx_price_timestamp', 'timestamp'),
        Index('idx_price_symbol_timeframe', 'symbol', 'timeframe'),
    )
    
    def __repr__(self):
        return f"<PriceData(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"


class PortfolioPosition(Base):
    """
    Portfolio positions tracking.
    
    Tracks:
    - Current holdings
    - Position sizing
    - Cost basis
    - P&L tracking
    """
    __tablename__ = "portfolio_positions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(50), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Position details
    quantity = Column(Numeric(precision=20, scale=8), nullable=False)
    avg_cost = Column(Numeric(precision=20, scale=8), nullable=False)  # Average cost basis
    current_price = Column(Numeric(precision=20, scale=8))
    
    # P&L tracking
    unrealized_pnl = Column(Numeric(precision=20, scale=2))
    realized_pnl = Column(Numeric(precision=20, scale=2), default=0)
    
    # Position metadata
    first_acquired = Column(DateTime(timezone=True), nullable=False)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True, index=True)
    
    # Risk metrics
    position_value = Column(Numeric(precision=20, scale=2))
    weight = Column(Float)  # Portfolio weight (0-1)
    
    # Relationships
    trades = relationship("Trade", back_populates="position", cascade="all, delete-orphan")
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'symbol', name='uq_portfolio_symbol'),
        CheckConstraint('quantity >= 0', name='check_quantity_non_negative'),
        Index('idx_position_portfolio_symbol', 'portfolio_id', 'symbol'),
        Index('idx_position_active', 'portfolio_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<PortfolioPosition(portfolio='{self.portfolio_id}', symbol='{self.symbol}', qty={self.quantity})>"


class Trade(Base):
    """
    Trade execution records.
    
    Complete audit trail for:
    - Trade execution
    - Order management
    - Transaction costs
    - Regulatory compliance
    """
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(50), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('portfolio_positions.id'), nullable=True)
    
    # Trade details
    symbol = Column(String(20), nullable=False, index=True)
    trade_type = Column(SQLEnum(TradeType), nullable=False)
    order_type = Column(SQLEnum(OrderType), nullable=False)
    
    quantity = Column(Numeric(precision=20, scale=8), nullable=False)
    price = Column(Numeric(precision=20, scale=8), nullable=False)
    
    # Costs and fees
    commission = Column(Numeric(precision=20, scale=2), default=0)
    slippage = Column(Numeric(precision=20, scale=2), default=0)
    total_cost = Column(Numeric(precision=20, scale=2), nullable=False)
    
    # Execution details
    executed_at = Column(DateTime(timezone=True), nullable=False, index=True)
    order_id = Column(String(100), unique=True)
    execution_venue = Column(String(50))  # Exchange/broker
    
    # Strategy context
    strategy_name = Column(String(100))
    signal_id = Column(String(100))
    
    # Relationships
    position = relationship("PortfolioPosition", back_populates="trades")
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    __table_args__ = (
        CheckConstraint('quantity > 0', name='check_trade_quantity_positive'),
        CheckConstraint('price > 0', name='check_trade_price_positive'),
        Index('idx_trade_portfolio_symbol', 'portfolio_id', 'symbol'),
        Index('idx_trade_executed_at', 'executed_at'),
        Index('idx_trade_strategy', 'strategy_name'),
    )
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', type='{self.trade_type.value}', qty={self.quantity}, price={self.price})>"


class CompanyFundamental(Base):
    """
    Company fundamental data.
    
    Stores:
    - Financial statements
    - Key metrics
    - Valuation ratios
    - Corporate actions
    """
    __tablename__ = "company_fundamentals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    report_date = Column(DateTime(timezone=True), nullable=False)
    fiscal_period = Column(String(10))  # e.g., 'Q1', 'Q2', 'FY'
    
    # Company info
    company_name = Column(String(200))
    sector = Column(String(100), index=True)
    industry = Column(String(100), index=True)
    market_cap = Column(Numeric(precision=20, scale=2))
    
    # Income statement
    revenue = Column(Numeric(precision=20, scale=2))
    gross_profit = Column(Numeric(precision=20, scale=2))
    operating_income = Column(Numeric(precision=20, scale=2))
    net_income = Column(Numeric(precision=20, scale=2))
    ebitda = Column(Numeric(precision=20, scale=2))
    eps = Column(Numeric(precision=10, scale=4))  # Earnings per share
    
    # Balance sheet
    total_assets = Column(Numeric(precision=20, scale=2))
    total_liabilities = Column(Numeric(precision=20, scale=2))
    total_equity = Column(Numeric(precision=20, scale=2))
    cash = Column(Numeric(precision=20, scale=2))
    debt = Column(Numeric(precision=20, scale=2))
    
    # Cash flow
    operating_cash_flow = Column(Numeric(precision=20, scale=2))
    investing_cash_flow = Column(Numeric(precision=20, scale=2))
    financing_cash_flow = Column(Numeric(precision=20, scale=2))
    free_cash_flow = Column(Numeric(precision=20, scale=2))
    
    # Valuation ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    peg_ratio = Column(Float)
    dividend_yield = Column(Float)
    
    # Growth metrics
    revenue_growth_yoy = Column(Float)
    earnings_growth_yoy = Column(Float)
    
    # Additional data
    metadata_json = Column(JSON)  # Flexible storage for additional metrics
    
    # Source tracking
    source = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        UniqueConstraint('symbol', 'report_date', 'fiscal_period', name='uq_fundamental_symbol_date_period'),
        Index('idx_fundamental_symbol_date', 'symbol', 'report_date'),
        Index('idx_fundamental_sector', 'sector'),
    )
    
    def __repr__(self):
        return f"<CompanyFundamental(symbol='{self.symbol}', date='{self.report_date}', revenue={self.revenue})>"


class VaRCalculation(Base):
    """
    Value at Risk calculation history.
    
    Stores:
    - VaR results
    - Method used
    - Portfolio context
    - Historical tracking
    """
    __tablename__ = "var_calculations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(50), nullable=False, index=True)
    
    # VaR details
    calculation_date = Column(DateTime(timezone=True), nullable=False, index=True)
    method = Column(SQLEnum(VaRMethodType), nullable=False)
    confidence_level = Column(Float, nullable=False)  # e.g., 0.95 for 95%
    time_horizon_days = Column(Integer, nullable=False)
    
    # Results
    var_amount = Column(Numeric(precision=20, scale=2), nullable=False)
    var_percentage = Column(Float, nullable=False)
    expected_shortfall = Column(Numeric(precision=20, scale=2))  # CVaR
    portfolio_value = Column(Numeric(precision=20, scale=2), nullable=False)
    
    # Method-specific metadata
    parameters = Column(JSON)  # Store method-specific parameters
    
    # Position-level VaR (if calculated)
    position_contributions = Column(JSON)  # Symbol -> VaR contribution
    
    # Validation metrics
    is_backtested = Column(Boolean, default=False)
    breach_count = Column(Integer)
    accuracy_score = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    __table_args__ = (
        CheckConstraint('confidence_level > 0 AND confidence_level < 1', name='check_confidence_valid'),
        CheckConstraint('time_horizon_days > 0', name='check_horizon_positive'),
        Index('idx_var_portfolio_date', 'portfolio_id', 'calculation_date'),
        Index('idx_var_method', 'method'),
    )
    
    def __repr__(self):
        return f"<VaRCalculation(portfolio='{self.portfolio_id}', method='{self.method.value}', var={self.var_amount})>"


class PerformanceMetric(Base):
    """
    Portfolio performance metrics tracking.
    
    Tracks:
    - Returns
    - Risk metrics
    - Sharpe/Sortino ratios
    - Drawdowns
    """
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(50), nullable=False, index=True)
    metric_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Return metrics
    daily_return = Column(Float)
    cumulative_return = Column(Float)
    annualized_return = Column(Float)
    
    # Risk metrics
    volatility = Column(Float)  # Annualized
    downside_deviation = Column(Float)
    max_drawdown = Column(Float)
    current_drawdown = Column(Float)
    
    # Risk-adjusted returns
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    calmar_ratio = Column(Float)
    
    # Additional metrics
    beta = Column(Float)  # vs benchmark
    alpha = Column(Float)  # vs benchmark
    information_ratio = Column(Float)
    treynor_ratio = Column(Float)
    
    # Portfolio characteristics
    portfolio_value = Column(Numeric(precision=20, scale=2))
    num_positions = Column(Integer)
    turnover = Column(Float)
    
    # Benchmark comparison
    benchmark_return = Column(Float)
    excess_return = Column(Float)
    tracking_error = Column(Float)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'metric_date', name='uq_metric_portfolio_date'),
        Index('idx_metric_portfolio_date', 'portfolio_id', 'metric_date'),
    )
    
    def __repr__(self):
        return f"<PerformanceMetric(portfolio='{self.portfolio_id}', date='{self.metric_date}', return={self.daily_return})>"


class PortfolioOptimization(Base):
    """
    Portfolio optimization results history.
    
    Stores:
    - Optimal weights
    - Optimization method
    - Constraints used
    - Performance expectations
    """
    __tablename__ = "portfolio_optimizations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String(50), nullable=False, index=True)
    
    # Optimization details
    optimization_date = Column(DateTime(timezone=True), nullable=False, index=True)
    method = Column(SQLEnum(OptimizationMethodType), nullable=False)
    
    # Results
    optimal_weights = Column(JSON, nullable=False)  # {symbol: weight}
    expected_return = Column(Float, nullable=False)
    expected_volatility = Column(Float, nullable=False)
    expected_sharpe = Column(Float)
    
    # Constraints used
    constraints = Column(JSON)  # Optimization constraints
    bounds = Column(JSON)  # Weight bounds
    
    # Target metrics (if applicable)
    target_return = Column(Float)
    target_risk = Column(Float)
    
    # Optimization metadata
    success = Column(Boolean, nullable=False)
    message = Column(Text)
    computation_time = Column(Float)  # seconds
    
    # Implementation tracking
    is_implemented = Column(Boolean, default=False)
    implemented_at = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)
    
    __table_args__ = (
        Index('idx_optimization_portfolio_date', 'portfolio_id', 'optimization_date'),
        Index('idx_optimization_method', 'method'),
        Index('idx_optimization_implemented', 'is_implemented'),
    )
    
    def __repr__(self):
        return f"<PortfolioOptimization(portfolio='{self.portfolio_id}', method='{self.method.value}', sharpe={self.expected_sharpe})>"


class DocumentEmbedding(Base):
    """
    Document embeddings for RAG and semantic search.
    
    Stores:
    - Text embeddings
    - Document metadata
    - Vector search optimization
    """
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Document identification
    document_id = Column(String(100), unique=True, nullable=False, index=True)
    document_type = Column(String(50), nullable=False, index=True)  # 'sec_filing', 'research', 'news'
    symbol = Column(String(20), index=True)  # Associated company
    
    # Document content
    title = Column(Text)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), unique=True)  # SHA-256 hash for deduplication
    
    # Embedding (stored locally, synced to vector DB)
    embedding_model = Column(String(100))  # Model used for embedding
    embedding_dim = Column(Integer)  # Embedding dimension
    
    # Metadata
    source_url = Column(Text)
    publication_date = Column(DateTime(timezone=True))
    author = Column(String(200))
    
    # Search optimization
    keywords = Column(JSON)  # Extracted keywords
    summary = Column(Text)
    
    # Vector DB sync
    vector_db_id = Column(String(100))  # ID in vector database
    vector_db_synced = Column(Boolean, default=False)
    last_synced_at = Column(DateTime(timezone=True))
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_document_type_symbol', 'document_type', 'symbol'),
        Index('idx_document_publication', 'publication_date'),
        Index('idx_document_synced', 'vector_db_synced'),
    )
    
    def __repr__(self):
        return f"<DocumentEmbedding(id='{self.document_id}', type='{self.document_type}', symbol='{self.symbol}')>"


# Export all models
__all__ = [
    "Base",
    "TimeFrame",
    "TradeType",
    "OrderType",
    "VaRMethodType",
    "OptimizationMethodType",
    "PriceData",
    "PortfolioPosition",
    "Trade",
    "CompanyFundamental",
    "VaRCalculation",
    "PerformanceMetric",
    "PortfolioOptimization",
    "DocumentEmbedding",
]