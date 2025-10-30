"""
Derivatives Platform - SQLAlchemy ORM Models

Production-grade database models for derivatives trading platform.
Maps to schema.sql tables with proper relationships and constraints.
"""

from sqlalchemy import (
    Column, Integer, BigInteger, String, Numeric, DateTime, Boolean,
    CheckConstraint, Index, ForeignKey, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Optional

Base = declarative_base()


class OptionTrade(Base):
    """Options trade execution record"""
    __tablename__ = 'option_trades'
    
    id = Column(BigInteger, primary_key=True)
    trade_id = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Option identification
    symbol = Column(String(50), nullable=False, index=True)
    underlying = Column(String(20), nullable=False, index=True)
    strike = Column(Numeric(12, 4), nullable=False)
    expiry = Column(DateTime, nullable=False, index=True)
    option_type = Column(String(10), nullable=False)
    exotic_type = Column(String(50))
    
    # Trade details
    action = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(12, 4), nullable=False)
    premium_paid = Column(Numeric(18, 2))
    commission = Column(Numeric(12, 2))
    fees = Column(Numeric(12, 2))
    
    # Greeks
    delta = Column(Numeric(10, 6))
    gamma = Column(Numeric(10, 6))
    theta = Column(Numeric(10, 6))
    vega = Column(Numeric(10, 6))
    rho = Column(Numeric(10, 6))
    
    # P&L
    entry_pnl = Column(Numeric(18, 2))
    current_pnl = Column(Numeric(18, 2))
    realized_pnl = Column(Numeric(18, 2))
    
    # Performance
    calculation_time_us = Column(Integer)
    execution_venue = Column(String(50))
    slippage_bps = Column(Integer)
    
    # Metadata
    strategy = Column(String(100), index=True)
    notes = Column(Text)
    created_by = Column(String(100))
    
    __table_args__ = (
        CheckConstraint("option_type IN ('call', 'put')", name='check_option_type'),
        CheckConstraint("action IN ('buy', 'sell', 'exercise')", name='check_action'),
        Index('idx_trades_timestamp', 'timestamp', postgresql_using='btree'),
    )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'underlying': self.underlying,
            'strike': float(self.strike),
            'expiry': self.expiry.isoformat(),
            'option_type': self.option_type,
            'action': self.action,
            'quantity': self.quantity,
            'price': float(self.price),
            'greeks': {
                'delta': float(self.delta) if self.delta else None,
                'gamma': float(self.gamma) if self.gamma else None,
                'theta': float(self.theta) if self.theta else None,
                'vega': float(self.vega) if self.vega else None,
                'rho': float(self.rho) if self.rho else None
            },
            'pnl': float(self.total_pnl) if hasattr(self, 'total_pnl') else None
        }


class Position(Base):
    """Current options position"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), unique=True, nullable=False)
    underlying = Column(String(20), nullable=False, index=True)
    strike = Column(Numeric(12, 4), nullable=False)
    expiry = Column(DateTime, nullable=False, index=True)
    option_type = Column(String(10), nullable=False)
    exotic_type = Column(String(50))
    
    # Position
    quantity = Column(Integer, nullable=False, default=0)
    average_entry_price = Column(Numeric(12, 4))
    current_market_price = Column(Numeric(12, 4))
    
    # Valuation
    market_value = Column(Numeric(18, 2))
    cost_basis = Column(Numeric(18, 2))
    unrealized_pnl = Column(Numeric(18, 2))
    realized_pnl = Column(Numeric(18, 2))
    
    # Greeks
    delta = Column(Numeric(10, 6))
    gamma = Column(Numeric(10, 6))
    theta = Column(Numeric(10, 6))
    vega = Column(Numeric(10, 6))
    rho = Column(Numeric(10, 6))
    
    # Risk
    notional_value = Column(Numeric(18, 2))
    var_contribution = Column(Numeric(18, 2))
    
    # Timestamps
    opened_at = Column(DateTime(timezone=True))
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def update_greeks(self, greeks: Dict):
        """Update Greeks from calculation"""
        self.delta = greeks.get('delta')
        self.gamma = greeks.get('gamma')
        self.theta = greeks.get('theta')
        self.vega = greeks.get('vega')
        self.rho = greeks.get('rho')
        self.last_updated = datetime.utcnow()


class GreeksHistory(Base):
    """Historical Greeks calculations"""
    __tablename__ = 'greeks_history'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    underlying = Column(String(20), nullable=False)
    
    # Greeks
    delta = Column(Numeric(10, 6), nullable=False)
    gamma = Column(Numeric(10, 6))
    theta = Column(Numeric(10, 6))
    vega = Column(Numeric(10, 6))
    rho = Column(Numeric(10, 6))
    
    # Market conditions
    spot_price = Column(Numeric(12, 4))
    implied_vol = Column(Numeric(8, 6))
    time_to_expiry = Column(Numeric(10, 6))
    
    # Performance
    calculation_time_us = Column(Integer)
    calculation_method = Column(String(50))
    accuracy_vs_bs = Column(Numeric(8, 6))


class VolatilitySurface(Base):
    """Stored volatility surfaces"""
    __tablename__ = 'volatility_surfaces'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    underlying = Column(String(20), nullable=False, index=True)
    
    # Surface data
    surface_data = Column(JSONB, nullable=False)
    strikes = Column(JSONB, nullable=False)
    maturities = Column(JSONB, nullable=False)
    
    # Construction
    construction_method = Column(String(20), nullable=False)
    construction_time_ms = Column(Numeric(10, 3))
    market_quotes_used = Column(JSONB)
    
    # Quality
    arbitrage_free = Column(Boolean, default=True)
    rmse = Column(Numeric(10, 6))
    sabr_params = Column(JSONB)


class PnLTracking(Base):
    """P&L tracking over time"""
    __tablename__ = 'pnl_tracking'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    
    # P&L
    realized_pnl = Column(Numeric(18, 2), nullable=False, default=0)
    unrealized_pnl = Column(Numeric(18, 2), nullable=False, default=0)
    total_pnl = Column(Numeric(18, 2), nullable=False, default=0)
    
    # Strategy
    strategy = Column(String(100), index=True)
    
    # Greeks summary
    total_delta = Column(Numeric(12, 4))
    total_gamma = Column(Numeric(12, 4))
    total_vega = Column(Numeric(12, 4))
    total_theta = Column(Numeric(12, 4))
    
    # Position summary
    positions_count = Column(Integer)
    notional_exposure = Column(Numeric(18, 2))
    
    # Risk metrics
    var_1day = Column(Numeric(18, 2))
    max_drawdown = Column(Numeric(18, 2))
    sharpe_ratio = Column(Numeric(8, 4))


class PerformanceMetric(Base):
    """System performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(BigInteger, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Numeric(18, 6), nullable=False)
    metric_unit = Column(String(20))
    
    # Context
    component = Column(String(50))
    additional_data = Column(JSONB)


# Example usage
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create engine (use actual connection string in production)
    engine = create_engine('postgresql://axiom_dev:password@localhost/axiom_derivatives')
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Example: Insert a trade
    trade = OptionTrade(
        trade_id='T20241029001',
        symbol='SPY241115C00450000',
        underlying='SPY',
        strike=450.00,
        expiry=datetime(2024, 11, 15),
        option_type='call',
        action='buy',
        quantity=10,
        price=5.50,
        delta=0.52,
        gamma=0.015,
        calculation_time_us=85,
        strategy='delta_neutral'
    )
    
    session.add(trade)
    session.commit()
    
    print(f"✓ Trade {trade.trade_id} recorded")
    print(f"  Greeks calculated in {trade.calculation_time_us}us")
    
    session.close()