-- Derivatives Platform PostgreSQL Schema
-- Production-grade database for sub-100 microsecond derivatives analytics
-- Author: Senior Quant Developer
-- Date: 2024-10-29

-- =============================================================================
-- CORE DERIVATIVES TABLES
-- =============================================================================

-- Options trades table (all executed trades)
CREATE TABLE IF NOT EXISTS option_trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Option identification
    symbol VARCHAR(50) NOT NULL,
    underlying VARCHAR(20) NOT NULL,
    strike DECIMAL(12, 4) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(10) NOT NULL CHECK (option_type IN ('call', 'put')),
    exotic_type VARCHAR(50),  -- NULL for vanilla, else barrier/asian/etc
    
    -- Trade details
    action VARCHAR(10) NOT NULL CHECK (action IN ('buy', 'sell', 'exercise')),
    quantity INTEGER NOT NULL,
    price DECIMAL(12, 4) NOT NULL,
    premium_paid DECIMAL(18, 2),
    commission DECIMAL(12, 2),
    fees DECIMAL(12, 2),
    
    -- Greeks at trade time
    delta DECIMAL(10, 6),
    gamma DECIMAL(10, 6),
    theta DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    rho DECIMAL(10, 6),
    
    -- P&L tracking
    entry_pnl DECIMAL(18, 2),
    current_pnl DECIMAL(18, 2),
    realized_pnl DECIMAL(18, 2),
    
    -- Performance metrics
    calculation_time_us INTEGER,  -- Microseconds
    execution_venue VARCHAR(50),
    slippage_bps INTEGER,  -- Basis points
    
    -- Metadata
    strategy VARCHAR(100),
    notes TEXT,
    created_by VARCHAR(100)
);

-- Indexes for fast queries
CREATE INDEX idx_trades_timestamp ON option_trades(timestamp DESC);
CREATE INDEX idx_trades_underlying ON option_trades(underlying);
CREATE INDEX idx_trades_symbol ON option_trades(symbol);
CREATE INDEX idx_trades_expiry ON option_trades(expiry);
CREATE INDEX idx_trades_strategy ON option_trades(strategy);

-- Partition by month for scalability
CREATE TABLE option_trades_2024_11 PARTITION OF option_trades
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

-- =============================================================================
-- POSITIONS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) UNIQUE NOT NULL,
    underlying VARCHAR(20) NOT NULL,
    strike DECIMAL(12, 4) NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(10) NOT NULL,
    exotic_type VARCHAR(50),
    
    -- Position details
    quantity INTEGER NOT NULL DEFAULT 0,
    average_entry_price DECIMAL(12, 4),
    current_market_price DECIMAL(12, 4),
    
    -- Valuation
    market_value DECIMAL(18, 2),
    cost_basis DECIMAL(18, 2),
    unrealized_pnl DECIMAL(18, 2),
    realized_pnl DECIMAL(18, 2),
    
    -- Current Greeks
    delta DECIMAL(10, 6),
    gamma DECIMAL(10, 6),
    theta DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    rho DECIMAL(10, 6),
    
    -- Risk metrics
    notional_value DECIMAL(18, 2),
    var_contribution DECIMAL(18, 2),
    
    -- Timestamps
    opened_at TIMESTAMPTZ,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT positive_quantity CHECK (quantity >= 0 OR quantity <= 0)  -- Allow shorts
);

CREATE INDEX idx_positions_underlying ON positions(underlying);
CREATE INDEX idx_positions_expiry ON positions(expiry);

-- =============================================================================
-- GREEKS HISTORY (Time Series)
-- =============================================================================

CREATE TABLE IF NOT EXISTS greeks_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol VARCHAR(50) NOT NULL,
    underlying VARCHAR(20) NOT NULL,
    
    -- Greeks
    delta DECIMAL(10, 6) NOT NULL,
    gamma DECIMAL(10, 6),
    theta DECIMAL(10, 6),
    vega DECIMAL(10, 6),
    rho DECIMAL(10, 6),
    
    -- Market conditions
    spot_price DECIMAL(12, 4),
    implied_vol DECIMAL(8, 6),
    time_to_expiry DECIMAL(10, 6),
    
    -- Performance
    calculation_time_us INTEGER,
    calculation_method VARCHAR(50),  -- 'ultra_fast', 'ensemble', 'black_scholes'
    
    -- Quality
    accuracy_vs_bs DECIMAL(8, 6)  -- Accuracy vs Black-Scholes
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions
CREATE TABLE greeks_history_2024_11 PARTITION OF greeks_history
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

CREATE INDEX idx_greeks_history_timestamp ON greeks_history(timestamp DESC);
CREATE INDEX idx_greeks_history_symbol ON greeks_history(symbol);

-- =============================================================================
-- VOLATILITY SURFACES
-- =============================================================================

CREATE TABLE IF NOT EXISTS volatility_surfaces (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    underlying VARCHAR(20) NOT NULL,
    
    -- Surface data (stored as JSON for flexibility)
    surface_data JSONB NOT NULL,  -- 2D array of implied vols
    strikes JSONB NOT NULL,  -- Array of strike prices
    maturities JSONB NOT NULL,  -- Array of time to maturity
    
    -- Construction metadata
    construction_method VARCHAR(20) NOT NULL,  -- 'GAN', 'SABR', 'interpolation'
    construction_time_ms DECIMAL(10, 3),
    market_quotes_used JSONB,  -- Input quotes
    
    -- Quality metrics
    arbitrage_free BOOLEAN DEFAULT TRUE,
    rmse DECIMAL(10, 6),  -- Root mean square error vs market
    
    -- Calibration (for SABR)
    sabr_params JSONB
);

CREATE INDEX idx_vol_surfaces_timestamp ON volatility_surfaces(timestamp DESC);
CREATE INDEX idx_vol_surfaces_underlying ON volatility_surfaces(underlying);

-- =============================================================================
-- P&L TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS pnl_tracking (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- P&L breakdown
    realized_pnl DECIMAL(18, 2) NOT NULL DEFAULT 0,
    unrealized_pnl DECIMAL(18, 2) NOT NULL DEFAULT 0,
    total_pnl DECIMAL(18, 2) NOT NULL DEFAULT 0,
    
    -- By strategy
    strategy VARCHAR(100),
    
    -- Greeks summary
    total_delta DECIMAL(12, 4),
    total_gamma DECIMAL(12, 4),
    total_vega DECIMAL(12, 4),
    total_theta DECIMAL(12, 4),
    
    -- Position summary
    positions_count INTEGER,
    notional_exposure DECIMAL(18, 2),
    
    -- Risk metrics
    var_1day DECIMAL(18, 2),
    max_drawdown DECIMAL(18, 2),
    sharpe_ratio DECIMAL(8, 4)
);

CREATE INDEX idx_pnl_timestamp ON pnl_tracking(timestamp DESC);
CREATE INDEX idx_pnl_strategy ON pnl_tracking(strategy);

-- =============================================================================
-- PERFORMANCE METRICS (System monitoring)
-- =============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(18, 6) NOT NULL,
    metric_unit VARCHAR(20),  -- 'microseconds', 'milliseconds', 'percent', etc
    
    -- Context
    component VARCHAR(50),  -- 'greeks', 'exotic', 'surface', etc
    additional_data JSONB
);

CREATE INDEX idx_perf_metrics_timestamp ON performance_metrics(timestamp DESC);
CREATE INDEX idx_perf_metrics_name ON performance_metrics(metric_name);

-- =============================================================================
-- MARKET DATA CACHE (Materialized for speed)
-- =============================================================================

CREATE MATERIALIZED VIEW option_chain_current AS
SELECT 
    symbol,
    underlying,
    strike,
    expiry,
    option_type,
    bid,
    ask,
    last_price,
    implied_vol,
    delta,
    gamma,
    volume,
    open_interest,
    MAX(timestamp) as last_update
FROM market_data_feed
GROUP BY symbol, underlying, strike, expiry, option_type,
         bid, ask, last_price, implied_vol, delta, gamma,
         volume, open_interest;

CREATE UNIQUE INDEX ON option_chain_current(symbol);

-- Refresh every minute
CREATE OR REPLACE FUNCTION refresh_option_chain()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY option_chain_current;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Calculate portfolio Greeks
CREATE OR REPLACE FUNCTION calculate_portfolio_greeks()
RETURNS TABLE(
    total_delta DECIMAL(12, 4),
    total_gamma DECIMAL(12, 4),
    total_vega DECIMAL(12, 4),
    total_theta DECIMAL(12, 4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(quantity * delta), 0) as total_delta,
        COALESCE(SUM(quantity * gamma), 0) as total_gamma,
        COALESCE(SUM(quantity * vega), 0) as total_vega,
        COALESCE(SUM(quantity * theta), 0) as total_theta
    FROM positions
    WHERE quantity <> 0;
END;
$$ LANGUAGE plpgsql;

-- Get current P&L
CREATE OR REPLACE FUNCTION get_current_pnl()
RETURNS DECIMAL(18, 2) AS $$
DECLARE
    total DECIMAL(18, 2);
BEGIN
    SELECT COALESCE(SUM(unrealized_pnl), 0) INTO total
    FROM positions;
    
    RETURN total;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Create default strategies
INSERT INTO pnl_tracking (strategy, realized_pnl, unrealized_pnl, total_pnl, positions_count)
VALUES 
    ('delta_neutral', 0, 0, 0, 0),
    ('market_making', 0, 0, 0, 0),
    ('volatility_arbitrage', 0, 0, 0, 0)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- GRANTS (Security)
-- =============================================================================

-- Application user (read/write)
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO axiom_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO axiom_app;

-- Read-only user (analytics)
GRANT SELECT ON ALL TABLES IN SCHEMA public TO axiom_readonly;

-- =============================================================================
-- MAINTENANCE
-- =============================================================================

-- Auto-vacuum for performance
ALTER TABLE option_trades SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE greeks_history SET (autovacuum_vacuum_scale_factor = 0.02);

-- Retention policy (keep 2 years)
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    DELETE FROM greeks_history 
    WHERE timestamp < NOW() - INTERVAL '2 years';
    
    DELETE FROM performance_metrics
    WHERE timestamp < NOW() - INTERVAL '6 months';
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup (run monthly)
-- In production: Use pg_cron or external scheduler

COMMENT ON TABLE option_trades IS 'All options trades executed by the platform';
COMMENT ON TABLE positions IS 'Current options positions';
COMMENT ON TABLE greeks_history IS 'Historical Greeks calculations (partitioned by month)';
COMMENT ON TABLE volatility_surfaces IS 'Implied volatility surfaces over time';
COMMENT ON TABLE pnl_tracking IS 'Daily P&L tracking by strategy';
COMMENT ON TABLE performance_metrics IS 'System performance metrics';