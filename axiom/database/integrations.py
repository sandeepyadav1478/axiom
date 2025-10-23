"""
Data integration layer for existing models.

Seamlessly integrates:
- VaR calculations → PostgreSQL
- Portfolio optimizations → PostgreSQL
- Company research → Vector DB
- Performance metrics → PostgreSQL
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal

import numpy as np
import pandas as pd

from .models import (
    VaRCalculation,
    PortfolioOptimization,
    PerformanceMetric,
    PortfolioPosition,
    Trade,
    PriceData,
    CompanyFundamental,
    DocumentEmbedding,
    VaRMethodType,
    OptimizationMethodType,
    TradeType,
    OrderType,
)
from .session import SessionManager
from .vector_store import get_vector_store
from ..models.risk.var_models import VaRResult, VaRMethod
from ..models.portfolio.optimization import OptimizationResult, OptimizationMethod, PortfolioMetrics

logger = logging.getLogger(__name__)


class VaRIntegration:
    """
    Integration layer for VaR calculations.
    
    Automatically stores VaR results in PostgreSQL for:
    - Historical tracking
    - Compliance reporting
    - Risk monitoring
    - Backtesting
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize VaR integration."""
        self.session = session_manager or SessionManager()
    
    def store_var_result(
        self,
        portfolio_id: str,
        var_result: VaRResult,
        position_contributions: Optional[Dict[str, float]] = None
    ) -> VaRCalculation:
        """
        Store VaR calculation result in database.
        
        Args:
            portfolio_id: Portfolio identifier
            var_result: VaR calculation result
            position_contributions: Per-position VaR contributions
            
        Returns:
            Stored VaRCalculation object
        """
        # Map VaR method enum
        method_mapping = {
            VaRMethod.PARAMETRIC: VaRMethodType.PARAMETRIC,
            VaRMethod.HISTORICAL: VaRMethodType.HISTORICAL,
            VaRMethod.MONTE_CARLO: VaRMethodType.MONTE_CARLO,
        }
        
        var_calc = VaRCalculation(
            portfolio_id=portfolio_id,
            calculation_date=datetime.fromisoformat(var_result.calculation_timestamp),
            method=method_mapping[var_result.method],
            confidence_level=var_result.confidence_level,
            time_horizon_days=var_result.time_horizon_days,
            var_amount=Decimal(str(var_result.var_amount)),
            var_percentage=var_result.var_percentage,
            expected_shortfall=Decimal(str(var_result.expected_shortfall)) if var_result.expected_shortfall else None,
            portfolio_value=Decimal(str(var_result.portfolio_value)),
            parameters=var_result.metadata,
            position_contributions=position_contributions,
        )
        
        self.session.add(var_calc)
        self.session.commit()
        
        logger.info(f"Stored VaR calculation for portfolio {portfolio_id}: ${var_result.var_amount:,.2f}")
        
        return var_calc
    
    def get_var_history(
        self,
        portfolio_id: str,
        days: int = 30,
        method: Optional[VaRMethodType] = None
    ) -> List[VaRCalculation]:
        """
        Get VaR calculation history.
        
        Args:
            portfolio_id: Portfolio identifier
            days: Number of days to look back
            method: Filter by specific method
            
        Returns:
            List of VaR calculations
        """
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = self.session.query(VaRCalculation).filter(
            VaRCalculation.portfolio_id == portfolio_id,
            VaRCalculation.calculation_date >= cutoff_date
        )
        
        if method:
            query = query.filter(VaRCalculation.method == method)
        
        return query.order_by(VaRCalculation.calculation_date.desc()).all()
    
    def get_latest_var(
        self,
        portfolio_id: str,
        method: Optional[VaRMethodType] = None
    ) -> Optional[VaRCalculation]:
        """Get latest VaR calculation."""
        query = self.session.query(VaRCalculation).filter(
            VaRCalculation.portfolio_id == portfolio_id
        )
        
        if method:
            query = query.filter(VaRCalculation.method == method)
        
        return query.order_by(VaRCalculation.calculation_date.desc()).first()


class PortfolioIntegration:
    """
    Integration layer for portfolio optimization.
    
    Stores optimization results for:
    - Performance tracking
    - Rebalancing decisions
    - Historical analysis
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize portfolio integration."""
        self.session = session_manager or SessionManager()
    
    def store_optimization_result(
        self,
        portfolio_id: str,
        opt_result: OptimizationResult,
        constraints: Optional[Dict[str, Any]] = None,
        bounds: Optional[Dict[str, Any]] = None
    ) -> PortfolioOptimization:
        """
        Store portfolio optimization result.
        
        Args:
            portfolio_id: Portfolio identifier
            opt_result: Optimization result
            constraints: Constraints used
            bounds: Weight bounds used
            
        Returns:
            Stored PortfolioOptimization object
        """
        # Map optimization method enum
        method_mapping = {
            OptimizationMethod.MAX_SHARPE: OptimizationMethodType.MAX_SHARPE,
            OptimizationMethod.MIN_VOLATILITY: OptimizationMethodType.MIN_VOLATILITY,
            OptimizationMethod.MAX_RETURN: OptimizationMethodType.MAX_RETURN,
            OptimizationMethod.EFFICIENT_RETURN: OptimizationMethodType.EFFICIENT_RETURN,
            OptimizationMethod.EFFICIENT_RISK: OptimizationMethodType.EFFICIENT_RISK,
            OptimizationMethod.RISK_PARITY: OptimizationMethodType.RISK_PARITY,
            OptimizationMethod.MIN_CVaR: OptimizationMethodType.MIN_CVAR,
        }
        
        portfolio_opt = PortfolioOptimization(
            portfolio_id=portfolio_id,
            optimization_date=datetime.fromisoformat(opt_result.timestamp),
            method=method_mapping[opt_result.method],
            optimal_weights=opt_result.get_weights_dict(),
            expected_return=opt_result.metrics.expected_return,
            expected_volatility=opt_result.metrics.volatility,
            expected_sharpe=opt_result.metrics.sharpe_ratio,
            constraints=constraints,
            bounds=bounds,
            success=opt_result.success,
            message=opt_result.message,
            computation_time=opt_result.computation_time,
        )
        
        self.session.add(portfolio_opt)
        self.session.commit()
        
        logger.info(f"Stored optimization for portfolio {portfolio_id}: Sharpe={opt_result.metrics.sharpe_ratio:.3f}")
        
        return portfolio_opt
    
    def store_performance_metrics(
        self,
        portfolio_id: str,
        metrics: PortfolioMetrics,
        portfolio_value: Decimal,
        num_positions: int,
        metric_date: Optional[datetime] = None
    ) -> PerformanceMetric:
        """
        Store portfolio performance metrics.
        
        Args:
            portfolio_id: Portfolio identifier
            metrics: Portfolio metrics
            portfolio_value: Current portfolio value
            num_positions: Number of positions
            metric_date: Metric date (default: now)
            
        Returns:
            Stored PerformanceMetric object
        """
        perf_metric = PerformanceMetric(
            portfolio_id=portfolio_id,
            metric_date=metric_date or datetime.now(),
            annualized_return=metrics.expected_return,
            volatility=metrics.volatility,
            max_drawdown=metrics.max_drawdown,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            calmar_ratio=metrics.calmar_ratio,
            beta=metrics.beta,
            alpha=metrics.alpha,
            information_ratio=metrics.information_ratio,
            treynor_ratio=metrics.treynor_ratio,
            portfolio_value=portfolio_value,
            num_positions=num_positions,
        )
        
        self.session.add(perf_metric)
        self.session.commit()
        
        logger.info(f"Stored performance metrics for portfolio {portfolio_id}")
        
        return perf_metric
    
    def store_position(
        self,
        portfolio_id: str,
        symbol: str,
        quantity: Decimal,
        avg_cost: Decimal,
        current_price: Optional[Decimal] = None
    ) -> PortfolioPosition:
        """
        Store or update portfolio position.
        
        Args:
            portfolio_id: Portfolio identifier
            symbol: Asset symbol
            quantity: Position quantity
            avg_cost: Average cost basis
            current_price: Current market price
            
        Returns:
            Stored PortfolioPosition object
        """
        # Check if position exists
        existing = self.session.query(PortfolioPosition).filter(
            PortfolioPosition.portfolio_id == portfolio_id,
            PortfolioPosition.symbol == symbol,
            PortfolioPosition.is_active == True
        ).first()
        
        if existing:
            # Update existing position
            existing.quantity = quantity
            existing.avg_cost = avg_cost
            existing.current_price = current_price
            existing.last_updated = datetime.now()
            
            if current_price:
                position_value = quantity * current_price
                existing.position_value = position_value
                existing.unrealized_pnl = position_value - (quantity * avg_cost)
            
            position = existing
        else:
            # Create new position
            position = PortfolioPosition(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=quantity,
                avg_cost=avg_cost,
                current_price=current_price,
                first_acquired=datetime.now(),
                is_active=True
            )
            
            if current_price:
                position.position_value = quantity * current_price
                position.unrealized_pnl = position.position_value - (quantity * avg_cost)
            
            self.session.add(position)
        
        self.session.commit()
        
        logger.info(f"Stored position for {portfolio_id}/{symbol}: {quantity}")
        
        return position
    
    def store_trade(
        self,
        portfolio_id: str,
        symbol: str,
        trade_type: TradeType,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal('0'),
        executed_at: Optional[datetime] = None
    ) -> Trade:
        """
        Store trade execution.
        
        Args:
            portfolio_id: Portfolio identifier
            symbol: Asset symbol
            trade_type: Buy/Sell/Short/Cover
            order_type: Market/Limit/Stop
            quantity: Trade quantity
            price: Execution price
            commission: Trading commission
            executed_at: Execution timestamp
            
        Returns:
            Stored Trade object
        """
        total_cost = (quantity * price) + commission
        
        trade = Trade(
            portfolio_id=portfolio_id,
            symbol=symbol,
            trade_type=trade_type,
            order_type=order_type,
            quantity=quantity,
            price=price,
            commission=commission,
            total_cost=total_cost,
            executed_at=executed_at or datetime.now(),
        )
        
        self.session.add(trade)
        self.session.commit()
        
        logger.info(f"Stored trade for {portfolio_id}: {trade_type.value} {quantity} {symbol} @ {price}")
        
        return trade


class MarketDataIntegration:
    """
    Integration layer for market data.
    
    Stores price data and fundamentals for:
    - Historical analysis
    - Backtesting
    - Research
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize market data integration."""
        self.session = session_manager or SessionManager()
    
    def store_price_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str = "unknown"
    ) -> int:
        """
        Store OHLCV price data from DataFrame.
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Asset symbol
            source: Data source
            
        Returns:
            Number of records stored
        """
        from .models import TimeFrame
        
        records = []
        for idx, row in df.iterrows():
            price_data = PriceData(
                symbol=symbol,
                timestamp=idx if isinstance(idx, datetime) else datetime.fromisoformat(str(idx)),
                timeframe=TimeFrame.DAY_1,
                open=Decimal(str(row['Open'])),
                high=Decimal(str(row['High'])),
                low=Decimal(str(row['Low'])),
                close=Decimal(str(row['Close'])),
                volume=Decimal(str(row['Volume'])),
                adj_close=Decimal(str(row.get('Adj Close', row['Close']))),
                source=source,
            )
            records.append(price_data)
        
        # Bulk insert
        self.session.bulk_insert(records)
        self.session.commit()
        
        logger.info(f"Stored {len(records)} price records for {symbol}")
        
        return len(records)
    
    def store_fundamentals(
        self,
        symbol: str,
        data: Dict[str, Any],
        report_date: datetime,
        source: str = "unknown"
    ) -> CompanyFundamental:
        """
        Store company fundamental data.
        
        Args:
            symbol: Company symbol
            data: Fundamental data dictionary
            report_date: Report date
            source: Data source
            
        Returns:
            Stored CompanyFundamental object
        """
        fundamental = CompanyFundamental(
            symbol=symbol,
            report_date=report_date,
            company_name=data.get('company_name'),
            sector=data.get('sector'),
            industry=data.get('industry'),
            market_cap=Decimal(str(data['market_cap'])) if data.get('market_cap') else None,
            revenue=Decimal(str(data['revenue'])) if data.get('revenue') else None,
            net_income=Decimal(str(data['net_income'])) if data.get('net_income') else None,
            eps=Decimal(str(data['eps'])) if data.get('eps') else None,
            pe_ratio=data.get('pe_ratio'),
            source=source,
            metadata_json=data,
        )
        
        self.session.add(fundamental)
        self.session.commit()
        
        logger.info(f"Stored fundamentals for {symbol}")
        
        return fundamental


class VectorIntegration:
    """
    Integration layer for vector embeddings.
    
    Manages:
    - Document embeddings
    - Company embeddings
    - Semantic search
    - RAG support
    """
    
    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        vector_store_type: Optional[str] = None
    ):
        """Initialize vector integration."""
        self.session = session_manager or SessionManager()
        self.vector_store = get_vector_store(vector_store_type)
    
    def store_document_embedding(
        self,
        document_id: str,
        document_type: str,
        content: str,
        embedding: np.ndarray,
        embedding_model: str,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentEmbedding:
        """
        Store document embedding in both PostgreSQL and Vector DB.
        
        Args:
            document_id: Unique document ID
            document_type: Document type ('sec_filing', 'research', 'news')
            content: Document content
            embedding: Document embedding vector
            embedding_model: Model used for embedding
            symbol: Associated company symbol
            metadata: Additional metadata
            
        Returns:
            Stored DocumentEmbedding object
        """
        import hashlib
        
        # Create content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if document exists
        existing = self.session.query(DocumentEmbedding).filter(
            DocumentEmbedding.content_hash == content_hash
        ).first()
        
        if existing:
            logger.info(f"Document {document_id} already exists")
            return existing
        
        # Store in PostgreSQL
        doc_embedding = DocumentEmbedding(
            document_id=document_id,
            document_type=document_type,
            symbol=symbol,
            content=content,
            content_hash=content_hash,
            embedding_model=embedding_model,
            embedding_dim=len(embedding),
            title=metadata.get('title') if metadata else None,
            source_url=metadata.get('source_url') if metadata else None,
            keywords=metadata.get('keywords') if metadata else None,
            summary=metadata.get('summary') if metadata else None,
        )
        
        self.session.add(doc_embedding)
        self.session.commit()
        
        # Store in Vector DB
        collection_name = f"documents_{document_type}"
        
        # Ensure collection exists
        try:
            self.vector_store.create_collection(
                name=collection_name,
                dimension=len(embedding)
            )
        except:
            pass  # Collection may already exist
        
        # Upsert to vector store
        vector_metadata = {
            'document_id': document_id,
            'symbol': symbol,
            'content': content[:1000],  # Store snippet
            **(metadata or {})
        }
        
        self.vector_store.upsert(
            collection_name=collection_name,
            vectors=[embedding],
            ids=[document_id],
            metadata=[vector_metadata]
        )
        
        # Update sync status
        doc_embedding.vector_db_synced = True
        doc_embedding.last_synced_at = datetime.now()
        doc_embedding.vector_db_id = document_id
        self.session.commit()
        
        logger.info(f"Stored document embedding: {document_id}")
        
        return doc_embedding
    
    def search_documents(
        self,
        query_embedding: np.ndarray,
        document_type: str,
        top_k: int = 10,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            document_type: Document type to search
            top_k: Number of results
            symbol: Filter by company symbol
            
        Returns:
            List of similar documents with scores
        """
        collection_name = f"documents_{document_type}"
        
        # Build filter
        filter_dict = {}
        if symbol:
            filter_dict['symbol'] = symbol
        
        # Search vector store
        results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter_dict if filter_dict else None
        )
        
        logger.info(f"Found {len(results)} similar documents")
        
        return results


# Export
__all__ = [
    "VaRIntegration",
    "PortfolioIntegration",
    "MarketDataIntegration",
    "VectorIntegration",
]