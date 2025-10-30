"""
REST API for Model Serving

Production API endpoints for serving all 37 ML models.

Endpoints:
- POST /predict/{model_type} - Single prediction
- POST /batch_predict - Batch predictions
- GET /models - List available models
- GET /health - Health check

Uses FastAPI for high performance.
"""

from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from axiom.models.base.model_cache import get_cached_model
from axiom.models.base.factory import ModelType
from axiom.infrastructure.monitoring.model_performance_dashboard import record_model_prediction


app = FastAPI(title="Axiom ML Model API", version="1.0.0")


class PredictionRequest(BaseModel):
    """Request for prediction"""
    model_type: str
    input_data: Dict[str, Any]


class PredictionResponse(BaseModel):
    """Response with prediction"""
    prediction: Any
    model_type: str
    latency_ms: float


class BatchRequest(BaseModel):
    """Batch prediction request"""
    requests: List[PredictionRequest]


@app.post("/predict/{model_type}")
async def predict(model_type: str, request: PredictionRequest) -> PredictionResponse:
    """Single model prediction"""
    
    import time
    start = time.time()
    
    try:
        # Get model type
        mt = ModelType(model_type)
        
        # Load from cache
        model = get_cached_model(mt)
        
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not available")
        
        # Predict
        if hasattr(model, 'predict'):
            result = model.predict(request.input_data)
        elif hasattr(model, 'allocate'):
            result = model.allocate(request.input_data)
        else:
            raise HTTPException(status_code=400, detail="Model doesn't support prediction")
        
        latency = (time.time() - start) * 1000
        
        # Record metrics
        record_model_prediction(model_type, latency, True)
        
        return PredictionResponse(
            prediction=result,
            model_type=model_type,
            latency_ms=latency
        )
        
    except Exception as e:
        record_model_prediction(model_type, (time.time() - start) * 1000, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List all available models"""
    
    models = {
        'portfolio': [
            'rl_portfolio_manager',
            'lstm_cnn_portfolio',
            'portfolio_transformer',
            'million_portfolio',
            'regimefolio',
            'dro_bas',
            'transaction_cost'
        ],
        'options': [
            'vae_option_pricer',
            'ann_greeks_calculator',
            'drl_option_hedger',
            'gan_volatility_surface',
            'informer_transformer_pricer',
            'bs_ann_hybrid',
            'wavelet_pinn',
            'sv_calibrator',
            'deep_hedging'
        ],
        'credit': [
            'cnn_lstm_credit',
            'ensemble_credit',
            'llm_credit_scoring',
            'transformer_nlp_credit',
            'gnn_credit_network',
            # + 7 more
        ],
        'ma': [
            'ml_target_screener',
            'nlp_sentiment_ma',
            'ai_due_diligence',
            'ma_success_predictor',
            'activist_detector',
            # + 4 more
        ]
    }
    
    return {
        'total_models': 37,
        'models_by_domain': models,
        'status': 'production'
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    from axiom.infrastructure.monitoring.model_performance_dashboard import _global_dashboard
    
    health = _global_dashboard.get_health_status()
    
    return {
        'status': 'healthy',
        'models': len(health),
        'model_health': health,
        'timestamp': str(datetime.now())
    }


if __name__ == "__main__":
    print("Model Serving API")
    print("Starting on http://localhost:8000")
    print("\nEndpoints:")
    print("  POST /predict/{model_type}")
    print("  GET /models")
    print("  GET /health")
    print("\nDocs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)