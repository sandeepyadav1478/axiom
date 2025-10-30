"""
MLOps Experiment Tracking Integration

Integrates MLflow for experiment tracking, model registry, and artifact management.
This module provides a unified interface for tracking all ML experiments across
the platform, leveraging open-source MLflow instead of building custom solutions.

Benefits:
- Automatic experiment logging
- Model versioning and registry
- Hyperparameter tracking
- Artifact storage
- Comparison and visualization
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import json

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class AxiomMLflowTracker:
    """
    Unified MLflow tracking for Axiom models
    
    Usage:
        tracker = AxiomMLflowTracker(experiment_name="portfolio_optimization")
        
        with tracker.start_run(run_name="rl_ppo_v1"):
            tracker.log_params(config.__dict__)
            
            for epoch in range(epochs):
                tracker.log_metrics({"loss": loss}, step=epoch)
            
            tracker.log_model(model, "rl_portfolio_manager")
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow required: pip install mlflow")
        
        self.experiment_name = experiment_name
        
        # Set tracking URI (default: local ./mlruns)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location
            )
        except:
            self.experiment = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start MLflow run
        
        Use as context manager:
            with tracker.start_run("model_v1"):
                # training code
        """
        return mlflow.start_run(run_name=run_name, tags=tags)
        
    def log_params(self, params: Dict[str, Any]):
        """Log parameters (hyperparameters, config)"""
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics (loss, accuracy, Sharpe ratio, etc.)"""
        mlflow.log_metrics(metrics, step=step)
        
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log model to MLflow
        
        Args:
            model: PyTorch model or any sklearn-compatible model
            artifact_path: Path within run artifacts
            registered_model_name: Register in model registry
        """
        # Detect model type and log appropriately
        if hasattr(model, 'state_dict'):  # PyTorch
            mlflow.pytorch.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        elif hasattr(model, 'get_params'):  # Sklearn-like
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        else:
            # Generic pickle
            import cloudpickle
            mlflow.log_artifact(artifact_path)
            
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (plots, reports, etc.)"""
        mlflow.log_artifact(local_path, artifact_path)
        
    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib/plotly figure"""
        mlflow.log_figure(figure, artifact_file)
        
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(AxiomMLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to loggable type
                if isinstance(v, (int, float, str, bool)):
                    items.append((new_key, v))
                else:
                    items.append((new_key, str(v)))
        return dict(items)


# Convenience decorators
def mlflow_track(experiment_name: str):
    """
    Decorator to automatically track function execution
    
    Usage:
        @mlflow_track("portfolio_training")
        def train_model(config):
            # training code
            return metrics
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = AxiomMLflowTracker(experiment_name)
            with tracker.start_run(run_name=func.__name__):
                # Log function args as params
                tracker.log_params(kwargs)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log results if dict
                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    if metrics:
                        tracker.log_metrics(metrics)
                
                return result
        return wrapper
    return decorator


# Example integration with our models
def train_portfolio_transformer_with_mlflow(
    model,
    config,
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Example: Training Portfolio Transformer with MLflow tracking
    
    This shows how to integrate our custom models with MLflow.
    """
    tracker = AxiomMLflowTracker("portfolio_transformer")
    
    with tracker.start_run("transformer_v1", tags={"model": "portfolio", "version": "1.0"}):
        # Log hyperparameters
        tracker.log_params({
            "d_model": config.d_model,
            "nhead": config.nhead,
            "learning_rate": config.learning_rate,
            "n_assets": config.n_assets,
            "lookback_window": config.lookback_window
        })
        
        # Training loop
        for epoch in range(config.epochs):
            # Train
            loss, sharpe = train_one_epoch(model, X_train, y_train)
            
            # Log metrics
            tracker.log_metrics({
                "train_loss": loss,
                "train_sharpe": sharpe
            }, step=epoch)
            
            # Validation
            val_loss, val_sharpe = validate(model, X_val, y_val)
            tracker.log_metrics({
                "val_loss": val_loss,
                "val_sharpe": val_sharpe
            }, step=epoch)
        
        # Log final model
        tracker.log_model(
            model,
            "model",
            registered_model_name="PortfolioTransformer"
        )
        
        # Log performance plots
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(losses)
        tracker.log_figure(fig, "training_curve.png")
        
        return {"final_sharpe": val_sharpe}


if __name__ == "__main__":
    print("MLflow Experiment Tracking - Example")
    
    if not MLFLOW_AVAILABLE:
        print("Install: pip install mlflow")
    else:
        # Initialize tracker
        tracker = AxiomMLflowTracker("test_experiment")
        
        # Example run
        with tracker.start_run("example_run"):
            tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
            tracker.log_metrics({"accuracy": 0.95, "loss": 0.05})
            print("✓ Logged to MLflow")
            print("View: mlflow ui")