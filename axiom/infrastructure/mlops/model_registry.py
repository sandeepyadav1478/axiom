"""
Model Registry for Production ML

Leverages MLflow Model Registry for model versioning and lifecycle management.

MLflow Registry provides:
- Model versioning
- Stage transitions (Staging → Production)
- Model lineage tracking
- Annotations and descriptions
- Access control

Used by: Databricks, Netflix, and thousands of companies.
We leverage it instead of building custom registry.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ModelStage(Enum):
    """Model lifecycle stages"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class AxiomModelRegistry:
    """
    Unified model registry using MLflow
    
    Manages lifecycle of all 19+ ML models in production.
    
    Usage:
        registry = AxiomModelRegistry()
        
        # Register model
        registry.register_model(
            model_name="portfolio_transformer",
            model_uri="runs:/abc123/model",
            description="Portfolio Transformer v1.0"
        )
        
        # Promote to production
        registry.transition_stage(
            model_name="portfolio_transformer",
            version=1,
            stage="Production"
        )
        
        # Load production model
        model = registry.load_production_model("portfolio_transformer")
    """
    
    def __init__(self, tracking_uri: Optional[str] = None):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow required: pip install mlflow")
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
    
    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Register model in registry
        
        Args:
            model_name: Name for model (e.g., 'portfolio_transformer')
            model_uri: URI to model artifact (e.g., 'runs:/run_id/model')
            description: Model description
            tags: Metadata tags
            
        Returns:
            Version number
        """
        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Add description
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=result.version,
                    description=description
                )
            
            return result.version
            
        except Exception as e:
            print(f"Model registration failed: {e}")
            return -1
    
    def transition_stage(
        self,
        model_name: str,
        version: int,
        stage: str,  # Staging, Production, Archived
        archive_existing: bool = True
    ):
        """
        Transition model to new stage
        
        Args:
            model_name: Model name
            version: Model version
            stage: Target stage
            archive_existing: Archive existing production models
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
    
    def load_production_model(self, model_name: str):
        """
        Load production version of model
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded model
        """
        model_uri = f"models:/{model_name}/Production"
        
        try:
            # Try PyTorch first
            model = mlflow.pytorch.load_model(model_uri)
            return model
        except:
            try:
                # Try sklearn
                model = mlflow.sklearn.load_model(model_uri)
                return model
            except:
                # Generic pyfunc
                model = mlflow.pyfunc.load_model(model_uri)
                return model
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        models = self.client.search_registered_models()
        
        return [
            {
                'name': model.name,
                'latest_version': model.latest_versions[0].version if model.latest_versions else None,
                'stage': model.latest_versions[0].current_stage if model.latest_versions else None,
                'description': model.description
            }
            for model in models
        ]
    
    def get_model_info(self, model_name: str, version: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed model information"""
        if version:
            model_version = self.client.get_model_version(model_name, version)
        else:
            # Get latest
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if not versions:
                return {}
            model_version = versions[0]
        
        return {
            'name': model_version.name,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'description': model_version.description,
            'run_id': model_version.run_id,
            'created_at': model_version.creation_timestamp,
            'tags': model_version.tags
        }


# Quick example
if __name__ == "__main__":
    print("Model Registry using MLflow")
    print("=" * 60)
    
    if not MLFLOW_AVAILABLE:
        print("Install: pip install mlflow")
    else:
        print("Model Registry Features:")
        print("  • Version control for models")
        print("  • Stage management (Staging/Production)")
        print("  • Model lineage tracking")
        print("  • Access control")
        
        print("\nLeveraging MLflow Model Registry")
        print("Used by: Databricks, Netflix, thousands of companies")
        print("\nWe integrate instead of building custom.")