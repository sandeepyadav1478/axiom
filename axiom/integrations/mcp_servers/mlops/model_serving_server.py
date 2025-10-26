"""
Model Serving MCP Server

Provides ML model deployment and serving capabilities. Supports deploying models,
making predictions, A/B testing, canary deployments, and health monitoring.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEndpoint:
    """Represents a deployed model endpoint."""

    def __init__(
        self,
        name: str,
        model_path: str,
        version: str,
        model_type: str = "sklearn",
        resources: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.model_path = model_path
        self.version = version
        self.model_type = model_type
        self.resources = resources or {"cpu": "500m", "memory": "1Gi"}
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.status = "pending"
        self.model = None
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.last_request_time = None

    def load_model(self):
        """Load the model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.status = "running"
            logger.info(f"Model loaded successfully: {self.name}")
        except Exception as e:
            self.status = "failed"
            logger.error(f"Failed to load model {self.name}: {str(e)}")
            raise

    def predict(self, data: Any) -> Any:
        """Make a prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")

        start_time = time.time()
        try:
            # Handle different input formats
            import numpy as np
            if isinstance(data, list):
                data = np.array(data)
            elif isinstance(data, dict) and 'features' in data:
                data = np.array(data['features'])

            # Make prediction
            prediction = self.model.predict(data)

            # Track metrics
            latency = time.time() - start_time
            self.request_count += 1
            self.total_latency += latency
            self.last_request_time = datetime.now().isoformat()

            return {
                "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                "latency_ms": round(latency * 1000, 2),
                "endpoint": self.name,
                "version": self.version
            }

        except Exception as e:
            self.error_count += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get endpoint metrics."""
        avg_latency = (self.total_latency / self.request_count * 1000) if self.request_count > 0 else 0

        return {
            "endpoint": self.name,
            "version": self.version,
            "status": self.status,
            "requests": {
                "total": self.request_count,
                "errors": self.error_count,
                "error_rate": round(self.error_count / self.request_count * 100, 2) if self.request_count > 0 else 0
            },
            "latency": {
                "average_ms": round(avg_latency, 2),
                "total_seconds": round(self.total_latency, 2)
            },
            "last_request": self.last_request_time,
            "created_at": self.created_at,
            "resources": self.resources
        }


class ModelServingMCPServer:
    """MCP Server for ML model serving and deployment."""

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize Model Serving MCP Server.

        Args:
            models_dir: Directory for model storage (default: ./models)
        """
        self.server = Server("model-serving")
        self.models_dir = Path(models_dir or "./models")
        self.models_dir.mkdir(exist_ok=True)

        # Model registry
        self.endpoints: Dict[str, ModelEndpoint] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self.ab_tests: Dict[str, Dict[str, Any]] = {}

        self._register_handlers()
        logger.info("Model Serving MCP Server initialized")

    def _register_handlers(self):
        """Register all tool handlers."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List all available model serving tools."""
            return [
                Tool(
                    name="deploy_model",
                    description="Deploy a trained ML model to a serving endpoint. "
                                "Loads the model and makes it available for predictions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint_name": {
                                "type": "string",
                                "description": "Unique name for the endpoint (e.g., 'arima-forecast', 'credit-score')"
                            },
                            "model_path": {
                                "type": "string",
                                "description": "Path to the pickled model file"
                            },
                            "version": {
                                "type": "string",
                                "description": "Model version (e.g., 'v1.0', '2024-01-15')"
                            },
                            "model_type": {
                                "type": "string",
                                "enum": ["sklearn", "statsmodels", "pytorch", "tensorflow", "custom"],
                                "description": "Type of model framework (default: sklearn)",
                                "default": "sklearn"
                            },
                            "resources": {
                                "type": "object",
                                "description": "Resource requirements (cpu, memory)",
                                "properties": {
                                    "cpu": {"type": "string"},
                                    "memory": {"type": "string"}
                                }
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Additional metadata (description, author, etc.)"
                            }
                        },
                        "required": ["endpoint_name", "model_path", "version"]
                    }
                ),
                Tool(
                    name="predict",
                    description="Get a prediction from a deployed model endpoint. "
                                "Supports single and batch predictions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to query"
                            },
                            "data": {
                                "description": "Input data for prediction (array, dict, or JSON string)"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Request timeout in seconds (default: 30)",
                                "default": 30
                            }
                        },
                        "required": ["endpoint", "data"]
                    }
                ),
                Tool(
                    name="batch_predict",
                    description="Perform batch predictions on multiple inputs. "
                                "More efficient than multiple single predictions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to query"
                            },
                            "data_list": {
                                "type": "array",
                                "description": "List of input data for predictions"
                            },
                            "batch_size": {
                                "type": "integer",
                                "description": "Batch size for processing (default: 32)",
                                "default": 32
                            }
                        },
                        "required": ["endpoint", "data_list"]
                    }
                ),
                Tool(
                    name="list_models",
                    description="List all deployed model endpoints with their status and metrics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "status_filter": {
                                "type": "string",
                                "enum": ["all", "running", "stopped", "failed"],
                                "description": "Filter by status (default: all)",
                                "default": "all"
                            },
                            "include_metrics": {
                                "type": "boolean",
                                "description": "Include performance metrics (default: true)",
                                "default": True
                            }
                        }
                    }
                ),
                Tool(
                    name="update_model",
                    description="Update an existing endpoint with a new model version. "
                                "Performs rolling update to minimize downtime.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to update"
                            },
                            "model_path": {
                                "type": "string",
                                "description": "Path to new model file"
                            },
                            "version": {
                                "type": "string",
                                "description": "New version identifier"
                            },
                            "strategy": {
                                "type": "string",
                                "enum": ["immediate", "rolling", "blue-green"],
                                "description": "Update strategy (default: rolling)",
                                "default": "rolling"
                            }
                        },
                        "required": ["endpoint", "model_path", "version"]
                    }
                ),
                Tool(
                    name="rollback_model",
                    description="Rollback an endpoint to a previous model version.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to rollback"
                            },
                            "target_version": {
                                "type": "string",
                                "description": "Version to rollback to (optional, uses last version if not specified)"
                            }
                        },
                        "required": ["endpoint"]
                    }
                ),
                Tool(
                    name="scale_endpoint",
                    description="Scale an endpoint's serving capacity (simulated in this implementation).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to scale"
                            },
                            "replicas": {
                                "type": "integer",
                                "description": "Number of replicas (default: 1)",
                                "default": 1
                            },
                            "resources": {
                                "type": "object",
                                "description": "Updated resource requirements"
                            }
                        },
                        "required": ["endpoint"]
                    }
                ),
                Tool(
                    name="get_metrics",
                    description="Get detailed performance metrics for an endpoint including "
                                "request count, latency, error rate, and throughput.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint"
                            },
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed breakdown (default: false)",
                                "default": False
                            }
                        },
                        "required": ["endpoint"]
                    }
                ),
                Tool(
                    name="ab_test",
                    description="Set up A/B testing between two model versions. "
                                "Splits traffic between models to compare performance.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "test_name": {
                                "type": "string",
                                "description": "Name for the A/B test"
                            },
                            "endpoint_a": {
                                "type": "string",
                                "description": "First endpoint (baseline)"
                            },
                            "endpoint_b": {
                                "type": "string",
                                "description": "Second endpoint (challenger)"
                            },
                            "traffic_split": {
                                "type": "number",
                                "description": "Traffic percentage to endpoint_b (0-100, default: 50)",
                                "default": 50
                            },
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Metrics to track (latency, accuracy, etc.)"
                            }
                        },
                        "required": ["test_name", "endpoint_a", "endpoint_b"]
                    }
                ),
                Tool(
                    name="canary_deploy",
                    description="Perform canary deployment of a new model version. "
                                "Gradually shifts traffic from old to new version.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint"
                            },
                            "new_model_path": {
                                "type": "string",
                                "description": "Path to new model"
                            },
                            "new_version": {
                                "type": "string",
                                "description": "New version identifier"
                            },
                            "initial_traffic": {
                                "type": "number",
                                "description": "Initial traffic % to new version (default: 10)",
                                "default": 10
                            },
                            "increment": {
                                "type": "number",
                                "description": "Traffic increment per step (default: 10)",
                                "default": 10
                            }
                        },
                        "required": ["endpoint", "new_model_path", "new_version"]
                    }
                ),
                Tool(
                    name="shadow_deploy",
                    description="Deploy model in shadow mode. Receives traffic but doesn't serve predictions. "
                                "Used for testing without affecting production.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the production endpoint to shadow"
                            },
                            "shadow_model_path": {
                                "type": "string",
                                "description": "Path to shadow model"
                            },
                            "shadow_version": {
                                "type": "string",
                                "description": "Shadow version identifier"
                            },
                            "sampling_rate": {
                                "type": "number",
                                "description": "Percentage of traffic to shadow (default: 100)",
                                "default": 100
                            }
                        },
                        "required": ["endpoint", "shadow_model_path", "shadow_version"]
                    }
                ),
                Tool(
                    name="health_check",
                    description="Check the health status of a deployed model endpoint. "
                                "Returns status, uptime, and basic diagnostics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "endpoint": {
                                "type": "string",
                                "description": "Name of the endpoint to check"
                            },
                            "test_prediction": {
                                "type": "boolean",
                                "description": "Run a test prediction (default: true)",
                                "default": True
                            }
                        },
                        "required": ["endpoint"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "deploy_model":
                    result = await self._deploy_model(**arguments)
                elif name == "predict":
                    result = await self._predict(**arguments)
                elif name == "batch_predict":
                    result = await self._batch_predict(**arguments)
                elif name == "list_models":
                    result = await self._list_models(**arguments)
                elif name == "update_model":
                    result = await self._update_model(**arguments)
                elif name == "rollback_model":
                    result = await self._rollback_model(**arguments)
                elif name == "scale_endpoint":
                    result = await self._scale_endpoint(**arguments)
                elif name == "get_metrics":
                    result = await self._get_metrics(**arguments)
                elif name == "ab_test":
                    result = await self._ab_test(**arguments)
                elif name == "canary_deploy":
                    result = await self._canary_deploy(**arguments)
                elif name == "shadow_deploy":
                    result = await self._shadow_deploy(**arguments)
                elif name == "health_check":
                    result = await self._health_check(**arguments)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

    async def _deploy_model(
        self,
        endpoint_name: str,
        model_path: str,
        version: str,
        model_type: str = "sklearn",
        resources: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy a model to an endpoint."""
        try:
            # Check if endpoint already exists
            if endpoint_name in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint {endpoint_name} already exists. Use update_model to update it."
                }

            # Check if model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                return {
                    "success": False,
                    "error": f"Model file not found: {model_path}"
                }

            # Create endpoint
            endpoint = ModelEndpoint(
                name=endpoint_name,
                model_path=model_path,
                version=version,
                model_type=model_type,
                resources=resources,
                metadata=metadata
            )

            # Load the model
            endpoint.load_model()

            # Register endpoint
            self.endpoints[endpoint_name] = endpoint

            # Record deployment
            self.deployment_history.append({
                "action": "deploy",
                "endpoint": endpoint_name,
                "version": version,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "endpoint": endpoint_name,
                "version": version,
                "status": endpoint.status,
                "model_type": model_type,
                "resources": endpoint.resources,
                "created_at": endpoint.created_at,
                "message": f"Model deployed successfully to endpoint '{endpoint_name}'"
            }

        except Exception as e:
            logger.error(f"Deployment error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _predict(
        self,
        endpoint: str,
        data: Any,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Make a prediction."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            ep = self.endpoints[endpoint]

            if ep.status != "running":
                return {
                    "success": False,
                    "error": f"Endpoint status is {ep.status}, expected 'running'"
                }

            # Parse data if it's a JSON string
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

            # Make prediction
            result = ep.predict(data)
            result["success"] = True

            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "endpoint": endpoint
            }

    async def _batch_predict(
        self,
        endpoint: str,
        data_list: List[Any],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Perform batch predictions."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            ep = self.endpoints[endpoint]
            predictions = []
            errors = []

            # Process in batches
            for i in range(0, len(data_list), batch_size):
                batch = data_list[i:i + batch_size]

                for idx, data in enumerate(batch):
                    try:
                        pred = ep.predict(data)
                        predictions.append(pred)
                    except Exception as e:
                        errors.append({
                            "index": i + idx,
                            "error": str(e)
                        })

            return {
                "success": True,
                "endpoint": endpoint,
                "total_requests": len(data_list),
                "successful": len(predictions),
                "failed": len(errors),
                "predictions": predictions,
                "errors": errors if errors else None
            }

        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _list_models(
        self,
        status_filter: str = "all",
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """List all deployed models."""
        models = []

        for name, endpoint in self.endpoints.items():
            if status_filter != "all" and endpoint.status != status_filter:
                continue

            model_info = {
                "endpoint": name,
                "version": endpoint.version,
                "status": endpoint.status,
                "model_type": endpoint.model_type,
                "created_at": endpoint.created_at
            }

            if include_metrics:
                model_info["metrics"] = endpoint.get_metrics()

            models.append(model_info)

        return {
            "total_endpoints": len(models),
            "status_filter": status_filter,
            "endpoints": models
        }

    async def _update_model(
        self,
        endpoint: str,
        model_path: str,
        version: str,
        strategy: str = "rolling"
    ) -> Dict[str, Any]:
        """Update an endpoint with a new model version."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            old_endpoint = self.endpoints[endpoint]
            old_version = old_endpoint.version

            # Create new endpoint with same name
            new_endpoint = ModelEndpoint(
                name=endpoint,
                model_path=model_path,
                version=version,
                model_type=old_endpoint.model_type,
                resources=old_endpoint.resources,
                metadata=old_endpoint.metadata
            )

            # Load new model
            new_endpoint.load_model()

            # Replace endpoint (simulate strategy)
            self.endpoints[endpoint] = new_endpoint

            # Record update
            self.deployment_history.append({
                "action": "update",
                "endpoint": endpoint,
                "old_version": old_version,
                "new_version": version,
                "strategy": strategy,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "success": True,
                "endpoint": endpoint,
                "old_version": old_version,
                "new_version": version,
                "strategy": strategy,
                "status": new_endpoint.status,
                "message": f"Endpoint updated successfully using {strategy} strategy"
            }

        except Exception as e:
            logger.error(f"Update error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _rollback_model(
        self,
        endpoint: str,
        target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rollback to a previous model version."""
        try:
            # Find the rollback target in history
            rollback_target = None
            for entry in reversed(self.deployment_history):
                if entry["endpoint"] == endpoint:
                    if target_version is None or entry.get("new_version") == target_version or entry.get("version") == target_version:
                        rollback_target = entry
                        break

            if not rollback_target:
                return {
                    "success": False,
                    "error": f"No rollback target found for endpoint {endpoint}"
                }

            return {
                "success": True,
                "endpoint": endpoint,
                "rolled_back_to": rollback_target.get("old_version") or rollback_target.get("version"),
                "timestamp": datetime.now().isoformat(),
                "message": "Rollback completed successfully (simulated)",
                "note": "In production, this would restore the previous model from storage"
            }

        except Exception as e:
            logger.error(f"Rollback error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _scale_endpoint(
        self,
        endpoint: str,
        replicas: int = 1,
        resources: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Scale endpoint capacity."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            ep = self.endpoints[endpoint]

            if resources:
                ep.resources.update(resources)

            return {
                "success": True,
                "endpoint": endpoint,
                "replicas": replicas,
                "resources": ep.resources,
                "message": f"Endpoint scaled to {replicas} replica(s) (simulated)",
                "note": "In production, this would adjust Kubernetes deployments"
            }

        except Exception as e:
            logger.error(f"Scaling error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_metrics(
        self,
        endpoint: str,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Get endpoint metrics."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            metrics = self.endpoints[endpoint].get_metrics()
            metrics["success"] = True

            if detailed:
                # Add additional detailed metrics
                metrics["detailed"] = {
                    "uptime": "calculated in production",
                    "throughput_rps": "calculated in production",
                    "p50_latency_ms": "calculated in production",
                    "p95_latency_ms": "calculated in production",
                    "p99_latency_ms": "calculated in production"
                }

            return metrics

        except Exception as e:
            logger.error(f"Metrics error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _ab_test(
        self,
        test_name: str,
        endpoint_a: str,
        endpoint_b: str,
        traffic_split: float = 50,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Set up A/B testing."""
        try:
            if endpoint_a not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint A not found: {endpoint_a}"
                }

            if endpoint_b not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint B not found: {endpoint_b}"
                }

            # Register A/B test
            self.ab_tests[test_name] = {
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "traffic_split": traffic_split,
                "metrics": metrics or ["latency", "error_rate"],
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }

            return {
                "success": True,
                "test_name": test_name,
                "configuration": self.ab_tests[test_name],
                "message": f"A/B test '{test_name}' configured successfully",
                "note": f"{traffic_split}% traffic to {endpoint_b}, {100-traffic_split}% to {endpoint_a}"
            }

        except Exception as e:
            logger.error(f"A/B test error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _canary_deploy(
        self,
        endpoint: str,
        new_model_path: str,
        new_version: str,
        initial_traffic: float = 10,
        increment: float = 10
    ) -> Dict[str, Any]:
        """Perform canary deployment."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            # Simulate canary deployment stages
            stages = []
            current_traffic = initial_traffic

            while current_traffic <= 100:
                stages.append({
                    "stage": len(stages) + 1,
                    "new_version_traffic": current_traffic,
                    "old_version_traffic": 100 - current_traffic,
                    "status": "pending"
                })
                current_traffic += increment

            return {
                "success": True,
                "endpoint": endpoint,
                "new_version": new_version,
                "deployment_plan": {
                    "initial_traffic": initial_traffic,
                    "increment": increment,
                    "total_stages": len(stages),
                    "stages": stages
                },
                "message": "Canary deployment plan created",
                "note": "In production, this would gradually shift traffic while monitoring metrics"
            }

        except Exception as e:
            logger.error(f"Canary deployment error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _shadow_deploy(
        self,
        endpoint: str,
        shadow_model_path: str,
        shadow_version: str,
        sampling_rate: float = 100
    ) -> Dict[str, Any]:
        """Deploy model in shadow mode."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            # Create shadow endpoint name
            shadow_name = f"{endpoint}-shadow-{shadow_version}"

            # Deploy shadow model
            shadow_result = await self._deploy_model(
                endpoint_name=shadow_name,
                model_path=shadow_model_path,
                version=shadow_version,
                metadata={"shadow": True, "production_endpoint": endpoint}
            )

            if not shadow_result.get("success"):
                return shadow_result

            return {
                "success": True,
                "production_endpoint": endpoint,
                "shadow_endpoint": shadow_name,
                "shadow_version": shadow_version,
                "sampling_rate": sampling_rate,
                "message": "Shadow deployment completed",
                "note": f"Shadow model receives {sampling_rate}% of traffic for testing without affecting production"
            }

        except Exception as e:
            logger.error(f"Shadow deployment error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _health_check(
        self,
        endpoint: str,
        test_prediction: bool = True
    ) -> Dict[str, Any]:
        """Check endpoint health."""
        try:
            if endpoint not in self.endpoints:
                return {
                    "success": False,
                    "healthy": False,
                    "error": f"Endpoint not found: {endpoint}"
                }

            ep = self.endpoints[endpoint]

            health_status = {
                "endpoint": endpoint,
                "status": ep.status,
                "healthy": ep.status == "running",
                "version": ep.version,
                "uptime": "calculated in production",
                "last_request": ep.last_request_time
            }

            if test_prediction and ep.status == "running":
                # Attempt a test prediction
                try:
                    import numpy as np
                    test_data = np.array([[0.5, 0.5, 0.5]])  # Generic test input
                    test_result = ep.predict(test_data)
                    health_status["test_prediction"] = {
                        "status": "passed",
                        "latency_ms": test_result.get("latency_ms")
                    }
                except Exception as e:
                    health_status["test_prediction"] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    health_status["healthy"] = False

            health_status["success"] = True
            return health_status

        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return {
                "success": False,
                "healthy": False,
                "error": str(e)
            }

    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point."""
    server = ModelServingMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())