"""Google Cloud Platform MCP Server Implementation.

Provides GCP operations through MCP protocol:
- Cloud Storage operations
- Compute Engine VM management
- BigQuery data analytics
- Cloud Functions deployment and invocation
"""

import base64
import json
import logging
from typing import Any, Optional

try:
    from google.cloud import storage, compute_v1, bigquery, functions_v1
    from google.api_core import exceptions as google_exceptions
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class GCPMCPServer:
    """Google Cloud Platform MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not GCP_AVAILABLE:
            raise ImportError(
                "Google Cloud libraries required. Install with: "
                "pip install google-cloud-storage google-cloud-compute google-cloud-bigquery"
            )
        
        self.config = config
        self.project_id = config.get("project_id")
        self.credentials_path = config.get("credentials_path")
        
        # Initialize clients (lazy loading)
        self._storage_client: Optional[storage.Client] = None
        self._compute_client: Optional[compute_v1.InstancesClient] = None
        self._bigquery_client: Optional[bigquery.Client] = None
        self._functions_client: Optional[functions_v1.CloudFunctionsServiceClient] = None

    def _get_storage_client(self) -> storage.Client:
        """Get or create Cloud Storage client."""
        if self._storage_client is None:
            if self.credentials_path:
                self._storage_client = storage.Client.from_service_account_json(
                    self.credentials_path, project=self.project_id
                )
            else:
                self._storage_client = storage.Client(project=self.project_id)
        return self._storage_client

    def _get_compute_client(self) -> compute_v1.InstancesClient:
        """Get or create Compute Engine client."""
        if self._compute_client is None:
            if self.credentials_path:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._compute_client = compute_v1.InstancesClient(credentials=credentials)
            else:
                self._compute_client = compute_v1.InstancesClient()
        return self._compute_client

    def _get_bigquery_client(self) -> bigquery.Client:
        """Get or create BigQuery client."""
        if self._bigquery_client is None:
            if self.credentials_path:
                self._bigquery_client = bigquery.Client.from_service_account_json(
                    self.credentials_path, project=self.project_id
                )
            else:
                self._bigquery_client = bigquery.Client(project=self.project_id)
        return self._bigquery_client

    def _get_functions_client(self) -> functions_v1.CloudFunctionsServiceClient:
        """Get or create Cloud Functions client."""
        if self._functions_client is None:
            if self.credentials_path:
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._functions_client = functions_v1.CloudFunctionsServiceClient(
                    credentials=credentials
                )
            else:
                self._functions_client = functions_v1.CloudFunctionsServiceClient()
        return self._functions_client

    # ===== CLOUD STORAGE OPERATIONS =====

    async def storage_upload(
        self,
        bucket_name: str,
        blob_name: str,
        data: Optional[str] = None,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Upload file to Cloud Storage.

        Args:
            bucket_name: Cloud Storage bucket name
            blob_name: Blob name (path in bucket)
            data: String data to upload (alternative to file_path)
            file_path: Local file path to upload
            content_type: Content type (MIME type)
            metadata: Blob metadata

        Returns:
            Upload result
        """
        try:
            client = self._get_storage_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if content_type:
                blob.content_type = content_type
            if metadata:
                blob.metadata = metadata
            
            if data is not None:
                blob.upload_from_string(data)
                size = len(data.encode('utf-8'))
            elif file_path is not None:
                blob.upload_from_filename(file_path)
                import os
                size = os.path.getsize(file_path)
            else:
                return {
                    "success": False,
                    "error": "Either 'data' or 'file_path' must be provided",
                }
            
            return {
                "success": True,
                "bucket": bucket_name,
                "blob_name": blob_name,
                "public_url": blob.public_url,
                "size_bytes": size,
                "generation": blob.generation,
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Cloud Storage upload failed: {e}")
            return {
                "success": False,
                "error": f"Cloud Storage error: {str(e)}",
                "bucket": bucket_name,
                "blob_name": blob_name,
            }
        except Exception as e:
            logger.error(f"Failed to upload to Cloud Storage: {e}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}",
                "bucket": bucket_name,
                "blob_name": blob_name,
            }

    async def storage_download(
        self,
        bucket_name: str,
        blob_name: str,
        file_path: Optional[str] = None,
        return_content: bool = False,
    ) -> dict[str, Any]:
        """Download file from Cloud Storage.

        Args:
            bucket_name: Cloud Storage bucket name
            blob_name: Blob name (path in bucket)
            file_path: Local file path to save (optional)
            return_content: Return content as string in response

        Returns:
            Download result
        """
        try:
            client = self._get_storage_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                return {
                    "success": False,
                    "error": f"Blob '{blob_name}' not found in bucket '{bucket_name}'",
                    "bucket": bucket_name,
                    "blob_name": blob_name,
                }
            
            if file_path:
                blob.download_to_filename(file_path)
                result = {
                    "success": True,
                    "bucket": bucket_name,
                    "blob_name": blob_name,
                    "file_path": file_path,
                }
            else:
                content = blob.download_as_bytes()
                
                result = {
                    "success": True,
                    "bucket": bucket_name,
                    "blob_name": blob_name,
                    "size_bytes": blob.size,
                    "content_type": blob.content_type,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "metadata": blob.metadata or {},
                }
                
                if return_content:
                    try:
                        result["content"] = content.decode('utf-8')
                    except UnicodeDecodeError:
                        result["content"] = base64.b64encode(content).decode('utf-8')
                        result["content_encoding"] = "base64"
            
            return result

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Cloud Storage download failed: {e}")
            return {
                "success": False,
                "error": f"Cloud Storage error: {str(e)}",
                "bucket": bucket_name,
                "blob_name": blob_name,
            }
        except Exception as e:
            logger.error(f"Failed to download from Cloud Storage: {e}")
            return {
                "success": False,
                "error": f"Download failed: {str(e)}",
                "bucket": bucket_name,
                "blob_name": blob_name,
            }

    async def storage_list(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """List blobs in Cloud Storage bucket.

        Args:
            bucket_name: Cloud Storage bucket name
            prefix: Filter by prefix (folder path)
            max_results: Maximum number of blobs to return

        Returns:
            List of blobs
        """
        try:
            client = self._get_storage_client()
            bucket = client.bucket(bucket_name)
            
            blobs_iter = bucket.list_blobs(prefix=prefix, max_results=max_results)
            
            blobs = []
            for blob in blobs_iter:
                blobs.append({
                    "name": blob.name,
                    "size_bytes": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "content_type": blob.content_type,
                    "storage_class": blob.storage_class,
                })
            
            return {
                "success": True,
                "bucket": bucket_name,
                "prefix": prefix,
                "blobs": blobs,
                "count": len(blobs),
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Cloud Storage list failed: {e}")
            return {
                "success": False,
                "error": f"Cloud Storage error: {str(e)}",
                "bucket": bucket_name,
            }
        except Exception as e:
            logger.error(f"Failed to list Cloud Storage blobs: {e}")
            return {
                "success": False,
                "error": f"List failed: {str(e)}",
                "bucket": bucket_name,
            }

    # ===== COMPUTE ENGINE OPERATIONS =====

    async def compute_list_instances(
        self,
        zone: str,
        filter_expr: Optional[str] = None,
    ) -> dict[str, Any]:
        """List Compute Engine instances.

        Args:
            zone: GCP zone (e.g., 'us-central1-a')
            filter_expr: Filter expression

        Returns:
            List of instances
        """
        try:
            client = self._get_compute_client()
            
            request = compute_v1.ListInstancesRequest(
                project=self.project_id,
                zone=zone,
                filter=filter_expr,
            )
            
            instances_list = client.list(request=request)
            
            instances = []
            for instance in instances_list:
                instances.append({
                    "name": instance.name,
                    "status": instance.status,
                    "machine_type": instance.machine_type.split('/')[-1],
                    "zone": zone,
                    "internal_ip": instance.network_interfaces[0].network_i_p if instance.network_interfaces else None,
                    "external_ip": (
                        instance.network_interfaces[0].access_configs[0].nat_i_p
                        if instance.network_interfaces and instance.network_interfaces[0].access_configs
                        else None
                    ),
                    "creation_timestamp": instance.creation_timestamp,
                })
            
            return {
                "success": True,
                "zone": zone,
                "instances": instances,
                "count": len(instances),
                "project_id": self.project_id,
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Compute Engine list failed: {e}")
            return {
                "success": False,
                "error": f"Compute Engine error: {str(e)}",
                "zone": zone,
            }
        except Exception as e:
            logger.error(f"Failed to list Compute Engine instances: {e}")
            return {
                "success": False,
                "error": f"List failed: {str(e)}",
                "zone": zone,
            }

    async def compute_start(
        self,
        zone: str,
        instance_name: str,
    ) -> dict[str, Any]:
        """Start Compute Engine instance.

        Args:
            zone: GCP zone
            instance_name: Instance name

        Returns:
            Start result
        """
        try:
            client = self._get_compute_client()
            
            request = compute_v1.StartInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_name,
            )
            
            operation = client.start(request=request)
            
            return {
                "success": True,
                "instance_name": instance_name,
                "zone": zone,
                "operation_name": operation.name,
                "message": f"Instance {instance_name} is starting",
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Compute Engine start failed: {e}")
            return {
                "success": False,
                "error": f"Compute Engine error: {str(e)}",
                "instance_name": instance_name,
                "zone": zone,
            }
        except Exception as e:
            logger.error(f"Failed to start Compute Engine instance: {e}")
            return {
                "success": False,
                "error": f"Start failed: {str(e)}",
                "instance_name": instance_name,
                "zone": zone,
            }

    async def compute_stop(
        self,
        zone: str,
        instance_name: str,
    ) -> dict[str, Any]:
        """Stop Compute Engine instance.

        Args:
            zone: GCP zone
            instance_name: Instance name

        Returns:
            Stop result
        """
        try:
            client = self._get_compute_client()
            
            request = compute_v1.StopInstanceRequest(
                project=self.project_id,
                zone=zone,
                instance=instance_name,
            )
            
            operation = client.stop(request=request)
            
            return {
                "success": True,
                "instance_name": instance_name,
                "zone": zone,
                "operation_name": operation.name,
                "message": f"Instance {instance_name} is stopping",
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Compute Engine stop failed: {e}")
            return {
                "success": False,
                "error": f"Compute Engine error: {str(e)}",
                "instance_name": instance_name,
                "zone": zone,
            }
        except Exception as e:
            logger.error(f"Failed to stop Compute Engine instance: {e}")
            return {
                "success": False,
                "error": f"Stop failed: {str(e)}",
                "instance_name": instance_name,
                "zone": zone,
            }

    # ===== BIGQUERY OPERATIONS =====

    async def bigquery_query(
        self,
        query: str,
        use_legacy_sql: bool = False,
        max_results: int = 1000,
    ) -> dict[str, Any]:
        """Run BigQuery SQL query.

        Args:
            query: SQL query string
            use_legacy_sql: Use legacy SQL syntax
            max_results: Maximum number of results

        Returns:
            Query results
        """
        try:
            client = self._get_bigquery_client()
            
            job_config = bigquery.QueryJobConfig(
                use_legacy_sql=use_legacy_sql,
                maximum_bytes_billed=10_000_000_000,  # 10 GB limit
            )
            
            query_job = client.query(query, job_config=job_config)
            results = query_job.result(max_results=max_results)
            
            # Convert to list of dicts
            rows = []
            for row in results:
                rows.append(dict(row))
            
            return {
                "success": True,
                "query": query,
                "rows": rows,
                "row_count": len(rows),
                "total_rows": results.total_rows,
                "bytes_processed": query_job.total_bytes_processed,
                "bytes_billed": query_job.total_bytes_billed,
                "job_id": query_job.job_id,
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"BigQuery query failed: {e}")
            return {
                "success": False,
                "error": f"BigQuery error: {str(e)}",
                "query": query,
            }
        except Exception as e:
            logger.error(f"Failed to execute BigQuery query: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}",
                "query": query,
            }

    async def bigquery_load(
        self,
        dataset_id: str,
        table_id: str,
        source_uri: Optional[str] = None,
        data: Optional[list[dict[str, Any]]] = None,
        source_format: str = "CSV",
        write_disposition: str = "WRITE_APPEND",
        schema: Optional[list[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Load data into BigQuery table.

        Args:
            dataset_id: Dataset ID
            table_id: Table ID
            source_uri: GCS URI (e.g., 'gs://bucket/file.csv')
            data: Direct data to load (alternative to source_uri)
            source_format: Source format (CSV, JSON, PARQUET, etc.)
            write_disposition: Write disposition (WRITE_APPEND, WRITE_TRUNCATE, WRITE_EMPTY)
            schema: Table schema

        Returns:
            Load result
        """
        try:
            client = self._get_bigquery_client()
            
            table_ref = client.dataset(dataset_id).table(table_id)
            
            job_config = bigquery.LoadJobConfig(
                source_format=getattr(bigquery.SourceFormat, source_format),
                write_disposition=getattr(bigquery.WriteDisposition, write_disposition),
            )
            
            if schema:
                # Convert schema format
                schema_fields = []
                for field in schema:
                    schema_fields.append(
                        bigquery.SchemaField(
                            field['name'],
                            field['type'],
                            mode=field.get('mode', 'NULLABLE')
                        )
                    )
                job_config.schema = schema_fields
            
            if source_uri:
                load_job = client.load_table_from_uri(
                    source_uri,
                    table_ref,
                    job_config=job_config,
                )
            elif data:
                load_job = client.load_table_from_json(
                    data,
                    table_ref,
                    job_config=job_config,
                )
            else:
                return {
                    "success": False,
                    "error": "Either 'source_uri' or 'data' must be provided",
                }
            
            # Wait for job to complete
            load_job.result()
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "table_id": table_id,
                "rows_loaded": load_job.output_rows,
                "job_id": load_job.job_id,
                "message": f"Loaded {load_job.output_rows} rows into {dataset_id}.{table_id}",
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"BigQuery load failed: {e}")
            return {
                "success": False,
                "error": f"BigQuery error: {str(e)}",
                "dataset_id": dataset_id,
                "table_id": table_id,
            }
        except Exception as e:
            logger.error(f"Failed to load data into BigQuery: {e}")
            return {
                "success": False,
                "error": f"Load failed: {str(e)}",
                "dataset_id": dataset_id,
                "table_id": table_id,
            }

    # ===== CLOUD FUNCTIONS OPERATIONS =====

    async def function_deploy(
        self,
        function_name: str,
        region: str,
        source_archive_url: str,
        entry_point: str,
        runtime: str = "python311",
        environment_variables: Optional[dict[str, str]] = None,
        timeout: int = 60,
        memory_mb: int = 256,
    ) -> dict[str, Any]:
        """Deploy Cloud Function.

        Args:
            function_name: Function name
            region: GCP region (e.g., 'us-central1')
            source_archive_url: GCS URL to source archive (e.g., 'gs://bucket/source.zip')
            entry_point: Entry point function name
            runtime: Runtime (e.g., 'python311', 'nodejs18')
            environment_variables: Environment variables
            timeout: Timeout in seconds
            memory_mb: Memory in MB

        Returns:
            Deployment result
        """
        try:
            client = self._get_functions_client()
            
            parent = f"projects/{self.project_id}/locations/{region}"
            function_path = f"{parent}/functions/{function_name}"
            
            function = functions_v1.CloudFunction(
                name=function_path,
                entry_point=entry_point,
                runtime=runtime,
                source_archive_url=source_archive_url,
                timeout=f"{timeout}s",
                available_memory_mb=memory_mb,
            )
            
            if environment_variables:
                function.environment_variables = environment_variables
            
            # Check if function exists
            try:
                client.get_function(name=function_path)
                # Update existing function
                operation = client.update_function(function=function)
                action = "updated"
            except google_exceptions.NotFound:
                # Create new function
                operation = client.create_function(
                    location=parent,
                    function=function,
                )
                action = "created"
            
            # Wait for operation (with timeout)
            result = operation.result(timeout=300)
            
            return {
                "success": True,
                "function_name": function_name,
                "region": region,
                "action": action,
                "runtime": runtime,
                "entry_point": entry_point,
                "message": f"Function {action} successfully",
            }

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Cloud Function deploy failed: {e}")
            return {
                "success": False,
                "error": f"Cloud Function error: {str(e)}",
                "function_name": function_name,
            }
        except Exception as e:
            logger.error(f"Failed to deploy Cloud Function: {e}")
            return {
                "success": False,
                "error": f"Deploy failed: {str(e)}",
                "function_name": function_name,
            }

    async def function_invoke(
        self,
        function_name: str,
        region: str,
        data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Invoke Cloud Function.

        Args:
            function_name: Function name
            region: GCP region
            data: Input data for function

        Returns:
            Invocation result
        """
        try:
            # Cloud Functions invocation typically done via HTTP
            # This is a simplified version
            client = self._get_functions_client()
            
            function_path = f"projects/{self.project_id}/locations/{region}/functions/{function_name}"
            
            request = functions_v1.CallFunctionRequest(
                name=function_path,
                data=json.dumps(data) if data else "",
            )
            
            response = client.call_function(request=request)
            
            result = {
                "success": True,
                "function_name": function_name,
                "region": region,
                "execution_id": response.execution_id,
            }
            
            if response.result:
                try:
                    result["response"] = json.loads(response.result)
                except json.JSONDecodeError:
                    result["response"] = response.result
            
            if response.error:
                result["function_error"] = response.error
            
            return result

        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Cloud Function invoke failed: {e}")
            return {
                "success": False,
                "error": f"Cloud Function error: {str(e)}",
                "function_name": function_name,
            }
        except Exception as e:
            logger.error(f"Failed to invoke Cloud Function: {e}")
            return {
                "success": False,
                "error": f"Invoke failed: {str(e)}",
                "function_name": function_name,
            }


def get_server_definition() -> dict[str, Any]:
    """Get GCP MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "gcp",
        "category": "cloud",
        "description": "Google Cloud Platform operations (Storage, Compute, BigQuery, Functions)",
        "tools": [
            # Cloud Storage Operations
            {
                "name": "storage_upload",
                "description": "Upload file to Cloud Storage bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket_name": {
                            "type": "string",
                            "description": "Cloud Storage bucket name",
                        },
                        "blob_name": {
                            "type": "string",
                            "description": "Blob name (path in bucket)",
                        },
                        "data": {
                            "type": "string",
                            "description": "String data to upload (alternative to file_path)",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Local file path to upload",
                        },
                        "content_type": {
                            "type": "string",
                            "description": "Content type (MIME type)",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Blob metadata",
                        },
                    },
                    "required": ["bucket_name", "blob_name"],
                },
            },
            {
                "name": "storage_download",
                "description": "Download file from Cloud Storage bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket_name": {
                            "type": "string",
                            "description": "Cloud Storage bucket name",
                        },
                        "blob_name": {
                            "type": "string",
                            "description": "Blob name (path in bucket)",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Local file path to save (optional)",
                        },
                        "return_content": {
                            "type": "boolean",
                            "description": "Return content in response",
                            "default": False,
                        },
                    },
                    "required": ["bucket_name", "blob_name"],
                },
            },
            {
                "name": "storage_list",
                "description": "List blobs in Cloud Storage bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket_name": {
                            "type": "string",
                            "description": "Cloud Storage bucket name",
                        },
                        "prefix": {
                            "type": "string",
                            "description": "Filter by prefix (folder path)",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of blobs",
                            "default": 100,
                        },
                    },
                    "required": ["bucket_name"],
                },
            },
            # Compute Engine Operations
            {
                "name": "compute_list_instances",
                "description": "List Compute Engine instances",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone": {
                            "type": "string",
                            "description": "GCP zone (e.g., 'us-central1-a')",
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Filter expression",
                        },
                    },
                    "required": ["zone"],
                },
            },
            {
                "name": "compute_start",
                "description": "Start Compute Engine instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone": {
                            "type": "string",
                            "description": "GCP zone",
                        },
                        "instance_name": {
                            "type": "string",
                            "description": "Instance name",
                        },
                    },
                    "required": ["zone", "instance_name"],
                },
            },
            {
                "name": "compute_stop",
                "description": "Stop Compute Engine instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zone": {
                            "type": "string",
                            "description": "GCP zone",
                        },
                        "instance_name": {
                            "type": "string",
                            "description": "Instance name",
                        },
                    },
                    "required": ["zone", "instance_name"],
                },
            },
            # BigQuery Operations
            {
                "name": "bigquery_query",
                "description": "Run BigQuery SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query string",
                        },
                        "use_legacy_sql": {
                            "type": "boolean",
                            "description": "Use legacy SQL syntax",
                            "default": False,
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 1000,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "bigquery_load",
                "description": "Load data into BigQuery table",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Dataset ID",
                        },
                        "table_id": {
                            "type": "string",
                            "description": "Table ID",
                        },
                        "source_uri": {
                            "type": "string",
                            "description": "GCS URI (e.g., 'gs://bucket/file.csv')",
                        },
                        "data": {
                            "type": "array",
                            "description": "Direct data to load",
                        },
                        "source_format": {
                            "type": "string",
                            "description": "Source format (CSV, JSON, PARQUET)",
                            "default": "CSV",
                        },
                        "write_disposition": {
                            "type": "string",
                            "description": "Write disposition (WRITE_APPEND, WRITE_TRUNCATE, WRITE_EMPTY)",
                            "default": "WRITE_APPEND",
                        },
                        "schema": {
                            "type": "array",
                            "description": "Table schema",
                        },
                    },
                    "required": ["dataset_id", "table_id"],
                },
            },
            # Cloud Functions Operations
            {
                "name": "function_deploy",
                "description": "Deploy Cloud Function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Function name",
                        },
                        "region": {
                            "type": "string",
                            "description": "GCP region (e.g., 'us-central1')",
                        },
                        "source_archive_url": {
                            "type": "string",
                            "description": "GCS URL to source archive",
                        },
                        "entry_point": {
                            "type": "string",
                            "description": "Entry point function name",
                        },
                        "runtime": {
                            "type": "string",
                            "description": "Runtime (e.g., 'python311', 'nodejs18')",
                            "default": "python311",
                        },
                        "environment_variables": {
                            "type": "object",
                            "description": "Environment variables",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 60,
                        },
                        "memory_mb": {
                            "type": "integer",
                            "description": "Memory in MB",
                            "default": 256,
                        },
                    },
                    "required": ["function_name", "region", "source_archive_url", "entry_point"],
                },
            },
            {
                "name": "function_invoke",
                "description": "Invoke Cloud Function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Function name",
                        },
                        "region": {
                            "type": "string",
                            "description": "GCP region",
                        },
                        "data": {
                            "type": "object",
                            "description": "Input data for function",
                        },
                    },
                    "required": ["function_name", "region"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "high",
            "category": "cloud",
            "requires": [
                "google-cloud-storage",
                "google-cloud-compute",
                "google-cloud-bigquery",
            ],
        },
    }