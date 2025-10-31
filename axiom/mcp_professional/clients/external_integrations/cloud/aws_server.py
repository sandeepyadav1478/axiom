"""AWS MCP Server Implementation.

Provides AWS cloud infrastructure operations through MCP protocol:
- S3 storage operations
- EC2 instance management
- Lambda function deployment and invocation
- CloudWatch metrics monitoring
"""

import base64
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class AWSMCPServer:
    """AWS MCP server implementation."""

    def __init__(self, config: dict[str, Any]):
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for AWS MCP server. Install with: pip install boto3"
            )
        
        self.config = config
        self.region = config.get("region", "us-east-1")
        self.access_key_id = config.get("access_key_id")
        self.secret_access_key = config.get("secret_access_key")
        self.profile = config.get("profile")
        
        # Initialize boto3 session
        session_kwargs = {"region_name": self.region}
        if self.profile:
            session_kwargs["profile_name"] = self.profile
        elif self.access_key_id and self.secret_access_key:
            session_kwargs["aws_access_key_id"] = self.access_key_id
            session_kwargs["aws_secret_access_key"] = self.secret_access_key
        
        self.session = boto3.Session(**session_kwargs)
        
        # Initialize clients (lazy loading)
        self._s3_client: Optional[Any] = None
        self._ec2_client: Optional[Any] = None
        self._lambda_client: Optional[Any] = None
        self._cloudwatch_client: Optional[Any] = None

    def _get_s3_client(self):
        """Get or create S3 client."""
        if self._s3_client is None:
            self._s3_client = self.session.client("s3")
        return self._s3_client

    def _get_ec2_client(self):
        """Get or create EC2 client."""
        if self._ec2_client is None:
            self._ec2_client = self.session.client("ec2")
        return self._ec2_client

    def _get_lambda_client(self):
        """Get or create Lambda client."""
        if self._lambda_client is None:
            self._lambda_client = self.session.client("lambda")
        return self._lambda_client

    def _get_cloudwatch_client(self):
        """Get or create CloudWatch client."""
        if self._cloudwatch_client is None:
            self._cloudwatch_client = self.session.client("cloudwatch")
        return self._cloudwatch_client

    # ===== S3 OPERATIONS =====

    async def s3_upload(
        self,
        bucket: str,
        key: str,
        data: Optional[str] = None,
        file_path: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Upload file to S3.

        Args:
            bucket: S3 bucket name
            key: Object key (path in S3)
            data: String data to upload (alternative to file_path)
            file_path: Local file path to upload
            content_type: Content type (MIME type)
            metadata: Object metadata

        Returns:
            Upload result
        """
        try:
            s3 = self._get_s3_client()
            
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            if metadata:
                extra_args["Metadata"] = metadata
            
            if data is not None:
                # Upload string data
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=data.encode('utf-8'),
                    **extra_args
                )
                size = len(data.encode('utf-8'))
            elif file_path is not None:
                # Upload file
                s3.upload_file(file_path, bucket, key, ExtraArgs=extra_args if extra_args else None)
                import os
                size = os.path.getsize(file_path)
            else:
                return {
                    "success": False,
                    "error": "Either 'data' or 'file_path' must be provided",
                }
            
            # Get object URL
            url = f"s3://{bucket}/{key}"
            
            return {
                "success": True,
                "bucket": bucket,
                "key": key,
                "url": url,
                "size_bytes": size,
                "region": self.region,
            }

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return {
                "success": False,
                "error": f"S3 upload error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "bucket": bucket,
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}",
                "bucket": bucket,
                "key": key,
            }

    async def s3_download(
        self,
        bucket: str,
        key: str,
        file_path: Optional[str] = None,
        return_content: bool = False,
    ) -> dict[str, Any]:
        """Download file from S3.

        Args:
            bucket: S3 bucket name
            key: Object key (path in S3)
            file_path: Local file path to save (optional)
            return_content: Return content as string in response

        Returns:
            Download result
        """
        try:
            s3 = self._get_s3_client()
            
            if file_path:
                # Download to file
                s3.download_file(bucket, key, file_path)
                result = {
                    "success": True,
                    "bucket": bucket,
                    "key": key,
                    "file_path": file_path,
                }
            else:
                # Get object
                response = s3.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()
                
                result = {
                    "success": True,
                    "bucket": bucket,
                    "key": key,
                    "size_bytes": len(content),
                    "content_type": response.get('ContentType'),
                    "last_modified": response.get('LastModified').isoformat() if response.get('LastModified') else None,
                    "metadata": response.get('Metadata', {}),
                }
                
                if return_content:
                    try:
                        result["content"] = content.decode('utf-8')
                    except UnicodeDecodeError:
                        result["content"] = base64.b64encode(content).decode('utf-8')
                        result["content_encoding"] = "base64"
            
            return result

        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return {
                "success": False,
                "error": f"S3 download error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "bucket": bucket,
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return {
                "success": False,
                "error": f"Download failed: {str(e)}",
                "bucket": bucket,
                "key": key,
            }

    async def s3_list(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        max_keys: int = 100,
    ) -> dict[str, Any]:
        """List objects in S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Filter by prefix (folder path)
            max_keys: Maximum number of objects to return

        Returns:
            List of objects
        """
        try:
            s3 = self._get_s3_client()
            
            kwargs = {
                "Bucket": bucket,
                "MaxKeys": max_keys,
            }
            if prefix:
                kwargs["Prefix"] = prefix
            
            response = s3.list_objects_v2(**kwargs)
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    "key": obj['Key'],
                    "size_bytes": obj['Size'],
                    "last_modified": obj['LastModified'].isoformat(),
                    "storage_class": obj.get('StorageClass', 'STANDARD'),
                })
            
            return {
                "success": True,
                "bucket": bucket,
                "prefix": prefix,
                "objects": objects,
                "count": len(objects),
                "is_truncated": response.get('IsTruncated', False),
            }

        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            return {
                "success": False,
                "error": f"S3 list error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "bucket": bucket,
            }
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return {
                "success": False,
                "error": f"List failed: {str(e)}",
                "bucket": bucket,
            }

    async def s3_delete(
        self,
        bucket: str,
        key: str,
    ) -> dict[str, Any]:
        """Delete object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key to delete

        Returns:
            Deletion result
        """
        try:
            s3 = self._get_s3_client()
            
            s3.delete_object(Bucket=bucket, Key=key)
            
            return {
                "success": True,
                "bucket": bucket,
                "key": key,
                "message": "Object deleted successfully",
            }

        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            return {
                "success": False,
                "error": f"S3 delete error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "bucket": bucket,
                "key": key,
            }
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return {
                "success": False,
                "error": f"Delete failed: {str(e)}",
                "bucket": bucket,
                "key": key,
            }

    # ===== EC2 OPERATIONS =====

    async def ec2_list_instances(
        self,
        filters: Optional[list[dict[str, Any]]] = None,
        max_results: int = 100,
    ) -> dict[str, Any]:
        """List EC2 instances.

        Args:
            filters: EC2 filters (e.g., [{"Name": "instance-state-name", "Values": ["running"]}])
            max_results: Maximum number of instances to return

        Returns:
            List of EC2 instances
        """
        try:
            ec2 = self._get_ec2_client()
            
            kwargs = {"MaxResults": max_results}
            if filters:
                kwargs["Filters"] = filters
            
            response = ec2.describe_instances(**kwargs)
            
            instances = []
            for reservation in response.get('Reservations', []):
                for instance in reservation.get('Instances', []):
                    # Extract name tag
                    name = "N/A"
                    for tag in instance.get('Tags', []):
                        if tag['Key'] == 'Name':
                            name = tag['Value']
                            break
                    
                    instances.append({
                        "instance_id": instance['InstanceId'],
                        "name": name,
                        "state": instance['State']['Name'],
                        "instance_type": instance['InstanceType'],
                        "availability_zone": instance['Placement']['AvailabilityZone'],
                        "private_ip": instance.get('PrivateIpAddress'),
                        "public_ip": instance.get('PublicIpAddress'),
                        "launch_time": instance['LaunchTime'].isoformat(),
                    })
            
            return {
                "success": True,
                "instances": instances,
                "count": len(instances),
                "region": self.region,
            }

        except ClientError as e:
            logger.error(f"EC2 list failed: {e}")
            return {
                "success": False,
                "error": f"EC2 list error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
            }
        except Exception as e:
            logger.error(f"Failed to list EC2 instances: {e}")
            return {
                "success": False,
                "error": f"List failed: {str(e)}",
            }

    async def ec2_start_instance(
        self,
        instance_id: str,
    ) -> dict[str, Any]:
        """Start EC2 instance.

        Args:
            instance_id: EC2 instance ID

        Returns:
            Start result
        """
        try:
            ec2 = self._get_ec2_client()
            
            response = ec2.start_instances(InstanceIds=[instance_id])
            
            state_change = response['StartingInstances'][0]
            
            return {
                "success": True,
                "instance_id": instance_id,
                "previous_state": state_change['PreviousState']['Name'],
                "current_state": state_change['CurrentState']['Name'],
                "message": f"Instance {instance_id} is starting",
            }

        except ClientError as e:
            logger.error(f"EC2 start failed: {e}")
            return {
                "success": False,
                "error": f"EC2 start error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "instance_id": instance_id,
            }
        except Exception as e:
            logger.error(f"Failed to start EC2 instance: {e}")
            return {
                "success": False,
                "error": f"Start failed: {str(e)}",
                "instance_id": instance_id,
            }

    async def ec2_stop_instance(
        self,
        instance_id: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Stop EC2 instance.

        Args:
            instance_id: EC2 instance ID
            force: Force stop (hibernate instead of graceful shutdown)

        Returns:
            Stop result
        """
        try:
            ec2 = self._get_ec2_client()
            
            kwargs = {"InstanceIds": [instance_id]}
            if force:
                kwargs["Force"] = True
            
            response = ec2.stop_instances(**kwargs)
            
            state_change = response['StoppingInstances'][0]
            
            return {
                "success": True,
                "instance_id": instance_id,
                "previous_state": state_change['PreviousState']['Name'],
                "current_state": state_change['CurrentState']['Name'],
                "message": f"Instance {instance_id} is stopping",
                "force": force,
            }

        except ClientError as e:
            logger.error(f"EC2 stop failed: {e}")
            return {
                "success": False,
                "error": f"EC2 stop error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "instance_id": instance_id,
            }
        except Exception as e:
            logger.error(f"Failed to stop EC2 instance: {e}")
            return {
                "success": False,
                "error": f"Stop failed: {str(e)}",
                "instance_id": instance_id,
            }

    async def ec2_create_instance(
        self,
        image_id: str,
        instance_type: str,
        key_name: Optional[str] = None,
        security_group_ids: Optional[list[str]] = None,
        subnet_id: Optional[str] = None,
        name: Optional[str] = None,
        user_data: Optional[str] = None,
        min_count: int = 1,
        max_count: int = 1,
    ) -> dict[str, Any]:
        """Create EC2 instance.

        Args:
            image_id: AMI ID
            instance_type: Instance type (e.g., 't2.micro')
            key_name: SSH key pair name
            security_group_ids: Security group IDs
            subnet_id: Subnet ID
            name: Instance name tag
            user_data: User data script
            min_count: Minimum number of instances
            max_count: Maximum number of instances

        Returns:
            Created instance information
        """
        try:
            ec2 = self._get_ec2_client()
            
            kwargs = {
                "ImageId": image_id,
                "InstanceType": instance_type,
                "MinCount": min_count,
                "MaxCount": max_count,
            }
            
            if key_name:
                kwargs["KeyName"] = key_name
            if security_group_ids:
                kwargs["SecurityGroupIds"] = security_group_ids
            if subnet_id:
                kwargs["SubnetId"] = subnet_id
            if user_data:
                kwargs["UserData"] = user_data
            
            # Add name tag
            if name:
                kwargs["TagSpecifications"] = [
                    {
                        "ResourceType": "instance",
                        "Tags": [{"Key": "Name", "Value": name}],
                    }
                ]
            
            response = ec2.run_instances(**kwargs)
            
            instances = []
            for instance in response['Instances']:
                instances.append({
                    "instance_id": instance['InstanceId'],
                    "state": instance['State']['Name'],
                    "instance_type": instance['InstanceType'],
                    "availability_zone": instance['Placement']['AvailabilityZone'],
                    "private_ip": instance.get('PrivateIpAddress'),
                })
            
            return {
                "success": True,
                "instances": instances,
                "count": len(instances),
                "image_id": image_id,
                "instance_type": instance_type,
                "message": f"Created {len(instances)} instance(s)",
            }

        except ClientError as e:
            logger.error(f"EC2 create failed: {e}")
            return {
                "success": False,
                "error": f"EC2 create error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
            }
        except Exception as e:
            logger.error(f"Failed to create EC2 instance: {e}")
            return {
                "success": False,
                "error": f"Create failed: {str(e)}",
            }

    # ===== LAMBDA OPERATIONS =====

    async def lambda_invoke(
        self,
        function_name: str,
        payload: Optional[dict[str, Any]] = None,
        invocation_type: str = "RequestResponse",
    ) -> dict[str, Any]:
        """Invoke Lambda function.

        Args:
            function_name: Lambda function name or ARN
            payload: Function input payload
            invocation_type: Invocation type (RequestResponse, Event, DryRun)

        Returns:
            Invocation result
        """
        try:
            lambda_client = self._get_lambda_client()
            
            kwargs = {
                "FunctionName": function_name,
                "InvocationType": invocation_type,
            }
            
            if payload:
                kwargs["Payload"] = json.dumps(payload).encode('utf-8')
            
            response = lambda_client.invoke(**kwargs)
            
            result = {
                "success": True,
                "function_name": function_name,
                "status_code": response['StatusCode'],
                "invocation_type": invocation_type,
            }
            
            # Read response payload
            if 'Payload' in response:
                payload_data = response['Payload'].read()
                try:
                    result["response"] = json.loads(payload_data.decode('utf-8'))
                except json.JSONDecodeError:
                    result["response"] = payload_data.decode('utf-8')
            
            if 'FunctionError' in response:
                result["function_error"] = response['FunctionError']
            
            return result

        except ClientError as e:
            logger.error(f"Lambda invoke failed: {e}")
            return {
                "success": False,
                "error": f"Lambda invoke error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "function_name": function_name,
            }
        except Exception as e:
            logger.error(f"Failed to invoke Lambda: {e}")
            return {
                "success": False,
                "error": f"Invoke failed: {str(e)}",
                "function_name": function_name,
            }

    async def lambda_deploy(
        self,
        function_name: str,
        zip_file_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_key: Optional[str] = None,
        runtime: str = "python3.11",
        handler: str = "lambda_function.lambda_handler",
        role_arn: Optional[str] = None,
        description: Optional[str] = None,
        timeout: int = 30,
        memory_size: int = 128,
        environment: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Deploy Lambda function.

        Args:
            function_name: Function name
            zip_file_path: Local path to deployment package
            s3_bucket: S3 bucket containing deployment package
            s3_key: S3 key of deployment package
            runtime: Lambda runtime
            handler: Function handler
            role_arn: IAM role ARN
            description: Function description
            timeout: Timeout in seconds
            memory_size: Memory size in MB
            environment: Environment variables

        Returns:
            Deployment result
        """
        try:
            lambda_client = self._get_lambda_client()
            
            # Check if function exists
            try:
                lambda_client.get_function(FunctionName=function_name)
                function_exists = True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    function_exists = False
                else:
                    raise
            
            # Prepare code
            code = {}
            if zip_file_path:
                with open(zip_file_path, 'rb') as f:
                    code['ZipFile'] = f.read()
            elif s3_bucket and s3_key:
                code['S3Bucket'] = s3_bucket
                code['S3Key'] = s3_key
            else:
                return {
                    "success": False,
                    "error": "Either zip_file_path or (s3_bucket and s3_key) must be provided",
                }
            
            if function_exists:
                # Update function code
                response = lambda_client.update_function_code(
                    FunctionName=function_name,
                    **code
                )
                
                # Update configuration if needed
                config_kwargs = {
                    "FunctionName": function_name,
                    "Runtime": runtime,
                    "Handler": handler,
                    "Timeout": timeout,
                    "MemorySize": memory_size,
                }
                if role_arn:
                    config_kwargs["Role"] = role_arn
                if description:
                    config_kwargs["Description"] = description
                if environment:
                    config_kwargs["Environment"] = {"Variables": environment}
                
                lambda_client.update_function_configuration(**config_kwargs)
                
                action = "updated"
            else:
                # Create function
                if not role_arn:
                    return {
                        "success": False,
                        "error": "role_arn is required for creating new functions",
                    }
                
                kwargs = {
                    "FunctionName": function_name,
                    "Runtime": runtime,
                    "Role": role_arn,
                    "Handler": handler,
                    "Code": code,
                    "Timeout": timeout,
                    "MemorySize": memory_size,
                }
                if description:
                    kwargs["Description"] = description
                if environment:
                    kwargs["Environment"] = {"Variables": environment}
                
                response = lambda_client.create_function(**kwargs)
                action = "created"
            
            return {
                "success": True,
                "function_name": function_name,
                "function_arn": response['FunctionArn'],
                "runtime": response['Runtime'],
                "handler": response['Handler'],
                "action": action,
                "message": f"Function {action} successfully",
            }

        except ClientError as e:
            logger.error(f"Lambda deploy failed: {e}")
            return {
                "success": False,
                "error": f"Lambda deploy error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "function_name": function_name,
            }
        except Exception as e:
            logger.error(f"Failed to deploy Lambda: {e}")
            return {
                "success": False,
                "error": f"Deploy failed: {str(e)}",
                "function_name": function_name,
            }

    async def lambda_list(
        self,
        max_items: int = 50,
    ) -> dict[str, Any]:
        """List Lambda functions.

        Args:
            max_items: Maximum number of functions to return

        Returns:
            List of Lambda functions
        """
        try:
            lambda_client = self._get_lambda_client()
            
            response = lambda_client.list_functions(MaxItems=max_items)
            
            functions = []
            for func in response.get('Functions', []):
                functions.append({
                    "function_name": func['FunctionName'],
                    "function_arn": func['FunctionArn'],
                    "runtime": func['Runtime'],
                    "handler": func['Handler'],
                    "code_size": func['CodeSize'],
                    "description": func.get('Description', ''),
                    "timeout": func['Timeout'],
                    "memory_size": func['MemorySize'],
                    "last_modified": func['LastModified'],
                })
            
            return {
                "success": True,
                "functions": functions,
                "count": len(functions),
                "region": self.region,
            }

        except ClientError as e:
            logger.error(f"Lambda list failed: {e}")
            return {
                "success": False,
                "error": f"Lambda list error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
            }
        except Exception as e:
            logger.error(f"Failed to list Lambda functions: {e}")
            return {
                "success": False,
                "error": f"List failed: {str(e)}",
            }

    # ===== CLOUDWATCH OPERATIONS =====

    async def cloudwatch_get_metrics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: Optional[list[dict[str, str]]] = None,
        statistic: str = "Average",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        period: int = 300,
    ) -> dict[str, Any]:
        """Get CloudWatch metrics.

        Args:
            namespace: Metric namespace (e.g., 'AWS/EC2', 'AWS/Lambda')
            metric_name: Metric name
            dimensions: Metric dimensions (e.g., [{"Name": "InstanceId", "Value": "i-1234"}])
            statistic: Statistic type (Average, Sum, Minimum, Maximum, SampleCount)
            start_time: Start time (ISO format, default: 1 hour ago)
            end_time: End time (ISO format, default: now)
            period: Period in seconds

        Returns:
            Metric datapoints
        """
        try:
            cloudwatch = self._get_cloudwatch_client()
            
            # Parse times
            if end_time:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            else:
                end_dt = datetime.utcnow()
            
            if start_time:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            else:
                start_dt = end_dt - timedelta(hours=1)
            
            kwargs = {
                "Namespace": namespace,
                "MetricName": metric_name,
                "Statistics": [statistic],
                "StartTime": start_dt,
                "EndTime": end_dt,
                "Period": period,
            }
            
            if dimensions:
                kwargs["Dimensions"] = dimensions
            
            response = cloudwatch.get_metric_statistics(**kwargs)
            
            # Sort datapoints by timestamp
            datapoints = sorted(
                response['Datapoints'],
                key=lambda x: x['Timestamp']
            )
            
            # Format datapoints
            formatted_datapoints = []
            for dp in datapoints:
                formatted_datapoints.append({
                    "timestamp": dp['Timestamp'].isoformat(),
                    "value": dp.get(statistic, 0),
                    "unit": dp.get('Unit', ''),
                })
            
            return {
                "success": True,
                "namespace": namespace,
                "metric_name": metric_name,
                "statistic": statistic,
                "datapoints": formatted_datapoints,
                "count": len(formatted_datapoints),
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
            }

        except ClientError as e:
            logger.error(f"CloudWatch metrics failed: {e}")
            return {
                "success": False,
                "error": f"CloudWatch error: {e.response['Error']['Message']}",
                "error_code": e.response['Error']['Code'],
                "namespace": namespace,
                "metric_name": metric_name,
            }
        except Exception as e:
            logger.error(f"Failed to get CloudWatch metrics: {e}")
            return {
                "success": False,
                "error": f"Metrics query failed: {str(e)}",
                "namespace": namespace,
                "metric_name": metric_name,
            }


def get_server_definition() -> dict[str, Any]:
    """Get AWS MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "aws",
        "category": "cloud",
        "description": "AWS cloud infrastructure operations (S3, EC2, Lambda, CloudWatch)",
        "tools": [
            # S3 Operations
            {
                "name": "s3_upload",
                "description": "Upload file to S3 bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "S3 bucket name",
                        },
                        "key": {
                            "type": "string",
                            "description": "Object key (path in S3)",
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
                            "description": "Object metadata",
                        },
                    },
                    "required": ["bucket", "key"],
                },
            },
            {
                "name": "s3_download",
                "description": "Download file from S3 bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "S3 bucket name",
                        },
                        "key": {
                            "type": "string",
                            "description": "Object key (path in S3)",
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
                    "required": ["bucket", "key"],
                },
            },
            {
                "name": "s3_list",
                "description": "List objects in S3 bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "S3 bucket name",
                        },
                        "prefix": {
                            "type": "string",
                            "description": "Filter by prefix (folder path)",
                        },
                        "max_keys": {
                            "type": "integer",
                            "description": "Maximum number of objects",
                            "default": 100,
                        },
                    },
                    "required": ["bucket"],
                },
            },
            {
                "name": "s3_delete",
                "description": "Delete object from S3 bucket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bucket": {
                            "type": "string",
                            "description": "S3 bucket name",
                        },
                        "key": {
                            "type": "string",
                            "description": "Object key to delete",
                        },
                    },
                    "required": ["bucket", "key"],
                },
            },
            # EC2 Operations
            {
                "name": "ec2_list_instances",
                "description": "List EC2 instances",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "description": "EC2 filters",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of instances",
                            "default": 100,
                        },
                    },
                },
            },
            {
                "name": "ec2_start_instance",
                "description": "Start EC2 instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instance_id": {
                            "type": "string",
                            "description": "EC2 instance ID",
                        }
                    },
                    "required": ["instance_id"],
                },
            },
            {
                "name": "ec2_stop_instance",
                "description": "Stop EC2 instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instance_id": {
                            "type": "string",
                            "description": "EC2 instance ID",
                        },
                        "force": {
                            "type": "boolean",
                            "description": "Force stop",
                            "default": False,
                        },
                    },
                    "required": ["instance_id"],
                },
            },
            {
                "name": "ec2_create_instance",
                "description": "Create EC2 instance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "string",
                            "description": "AMI ID",
                        },
                        "instance_type": {
                            "type": "string",
                            "description": "Instance type (e.g., 't2.micro')",
                        },
                        "key_name": {
                            "type": "string",
                            "description": "SSH key pair name",
                        },
                        "security_group_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Security group IDs",
                        },
                        "subnet_id": {
                            "type": "string",
                            "description": "Subnet ID",
                        },
                        "name": {
                            "type": "string",
                            "description": "Instance name tag",
                        },
                        "user_data": {
                            "type": "string",
                            "description": "User data script",
                        },
                        "min_count": {
                            "type": "integer",
                            "description": "Minimum number of instances",
                            "default": 1,
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "Maximum number of instances",
                            "default": 1,
                        },
                    },
                    "required": ["image_id", "instance_type"],
                },
            },
            # Lambda Operations
            {
                "name": "lambda_invoke",
                "description": "Invoke Lambda function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Lambda function name or ARN",
                        },
                        "payload": {
                            "type": "object",
                            "description": "Function input payload",
                        },
                        "invocation_type": {
                            "type": "string",
                            "enum": ["RequestResponse", "Event", "DryRun"],
                            "description": "Invocation type",
                            "default": "RequestResponse",
                        },
                    },
                    "required": ["function_name"],
                },
            },
            {
                "name": "lambda_deploy",
                "description": "Deploy Lambda function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Function name",
                        },
                        "zip_file_path": {
                            "type": "string",
                            "description": "Local path to deployment package",
                        },
                        "s3_bucket": {
                            "type": "string",
                            "description": "S3 bucket containing deployment package",
                        },
                        "s3_key": {
                            "type": "string",
                            "description": "S3 key of deployment package",
                        },
                        "runtime": {
                            "type": "string",
                            "description": "Lambda runtime",
                            "default": "python3.11",
                        },
                        "handler": {
                            "type": "string",
                            "description": "Function handler",
                            "default": "lambda_function.lambda_handler",
                        },
                        "role_arn": {
                            "type": "string",
                            "description": "IAM role ARN",
                        },
                        "description": {
                            "type": "string",
                            "description": "Function description",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 30,
                        },
                        "memory_size": {
                            "type": "integer",
                            "description": "Memory size in MB",
                            "default": 128,
                        },
                        "environment": {
                            "type": "object",
                            "description": "Environment variables",
                        },
                    },
                    "required": ["function_name"],
                },
            },
            {
                "name": "lambda_list",
                "description": "List Lambda functions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_items": {
                            "type": "integer",
                            "description": "Maximum number of functions",
                            "default": 50,
                        }
                    },
                },
            },
            # CloudWatch Operations
            {
                "name": "cloudwatch_get_metrics",
                "description": "Get CloudWatch metrics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {
                            "type": "string",
                            "description": "Metric namespace (e.g., 'AWS/EC2', 'AWS/Lambda')",
                        },
                        "metric_name": {
                            "type": "string",
                            "description": "Metric name",
                        },
                        "dimensions": {
                            "type": "array",
                            "description": "Metric dimensions",
                        },
                        "statistic": {
                            "type": "string",
                            "enum": ["Average", "Sum", "Minimum", "Maximum", "SampleCount"],
                            "description": "Statistic type",
                            "default": "Average",
                        },
                        "start_time": {
                            "type": "string",
                            "description": "Start time (ISO format)",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "End time (ISO format)",
                        },
                        "period": {
                            "type": "integer",
                            "description": "Period in seconds",
                            "default": 300,
                        },
                    },
                    "required": ["namespace", "metric_name"],
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "high",
            "category": "cloud",
            "requires": ["boto3", "botocore"],
        },
    }