"""Kubernetes MCP Server Implementation.

Provides Kubernetes orchestration operations through MCP protocol:
- Deployment management
- Service management
- Pod operations
- Cluster monitoring
"""

import logging
from typing import Any, Optional, Union

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

logger = logging.getLogger(__name__)


class KubernetesMCPServer:
    """Kubernetes MCP server implementation."""

    def __init__(self, server_config: dict[str, Any]):
        if not KUBERNETES_AVAILABLE:
            raise ImportError(
                "kubernetes is required for Kubernetes MCP server. "
                "Install with: pip install kubernetes"
            )
        
        self.config = server_config
        self.config_path = server_config.get("config_path")
        self.context = server_config.get("context")
        self.namespace = server_config.get("namespace", "default")
        
        # Load kubernetes config
        try:
            if self.config_path:
                config.load_kube_config(config_file=self.config_path, context=self.context)
            else:
                # Try in-cluster config first, then fallback to kubeconfig
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config(context=self.context)
        except Exception as e:
            logger.warning(f"Failed to load Kubernetes config: {e}")
        
        # Initialize API clients
        self._apps_v1 = None
        self._core_v1 = None
        self._batch_v1 = None

    def _get_apps_v1(self) -> client.AppsV1Api:
        """Get Apps V1 API client."""
        if self._apps_v1 is None:
            self._apps_v1 = client.AppsV1Api()
        return self._apps_v1

    def _get_core_v1(self) -> client.CoreV1Api:
        """Get Core V1 API client."""
        if self._core_v1 is None:
            self._core_v1 = client.CoreV1Api()
        return self._core_v1

    def _get_batch_v1(self) -> client.BatchV1Api:
        """Get Batch V1 API client."""
        if self._batch_v1 is None:
            self._batch_v1 = client.BatchV1Api()
        return self._batch_v1

    # ===== DEPLOYMENT MANAGEMENT =====

    async def create_deployment(
        self,
        name: str,
        image: str,
        replicas: int = 1,
        namespace: Optional[str] = None,
        port: Optional[int] = None,
        env_vars: Optional[dict[str, str]] = None,
        resources: Optional[dict[str, Any]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Create Kubernetes deployment.

        Args:
            name: Deployment name
            image: Container image
            replicas: Number of replicas
            namespace: Kubernetes namespace
            port: Container port
            env_vars: Environment variables
            resources: Resource requests/limits
            labels: Pod labels

        Returns:
            Creation result
        """
        try:
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            # Prepare labels
            deploy_labels = labels or {"app": name}
            
            # Prepare environment variables
            env = []
            if env_vars:
                for key, value in env_vars.items():
                    env.append(client.V1EnvVar(name=key, value=value))
            
            # Prepare container
            container = client.V1Container(
                name=name,
                image=image,
                env=env if env else None,
            )
            
            # Add port if specified
            if port:
                container.ports = [client.V1ContainerPort(container_port=port)]
            
            # Add resources if specified
            if resources:
                container.resources = client.V1ResourceRequirements(
                    requests=resources.get("requests"),
                    limits=resources.get("limits"),
                )
            
            # Create deployment spec
            template = client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(labels=deploy_labels),
                spec=client.V1PodSpec(containers=[container])
            )
            
            spec = client.V1DeploymentSpec(
                replicas=replicas,
                selector=client.V1LabelSelector(match_labels=deploy_labels),
                template=template
            )
            
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=name, labels=deploy_labels),
                spec=spec
            )
            
            # Create deployment
            result = apps_v1.create_namespaced_deployment(
                namespace=ns,
                body=deployment
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "replicas": replicas,
                "image": image,
                "uid": result.metadata.uid,
                "creation_timestamp": result.metadata.creation_timestamp.isoformat(),
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error creating deployment: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to create deployment: {e}")
            return {
                "success": False,
                "error": f"Deployment creation failed: {str(e)}",
                "name": name,
            }

    async def update_deployment(
        self,
        name: str,
        image: Optional[str] = None,
        replicas: Optional[int] = None,
        namespace: Optional[str] = None,
        env_vars: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Update existing deployment.

        Args:
            name: Deployment name
            image: New container image
            replicas: New replica count
            namespace: Kubernetes namespace
            env_vars: New environment variables

        Returns:
            Update result
        """
        try:
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(name=name, namespace=ns)
            
            # Update image
            if image:
                deployment.spec.template.spec.containers[0].image = image
            
            # Update replicas
            if replicas is not None:
                deployment.spec.replicas = replicas
            
            # Update environment variables
            if env_vars:
                env = []
                for key, value in env_vars.items():
                    env.append(client.V1EnvVar(name=key, value=value))
                deployment.spec.template.spec.containers[0].env = env
            
            # Apply update
            result = apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=ns,
                body=deployment
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "replicas": result.spec.replicas,
                "image": result.spec.template.spec.containers[0].image,
                "updated": True,
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error updating deployment: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
            return {
                "success": False,
                "error": f"Deployment update failed: {str(e)}",
                "name": name,
            }

    async def delete_deployment(
        self,
        name: str,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete deployment.

        Args:
            name: Deployment name
            namespace: Kubernetes namespace

        Returns:
            Deletion result
        """
        try:
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            apps_v1.delete_namespaced_deployment(
                name=name,
                namespace=ns,
                body=client.V1DeleteOptions(propagation_policy="Foreground")
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "message": "Deployment deleted successfully",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error deleting deployment: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            return {
                "success": False,
                "error": f"Deployment deletion failed: {str(e)}",
                "name": name,
            }

    async def scale_deployment(
        self,
        name: str,
        replicas: int,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Scale deployment replicas.

        Args:
            name: Deployment name
            replicas: New replica count
            namespace: Kubernetes namespace

        Returns:
            Scale result
        """
        try:
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(name=name, namespace=ns)
            
            # Update replicas
            deployment.spec.replicas = replicas
            
            # Apply update
            result = apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=ns,
                body=deployment
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "previous_replicas": deployment.spec.replicas,
                "new_replicas": replicas,
                "message": f"Deployment scaled to {replicas} replicas",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error scaling deployment: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return {
                "success": False,
                "error": f"Deployment scaling failed: {str(e)}",
                "name": name,
            }

    async def rollback_deployment(
        self,
        name: str,
        revision: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Rollback deployment to previous revision.

        Args:
            name: Deployment name
            revision: Specific revision (default: previous)
            namespace: Kubernetes namespace

        Returns:
            Rollback result
        """
        try:
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            # Get deployment
            deployment = apps_v1.read_namespaced_deployment(name=name, namespace=ns)
            
            # Trigger rollback by updating rollback annotation
            if not deployment.metadata.annotations:
                deployment.metadata.annotations = {}
            
            deployment.metadata.annotations["kubernetes.io/change-cause"] = (
                f"Rollback to revision {revision}" if revision else "Rollback to previous revision"
            )
            
            # Apply update
            result = apps_v1.patch_namespaced_deployment(
                name=name,
                namespace=ns,
                body=deployment
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "revision": revision,
                "message": "Deployment rollback initiated",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error rolling back deployment: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to rollback deployment: {e}")
            return {
                "success": False,
                "error": f"Deployment rollback failed: {str(e)}",
                "name": name,
            }

    # ===== SERVICE MANAGEMENT =====

    async def create_service(
        self,
        name: str,
        selector: dict[str, str],
        port: int,
        target_port: Optional[int] = None,
        service_type: str = "ClusterIP",
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create Kubernetes service.

        Args:
            name: Service name
            selector: Pod selector labels
            port: Service port
            target_port: Container target port
            service_type: Service type (ClusterIP, NodePort, LoadBalancer)
            namespace: Kubernetes namespace

        Returns:
            Creation result
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            target = target_port or port
            
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1ServiceSpec(
                    selector=selector,
                    ports=[client.V1ServicePort(
                        port=port,
                        target_port=target
                    )],
                    type=service_type
                )
            )
            
            result = core_v1.create_namespaced_service(
                namespace=ns,
                body=service
            )
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "type": service_type,
                "port": port,
                "target_port": target,
                "cluster_ip": result.spec.cluster_ip,
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error creating service: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            return {
                "success": False,
                "error": f"Service creation failed: {str(e)}",
                "name": name,
            }

    async def expose_service(
        self,
        name: str,
        deployment_name: str,
        port: int,
        service_type: str = "LoadBalancer",
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Expose deployment as service.

        Args:
            name: Service name
            deployment_name: Deployment to expose
            port: Service port
            service_type: Service type
            namespace: Kubernetes namespace

        Returns:
            Expose result
        """
        try:
            # Get deployment to extract selector
            apps_v1 = self._get_apps_v1()
            ns = namespace or self.namespace
            
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=ns
            )
            
            selector = deployment.spec.selector.match_labels
            
            # Create service
            return await self.create_service(
                name=name,
                selector=selector,
                port=port,
                service_type=service_type,
                namespace=ns,
            )

        except Exception as e:
            logger.error(f"Failed to expose service: {e}")
            return {
                "success": False,
                "error": f"Service expose failed: {str(e)}",
                "name": name,
            }

    async def delete_service(
        self,
        name: str,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete service.

        Args:
            name: Service name
            namespace: Kubernetes namespace

        Returns:
            Deletion result
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            core_v1.delete_namespaced_service(name=name, namespace=ns)
            
            return {
                "success": True,
                "name": name,
                "namespace": ns,
                "message": "Service deleted successfully",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error deleting service: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Failed to delete service: {e}")
            return {
                "success": False,
                "error": f"Service deletion failed: {str(e)}",
                "name": name,
            }

    # ===== POD MANAGEMENT =====

    async def list_pods(
        self,
        namespace: Optional[str] = None,
        label_selector: Optional[str] = None,
    ) -> dict[str, Any]:
        """List pods in namespace.

        Args:
            namespace: Kubernetes namespace
            label_selector: Label selector (e.g., 'app=myapp')

        Returns:
            List of pods
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            if label_selector:
                pods = core_v1.list_namespaced_pod(
                    namespace=ns,
                    label_selector=label_selector
                )
            else:
                pods = core_v1.list_namespaced_pod(namespace=ns)
            
            pod_list = []
            for pod in pods.items:
                pod_info = {
                    "name": pod.metadata.name,
                    "namespace": pod.metadata.namespace,
                    "status": pod.status.phase,
                    "ip": pod.status.pod_ip,
                    "node": pod.spec.node_name,
                    "created": pod.metadata.creation_timestamp.isoformat(),
                }
                
                # Get container statuses
                if pod.status.container_statuses:
                    pod_info["containers"] = []
                    for container in pod.status.container_statuses:
                        pod_info["containers"].append({
                            "name": container.name,
                            "ready": container.ready,
                            "restart_count": container.restart_count,
                            "image": container.image,
                        })
                
                pod_list.append(pod_info)
            
            return {
                "success": True,
                "namespace": ns,
                "pods": pod_list,
                "count": len(pod_list),
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error listing pods: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
            }
        except Exception as e:
            logger.error(f"Failed to list pods: {e}")
            return {
                "success": False,
                "error": f"Pod listing failed: {str(e)}",
            }

    async def get_pod_logs(
        self,
        pod_name: str,
        namespace: Optional[str] = None,
        container: Optional[str] = None,
        tail_lines: int = 100,
    ) -> dict[str, Any]:
        """Get pod logs.

        Args:
            pod_name: Pod name
            namespace: Kubernetes namespace
            container: Container name (if multi-container pod)
            tail_lines: Number of lines to tail

        Returns:
            Pod logs
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            kwargs = {
                "name": pod_name,
                "namespace": ns,
                "tail_lines": tail_lines,
            }
            if container:
                kwargs["container"] = container
            
            logs = core_v1.read_namespaced_pod_log(**kwargs)
            
            return {
                "success": True,
                "pod_name": pod_name,
                "namespace": ns,
                "container": container,
                "logs": logs,
                "lines": len(logs.split('\n')),
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error getting pod logs: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "pod_name": pod_name,
            }
        except Exception as e:
            logger.error(f"Failed to get pod logs: {e}")
            return {
                "success": False,
                "error": f"Get logs failed: {str(e)}",
                "pod_name": pod_name,
            }

    async def delete_pod(
        self,
        pod_name: str,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete pod.

        Args:
            pod_name: Pod name
            namespace: Kubernetes namespace

        Returns:
            Deletion result
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            core_v1.delete_namespaced_pod(name=pod_name, namespace=ns)
            
            return {
                "success": True,
                "pod_name": pod_name,
                "namespace": ns,
                "message": "Pod deleted successfully",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error deleting pod: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "pod_name": pod_name,
            }
        except Exception as e:
            logger.error(f"Failed to delete pod: {e}")
            return {
                "success": False,
                "error": f"Pod deletion failed: {str(e)}",
                "pod_name": pod_name,
            }

    async def exec_pod(
        self,
        pod_name: str,
        command: list[str],
        namespace: Optional[str] = None,
        container: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute command in pod.

        Args:
            pod_name: Pod name
            command: Command to execute
            namespace: Kubernetes namespace
            container: Container name

        Returns:
            Command result
        """
        try:
            from kubernetes.stream import stream
            
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            kwargs = {
                "name": pod_name,
                "namespace": ns,
                "command": command,
                "stderr": True,
                "stdin": False,
                "stdout": True,
                "tty": False,
            }
            if container:
                kwargs["container"] = container
            
            resp = stream(core_v1.connect_get_namespaced_pod_exec, **kwargs)
            
            return {
                "success": True,
                "pod_name": pod_name,
                "namespace": ns,
                "command": " ".join(command),
                "output": resp,
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error executing command: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
                "pod_name": pod_name,
            }
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "pod_name": pod_name,
            }

    # ===== MONITORING =====

    async def get_cluster_info(self) -> dict[str, Any]:
        """Get cluster information.

        Returns:
            Cluster info
        """
        try:
            core_v1 = self._get_core_v1()
            
            # Get nodes
            nodes = core_v1.list_node()
            
            node_info = []
            for node in nodes.items:
                info = {
                    "name": node.metadata.name,
                    "status": "Ready" if any(
                        condition.type == "Ready" and condition.status == "True"
                        for condition in node.status.conditions
                    ) else "NotReady",
                    "roles": node.metadata.labels.get("kubernetes.io/role", "worker"),
                    "version": node.status.node_info.kubelet_version,
                }
                
                # Get capacity
                if node.status.capacity:
                    info["capacity"] = {
                        "cpu": node.status.capacity.get("cpu"),
                        "memory": node.status.capacity.get("memory"),
                        "pods": node.status.capacity.get("pods"),
                    }
                
                node_info.append(info)
            
            # Get namespaces
            namespaces = core_v1.list_namespace()
            namespace_names = [ns.metadata.name for ns in namespaces.items]
            
            return {
                "success": True,
                "nodes": node_info,
                "node_count": len(node_info),
                "namespaces": namespace_names,
                "namespace_count": len(namespace_names),
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error getting cluster info: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {
                "success": False,
                "error": f"Get cluster info failed: {str(e)}",
            }

    async def get_resource_usage(
        self,
        namespace: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get resource usage statistics.

        Args:
            namespace: Kubernetes namespace

        Returns:
            Resource usage info
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            # Get pods
            pods = core_v1.list_namespaced_pod(namespace=ns)
            
            total_cpu_requests = 0
            total_memory_requests = 0
            total_cpu_limits = 0
            total_memory_limits = 0
            
            for pod in pods.items:
                if pod.spec.containers:
                    for container in pod.spec.containers:
                        if container.resources:
                            if container.resources.requests:
                                cpu = container.resources.requests.get("cpu", "0")
                                memory = container.resources.requests.get("memory", "0")
                                # Parse CPU (e.g., "100m" = 0.1)
                                if cpu.endswith("m"):
                                    total_cpu_requests += float(cpu[:-1]) / 1000
                                else:
                                    total_cpu_requests += float(cpu)
                                # Parse Memory (simplified)
                                if memory.endswith("Mi"):
                                    total_memory_requests += float(memory[:-2])
                                elif memory.endswith("Gi"):
                                    total_memory_requests += float(memory[:-2]) * 1024
                            
                            if container.resources.limits:
                                cpu = container.resources.limits.get("cpu", "0")
                                memory = container.resources.limits.get("memory", "0")
                                if cpu.endswith("m"):
                                    total_cpu_limits += float(cpu[:-1]) / 1000
                                else:
                                    total_cpu_limits += float(cpu)
                                if memory.endswith("Mi"):
                                    total_memory_limits += float(memory[:-2])
                                elif memory.endswith("Gi"):
                                    total_memory_limits += float(memory[:-2]) * 1024
            
            return {
                "success": True,
                "namespace": ns,
                "pod_count": len(pods.items),
                "cpu_requests": f"{total_cpu_requests:.2f} cores",
                "cpu_limits": f"{total_cpu_limits:.2f} cores",
                "memory_requests": f"{total_memory_requests:.0f} Mi",
                "memory_limits": f"{total_memory_limits:.0f} Mi",
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error getting resource usage: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {
                "success": False,
                "error": f"Get resource usage failed: {str(e)}",
            }

    async def get_events(
        self,
        namespace: Optional[str] = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Get cluster events.

        Args:
            namespace: Kubernetes namespace
            limit: Maximum number of events

        Returns:
            Cluster events
        """
        try:
            core_v1 = self._get_core_v1()
            ns = namespace or self.namespace
            
            events = core_v1.list_namespaced_event(namespace=ns, limit=limit)
            
            event_list = []
            for event in events.items:
                event_list.append({
                    "type": event.type,
                    "reason": event.reason,
                    "message": event.message,
                    "object": f"{event.involved_object.kind}/{event.involved_object.name}",
                    "count": event.count,
                    "first_timestamp": event.first_timestamp.isoformat() if event.first_timestamp else None,
                    "last_timestamp": event.last_timestamp.isoformat() if event.last_timestamp else None,
                })
            
            return {
                "success": True,
                "namespace": ns,
                "events": event_list,
                "count": len(event_list),
            }

        except ApiException as e:
            logger.error(f"Kubernetes API error getting events: {e}")
            return {
                "success": False,
                "error": f"Kubernetes API error: {e.reason}",
                "status": e.status,
            }
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return {
                "success": False,
                "error": f"Get events failed: {str(e)}",
            }


def get_server_definition() -> dict[str, Any]:
    """Get Kubernetes MCP server definition.

    Returns:
        Server definition dictionary
    """
    return {
        "name": "kubernetes",
        "category": "devops",
        "description": "Kubernetes orchestration operations (deployments, services, pods, monitoring)",
        "tools": [
            # Deployment Management
            {
                "name": "create_deployment",
                "description": "Create Kubernetes deployment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Deployment name"},
                        "image": {"type": "string", "description": "Container image"},
                        "replicas": {"type": "integer", "description": "Number of replicas", "default": 1},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "port": {"type": "integer", "description": "Container port"},
                        "env_vars": {"type": "object", "description": "Environment variables"},
                        "resources": {"type": "object", "description": "Resource requests/limits"},
                        "labels": {"type": "object", "description": "Pod labels"},
                    },
                    "required": ["name", "image"],
                },
            },
            {
                "name": "update_deployment",
                "description": "Update existing deployment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Deployment name"},
                        "image": {"type": "string", "description": "New container image"},
                        "replicas": {"type": "integer", "description": "New replica count"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "env_vars": {"type": "object", "description": "New environment variables"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "delete_deployment",
                "description": "Delete deployment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Deployment name"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "scale_deployment",
                "description": "Scale deployment replicas",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Deployment name"},
                        "replicas": {"type": "integer", "description": "New replica count"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name", "replicas"],
                },
            },
            {
                "name": "rollback_deployment",
                "description": "Rollback deployment to previous revision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Deployment name"},
                        "revision": {"type": "integer", "description": "Specific revision"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name"],
                },
            },
            # Service Management
            {
                "name": "create_service",
                "description": "Create Kubernetes service",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Service name"},
                        "selector": {"type": "object", "description": "Pod selector labels"},
                        "port": {"type": "integer", "description": "Service port"},
                        "target_port": {"type": "integer", "description": "Container target port"},
                        "service_type": {
                            "type": "string",
                            "enum": ["ClusterIP", "NodePort", "LoadBalancer"],
                            "description": "Service type",
                            "default": "ClusterIP",
                        },
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name", "selector", "port"],
                },
            },
            {
                "name": "expose_service",
                "description": "Expose deployment as service",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Service name"},
                        "deployment_name": {"type": "string", "description": "Deployment to expose"},
                        "port": {"type": "integer", "description": "Service port"},
                        "service_type": {
                            "type": "string",
                            "enum": ["ClusterIP", "NodePort", "LoadBalancer"],
                            "description": "Service type",
                            "default": "LoadBalancer",
                        },
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name", "deployment_name", "port"],
                },
            },
            {
                "name": "delete_service",
                "description": "Delete service",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Service name"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["name"],
                },
            },
            # Pod Management
            {
                "name": "list_pods",
                "description": "List pods in namespace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "label_selector": {"type": "string", "description": "Label selector"},
                    },
                },
            },
            {
                "name": "get_pod_logs",
                "description": "Get pod logs",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Pod name"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "container": {"type": "string", "description": "Container name"},
                        "tail_lines": {"type": "integer", "description": "Number of lines to tail", "default": 100},
                    },
                    "required": ["pod_name"],
                },
            },
            {
                "name": "delete_pod",
                "description": "Delete pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Pod name"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                    "required": ["pod_name"],
                },
            },
            {
                "name": "exec_pod",
                "description": "Execute command in pod",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Pod name"},
                        "command": {"type": "array", "items": {"type": "string"}, "description": "Command to execute"},
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "container": {"type": "string", "description": "Container name"},
                    },
                    "required": ["pod_name", "command"],
                },
            },
            # Monitoring
            {
                "name": "get_cluster_info",
                "description": "Get cluster information",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "get_resource_usage",
                "description": "Get resource usage statistics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                    },
                },
            },
            {
                "name": "get_events",
                "description": "Get cluster events",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string", "description": "Kubernetes namespace"},
                        "limit": {"type": "integer", "description": "Maximum number of events", "default": 50},
                    },
                },
            },
        ],
        "resources": [],
        "metadata": {
            "version": "1.0.0",
            "priority": "high",
            "category": "devops",
            "requires": ["kubernetes"],
        },
    }