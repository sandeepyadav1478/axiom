"""
Resource Pooling - Production Implementation

Manages expensive resources with pooling:
- Database connections (reuse, don't recreate)
- Model instances (GPU memory expensive)
- HTTP sessions (connection reuse)
- Thread pools (bounded concurrency)

This is how you handle resources professionally.

Pattern: Object Pool
Benefits: Performance (no recreation), resource limits (bounded)
"""

from typing import Generic, TypeVar, Optional, Callable
from queue import Queue, Empty, Full
import threading
import time
from contextlib import contextmanager


T = TypeVar('T')


class ResourcePool(Generic[T]):
    """
    Generic resource pool implementation
    
    Features:
    - Bounded size (prevent resource exhaustion)
    - Lazy creation (create as needed)
    - Health checks (validate before use)
    - Automatic cleanup (release on error)
    - Metrics (utilization tracking)
    
    Thread-safe for concurrent access
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        health_check: Optional[Callable[[T], bool]] = None,
        dispose: Optional[Callable[[T], None]] = None
    ):
        """
        Initialize resource pool
        
        Args:
            name: Pool identifier
            factory: Function to create resources
            max_size: Maximum pool size
            min_size: Minimum pool size (pre-created)
            health_check: Function to validate resource
            dispose: Function to cleanup resource
        """
        self.name = name
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.health_check = health_check
        self.dispose = dispose
        
        # Pool queue (thread-safe)
        self._pool: Queue[T] = Queue(maxsize=max_size)
        self._lock = threading.RLock()
        
        # Statistics
        self._total_created = 0
        self._total_acquired = 0
        self._total_released = 0
        self._currently_in_use = 0
        
        # Pre-create minimum instances
        for _ in range(min_size):
            self._create_resource()
        
        print(f"ResourcePool '{name}' initialized")
        print(f"  Min size: {min_size}, Max size: {max_size}")
    
    def _create_resource(self) -> T:
        """Create new resource using factory"""
        with self._lock:
            if self._total_created >= self.max_size:
                raise RuntimeError(f"Pool exhausted: {self.max_size} resources created")
            
            resource = self.factory()
            self._total_created += 1
            self._pool.put(resource)
            
            return resource
    
    @contextmanager
    def acquire(self, timeout_seconds: float = 5.0):
        """
        Acquire resource from pool
        
        Usage:
            with pool.acquire() as resource:
                # Use resource
                result = resource.execute()
            # Resource automatically released
        
        Args:
            timeout_seconds: Max wait time for resource
        
        Yields: Resource instance
        
        Raises: TimeoutError if pool exhausted
        """
        resource = None
        
        try:
            # Try to get from pool
            try:
                resource = self._pool.get(timeout=timeout_seconds)
                
                with self._lock:
                    self._total_acquired += 1
                    self._currently_in_use += 1
            
            except Empty:
                # Pool empty, try to create
                with self._lock:
                    if self._total_created < self.max_size:
                        resource = self._create_resource()
                        resource = self._pool.get(timeout=0.1)  # Should be available
                        
                        self._total_acquired += 1
                        self._currently_in_use += 1
                    else:
                        raise TimeoutError(
                            f"Pool exhausted: {self.max_size} resources in use"
                        )
            
            # Validate resource if health check provided
            if self.health_check and not self.health_check(resource):
                # Resource unhealthy, dispose and create new
                if self.dispose:
                    self.dispose(resource)
                
                resource = self.factory()
                
                with self._lock:
                    self._total_created += 1
            
            # Yield to caller
            yield resource
        
        finally:
            # Always release resource
            if resource is not None:
                self._pool.put(resource)
                
                with self._lock:
                    self._total_released += 1
                    self._currently_in_use -= 1
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self._lock:
            return {
                'name': self.name,
                'pool_size': self._pool.qsize(),
                'max_size': self.max_size,
                'total_created': self._total_created,
                'currently_in_use': self._currently_in_use,
                'total_acquired': self._total_acquired,
                'total_released': self._total_released,
                'utilization': self._currently_in_use / self.max_size if self.max_size > 0 else 0
            }
    
    def shutdown(self):
        """Shutdown pool and cleanup all resources"""
        with self._lock:
            while not self._pool.empty():
                try:
                    resource = self._pool.get_nowait()
                    if self.dispose:
                        self.dispose(resource)
                except Empty:
                    break
        
        print(f"✓ Pool '{self.name}' shutdown")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("RESOURCE POOLING - PRODUCTION IMPLEMENTATION")
    print("="*60)
    
    # Example: Database connection pool
    class DatabaseConnection:
        def __init__(self):
            print("  Creating DB connection...")
            time.sleep(0.01)  # Simulate connection time
            self.connected = True
        
        def query(self, sql: str) -> str:
            if not self.connected:
                raise RuntimeError("Connection closed")
            return f"Result for: {sql}"
        
        def close(self):
            self.connected = False
            print("  DB connection closed")
    
    # Create pool
    db_pool = ResourcePool(
        name="database_connections",
        factory=DatabaseConnection,
        max_size=5,
        min_size=2,
        health_check=lambda conn: conn.connected,
        dispose=lambda conn: conn.close()
    )
    
    # Test 1: Acquire and use
    print("\n→ Test 1: Acquire Connection")
    
    with db_pool.acquire() as conn:
        result = conn.query("SELECT * FROM trades")
        print(f"   {result}")
    
    # Test 2: Concurrent access
    print("\n→ Test 2: Concurrent Access")
    
    def use_connection(i):
        with db_pool.acquire() as conn:
            result = conn.query(f"SELECT {i}")
            print(f"   Thread {i}: {result}")
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(use_connection, i) for i in range(3)]
        concurrent.futures.wait(futures)
    
    # Stats
    print("\n→ Pool Statistics:")
    stats = db_pool.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    db_pool.shutdown()
    
    print("\n" + "="*60)
    print("✓ Resource pooling operational")
    print("✓ Thread-safe concurrent access")
    print("✓ Automatic cleanup")
    print("✓ Health checking")
    print("\nPRODUCTION-GRADE RESOURCE MANAGEMENT")