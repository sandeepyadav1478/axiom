"""
Data Versioning for ML Reproducibility

Tracks versions of:
- Training datasets
- Feature sets
- Model inputs
- Preprocessing pipelines

Ensures reproducible ML experiments.

Uses DVC (Data Version Control) concepts with lightweight implementation.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path


@dataclass
class DataVersion:
    """Version metadata for dataset"""
    version_id: str
    dataset_name: str
    created_at: datetime
    num_samples: int
    num_features: int
    checksum: str
    metadata: Dict


class DataVersionControl:
    """Lightweight data versioning system"""
    
    def __init__(self, storage_path: str = "data_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.versions: Dict[str, DataVersion] = {}
    
    def version_dataset(
        self,
        dataset_name: str,
        data: any,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create version of dataset"""
        
        # Calculate checksum
        if hasattr(data, 'to_json'):
            data_str = data.to_json()
        else:
            data_str = str(data)
        
        checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        # Create version ID
        version_id = f"{dataset_name}_v{len(self.versions) + 1}_{checksum}"
        
        # Store version
        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            created_at=datetime.now(),
            num_samples=len(data) if hasattr(data, '__len__') else 0,
            num_features=data.shape[1] if hasattr(data, 'shape') and len(data.shape) > 1 else 0,
            checksum=checksum,
            metadata=metadata or {}
        )
        
        self.versions[version_id] = version
        
        # Save metadata
        self._save_version_metadata(version)
        
        return version_id
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Retrieve version metadata"""
        return self.versions.get(version_id)
    
    def list_versions(self, dataset_name: Optional[str] = None) -> list:
        """List all versions, optionally filtered by dataset"""
        
        versions = list(self.versions.values())
        
        if dataset_name:
            versions = [v for v in versions if v.dataset_name == dataset_name]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def _save_version_metadata(self, version: DataVersion):
        """Save version metadata to disk"""
        
        metadata_file = self.storage_path / f"{version.version_id}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'version_id': version.version_id,
                'dataset_name': version.dataset_name,
                'created_at': version.created_at.isoformat(),
                'num_samples': version.num_samples,
                'num_features': version.num_features,
                'checksum': version.checksum,
                'metadata': version.metadata
            }, f, indent=2)


if __name__ == "__main__":
    print("Data Versioning for ML")
    print("=" * 60)
    
    dvc = DataVersionControl()
    
    # Version some data
    import pandas as pd
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    
    v1 = dvc.version_dataset(
        'training_data',
        df,
        metadata={'experiment': 'test_1'}
    )
    
    print(f"\nVersioned dataset: {v1}")
    
    versions = dvc.list_versions('training_data')
    print(f"Total versions: {len(versions)}")
    
    print("\n✓ Data versioning for reproducibility")