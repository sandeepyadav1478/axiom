#!/bin/bash
# Automated Model Backup Script

BACKUP_DIR="model_backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

echo "Backing up Axiom ML models..."

# Backup MLflow models
echo "Backing up MLflow registry..."
mlflow models download --model-uri models:/portfolio_transformer/Production --dst $BACKUP_DIR/portfolio_transformer

# Backup model artifacts
echo "Backing up model artifacts..."
cp -r mlruns/ $BACKUP_DIR/mlruns/

# Backup feature store
echo "Backing up feature store..."
cp -r feature_repo/ $BACKUP_DIR/feature_repo/

# Create manifest
cat > $BACKUP_DIR/manifest.txt << EOF
Backup Date: $(date)
Models: 60
Components: MLflow registry, Model artifacts, Feature store
Backup Location: $BACKUP_DIR
EOF

echo "Backup complete: $BACKUP_DIR"