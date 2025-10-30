"""
Demo: Ensemble XGBoost + LightGBM Credit Risk Model

Demonstrates the production-grade ensemble credit risk model combining
XGBoost, LightGBM, Random Forest, and Gradient Boosting for superior
default prediction accuracy.

Based on research from:
M Zhu, Y Zhang, Y Gong, K Xing (IEEE 2024)
"Ensemble Methodology: Innovations in Credit Default Prediction"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from axiom.models.risk.ensemble_credit_model import (
        EnsembleCreditModel,
        EnsembleConfig,
        create_sample_credit_features
    )
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


def plot_model_comparison(history: dict):
    """Plot AUC comparison across models"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    aucs = []
    
    for model_name, auc_score in history.items():
        if auc_score is not None and 'auc' in model_name:
            clean_name = model_name.replace('_auc', '').replace('_', ' ').title()
            models.append(clean_name)
            aucs.append(auc_score)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(models)))
    bars = ax.barh(models, aucs, color=colors)
    
    ax.set_xlabel('AUC-ROC Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, auc_val in zip(bars, aucs):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f'{auc_val:.4f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """Plot feature importance"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = importance_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['importance'], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for name, model in models_dict.items():
        if model is not None:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    print("=" * 90)
    print("Ensemble Credit Risk Model Demo")
    print("XGBoost + LightGBM + Random Forest + Gradient Boosting")
    print("=" * 90)
    print()
    
    if not IMPORTS_AVAILABLE:
        print("ERROR: Required packages not available")
        print("Install: pip install xgboost lightgbm scikit-learn imbalanced-learn")
        return
    
    # Configuration
    print("1. Configuration")
    print("-" * 90)
    config = EnsembleConfig(
        xgb_n_estimators=300,
        lgb_n_estimators=300,
        rf_n_estimators=200,
        ensemble_method="both",  # Stacking AND voting
        use_smote=True,
        smote_sampling_strategy=0.5
    )
    print(f"  Base Models: XGBoost, LightGBM, Random Forest, Gradient Boosting")
    print(f"  Ensemble: {config.ensemble_method}")
    print(f"  SMOTE: {config.use_smote} (sampling strategy: {config.smote_sampling_strategy})")
    print()
    
    # Generate data
    print("2. Generating Credit Data")
    print("-" * 90)
    X, y = create_sample_credit_features(n_samples=3000, default_rate=0.15)
    
    # Split
    train_size = 2400
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    val_size = 400
    X_train, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train, y_val = y_train[:-val_size], y_train[-val_size:]
    
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Default rate: {y.mean():.1%}")
    print()
    
    # Initialize model
    print("3. Initializing Ensemble Model")
    print("-" * 90)
    model = EnsembleCreditModel(config)
    print("  ✓ XGBoost configured")
    print("  ✓ LightGBM configured")
    print("  ✓ Random Forest configured")
    print("  ✓ Gradient Boosting configured")
    print("  ✓ Stacking ensemble configured")
    print("  ✓ Voting ensemble configured")
    print()
    
    # Train
    print("4. Training All Models")
    print("-" * 90)
    history = model.train(X_train, y_train, X_val, y_val, verbose=1)
    print()
    
    # Evaluate on test
    print("5. Test Set Performance")
    print("-" * 90)
    
    # Individual models
    models_to_test = {
        'XGBoost': model.xgb_model,
        'LightGBM': model.lgb_model,
        'Random Forest': model.rf_model,
        'Gradient Boosting': model.gb_model
    }
    
    test_results = {}
    for name, m in models_to_test.items():
        results = model.evaluate(X_test, y_test, use_ensemble=False)
        # Need to use specific model for evaluation
        X_scaled = model.scaler.transform(X_test.values)
        y_pred_proba = m.predict_proba(X_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_results[name] = test_auc
        print(f"  {name:20s} AUC: {test_auc:.4f}")
    
    # Ensemble models
    if model.ensemble_model:
        ensemble_results = model.evaluate(X_test, y_test, use_ensemble=True)
        print(f"\n  {'Stacking Ensemble':20s} AUC: {ensemble_results['auc_roc']:.4f} ⭐")
        print(f"  {'':20s} Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"  {'':20s} Precision: {ensemble_results['precision']:.4f}")
        print(f"  {'':20s} Recall: {ensemble_results['recall']:.4f}")
        print(f"  {'':20s} F1-Score: {ensemble_results['f1_score']:.4f}")
    print()
    
    # Feature importance
    print("6. Top 15 Most Important Features")
    print("-" * 90)
    importance_df = model.get_feature_importance(top_n=15, feature_names=X.columns.tolist())
    for idx, row in importance_df.iterrows():
        print(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:.4f} "
              f"{'█' * int(row['importance'] * 50)}")
    print()
    
    # Business impact
    print("7. Business Impact Analysis")
    print("-" * 90)
    total_customers = len(y_test)
    actual_defaults = y_test.sum()
    
    y_pred = model.predict(X_test, threshold=0.5)
    caught_defaults = ((y_pred == 1) & (y_test == 1)).sum()
    false_positives = ((y_pred == 1) & (y_test == 0)).sum()
    
    avg_credit = 8000
    default_loss = 0.75
    review_cost = 75
    
    potential_loss = actual_defaults * avg_credit * default_loss
    prevented_loss = caught_defaults * avg_credit * default_loss
    review_costs = (caught_defaults + false_positives) * review_cost
    net_savings = prevented_loss - review_costs
    
    print(f"  Portfolio: {total_customers:,} customers")
    print(f"  Actual defaults: {int(actual_defaults):,}")
    print(f"  Caught by model: {int(caught_defaults):,} ({caught_defaults/actual_defaults*100:.1f}%)")
    print(f"  False positives: {int(false_positives):,}")
    print(f"\n  Financial Impact:")
    print(f"    Potential losses: ${potential_loss:,.0f}")
    print(f"    Prevented losses: ${prevented_loss:,.0f}")
    print(f"    Review costs: ${review_costs:,.0f}")
    print(f"    Net savings: ${net_savings:,.0f}")
    print(f"    ROI: {(net_savings/review_costs - 1)*100:.0f}%")
    print()
    
    # Visualizations
    print("8. Generating Visualizations")
    print("-" * 90)
    
    fig1 = plot_model_comparison(history)
    plt.savefig('ensemble_credit_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ Model comparison: ensemble_credit_comparison.png")
    
    fig2 = plot_feature_importance(importance_df, top_n=15)
    plt.savefig('ensemble_credit_features.png', dpi=150, bbox_inches='tight')
    print("  ✓ Feature importance: ensemble_credit_features.png")
    
    fig3 = plot_roc_curves(models_to_test, X_test, y_test)
    plt.savefig('ensemble_credit_roc.png', dpi=150, bbox_inches='tight')
    print("  ✓ ROC curves: ensemble_credit_roc.png")
    print()
    
    # Summary
    print("=" * 90)
    print("Ensemble Credit Model - Superior Performance Through Model Diversity")
    print()
    print("Key Results:")
    print(f"  • Best single model AUC: {max(test_results.values()):.4f}")
    print(f"  • Ensemble AUC: {ensemble_results['auc_roc']:.4f}")
    print(f"  • Improvement: {(ensemble_results['auc_roc']/max(test_results.values()) - 1)*100:.1f}%")
    print(f"  • Net savings: ${net_savings:,.0f} on test portfolio")
    print()
    print("Based on: Zhu et al. (IEEE 2024)")
    print("=" * 90)


if __name__ == "__main__":
    main()