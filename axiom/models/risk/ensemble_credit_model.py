"""
Ensemble Credit Risk Model: XGBoost + LightGBM + Stacking

Based on: M Zhu, Y Zhang, Y Gong, K Xing (IEEE 2024)
"Ensemble Methodology: Innovations in Credit Default Prediction 
Using LightGBM, XGBoost, and LocalEnsemble"
IEEE 4th International Conference, 2024

This implementation combines multiple gradient boosting algorithms in an optimized
ensemble for superior credit default prediction accuracy and robustness.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class EnsembleConfig:
    """Configuration for Ensemble Credit Model"""
    # XGBoost parameters
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LightGBM parameters
    lgb_n_estimators: int = 300
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.1
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    
    # Random Forest parameters
    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_min_samples_split: int = 10
    
    # Gradient Boosting parameters
    gb_n_estimators: int = 200
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 5
    
    # Ensemble strategy
    ensemble_method: str = "stacking"  # "stacking", "voting", or "both"
    voting_type: str = "soft"  # "soft" or "hard"
    
    # Class imbalance handling
    use_smote: bool = True
    smote_sampling_strategy: float = 0.5  # Minority class ratio after SMOTE
    use_undersampling: bool = False
    
    # Cross-validation
    cv_folds: int = 5
    random_state: int = 42


class EnsembleCreditModel:
    """
    Advanced Ensemble Credit Risk Model
    
    Combines XGBoost, LightGBM, Random Forest, and Gradient Boosting
    using stacking or voting strategies for optimal performance.
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        if not all([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE]):
            missing = []
            if not XGBOOST_AVAILABLE:
                missing.append("xgboost")
            if not LIGHTGBM_AVAILABLE:
                missing.append("lightgbm")
            if not SKLEARN_AVAILABLE:
                missing.append("scikit-learn and imbalanced-learn")
            raise ImportError(f"Missing required packages: {', '.join(missing)}")
            
        self.config = config or EnsembleConfig()
        
        # Initialize base models
        self.xgb_model = None
        self.lgb_model = None
        self.rf_model = None
        self.gb_model = None
        
        # Initialize ensemble
        self.ensemble_model = None
        self.voting_model = None
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.sampler = None
        
        # Feature importance
        self.feature_importance = {}
        
        # Training history
        self.history = {
            'xgb_auc': None,
            'lgb_auc': None,
            'rf_auc': None,
            'gb_auc': None,
            'ensemble_auc': None,
            'voting_auc': None
        }
        
    def _create_base_models(self):
        """Create individual base models"""
        # XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            random_state=self.config.random_state,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # LightGBM
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=self.config.lgb_n_estimators,
            num_leaves=self.config.lgb_num_leaves,
            learning_rate=self.config.lgb_learning_rate,
            subsample=self.config.lgb_subsample,
            colsample_bytree=self.config.lgb_colsample_bytree,
            random_state=self.config.random_state,
            verbose=-1
        )
        
        # Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_split=self.config.rf_min_samples_split,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.gb_model = GradientBoostingClassifier(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=self.config.gb_max_depth,
            random_state=self.config.random_state
        )
        
    def _handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE and/or undersampling
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Resampled (X, y)
        """
        if not self.config.use_smote and not self.config.use_undersampling:
            return X, y
        
        resampling_steps = []
        
        if self.config.use_smote:
            smote = SMOTE(
                sampling_strategy=self.config.smote_sampling_strategy,
                random_state=self.config.random_state
            )
            resampling_steps.append(('smote', smote))
        
        if self.config.use_undersampling:
            undersampler = RandomUnderSampler(
                random_state=self.config.random_state
            )
            resampling_steps.append(('undersample', undersampler))
        
        # Apply resampling
        if resampling_steps:
            for name, sampler in resampling_steps:
                X, y = sampler.fit_resample(X, y)
        
        return X, y
        
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: int = 1
    ) -> Dict[str, float]:
        """
        Train ensemble credit model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Verbosity level
            
        Returns:
            Training results with AUC scores for all models
        """
        # Convert to numpy
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, pd.Series):
            y_val = y_val.values
        
        # Flatten labels if needed
        if len(y_train.shape) > 1:
            y_train = y_train.ravel()
        if y_val is not None and len(y_val.shape) > 1:
            y_val = y_val.ravel()
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self._handle_class_imbalance(
            X_train_scaled, y_train
        )
        
        if verbose > 0:
            print(f"Original samples: {len(y_train)}, After balancing: {len(y_train_balanced)}")
            print(f"Default rate - Original: {y_train.mean():.2%}, Balanced: {y_train_balanced.mean():.2%}")
        
        # Create base models
        self._create_base_models()
        
        # Train individual models
        if verbose > 0:
            print("\nTraining base models...")
        
        # 1. XGBoost
        self.xgb_model.fit(X_train_balanced, y_train_balanced)
        if X_val is not None:
            xgb_pred = self.xgb_model.predict_proba(X_val_scaled)[:, 1]
            self.history['xgb_auc'] = roc_auc_score(y_val, xgb_pred)
            if verbose > 0:
                print(f"  XGBoost AUC: {self.history['xgb_auc']:.4f}")
        
        # 2. LightGBM
        self.lgb_model.fit(X_train_balanced, y_train_balanced)
        if X_val is not None:
            lgb_pred = self.lgb_model.predict_proba(X_val_scaled)[:, 1]
            self.history['lgb_auc'] = roc_auc_score(y_val, lgb_pred)
            if verbose > 0:
                print(f"  LightGBM AUC: {self.history['lgb_auc']:.4f}")
        
        # 3. Random Forest
        self.rf_model.fit(X_train_balanced, y_train_balanced)
        if X_val is not None:
            rf_pred = self.rf_model.predict_proba(X_val_scaled)[:, 1]
            self.history['rf_auc'] = roc_auc_score(y_val, rf_pred)
            if verbose > 0:
                print(f"  Random Forest AUC: {self.history['rf_auc']:.4f}")
        
        # 4. Gradient Boosting
        self.gb_model.fit(X_train_balanced, y_train_balanced)
        if X_val is not None:
            gb_pred = self.gb_model.predict_proba(X_val_scaled)[:, 1]
            self.history['gb_auc'] = roc_auc_score(y_val, gb_pred)
            if verbose > 0:
                print(f"  Gradient Boosting AUC: {self.history['gb_auc']:.4f}")
        
        # Create ensemble
        if verbose > 0:
            print("\nCreating ensemble models...")
        
        # Stacking Ensemble
        if self.config.ensemble_method in ["stacking", "both"]:
            base_estimators = [
                ('xgb', self.xgb_model),
                ('lgb', self.lgb_model),
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ]
            
            self.ensemble_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression(
                    random_state=self.config.random_state,
                    max_iter=1000
                ),
                cv=self.config.cv_folds
            )
            
            self.ensemble_model.fit(X_train_balanced, y_train_balanced)
            
            if X_val is not None:
                ensemble_pred = self.ensemble_model.predict_proba(X_val_scaled)[:, 1]
                self.history['ensemble_auc'] = roc_auc_score(y_val, ensemble_pred)
                if verbose > 0:
                    print(f"  Stacking Ensemble AUC: {self.history['ensemble_auc']:.4f}")
        
        # Voting Ensemble
        if self.config.ensemble_method in ["voting", "both"]:
            voting_estimators = [
                ('xgb', self.xgb_model),
                ('lgb', self.lgb_model),
                ('rf', self.rf_model),
                ('gb', self.gb_model)
            ]
            
            self.voting_model = VotingClassifier(
                estimators=voting_estimators,
                voting=self.config.voting_type
            )
            
            self.voting_model.fit(X_train_balanced, y_train_balanced)
            
            if X_val is not None:
                voting_pred = self.voting_model.predict_proba(X_val_scaled)[:, 1]
                self.history['voting_auc'] = roc_auc_score(y_val, voting_pred)
                if verbose > 0:
                    print(f"  Voting Ensemble AUC: {self.history['voting_auc']:.4f}")
        
        # Extract feature importance
        self._extract_feature_importance()
        
        return self.history
        
    def _extract_feature_importance(self):
        """Extract and aggregate feature importance from all models"""
        self.feature_importance = {
            'xgb': self.xgb_model.feature_importances_ if hasattr(self.xgb_model, 'feature_importances_') else None,
            'lgb': self.lgb_model.feature_importances_ if hasattr(self.lgb_model, 'feature_importances_') else None,
            'rf': self.rf_model.feature_importances_ if hasattr(self.rf_model, 'feature_importances_') else None,
            'gb': self.gb_model.feature_importances_ if hasattr(self.gb_model, 'feature_importances_') else None
        }
        
        # Calculate average importance
        importances = [imp for imp in self.feature_importance.values() if imp is not None]
        if importances:
            self.feature_importance['average'] = np.mean(importances, axis=0)
        
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Predict default probabilities
        
        Args:
            X: Features
            use_ensemble: Use ensemble model (True) or best base model (False)
            
        Returns:
            Default probabilities
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        if use_ensemble and self.ensemble_model is not None:
            probabilities = self.ensemble_model.predict_proba(X_scaled)[:, 1]
        elif use_ensemble and self.voting_model is not None:
            probabilities = self.voting_model.predict_proba(X_scaled)[:, 1]
        else:
            # Use best base model (typically LightGBM or XGBoost)
            probabilities = self.lgb_model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
        
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        threshold: float = 0.5,
        use_ensemble: bool = True
    ) -> np.ndarray:
        """
        Predict default labels
        
        Args:
            X: Features
            threshold: Classification threshold
            use_ensemble: Use ensemble model
            
        Returns:
            Predicted labels (0: no default, 1: default)
        """
        probabilities = self.predict_proba(X, use_ensemble=use_ensemble)
        return (probabilities > threshold).astype(int)
        
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        use_ensemble: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation
        
        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if len(y.shape) > 1:
            y = y.ravel()
        
        # Get predictions
        y_proba = self.predict_proba(X, use_ensemble=use_ensemble)
        y_pred = (y_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary'
        )
        accuracy = (y_pred == y).mean()
        
        return {
            'auc_roc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def get_feature_importance(
        self,
        top_n: int = 20,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get aggregated feature importance
        
        Args:
            top_n: Number of top features to return
            feature_names: Optional feature names
            
        Returns:
            DataFrame with feature importance rankings
        """
        if 'average' not in self.feature_importance:
            return pd.DataFrame()
        
        avg_importance = self.feature_importance['average']
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(avg_importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance,
            'xgb_importance': self.feature_importance.get('xgb', [0] * len(avg_importance)),
            'lgb_importance': self.feature_importance.get('lgb', [0] * len(avg_importance)),
            'rf_importance': self.feature_importance.get('rf', [0] * len(avg_importance)),
            'gb_importance': self.feature_importance.get('gb', [0] * len(avg_importance))
        })
        
        # Sort by average importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
        
    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        cv_folds: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation on all models
        
        Returns:
            CV scores for each model
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if len(y.shape) > 1:
            y = y.ravel()
        
        cv_folds = cv_folds or self.config.cv_folds
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_state)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Create base models
        self._create_base_models()
        
        # Cross-validate each model
        results = {}
        
        for name, model in [('XGBoost', self.xgb_model),
                            ('LightGBM', self.lgb_model),
                            ('RandomForest', self.rf_model),
                            ('GradientBoosting', self.gb_model)]:
            scores = cross_val_score(
                model, X_scaled, y,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            results[name] = scores
            print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
        
    def save(self, path: str):
        """Save ensemble model"""
        import joblib
        joblib.dump({
            'xgb_model': self.xgb_model,
            'lgb_model': self.lgb_model,
            'rf_model': self.rf_model,
            'gb_model': self.gb_model,
            'ensemble_model': self.ensemble_model,
            'voting_model': self.voting_model,
            'scaler': self.scaler,
            'config': self.config,
            'history': self.history,
            'feature_importance': self.feature_importance
        }, path)
        
    def load(self, path: str):
        """Load ensemble model"""
        import joblib
        checkpoint = joblib.load(path)
        self.xgb_model = checkpoint['xgb_model']
        self.lgb_model = checkpoint['lgb_model']
        self.rf_model = checkpoint['rf_model']
        self.gb_model = checkpoint['gb_model']
        self.ensemble_model = checkpoint['ensemble_model']
        self.voting_model = checkpoint['voting_model']
        self.scaler = checkpoint['scaler']
        self.history = checkpoint.get('history', {})
        self.feature_importance = checkpoint.get('feature_importance', {})


def create_sample_credit_features(
    n_samples: int = 1000,
    n_features: int = 20,
    default_rate: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create sample credit features for testing
    
    Returns:
        (X, y) where X is features and y is default labels
    """
    np.random.seed(42)
    
    feature_names = [
        'credit_limit', 'age', 'payment_history_6m', 'balance',
        'utilization_rate', 'num_accounts', 'inquiries_6m',
        'delinquency_2y', 'public_records', 'total_debt',
        'income', 'employment_length', 'home_ownership',
        'debt_to_income', 'revolving_balance', 'num_credit_lines',
        'months_since_last_delinq', 'num_active_accounts',
        'total_credit_limit', 'avg_account_age'
    ]
    
    X = pd.DataFrame()
    
    # Generate features with realistic distributions
    X['credit_limit'] = np.random.lognormal(9, 0.5, n_samples)
    X['age'] = np.random.normal(45, 15, n_samples).clip(18, 90)
    X['payment_history_6m'] = np.random.beta(8, 2, n_samples)
    X['balance'] = X['credit_limit'] * np.random.beta(2, 5, n_samples)
    X['utilization_rate'] = (X['balance'] / X['credit_limit']).clip(0, 1)
    X['num_accounts'] = np.random.poisson(3, n_samples)
    X['inquiries_6m'] = np.random.poisson(1, n_samples)
    X['delinquency_2y'] = np.random.poisson(0.3, n_samples)
    X['public_records'] = np.random.poisson(0.1, n_samples)
    X['total_debt'] = np.random.lognormal(10, 0.8, n_samples)
    X['income'] = np.random.lognormal(11, 0.6, n_samples)
    X['employment_length'] = np.random.exponential(5, n_samples).clip(0, 40)
    X['home_ownership'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    X['debt_to_income'] = (X['total_debt'] / X['income']).clip(0, 2)
    X['revolving_balance'] = X['balance'] * 0.8
    X['num_credit_lines'] = np.random.poisson(5, n_samples)
    X['months_since_last_delinq'] = np.random.exponential(24, n_samples)
    X['num_active_accounts'] = X['num_accounts'] * np.random.beta(8, 2, n_samples)
    X['total_credit_limit'] = X['credit_limit'] * X['num_accounts']
    X['avg_account_age'] = np.random.exponential(36, n_samples)
    
    # Generate labels (defaults)
    # Higher risk factors increase default probability
    risk_score = (
        0.3 * (X['utilization_rate'] > 0.8).astype(int) +
        0.2 * (X['delinquency_2y'] > 0).astype(int) +
        0.2 * (X['debt_to_income'] > 0.5).astype(int) +
        0.15 * (X['inquiries_6m'] > 2).astype(int) +
        0.15 * (X['payment_history_6m'] < 0.7).astype(int)
    )
    
    default_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.5)))
    y = (np.random.rand(n_samples) < default_prob).astype(int)
    
    # Adjust to target default rate
    current_rate = y.mean()
    if current_rate < default_rate:
        # Need more defaults
        non_defaults = np.where(y == 0)[0]
        n_to_flip = int((default_rate - current_rate) * n_samples)
        flip_indices = np.random.choice(non_defaults, size=min(n_to_flip, len(non_defaults)), replace=False)
        y[flip_indices] = 1
    elif current_rate > default_rate:
        # Need fewer defaults
        defaults = np.where(y == 1)[0]
        n_to_flip = int((current_rate - default_rate) * n_samples)
        flip_indices = np.random.choice(defaults, size=min(n_to_flip, len(defaults)), replace=False)
        y[flip_indices] = 0
    
    y = pd.Series(y, name='default')
    
    return X, y


# Example usage
if __name__ == "__main__":
    print("Ensemble Credit Risk Model - Example Usage")
    print("=" * 60)
    
    if not all([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SKLEARN_AVAILABLE]):
        print("ERROR: Missing required packages")
        print(f"  XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗'}")
        print(f"  LightGBM: {'✓' if LIGHTGBM_AVAILABLE else '✗'}")
        print(f"  sklearn: {'✓' if SKLEARN_AVAILABLE else '✗'}")
    else:
        # Create sample data
        print("\n1. Generating sample data...")
        X, y = create_sample_credit_features(n_samples=2000, default_rate=0.15)
        
        # Split
        train_size = 1600
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        val_size = 200
        X_train, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]
        
        print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        print(f"   Default rate: {y_train.mean():.1%}")
        
        # Initialize model
        print("\n2. Initializing Ensemble Model...")
        config = EnsembleConfig(
            ensemble_method="both",  # Both stacking and voting
            use_smote=True
        )
        model = EnsembleCreditModel(config)
        
        # Train
        print("\n3. Training ensemble...")
        history = model.train(X_train, y_train, X_val, y_val, verbose=1)
        
        # Evaluate
        print("\n4. Test set evaluation...")
        results = model.evaluate(X_test, y_test, use_ensemble=True)
        print(f"   AUC-ROC: {results['auc_roc']:.4f}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")
        
        # Feature importance
        print("\n5. Top 10 features:")
        importance_df = model.get_feature_importance(top_n=10, feature_names=X.columns.tolist())
        for idx, row in importance_df.iterrows():
            print(f"   {row['feature']:25s}: {row['importance']:.4f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")