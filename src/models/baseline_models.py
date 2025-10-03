"""
Baseline machine learning models for predictive maintenance.

This module implements traditional ML models like Random Forest and XGBoost.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaselineModel:
    """
    Base class for baseline machine learning models.
    """

    def __init__(self, config: dict, model_type: str = 'random_forest', task: str = 'classification'):
        """
        Initialize baseline model.

        Args:
            config: Configuration dictionary
            model_type: Type of model ('random_forest', 'xgboost')
            task: Task type ('classification', 'regression')
        """
        self.config = config
        self.model_type = model_type
        self.task = task
        self.model = None
        self.feature_importance = None

        # Get model configuration
        model_config = config['models'].get(model_type, {})
        self.model_params = model_config.copy()

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the appropriate model based on type and task."""
        logger.info(f"Initializing {self.model_type} for {self.task}")

        if self.model_type == 'random_forest':
            if self.task == 'classification':
                self.model = RandomForestClassifier(**self.model_params)
            else:
                self.model = RandomForestRegressor(**self.model_params)

        elif self.model_type == 'xgboost':
            if self.task == 'classification':
                self.model = xgb.XGBClassifier(**self.model_params)
            else:
                self.model = xgb.XGBRegressor(**self.model_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for XGBoost early stopping)
            y_val: Validation targets

        Returns:
            Training history/metrics
        """
        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

        # XGBoost supports early stopping with validation set
        if self.model_type == 'xgboost' and X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

            # Get evaluation results
            results = self.model.evals_result()

        else:
            self.model.fit(X_train, y_train)
            results = {}

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        logger.info("Training completed")
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        probabilities = self.model.predict_proba(X)
        return probabilities

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Input features
            y: True targets

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")

        predictions = self.predict(X)

        if self.task == 'classification':
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y, predictions, average='weighted', zero_division=0),
            }

            # Print classification report
            logger.info("\nClassification Report:")
            logger.info(f"\n{classification_report(y, predictions, zero_division=0)}")

        else:
            # Regression metrics
            metrics = {
                'mae': mean_absolute_error(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions)),
                'r2_score': r2_score(y, predictions),
            }

        logger.info(f"Metrics: {metrics}")
        return metrics

    def get_feature_importance(
        self,
        feature_names: Optional[list] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            feature_names: Names of features
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return pd.DataFrame()

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

        logger.info(f"\nTop {top_n} Important Features:")
        logger.info(f"\n{importance_df.to_string()}")

        return importance_df

    def save(self, filepath: str) -> None:
        """
        Save trained model.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task': self.task,
            'feature_importance': self.feature_importance,
            'model_params': self.model_params,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, config: dict) -> 'BaselineModel':
        """
        Load trained model.

        Args:
            filepath: Path to saved model
            config: Configuration dictionary

        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        instance = cls(config, model_data['model_type'], model_data['task'])
        instance.model = model_data['model']
        instance.feature_importance = model_data['feature_importance']
        instance.model_params = model_data['model_params']

        logger.info(f"Model loaded from {filepath}")
        return instance


class EnsembleModel:
    """
    Ensemble of multiple baseline models.
    """

    def __init__(self, models: list):
        """
        Initialize ensemble model.

        Args:
            models: List of trained BaselineModel instances
        """
        self.models = models

    def predict(self, X: np.ndarray, method: str = 'average') -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features
            method: Ensemble method ('average', 'voting')

        Returns:
            Ensemble predictions
        """
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        if method == 'average':
            ensemble_pred = np.mean(predictions, axis=0)
        elif method == 'voting':
            # Majority voting for classification
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        return ensemble_pred


if __name__ == "__main__":
    # Test baseline models
    from ..utils.config_loader import load_config
    from ..ingestion.data_loader import get_data_loader
    from ..preprocessing import FeatureEngineer

    config = load_config()

    # Load and prepare data
    loader = get_data_loader('cmapss', config)
    train_df, _, _ = loader.load_dataset('FD001')
    train_df = loader.add_rul_column(train_df)
    train_df = loader.add_labels(train_df)

    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]

    # Prepare features and target
    X = train_df[sensor_cols].values[:1000]  # Use subset for testing
    y = train_df['label_binary'].values[:1000]

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test Random Forest
    print("\n=== Random Forest Classifier ===")
    rf_model = BaselineModel(config, 'random_forest', 'classification')
    rf_model.train(X_train, y_train)
    metrics = rf_model.evaluate(X_test, y_test)
    print(f"Metrics: {metrics}")

    # Test XGBoost
    print("\n=== XGBoost Classifier ===")
    xgb_model = BaselineModel(config, 'xgboost', 'classification')
    xgb_model.train(X_train, y_train, X_test, y_test)
    metrics = xgb_model.evaluate(X_test, y_test)
    print(f"Metrics: {metrics}")
