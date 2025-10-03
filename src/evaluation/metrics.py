"""
Evaluation metrics module for predictive maintenance.

This module provides comprehensive metrics for failure prediction,
including classification metrics, regression metrics, and time-to-failure metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PredictiveMaintenanceMetrics:
    """
    Comprehensive metrics for predictive maintenance evaluation.
    """

    def __init__(self, config: dict):
        """
        Initialize metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.warning_threshold = config['evaluation'].get('warning_threshold_cycles', 50)

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)

        Returns:
            Dictionary of classification metrics
        """
        logger.info("Calculating classification metrics")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Add ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

        # Classification report
        logger.info(f"\nClassification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")

        return metrics

    def regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics for RUL prediction.

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values

        Returns:
            Dictionary of regression metrics
        """
        logger.info("Calculating regression metrics")

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred),
        }

        # Calculate additional metrics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['median_error'] = np.median(np.abs(errors))

        return metrics

    def time_to_failure_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate time-to-failure specific metrics.

        These metrics are crucial for predictive maintenance:
        - Early prediction accuracy
        - Late prediction accuracy
        - Warning lead time

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            threshold: Warning threshold (uses config default if None)

        Returns:
            Dictionary of TTF metrics
        """
        logger.info("Calculating time-to-failure metrics")

        if threshold is None:
            threshold = self.warning_threshold

        metrics = {}

        # Early predictions (RUL > threshold)
        early_mask = y_true > threshold
        if np.sum(early_mask) > 0:
            early_mae = mean_absolute_error(y_true[early_mask], y_pred[early_mask])
            metrics['early_prediction_mae'] = early_mae

        # Late predictions (RUL <= threshold)
        late_mask = y_true <= threshold
        if np.sum(late_mask) > 0:
            late_mae = mean_absolute_error(y_true[late_mask], y_pred[late_mask])
            metrics['late_prediction_mae'] = late_mae

        # Scoring function from NASA competition
        # Penalizes late predictions more than early predictions
        errors = y_pred - y_true
        score = self._scoring_function(errors)
        metrics['nasa_score'] = score

        # Detection accuracy (within threshold)
        detection_accuracy = np.mean(np.abs(errors) <= threshold)
        metrics['detection_accuracy'] = detection_accuracy

        return metrics

    def _scoring_function(self, errors: np.ndarray) -> float:
        """
        NASA C-MAPSS scoring function.

        Penalizes late predictions (negative errors) more than early predictions.

        Args:
            errors: Prediction errors (y_pred - y_true)

        Returns:
            Score value (lower is better)
        """
        scores = np.where(
            errors < 0,
            np.exp(-errors / 13) - 1,  # Late prediction penalty
            np.exp(errors / 10) - 1     # Early prediction penalty
        )
        return np.sum(scores)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task: str = 'regression',
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all relevant metrics based on task type.

        Args:
            y_true: True values
            y_pred: Predicted values
            task: Task type ('classification' or 'regression')
            y_proba: Predicted probabilities (for classification)

        Returns:
            Dictionary of all metrics
        """
        logger.info(f"Calculating all metrics for {task} task")

        all_metrics = {}

        if task == 'classification':
            all_metrics.update(self.classification_metrics(y_true, y_pred, y_proba))

        elif task == 'regression':
            all_metrics.update(self.regression_metrics(y_true, y_pred))
            all_metrics.update(self.time_to_failure_metrics(y_true, y_pred))

        else:
            raise ValueError(f"Unknown task type: {task}")

        logger.info(f"\nAll Metrics: {all_metrics}")
        return all_metrics

    def create_metrics_report(
        self,
        metrics: Dict[str, float],
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Create a formatted metrics report.

        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model

        Returns:
            DataFrame with metrics report
        """
        report_data = []

        for metric_name, metric_value in metrics.items():
            report_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': metric_value
            })

        report_df = pd.DataFrame(report_data)
        return report_df


class ModelComparator:
    """
    Compare multiple models based on various metrics.
    """

    def __init__(self, config: dict):
        """
        Initialize model comparator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = []

    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        training_time: Optional[float] = None,
        prediction_time: Optional[float] = None
    ) -> None:
        """
        Add model results.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            training_time: Training time in seconds
            prediction_time: Prediction time in seconds
        """
        result = {
            'model_name': model_name,
            'metrics': metrics,
            'training_time': training_time,
            'prediction_time': prediction_time,
        }

        self.results.append(result)
        logger.info(f"Added results for {model_name}")

    def get_comparison_table(self, metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create comparison table of all models.

        Args:
            metrics_to_compare: List of metrics to include (uses all if None)

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()

        comparison_data = []

        for result in self.results:
            row = {'Model': result['model_name']}

            # Add metrics
            if metrics_to_compare:
                for metric in metrics_to_compare:
                    row[metric] = result['metrics'].get(metric, np.nan)
            else:
                row.update(result['metrics'])

            # Add timing information
            if result['training_time']:
                row['Training_Time'] = result['training_time']
            if result['prediction_time']:
                row['Prediction_Time'] = result['prediction_time']

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")

        return comparison_df

    def get_best_model(self, metric: str, maximize: bool = True) -> Dict[str, any]:
        """
        Get best model based on a specific metric.

        Args:
            metric: Metric to optimize
            maximize: If True, higher is better; if False, lower is better

        Returns:
            Dictionary with best model information
        """
        if not self.results:
            logger.warning("No results available")
            return {}

        # Extract metric values
        metric_values = []
        for result in self.results:
            value = result['metrics'].get(metric)
            if value is not None:
                metric_values.append((result['model_name'], value, result))

        if not metric_values:
            logger.warning(f"Metric '{metric}' not found in any results")
            return {}

        # Find best model
        if maximize:
            best = max(metric_values, key=lambda x: x[1])
        else:
            best = min(metric_values, key=lambda x: x[1])

        best_model_name, best_value, best_result = best

        logger.info(f"\nBest model based on {metric}: {best_model_name} ({metric}={best_value:.4f})")

        return {
            'model_name': best_model_name,
            'metric': metric,
            'value': best_value,
            'all_metrics': best_result['metrics']
        }

    def save_comparison(self, filepath: str) -> None:
        """
        Save comparison results to CSV.

        Args:
            filepath: Path to save results
        """
        comparison_df = self.get_comparison_table()

        if not comparison_df.empty:
            comparison_df.to_csv(filepath, index=False)
            logger.info(f"Comparison saved to {filepath}")


if __name__ == "__main__":
    # Test metrics
    from ..utils.config_loader import load_config

    config = load_config()

    # Test classification metrics
    print("\n=== Classification Metrics ===")
    y_true_cls = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred_cls = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_proba_cls = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.95, 0.15, 0.6, 0.85, 0.3])

    metrics_calc = PredictiveMaintenanceMetrics(config)
    cls_metrics = metrics_calc.classification_metrics(y_true_cls, y_pred_cls, y_proba_cls)
    print(cls_metrics)

    # Test regression metrics
    print("\n=== Regression Metrics ===")
    y_true_reg = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1, 0])
    y_pred_reg = np.array([95, 75, 65, 35, 25, 12, 8, 3, 2, 1])

    reg_metrics = metrics_calc.regression_metrics(y_true_reg, y_pred_reg)
    ttf_metrics = metrics_calc.time_to_failure_metrics(y_true_reg, y_pred_reg)
    print(reg_metrics)
    print(ttf_metrics)
