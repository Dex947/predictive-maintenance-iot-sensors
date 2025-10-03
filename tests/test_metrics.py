"""
Unit tests for evaluation metrics module.
"""

import pytest
import numpy as np

from src.utils.config_loader import load_config
from src.evaluation import PredictiveMaintenanceMetrics, ModelComparator


@pytest.fixture
def config():
    """Load configuration for tests."""
    return load_config()


@pytest.fixture
def metrics_calc(config):
    """Create metrics calculator instance."""
    return PredictiveMaintenanceMetrics(config)


def test_classification_metrics(metrics_calc):
    """Test classification metrics calculation."""
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.95, 0.15, 0.6, 0.85, 0.3])

    metrics = metrics_calc.classification_metrics(y_true, y_pred, y_proba)

    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics

    # Check value ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_regression_metrics(metrics_calc):
    """Test regression metrics calculation."""
    y_true = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1, 0])
    y_pred = np.array([95, 75, 65, 35, 25, 12, 8, 3, 2, 1])

    metrics = metrics_calc.regression_metrics(y_true, y_pred)

    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2_score' in metrics

    # MAE should be positive
    assert metrics['mae'] > 0
    assert metrics['rmse'] > 0


def test_time_to_failure_metrics(metrics_calc):
    """Test TTF-specific metrics."""
    y_true = np.array([100, 80, 60, 40, 20, 10, 5, 2, 1, 0])
    y_pred = np.array([95, 75, 65, 35, 25, 12, 8, 3, 2, 1])

    metrics = metrics_calc.time_to_failure_metrics(y_true, y_pred)

    assert 'nasa_score' in metrics
    assert 'detection_accuracy' in metrics

    # Detection accuracy should be between 0 and 1
    assert 0 <= metrics['detection_accuracy'] <= 1


def test_model_comparator(config):
    """Test model comparator."""
    comparator = ModelComparator(config)

    # Add some results
    comparator.add_result('model_1', {'accuracy': 0.85, 'f1_score': 0.80}, 10.0, 0.1)
    comparator.add_result('model_2', {'accuracy': 0.90, 'f1_score': 0.88}, 15.0, 0.2)

    # Get comparison table
    comparison_df = comparator.get_comparison_table()

    assert len(comparison_df) == 2
    assert 'Model' in comparison_df.columns
    assert 'accuracy' in comparison_df.columns

    # Get best model
    best = comparator.get_best_model('accuracy', maximize=True)

    assert best['model_name'] == 'model_2'
    assert best['value'] == 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
