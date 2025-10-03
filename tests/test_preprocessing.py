"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np

from src.utils.config_loader import load_config
from src.preprocessing import FeatureEngineer, SequenceGenerator, DataScaler, split_data


@pytest.fixture
def config():
    """Load configuration for tests."""
    return load_config()


@pytest.fixture
def sample_data():
    """Create sample sensor data."""
    np.random.seed(42)

    data = []
    for unit_id in range(1, 4):
        for cycle in range(1, 51):
            row = {
                'unit_id': unit_id,
                'cycle': cycle,
                'sensor_1': np.random.randn(),
                'sensor_2': np.random.randn(),
                'RUL': 50 - cycle
            }
            data.append(row)

    return pd.DataFrame(data)


def test_feature_engineer_rolling_features(config, sample_data):
    """Test rolling feature creation."""
    engineer = FeatureEngineer(config)

    sensor_cols = ['sensor_1', 'sensor_2']
    features_df = engineer.create_rolling_features(sample_data, sensor_cols)

    # Check new columns created
    assert 'sensor_1_rolling_mean' in features_df.columns
    assert 'sensor_1_rolling_std' in features_df.columns

    # Check no NaN in results (min_periods=1)
    assert features_df['sensor_1_rolling_mean'].isna().sum() == 0


def test_feature_engineer_degradation_features(config, sample_data):
    """Test degradation feature creation."""
    engineer = FeatureEngineer(config)

    sensor_cols = ['sensor_1', 'sensor_2']
    features_df = engineer.create_degradation_features(sample_data, sensor_cols)

    assert 'sensor_1_cumulative_mean' in features_df.columns
    assert 'sensor_1_diff' in features_df.columns
    assert 'sensor_1_cumulative_change' in features_df.columns


def test_sequence_generator(config, sample_data):
    """Test sequence generation."""
    generator = SequenceGenerator(config)

    feature_cols = ['sensor_1', 'sensor_2']
    X, y = generator.create_sequences(sample_data, feature_cols, 'RUL')

    # Check output shape
    window_size = config['preprocessing']['window_size']
    assert X.shape[1] == window_size  # timesteps
    assert X.shape[2] == len(feature_cols)  # features
    assert len(X) == len(y)


def test_data_scaler(config, sample_data):
    """Test data scaling."""
    scaler = DataScaler('standard')

    feature_cols = ['sensor_1', 'sensor_2']
    scaled_df = scaler.fit_transform(sample_data, feature_cols)

    # Check scaling (mean ≈ 0, std ≈ 1 for standard scaler)
    assert abs(scaled_df['sensor_1'].mean()) < 0.1
    assert abs(scaled_df['sensor_1'].std() - 1.0) < 0.1


def test_split_data(sample_data):
    """Test data splitting."""
    train_df, val_df, test_df = split_data(
        sample_data,
        test_size=0.2,
        validation_size=0.1,
        shuffle=True,
        random_state=42
    )

    # Check no overlap in unit_ids
    train_units = set(train_df['unit_id'].unique())
    val_units = set(val_df['unit_id'].unique())
    test_units = set(test_df['unit_id'].unique())

    assert len(train_units & val_units) == 0
    assert len(train_units & test_units) == 0
    assert len(val_units & test_units) == 0

    # Check all data preserved
    total_rows = len(train_df) + len(val_df) + len(test_df)
    assert total_rows == len(sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
