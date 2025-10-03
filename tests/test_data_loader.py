"""
Unit tests for data loading module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config_loader import load_config
from src.ingestion.data_loader import CMAPSSDataLoader, GenericDataLoader, get_data_loader


@pytest.fixture
def config():
    """Load configuration for tests."""
    return load_config()


@pytest.fixture
def cmapss_loader(config):
    """Create CMAPSS data loader instance."""
    return CMAPSSDataLoader(config)


def test_cmapss_loader_initialization(cmapss_loader):
    """Test CMAPSS loader initialization."""
    assert cmapss_loader is not None
    assert len(cmapss_loader.col_names) == 26  # 2 index + 3 settings + 21 sensors


def test_add_rul_column(cmapss_loader):
    """Test RUL column addition."""
    # Create sample data
    df = pd.DataFrame({
        'unit_id': [1, 1, 1, 2, 2],
        'cycle': [1, 2, 3, 1, 2],
        'sensor_1': [1.0, 2.0, 3.0, 1.5, 2.5]
    })

    df_with_rul = cmapss_loader.add_rul_column(df)

    assert 'RUL' in df_with_rul.columns
    assert df_with_rul.loc[0, 'RUL'] == 2  # First cycle of unit 1, max cycle = 3
    assert df_with_rul.loc[2, 'RUL'] == 0  # Last cycle of unit 1


def test_add_labels(cmapss_loader):
    """Test label addition."""
    df = pd.DataFrame({
        'unit_id': [1, 1, 1, 1],
        'cycle': [1, 2, 3, 4],
        'RUL': [50, 30, 15, 5]
    })

    df_with_labels = cmapss_loader.add_labels(df, w1=30, w0=15)

    assert 'label_binary' in df_with_labels.columns
    assert 'RUL_clipped' in df_with_labels.columns

    # Check binary labels
    assert df_with_labels.loc[0, 'label_binary'] == 0  # RUL=50 > w1=30
    assert df_with_labels.loc[1, 'label_binary'] == 1  # RUL=30 <= w1=30
    assert df_with_labels.loc[3, 'label_binary'] == 1  # RUL=5 <= w1=30

    # Check clipped RUL
    assert df_with_labels.loc[0, 'RUL_clipped'] == 15  # Clipped at w0=15
    assert df_with_labels.loc[3, 'RUL_clipped'] == 5   # Not clipped


def test_get_data_loader(config):
    """Test data loader factory function."""
    cmapss_loader = get_data_loader('cmapss', config)
    assert isinstance(cmapss_loader, CMAPSSDataLoader)

    generic_loader = get_data_loader('generic', config)
    assert isinstance(generic_loader, GenericDataLoader)

    with pytest.raises(ValueError):
        get_data_loader('invalid_type', config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
