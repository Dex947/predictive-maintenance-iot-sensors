"""
Data scaling and normalization module.

This module provides utilities for scaling features for machine learning models.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataScaler:
    """
    Wrapper for different types of scalers with save/load functionality.
    """

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize data scaler.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.scaler = self._get_scaler(scaler_type)
        self.feature_cols = None

    def _get_scaler(self, scaler_type: str):
        """
        Get scaler instance based on type.

        Args:
            scaler_type: Type of scaler

        Returns:
            Scaler instance
        """
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }

        if scaler_type not in scalers:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        logger.info(f"Using {scaler_type} scaler")
        return scalers[scaler_type]

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit scaler and transform data.

        Args:
            df: Input dataframe
            feature_cols: Columns to scale
            exclude_cols: Columns to exclude from scaling

        Returns:
            Dataframe with scaled features
        """
        df_scaled = df.copy()

        # Determine columns to scale
        if exclude_cols:
            cols_to_scale = [col for col in feature_cols if col not in exclude_cols]
        else:
            cols_to_scale = feature_cols

        self.feature_cols = cols_to_scale

        logger.info(f"Fitting scaler on {len(cols_to_scale)} features")

        # Fit and transform
        df_scaled[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])

        return df_scaled

    def transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Transform data using fitted scaler.

        Args:
            df: Input dataframe
            feature_cols: Columns to scale (uses saved cols if None)

        Returns:
            Dataframe with scaled features
        """
        df_scaled = df.copy()

        cols_to_scale = feature_cols if feature_cols else self.feature_cols

        if cols_to_scale is None:
            raise ValueError("No feature columns specified. Fit scaler first or provide feature_cols.")

        logger.info(f"Transforming {len(cols_to_scale)} features")

        # Transform
        df_scaled[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df_scaled

    def inverse_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Inverse transform scaled data.

        Args:
            df: Scaled dataframe
            feature_cols: Columns to inverse scale

        Returns:
            Dataframe with original scale
        """
        df_original = df.copy()

        cols_to_scale = feature_cols if feature_cols else self.feature_cols

        if cols_to_scale is None:
            raise ValueError("No feature columns specified.")

        logger.info(f"Inverse transforming {len(cols_to_scale)} features")

        # Inverse transform
        df_original[cols_to_scale] = self.scaler.inverse_transform(df[cols_to_scale])

        return df_original

    def save(self, filepath: str) -> None:
        """
        Save fitted scaler to file.

        Args:
            filepath: Path to save scaler
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        scaler_data = {
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_cols': self.feature_cols
        }

        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)

        logger.info(f"Scaler saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DataScaler':
        """
        Load fitted scaler from file.

        Args:
            filepath: Path to saved scaler

        Returns:
            Loaded DataScaler instance
        """
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)

        instance = cls(scaler_data['scaler_type'])
        instance.scaler = scaler_data['scaler']
        instance.feature_cols = scaler_data['feature_cols']

        logger.info(f"Scaler loaded from {filepath}")
        return instance


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    group_col: str = 'unit_id',
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Splits by unit_id to ensure no data leakage.

    Args:
        df: Input dataframe
        test_size: Proportion of data for test set
        validation_size: Proportion of training data for validation set
        group_col: Column to group by for splitting
        shuffle: Whether to shuffle units before splitting
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting data: test={test_size}, val={validation_size}")

    # Get unique units
    units = df[group_col].unique()

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(units)

    # Calculate split indices
    n_units = len(units)
    n_test = int(n_units * test_size)
    n_val = int((n_units - n_test) * validation_size)

    # Split units
    test_units = units[:n_test]
    val_units = units[n_test:n_test + n_val]
    train_units = units[n_test + n_val:]

    # Split dataframes
    test_df = df[df[group_col].isin(test_units)].copy()
    val_df = df[df[group_col].isin(val_units)].copy()
    train_df = df[df[group_col].isin(train_units)].copy()

    logger.info(f"Train: {len(train_units)} units, {len(train_df)} samples")
    logger.info(f"Val: {len(val_units)} units, {len(val_df)} samples")
    logger.info(f"Test: {len(test_units)} units, {len(test_df)} samples")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test scaler
    from ..utils.config_loader import load_config
    from ..ingestion.data_loader import get_data_loader

    config = load_config()

    # Load sample data
    loader = get_data_loader('cmapss', config)
    train_df, _, _ = loader.load_dataset('FD001')

    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]

    # Test scaling
    scaler = DataScaler('standard')
    train_scaled = scaler.fit_transform(train_df, sensor_cols)

    print("\nOriginal data:")
    print(train_df[sensor_cols].describe())

    print("\nScaled data:")
    print(train_scaled[sensor_cols].describe())

    # Test save/load
    scaler.save('models/test_scaler.pkl')
    loaded_scaler = DataScaler.load('models/test_scaler.pkl')
    print("\nScaler saved and loaded successfully")
