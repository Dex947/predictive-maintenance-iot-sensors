"""
Feature engineering module for predictive maintenance.

This module provides functions for creating rolling window features,
statistical features, and other domain-specific features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for time-series sensor data.

    Creates rolling window statistics, degradation indicators,
    and other predictive features.
    """

    def __init__(self, config: dict):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window_size = config['preprocessing']['window_size']
        self.stat_features = config['preprocessing']['statistical_features']

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: str = 'unit_id'
    ) -> pd.DataFrame:
        """
        Create rolling window statistical features.

        Args:
            df: Input dataframe
            sensor_cols: List of sensor column names
            group_col: Column to group by (e.g., 'unit_id')

        Returns:
            Dataframe with added rolling features
        """
        logger.info(f"Creating rolling features with window size {self.window_size}")

        df_features = df.copy()
        new_features = {}

        for sensor in sensor_cols:
            grouped = df_features.groupby(group_col)[sensor]

            # Rolling statistics - collect in dict first to avoid fragmentation
            if 'mean' in self.stat_features:
                new_features[f'{sensor}_rolling_mean'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=1).mean()
                )

            if 'std' in self.stat_features:
                new_features[f'{sensor}_rolling_std'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=1).std()
                )

            if 'min' in self.stat_features:
                new_features[f'{sensor}_rolling_min'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=1).min()
                )

            if 'max' in self.stat_features:
                new_features[f'{sensor}_rolling_max'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=1).max()
                )

            if 'rms' in self.stat_features:
                new_features[f'{sensor}_rolling_rms'] = grouped.transform(
                    lambda x: np.sqrt((x ** 2).rolling(window=self.window_size, min_periods=1).mean())
                )

        # Concatenate all new features at once
        df_features = pd.concat([df_features, pd.DataFrame(new_features, index=df_features.index)], axis=1)

        logger.info(f"Created rolling features. New shape: {df_features.shape}")
        return df_features

    def create_statistical_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: str = 'unit_id'
    ) -> pd.DataFrame:
        """
        Create advanced statistical features per unit.

        Args:
            df: Input dataframe
            sensor_cols: List of sensor column names
            group_col: Column to group by

        Returns:
            Dataframe with added statistical features
        """
        logger.info("Creating advanced statistical features")

        df_features = df.copy()

        for sensor in sensor_cols:
            grouped = df_features.groupby(group_col)[sensor]

            if 'kurtosis' in self.stat_features:
                df_features[f'{sensor}_kurtosis'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=4).apply(stats.kurtosis, raw=True)
                )

            if 'skewness' in self.stat_features:
                df_features[f'{sensor}_skewness'] = grouped.transform(
                    lambda x: x.rolling(window=self.window_size, min_periods=3).apply(stats.skew, raw=True)
                )

        # Fill NaN values with 0 (from kurtosis/skewness calculations)
        df_features = df_features.fillna(0)

        logger.info(f"Created statistical features. New shape: {df_features.shape}")
        return df_features

    def create_degradation_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: str = 'unit_id'
    ) -> pd.DataFrame:
        """
        Create degradation indicator features.

        These features capture the trend and change over time.

        Args:
            df: Input dataframe
            sensor_cols: List of sensor column names
            group_col: Column to group by

        Returns:
            Dataframe with degradation features
        """
        logger.info("Creating degradation features")

        df_features = df.copy()
        new_features = {}

        for sensor in sensor_cols:
            grouped = df_features.groupby(group_col)[sensor]

            # Cumulative mean (trend) - use expanding window properly
            new_features[f'{sensor}_cumulative_mean'] = grouped.transform(
                lambda x: x.expanding().mean()
            )

            # Rate of change
            new_features[f'{sensor}_diff'] = grouped.transform(
                lambda x: x.diff()
            ).fillna(0)

            # Cumulative sum of absolute changes
            new_features[f'{sensor}_cumulative_change'] = grouped.transform(
                lambda x: x.diff().abs().cumsum()
            ).fillna(0)

        # Concatenate all new features at once
        df_features = pd.concat([df_features, pd.DataFrame(new_features, index=df_features.index)], axis=1)

        logger.info(f"Created degradation features. New shape: {df_features.shape}")
        return df_features

    def create_all_features(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        group_col: str = 'unit_id'
    ) -> pd.DataFrame:
        """
        Create all feature types.

        Args:
            df: Input dataframe
            sensor_cols: List of sensor column names
            group_col: Column to group by

        Returns:
            Dataframe with all engineered features
        """
        logger.info("Creating all features")

        df_features = df.copy()

        # Rolling features
        df_features = self.create_rolling_features(df_features, sensor_cols, group_col)

        # Statistical features
        df_features = self.create_statistical_features(df_features, sensor_cols, group_col)

        # Degradation features
        df_features = self.create_degradation_features(df_features, sensor_cols, group_col)

        logger.info(f"All features created. Final shape: {df_features.shape}")
        return df_features

    def remove_low_variance_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.

        Args:
            df: Input dataframe
            threshold: Variance threshold

        Returns:
            Tuple of (filtered dataframe, list of removed columns)
        """
        logger.info(f"Removing low variance features (threshold={threshold})")

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate variance
        variances = df[numeric_cols].var()

        # Find low variance columns
        low_var_cols = variances[variances < threshold].index.tolist()

        if low_var_cols:
            logger.info(f"Removing {len(low_var_cols)} low variance columns")
            df_filtered = df.drop(columns=low_var_cols)
        else:
            logger.info("No low variance columns found")
            df_filtered = df.copy()

        return df_filtered, low_var_cols


class SequenceGenerator:
    """
    Generate sequences for deep learning models.

    Creates sliding windows of time-series data for RNN/LSTM/CNN models.
    """

    def __init__(self, config: dict):
        """
        Initialize sequence generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.window_size = config['preprocessing']['window_size']
        self.stride = config['preprocessing'].get('stride', 1)

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'RUL',
        group_col: str = 'unit_id'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction.

        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            target_col: Target column name
            group_col: Column to group by

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        logger.info(f"Creating sequences with window_size={self.window_size}, stride={self.stride}")

        sequences = []
        targets = []

        # Group by unit
        for unit_id, group_df in df.groupby(group_col):
            # Extract features and target
            unit_features = group_df[feature_cols].values
            unit_targets = group_df[target_col].values

            # Create sequences with stride
            for i in range(0, len(unit_features) - self.window_size + 1, self.stride):
                sequences.append(unit_features[i:i + self.window_size])
                # Target is the last value in the window
                targets.append(unit_targets[i + self.window_size - 1])

        X = np.array(sequences)
        y = np.array(targets)

        logger.info(f"Created sequences: X shape={X.shape}, y shape={y.shape}")

        return X, y

    def create_sequences_classification(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'label_binary',
        group_col: str = 'unit_id'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for classification task.

        Args:
            df: Input dataframe
            feature_cols: List of feature column names
            label_col: Label column name
            group_col: Column to group by

        Returns:
            Tuple of (X_sequences, y_labels)
        """
        return self.create_sequences(df, feature_cols, label_col, group_col)


if __name__ == "__main__":
    # Test feature engineering
    from ..utils.config_loader import load_config
    from ..ingestion.data_loader import get_data_loader

    config = load_config()

    # Load sample data
    loader = get_data_loader('cmapss', config)
    train_df, _, _ = loader.load_dataset('FD001')

    # Add RUL
    train_df = loader.add_rul_column(train_df)
    train_df = loader.add_labels(train_df)

    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]

    # Test feature engineering
    engineer = FeatureEngineer(config)
    train_features = engineer.create_all_features(train_df, sensor_cols)

    print("\nOriginal shape:", train_df.shape)
    print("Feature engineered shape:", train_features.shape)
    print("\nNew columns:", [col for col in train_features.columns if col not in train_df.columns][:10])
