"""
Data ingestion module for predictive maintenance datasets.

This module handles downloading and loading various predictive maintenance datasets.
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config_loader import load_config

logger = get_logger(__name__)


class CMAPSSDataLoader:
    """
    Loader for NASA C-MAPSS Turbofan Engine Degradation Dataset.

    This class handles downloading, extracting, and loading the NASA C-MAPSS dataset.
    """

    def __init__(self, config: dict):
        """
        Initialize CMAPSS data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['paths']['raw_data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Column names for CMAPSS dataset
        self.index_names = ['unit_id', 'cycle']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names

    def download_dataset(self, force: bool = False) -> None:
        """
        Download NASA C-MAPSS dataset.

        Args:
            force: If True, download even if file exists
        """
        url = self.config['data_sources']['nasa_cmapss']['url']
        zip_path = self.data_dir / "cmapss.zip"

        if zip_path.exists() and not force:
            logger.info(f"Dataset already downloaded: {zip_path}")
            return

        logger.info(f"Downloading C-MAPSS dataset from {url}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            logger.info(f"Dataset downloaded successfully to {zip_path}")

            # Extract zip file
            self._extract_dataset(zip_path)

        except requests.RequestException as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def _extract_dataset(self, zip_path: Path) -> None:
        """
        Extract downloaded zip file.

        Args:
            zip_path: Path to zip file
        """
        logger.info(f"Extracting {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            # Move files from subfolder to data_dir if they're in a subfolder
            extracted_subfolders = [d for d in self.data_dir.iterdir() if d.is_dir() and 'Turbofan' in d.name]
            if extracted_subfolders:
                subfolder = extracted_subfolders[0]
                logger.info(f"Moving files from {subfolder.name} to {self.data_dir}")

                import shutil
                for file in subfolder.iterdir():
                    if file.is_file():
                        dest = self.data_dir / file.name
                        if not dest.exists():
                            shutil.move(str(file), str(dest))

                # Remove empty subfolder
                try:
                    subfolder.rmdir()
                except:
                    pass

            logger.info("Dataset extracted successfully")

        except zipfile.BadZipFile as e:
            logger.error(f"Error extracting dataset: {e}")
            raise

    def load_dataset(self, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load a specific subset of the C-MAPSS dataset.

        Args:
            subset: Dataset subset (FD001, FD002, FD003, FD004)

        Returns:
            Tuple of (train_df, test_df, rul_df)
        """
        logger.info(f"Loading C-MAPSS {subset} dataset")

        # Check if dataset exists, download if not
        train_file = self.data_dir / f"train_{subset}.txt"
        if not train_file.exists():
            logger.warning("Dataset not found. Attempting to download...")
            self.download_dataset()

        # Load training data
        train_df = pd.read_csv(
            self.data_dir / f"train_{subset}.txt",
            sep=r'\s+',
            header=None,
            names=self.col_names
        )

        # Load test data
        test_df = pd.read_csv(
            self.data_dir / f"test_{subset}.txt",
            sep=r'\s+',
            header=None,
            names=self.col_names
        )

        # Load RUL (Remaining Useful Life) for test data
        rul_df = pd.read_csv(
            self.data_dir / f"RUL_{subset}.txt",
            sep=r'\s+',
            header=None,
            names=['RUL']
        )

        logger.info(f"Loaded {subset}: Train shape={train_df.shape}, Test shape={test_df.shape}")

        return train_df, test_df, rul_df

    def add_rul_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Remaining Useful Life (RUL) column to dataframe.

        RUL is calculated as the reverse of the cycle count for each unit.

        Args:
            df: Input dataframe with 'unit_id' and 'cycle' columns

        Returns:
            Dataframe with added 'RUL' column
        """
        # Calculate max cycle for each unit
        max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']

        # Merge and calculate RUL
        df = df.merge(max_cycles, on='unit_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df = df.drop('max_cycle', axis=1)

        return df

    def add_labels(self, df: pd.DataFrame, w1: int = 30, w0: int = 15) -> pd.DataFrame:
        """
        Add binary and multiclass labels for failure prediction.

        Args:
            df: Input dataframe with 'RUL' column
            w1: Threshold for binary classification (RUL <= w1 means failure soon)
            w0: Threshold for piece-wise RUL clipping

        Returns:
            Dataframe with added label columns
        """
        # Binary label: 1 if failure within w1 cycles
        df['label_binary'] = (df['RUL'] <= w1).astype(int)

        # Piece-wise linear RUL (clip at w0)
        df['RUL_clipped'] = df['RUL'].clip(upper=w0)

        return df


class GenericDataLoader:
    """
    Generic data loader for CSV-based predictive maintenance datasets.

    This class can load datasets from Kaggle or local CSV files.
    """

    def __init__(self, config: dict):
        """
        Initialize generic data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['paths']['raw_data_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {filepath}")

        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded dataframe with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

    def download_kaggle_dataset(self, dataset_name: str, force: bool = False) -> None:
        """
        Download dataset from Kaggle using Kaggle API.

        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'user/dataset-name')
            force: If True, download even if exists

        Note:
            Requires kaggle.json credentials in ~/.kaggle/
        """
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")

        try:
            import kaggle

            output_dir = self.data_dir / dataset_name.split('/')[-1]

            if output_dir.exists() and not force:
                logger.info(f"Dataset already exists: {output_dir}")
                return

            kaggle.api.dataset_download_files(
                dataset_name,
                path=output_dir,
                unzip=True
            )

            logger.info(f"Dataset downloaded to {output_dir}")

        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {e}")
            raise


def get_data_loader(dataset_type: str, config: dict = None):
    """
    Factory function to get appropriate data loader.

    Args:
        dataset_type: Type of dataset ('cmapss', 'generic')
        config: Configuration dictionary

    Returns:
        Appropriate data loader instance
    """
    if config is None:
        config = load_config()

    loaders = {
        'cmapss': CMAPSSDataLoader,
        'generic': GenericDataLoader,
    }

    loader_class = loaders.get(dataset_type.lower())

    if loader_class is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return loader_class(config)


if __name__ == "__main__":
    # Test data loader
    config = load_config()

    # Test CMAPSS loader
    cmapss_loader = get_data_loader('cmapss', config)
    train_df, test_df, rul_df = cmapss_loader.load_dataset('FD001')

    print("\nTrain data:")
    print(train_df.head())

    print("\nTest data:")
    print(test_df.head())

    print("\nRUL data:")
    print(rul_df.head())
