"""
Configuration loader utility for predictive maintenance system.

This module provides functions to load and validate configuration files.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config/config.yaml

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        # Get project root directory
        current_dir = Path(__file__).parent.parent.parent
        config_path = current_dir / "config" / "config.yaml"
    else:
        current_dir = Path(config_path).parent.parent

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Convert all relative paths to absolute paths based on project root
        _resolve_paths(config, current_dir)

        # Create necessary directories
        _create_directories(config)

        logger.info("Configuration loaded successfully")
        return config

    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def _resolve_paths(config: Dict[str, Any], project_root: Path) -> None:
    """
    Convert all relative paths in config to absolute paths.

    Args:
        config: Configuration dictionary
        project_root: Project root directory
    """
    paths = config.get('paths', {})

    for key, value in paths.items():
        if value and isinstance(value, str):
            # Convert to absolute path if it's relative
            path = Path(value)
            if not path.is_absolute():
                paths[key] = str(project_root / value)
            else:
                paths[key] = str(path)


def _create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories based on configuration.

    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})

    directories = [
        paths.get('data_dir'),
        paths.get('raw_data_dir'),
        paths.get('processed_data_dir'),
        paths.get('models_dir', 'models'),
        paths.get('results_dir', 'results'),
        paths.get('logs_dir', 'logs'),
        paths.get('reports_dir', 'reports'),
    ]

    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")


def get_seed(config: Dict[str, Any]) -> int:
    """
    Get random seed from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Random seed value
    """
    return config.get('project', {}).get('seed', 42)


def get_data_path(config: Dict[str, Any], filename: str, processed: bool = False) -> Path:
    """
    Get full path for data file.

    Args:
        config: Configuration dictionary
        filename: Name of the data file
        processed: If True, returns path in processed_data_dir, else raw_data_dir

    Returns:
        Full path to data file
    """
    paths = config.get('paths', {})

    if processed:
        base_dir = paths.get('processed_data_dir', 'data/processed')
    else:
        base_dir = paths.get('raw_data_dir', 'data/raw')

    return Path(base_dir) / filename


def get_model_path(config: Dict[str, Any], model_name: str) -> Path:
    """
    Get full path for model file.

    Args:
        config: Configuration dictionary
        model_name: Name of the model

    Returns:
        Full path to model file
    """
    models_dir = config.get('paths', {}).get('models_dir', 'models')
    return Path(models_dir) / f"{model_name}.pkl"


def get_model_config(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    """
    Get configuration for specific model type.

    Args:
        config: Configuration dictionary
        model_type: Type of model (e.g., 'random_forest', 'lstm')

    Returns:
        Model-specific configuration dictionary
    """
    return config.get('models', {}).get(model_type, {})


if __name__ == "__main__":
    # Test configuration loader
    logging.basicConfig(level=logging.INFO)

    try:
        config = load_config()
        print("Configuration loaded successfully!")
        print(f"Project name: {config['project']['name']}")
        print(f"Random seed: {get_seed(config)}")

    except Exception as e:
        print(f"Error loading configuration: {e}")
