"""
Deep learning models for predictive maintenance.

This module implements LSTM, GRU, and 1D CNN models for time-series prediction.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeepLearningModel:
    """
    Base class for deep learning models.
    """

    def __init__(
        self,
        config: dict,
        model_type: str = 'lstm',
        task: str = 'regression',
        input_shape: Tuple[int, int] = None
    ):
        """
        Initialize deep learning model.

        Args:
            config: Configuration dictionary
            model_type: Type of model ('lstm', 'gru', 'cnn_1d')
            task: Task type ('classification', 'regression')
            input_shape: Shape of input (timesteps, features)
        """
        self.config = config
        self.model_type = model_type
        self.task = task
        self.input_shape = input_shape
        self.model = None
        self.history = None

        # Get model configuration
        self.model_config = config['models'].get(model_type, {})

        # Set random seeds for reproducibility
        seed = config['project']['seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_lstm(self) -> keras.Model:
        """
        Build LSTM model.

        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model")

        model = models.Sequential(name='LSTM_Model')

        units = self.model_config.get('units', [128, 64])
        dropout = self.model_config.get('dropout', 0.2)
        recurrent_dropout = self.model_config.get('recurrent_dropout', 0.2)

        # First LSTM layer
        model.add(layers.LSTM(
            units[0],
            input_shape=self.input_shape,
            return_sequences=len(units) > 1,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name='lstm_1'
        ))

        # Additional LSTM layers
        for i, unit in enumerate(units[1:], start=2):
            return_seq = i < len(units)
            model.add(layers.LSTM(
                unit,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f'lstm_{i}'
            ))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout, name='dropout_final'))

        # Output layer
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(1, activation='linear', name='output'))

        return model

    def build_gru(self) -> keras.Model:
        """
        Build GRU model.

        Returns:
            Compiled Keras model
        """
        logger.info("Building GRU model")

        model = models.Sequential(name='GRU_Model')

        units = self.model_config.get('units', [128, 64])
        dropout = self.model_config.get('dropout', 0.2)
        recurrent_dropout = self.model_config.get('recurrent_dropout', 0.2)

        # First GRU layer
        model.add(layers.GRU(
            units[0],
            input_shape=self.input_shape,
            return_sequences=len(units) > 1,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name='gru_1'
        ))

        # Additional GRU layers
        for i, unit in enumerate(units[1:], start=2):
            return_seq = i < len(units)
            model.add(layers.GRU(
                unit,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                name=f'gru_{i}'
            ))

        # Dense layers
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout, name='dropout_final'))

        # Output layer
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(1, activation='linear', name='output'))

        return model

    def build_cnn_1d(self) -> keras.Model:
        """
        Build 1D CNN model.

        Returns:
            Compiled Keras model
        """
        logger.info("Building 1D CNN model")

        model = models.Sequential(name='CNN1D_Model')

        filters_list = self.model_config.get('filters', [64, 128, 64])
        kernel_size = self.model_config.get('kernel_size', 3)
        pool_size = self.model_config.get('pool_size', 2)
        dropout = self.model_config.get('dropout', 0.3)

        # First Conv1D layer
        model.add(layers.Conv1D(
            filters_list[0],
            kernel_size,
            activation='relu',
            input_shape=self.input_shape,
            name='conv1d_1'
        ))
        model.add(layers.MaxPooling1D(pool_size, name='pool_1'))
        model.add(layers.Dropout(dropout, name='dropout_1'))

        # Additional Conv1D layers
        for i, filters in enumerate(filters_list[1:], start=2):
            model.add(layers.Conv1D(
                filters,
                kernel_size,
                activation='relu',
                name=f'conv1d_{i}'
            ))
            model.add(layers.MaxPooling1D(pool_size, name=f'pool_{i}'))
            model.add(layers.Dropout(dropout, name=f'dropout_{i}'))

        # Flatten and dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(dropout, name='dropout_final'))

        # Output layer
        if self.task == 'classification':
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            model.add(layers.Dense(1, activation='linear', name='output'))

        return model

    def build_model(self) -> keras.Model:
        """
        Build model based on model_type.

        Returns:
            Compiled Keras model
        """
        if self.input_shape is None:
            raise ValueError("input_shape must be provided")

        # Build model architecture
        if self.model_type == 'lstm':
            self.model = self.build_lstm()
        elif self.model_type == 'gru':
            self.model = self.build_gru()
        elif self.model_type == 'cnn_1d':
            self.model = self.build_cnn_1d()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Compile model
        learning_rate = self.model_config.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)

        if self.task == 'classification':
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        else:
            loss = 'mse'
            metrics = ['mae', 'mse']

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info(f"\nModel Summary:")
        self.model.summary(print_fn=logger.info)

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features (samples, timesteps, features)
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Verbosity mode

        Returns:
            Training history
        """
        if self.model is None:
            logger.info("Building model...")
            self.build_model()

        logger.info(f"Training {self.model_type} model")
        logger.info(f"Training samples: {X_train.shape[0]}")

        # Prepare callbacks
        callback_list = self._get_callbacks()

        # Validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        # Training parameters
        epochs = self.model_config.get('epochs', 100)
        batch_size = self.model_config.get('batch_size', 64)

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )

        self.history = history.history
        logger.info("Training completed")

        return self.history

    def _get_callbacks(self) -> List[callbacks.Callback]:
        """
        Get training callbacks.

        Returns:
            List of Keras callbacks
        """
        callback_list = []

        # Early stopping
        patience = self.model_config.get('patience', 15)
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)

        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)

        return callback_list

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features (samples, timesteps, features)

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built/trained. Call build_model() or train() first.")

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

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

        results = self.model.evaluate(X, y, verbose=0)
        metric_names = self.model.metrics_names

        metrics = dict(zip(metric_names, results))

        logger.info(f"Metrics: {metrics}")
        return metrics

    def save(self, filepath: str) -> None:
        """
        Save trained model.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)

        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'task': self.task,
            'input_shape': self.input_shape,
            'model_config': self.model_config,
            'history': self.history,
        }

        metadata_path = filepath.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, filepath: str, config: dict) -> 'DeepLearningModel':
        """
        Load trained model.

        Args:
            filepath: Path to saved model (without extension)
            config: Configuration dictionary

        Returns:
            Loaded model instance
        """
        filepath = Path(filepath)

        # Load Keras model
        model_path = filepath.with_suffix('.h5')
        keras_model = keras.models.load_model(model_path)

        # Load metadata
        metadata_path = filepath.with_suffix('.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            config,
            metadata['model_type'],
            metadata['task'],
            metadata['input_shape']
        )
        instance.model = keras_model
        instance.history = metadata['history']
        instance.model_config = metadata['model_config']

        logger.info(f"Model loaded from {model_path}")
        return instance


if __name__ == "__main__":
    # Test deep learning models
    from ..utils.config_loader import load_config

    config = load_config()

    # Create dummy data
    n_samples = 1000
    timesteps = 30
    n_features = 21

    X_train = np.random.randn(n_samples, timesteps, n_features).astype(np.float32)
    y_train = np.random.randint(0, 2, n_samples).astype(np.float32)

    X_val = np.random.randn(200, timesteps, n_features).astype(np.float32)
    y_val = np.random.randint(0, 2, 200).astype(np.float32)

    # Test LSTM
    print("\n=== LSTM Model ===")
    lstm_model = DeepLearningModel(config, 'lstm', 'classification', (timesteps, n_features))
    lstm_model.build_model()
    history = lstm_model.train(X_train, y_train, X_val, y_val, verbose=0)
    metrics = lstm_model.evaluate(X_val, y_val)
    print(f"Metrics: {metrics}")

    # Test GRU
    print("\n=== GRU Model ===")
    gru_model = DeepLearningModel(config, 'gru', 'classification', (timesteps, n_features))
    gru_model.build_model()

    # Test 1D CNN
    print("\n=== 1D CNN Model ===")
    cnn_model = DeepLearningModel(config, 'cnn_1d', 'classification', (timesteps, n_features))
    cnn_model.build_model()
