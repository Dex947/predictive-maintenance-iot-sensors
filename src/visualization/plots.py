"""
Visualization module for predictive maintenance.

This module provides comprehensive visualization functions including
time-series plots, feature importance, SHAP explanations, and model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import shap

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


class PredictiveMaintenanceVisualizer:
    """
    Comprehensive visualization for predictive maintenance analysis.
    """

    def __init__(self, config: dict, output_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            config: Configuration dictionary
            output_dir: Directory to save plots
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization config
        viz_config = config.get('visualization', {})
        self.plot_formats = viz_config.get('plot_formats', ['png'])
        self.dpi = viz_config.get('dpi', 300)
        self.figsize = tuple(viz_config.get('figsize', [12, 8]))

    def plot_sensor_degradation(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        unit_ids: Optional[List[int]] = None,
        save_name: str = 'sensor_degradation'
    ) -> None:
        """
        Plot sensor degradation over time for selected units.

        Args:
            df: Dataframe with sensor data
            sensor_cols: List of sensor columns to plot
            unit_ids: List of unit IDs to plot (plots first 3 if None)
            save_name: Filename for saving plot
        """
        logger.info("Plotting sensor degradation")

        if unit_ids is None:
            unit_ids = df['unit_id'].unique()[:3]

        n_sensors = min(len(sensor_cols), 6)  # Limit to 6 sensors for clarity
        selected_sensors = sensor_cols[:n_sensors]

        fig, axes = plt.subplots(n_sensors, 1, figsize=(self.figsize[0], n_sensors * 3))
        if n_sensors == 1:
            axes = [axes]

        for idx, sensor in enumerate(selected_sensors):
            for unit_id in unit_ids:
                unit_data = df[df['unit_id'] == unit_id]
                axes[idx].plot(unit_data['cycle'], unit_data[sensor], label=f'Unit {unit_id}', alpha=0.7)

            axes[idx].set_xlabel('Cycle')
            axes[idx].set_ylabel(sensor)
            axes[idx].set_title(f'{sensor} Degradation')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_rul_distribution(
        self,
        df: pd.DataFrame,
        rul_col: str = 'RUL',
        save_name: str = 'rul_distribution'
    ) -> None:
        """
        Plot RUL (Remaining Useful Life) distribution.

        Args:
            df: Dataframe with RUL column
            rul_col: Name of RUL column
            save_name: Filename for saving plot
        """
        logger.info("Plotting RUL distribution")

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Histogram
        axes[0].hist(df[rul_col], bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Remaining Useful Life (cycles)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('RUL Distribution')
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(df[rul_col], vert=True)
        axes[1].set_ylabel('Remaining Useful Life (cycles)')
        axes[1].set_title('RUL Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_name: str = 'feature_importance'
    ) -> None:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to plot
            save_name: Filename for saving plot
        """
        logger.info("Plotting feature importance")

        # Get top N features
        plot_df = importance_df.head(top_n).sort_values('importance')

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_name: str = 'confusion_matrix'
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            save_name: Filename for saving plot
        """
        logger.info("Plotting confusion matrix")

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels if labels else ['Class 0', 'Class 1'],
                   yticklabels=labels if labels else ['Class 0', 'Class 1'])

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_name: str = 'prediction_vs_actual'
    ) -> None:
        """
        Plot predicted vs actual values.

        Args:
            y_true: True values
            y_pred: Predicted values
            save_name: Filename for saving plot
        """
        logger.info("Plotting predictions vs actual")

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                    'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title('Predicted vs Actual RUL')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Error distribution
        errors = y_pred - y_true
        axes[1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error (Predicted - Actual)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Prediction Error Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'training_history'
    ) -> None:
        """
        Plot training history for deep learning models.

        Args:
            history: Training history dictionary
            save_name: Filename for saving plot
        """
        logger.info("Plotting training history")

        metrics = [key for key in history.keys() if not key.startswith('val_')]
        n_metrics = len(metrics)

        fig, axes = plt.subplots(n_metrics, 1, figsize=(self.figsize[0], n_metrics * 4))
        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            axes[idx].plot(history[metric], label=f'Train {metric}')

            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[idx].plot(history[val_metric], label=f'Val {metric}')

            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} over Epochs')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_shap_summary(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
        save_name: str = 'shap_summary'
    ) -> None:
        """
        Plot SHAP summary for model interpretability.

        Args:
            model: Trained model
            X: Feature data
            feature_names: Feature names
            max_display: Maximum features to display
            save_name: Filename for saving plot
        """
        logger.info("Creating SHAP summary plot")

        try:
            # Create SHAP explainer
            # Use a sample for efficiency
            shap_config = self.config.get('visualization', {}).get('shap', {})
            sample_size = min(shap_config.get('sample_size', 100), len(X))
            X_sample = X[:sample_size]

            # Tree explainer for tree-based models
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)

            shap_values = explainer.shap_values(X_sample)

            # Summary plot
            fig = plt.figure(figsize=self.figsize)
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )

            plt.tight_layout()
            self._save_figure(fig, save_name)
            plt.close()

            logger.info("SHAP summary plot created")

        except Exception as e:
            logger.warning(f"Could not create SHAP plot: {e}")

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str],
        save_name: str = 'model_comparison'
    ) -> None:
        """
        Plot model comparison across metrics.

        Args:
            comparison_df: DataFrame with model comparison
            metrics: List of metrics to plot
            save_name: Filename for saving plot
        """
        logger.info("Plotting model comparison")

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5))

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                data = comparison_df[['Model', metric]].sort_values(metric)

                axes[idx].barh(data['Model'], data[metric], color='steelblue')
                axes[idx].set_xlabel(metric)
                axes[idx].set_title(f'Model Comparison: {metric}')
                axes[idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        self._save_figure(fig, save_name)
        plt.close()

    def plot_interactive_timeseries(
        self,
        df: pd.DataFrame,
        sensor_cols: List[str],
        unit_id: int,
        save_name: str = 'interactive_timeseries'
    ) -> None:
        """
        Create interactive time-series plot using Plotly.

        Args:
            df: Dataframe with sensor data
            sensor_cols: List of sensor columns
            unit_id: Unit ID to plot
            save_name: Filename for saving plot
        """
        logger.info(f"Creating interactive time-series plot for unit {unit_id}")

        unit_data = df[df['unit_id'] == unit_id]

        fig = go.Figure()

        for sensor in sensor_cols[:10]:  # Limit to 10 sensors
            fig.add_trace(go.Scatter(
                x=unit_data['cycle'],
                y=unit_data[sensor],
                mode='lines',
                name=sensor
            ))

        fig.update_layout(
            title=f'Sensor Time Series for Unit {unit_id}',
            xaxis_title='Cycle',
            yaxis_title='Sensor Value',
            hovermode='x unified',
            height=600,
            width=1200
        )

        # Save as HTML
        if 'html' in self.plot_formats:
            output_path = self.output_dir / f"{save_name}.html"
            fig.write_html(output_path)
            logger.info(f"Interactive plot saved to {output_path}")

    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure in configured formats.

        Args:
            fig: Matplotlib figure
            filename: Base filename (without extension)
        """
        for fmt in self.plot_formats:
            if fmt in ['png', 'jpg', 'pdf', 'svg']:
                output_path = self.output_dir / f"{filename}.{fmt}"
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Plot saved to {output_path}")


if __name__ == "__main__":
    # Test visualizations
    from ..utils.config_loader import load_config
    from ..ingestion.data_loader import get_data_loader

    config = load_config()

    # Load sample data
    loader = get_data_loader('cmapss', config)
    train_df, _, _ = loader.load_dataset('FD001')
    train_df = loader.add_rul_column(train_df)

    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]

    # Test visualizer
    viz = PredictiveMaintenanceVisualizer(config, 'results')

    # Plot sensor degradation
    viz.plot_sensor_degradation(train_df, sensor_cols, unit_ids=[1, 2, 3])

    # Plot RUL distribution
    viz.plot_rul_distribution(train_df)

    print("Visualization tests completed!")
