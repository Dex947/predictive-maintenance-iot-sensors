"""
Main workflow pipeline for Predictive Maintenance System.

This script orchestrates the entire predictive maintenance pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training (baseline and deep learning)
4. Evaluation and comparison
5. Visualization and reporting
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config_loader import load_config, get_seed
from src.utils.logger import get_logger
from src.ingestion.data_loader import get_data_loader
from src.preprocessing import FeatureEngineer, SequenceGenerator, DataScaler, split_data
from src.models import BaselineModel, DeepLearningModel
from src.evaluation import PredictiveMaintenanceMetrics, ModelComparator
from src.visualization import PredictiveMaintenanceVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predictive Maintenance Pipeline')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='FD001',
                       choices=['FD001', 'FD002', 'FD003', 'FD004'],
                       help='NASA C-MAPSS dataset subset')
    parser.add_argument('--task', type=str, default='regression',
                       choices=['classification', 'regression'],
                       help='Task type')
    parser.add_argument('--models', nargs='+',
                       default=['random_forest', 'xgboost', 'lstm'],
                       help='Models to train')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline models')
    parser.add_argument('--skip-deep-learning', action='store_true',
                       help='Skip deep learning models')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = get_logger(__name__, config)

    logger.info("="*80)
    logger.info("PREDICTIVE MAINTENANCE PIPELINE STARTED")
    logger.info("="*80)

    # Set random seed
    seed = get_seed(config)
    np.random.seed(seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    viz = PredictiveMaintenanceVisualizer(config, str(output_dir))
    metrics_calc = PredictiveMaintenanceMetrics(config)
    comparator = ModelComparator(config)

    # -------------------------------------------------------------------------
    # 1. DATA LOADING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("="*80)

    loader = get_data_loader('cmapss', config)
    train_df, test_df, rul_df = loader.load_dataset(args.dataset)

    # Add RUL and labels
    train_df = loader.add_rul_column(train_df)
    train_df = loader.add_labels(train_df)

    logger.info(f"Training data: {train_df.shape}")
    logger.info(f"Test data: {test_df.shape}")

    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    logger.info(f"Number of sensors: {len(sensor_cols)}")

    # -------------------------------------------------------------------------
    # 2. EXPLORATORY VISUALIZATION
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 2: EXPLORATORY VISUALIZATION")
    logger.info("="*80)

    viz.plot_sensor_degradation(train_df, sensor_cols, unit_ids=[1, 2, 3],
                                save_name='01_sensor_degradation')
    viz.plot_rul_distribution(train_df, save_name='02_rul_distribution')

    # -------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("="*80)

    engineer = FeatureEngineer(config)

    logger.info("Creating features for training data...")
    train_features = engineer.create_all_features(train_df, sensor_cols)

    logger.info(f"Original features: {train_df.shape[1]}")
    logger.info(f"Engineered features: {train_features.shape[1]}")

    # Remove low variance features
    train_features, removed_cols = engineer.remove_low_variance_features(train_features)
    logger.info(f"Removed {len(removed_cols)} low variance features")

    # -------------------------------------------------------------------------
    # 4. DATA SPLITTING AND SCALING
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 4: DATA SPLITTING AND SCALING")
    logger.info("="*80)

    # Split data
    train_split, val_split, test_split = split_data(
        train_features,
        test_size=config['preprocessing']['test_split'],
        validation_size=config['preprocessing']['validation_split'],
        random_state=seed
    )

    # Prepare feature columns (exclude metadata and target columns)
    exclude_cols = ['unit_id', 'cycle', 'RUL', 'label_binary', 'RUL_clipped']
    feature_cols = [col for col in train_split.columns if col not in exclude_cols]

    # Scale features
    scaler = DataScaler(config['preprocessing']['scaler'])
    train_scaled = scaler.fit_transform(train_split, feature_cols)
    val_scaled = scaler.transform(val_split)
    test_scaled = scaler.transform(test_split)

    # Save scaler
    scaler.save(output_dir / 'scaler.pkl')

    # Prepare data for models
    target_col = 'label_binary' if args.task == 'classification' else 'RUL'

    X_train = train_scaled[feature_cols].values
    y_train = train_scaled[target_col].values

    X_val = val_scaled[feature_cols].values
    y_val = val_scaled[target_col].values

    X_test = test_scaled[feature_cols].values
    y_test = test_scaled[target_col].values

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    # -------------------------------------------------------------------------
    # 5. BASELINE MODELS
    # -------------------------------------------------------------------------
    if not args.skip_baseline:
        logger.info("\n" + "="*80)
        logger.info("STEP 5: BASELINE MODELS")
        logger.info("="*80)

        baseline_models = ['random_forest', 'xgboost']

        for model_name in baseline_models:
            if model_name not in args.models:
                continue

            logger.info(f"\n--- Training {model_name.upper()} ---")

            start_time = time.time()

            # Initialize and train model
            model = BaselineModel(config, model_name, args.task)
            model.train(X_train, y_train, X_val, y_val)

            training_time = time.time() - start_time

            # Evaluate
            start_time = time.time()
            test_metrics = model.evaluate(X_test, y_test)
            prediction_time = time.time() - start_time

            # Add to comparator
            comparator.add_result(model_name, test_metrics, training_time, prediction_time)

            # Feature importance
            importance_df = model.get_feature_importance(feature_cols)
            if not importance_df.empty:
                viz.plot_feature_importance(importance_df,
                                           save_name=f'03_{model_name}_feature_importance')

            # Save model
            model.save(output_dir / f'{model_name}_model.pkl')

            logger.info(f"Training time: {training_time:.2f}s")
            logger.info(f"Prediction time: {prediction_time:.2f}s")

    # -------------------------------------------------------------------------
    # 6. DEEP LEARNING MODELS
    # -------------------------------------------------------------------------
    if not args.skip_deep_learning:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: DEEP LEARNING MODELS")
        logger.info("="*80)

        # Prepare sequences
        logger.info("Creating sequences for deep learning...")
        seq_generator = SequenceGenerator(config)

        X_train_seq, y_train_seq = seq_generator.create_sequences(
            train_scaled, feature_cols, target_col
        )
        X_val_seq, y_val_seq = seq_generator.create_sequences(
            val_scaled, feature_cols, target_col
        )
        X_test_seq, y_test_seq = seq_generator.create_sequences(
            test_scaled, feature_cols, target_col
        )

        logger.info(f"Sequence shape: {X_train_seq.shape}")

        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

        dl_models = ['lstm', 'gru', 'cnn_1d']

        for model_name in dl_models:
            if model_name not in args.models:
                continue

            logger.info(f"\n--- Training {model_name.upper()} ---")

            start_time = time.time()

            # Initialize and train model
            model = DeepLearningModel(config, model_name, args.task, input_shape)
            history = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, verbose=1)

            training_time = time.time() - start_time

            # Plot training history
            viz.plot_training_history(history, save_name=f'04_{model_name}_training_history')

            # Evaluate
            start_time = time.time()
            test_metrics = model.evaluate(X_test_seq, y_test_seq)
            prediction_time = time.time() - start_time

            # Add to comparator
            comparator.add_result(model_name, test_metrics, training_time, prediction_time)

            # Save model
            model.save(str(output_dir / f'{model_name}_model'))

            logger.info(f"Training time: {training_time:.2f}s")
            logger.info(f"Prediction time: {prediction_time:.2f}s")

    # -------------------------------------------------------------------------
    # 7. MODEL COMPARISON AND VISUALIZATION
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 7: MODEL COMPARISON")
    logger.info("="*80)

    # Get comparison table
    comparison_df = comparator.get_comparison_table()

    # Save comparison
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    logger.info(f"\nModel comparison saved to {output_dir / 'model_comparison.csv'}")

    # Find best model
    if args.task == 'classification':
        best_model_info = comparator.get_best_model('f1_score', maximize=True)
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    else:
        best_model_info = comparator.get_best_model('rmse', maximize=False)
        metrics_to_plot = ['mae', 'rmse', 'r2_score']

    if best_model_info:
        logger.info(f"\nBest model: {best_model_info['model_name']}")
        logger.info(f"Metrics: {best_model_info['all_metrics']}")

    # Plot model comparison
    if len(comparison_df) > 1:
        viz.plot_model_comparison(comparison_df, metrics_to_plot,
                                 save_name='05_model_comparison')

    # -------------------------------------------------------------------------
    # 8. SAVE SUMMARY REPORT
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*80)
    logger.info("STEP 8: GENERATING SUMMARY REPORT")
    logger.info("="*80)

    summary = {
        'dataset': args.dataset,
        'task': args.task,
        'n_training_samples': len(train_df),
        'n_test_samples': len(test_df),
        'n_features': len(feature_cols),
        'n_sensors': len(sensor_cols),
        'models_trained': list(comparison_df['Model'].values) if not comparison_df.empty else [],
        'best_model': best_model_info if best_model_info else None,
        'seed': seed,
    }

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary saved to {output_dir / 'summary.json'}")

    # Create markdown report
    report_path = output_dir / 'summary.md'
    with open(report_path, 'w') as f:
        f.write("# Predictive Maintenance - Results Summary\n\n")
        f.write(f"**Dataset**: {args.dataset}\n\n")
        f.write(f"**Task**: {args.task}\n\n")
        f.write(f"**Random Seed**: {seed}\n\n")
        f.write(f"## Dataset Statistics\n\n")
        f.write(f"- Training samples: {len(train_df)}\n")
        f.write(f"- Test samples: {len(test_df)}\n")
        f.write(f"- Features: {len(feature_cols)}\n")
        f.write(f"- Sensors: {len(sensor_cols)}\n\n")

        f.write("## Model Comparison\n\n")
        if not comparison_df.empty:
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

        if best_model_info:
            f.write(f"## Best Model\n\n")
            f.write(f"**Model**: {best_model_info['model_name']}\n\n")
            f.write(f"**Best Metric**: {best_model_info['metric']} = {best_model_info['value']:.4f}\n\n")
            f.write("**All Metrics**:\n\n")
            for metric, value in best_model_info['all_metrics'].items():
                f.write(f"- {metric}: {value:.4f}\n")

    logger.info(f"Report saved to {report_path}")

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    main()
