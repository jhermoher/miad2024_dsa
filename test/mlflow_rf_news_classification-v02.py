#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 22:00:20 2024

@author: Equipo 16_DSA
"""

import os
import logging
import time
import numpy as np
import mlflow
import mlflow.sklearn
import gc
import json
from pathlib import Path
from datetime import datetime
from news_classifier import EnhancedNewsClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow and results configuration
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "rf_news_classification"
DATA_DIR = str(Path.home() / "data")
RESULTS_DIR = str(Path.home() / "results")

def run_experiment(params):
    """Run a single experiment with MLflow tracking"""
    try:
        # Set up MLflow
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Disable autologging
        mlflow.sklearn.autolog(disable=True)
        
        # Start MLflow run with timestamp
        run_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.time()
            run_id = run.info.run_id
            
            # Log parameters manually
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            logger.info(f"Started experiment with parameters: {params}")
            
            # Initialize and run classifier
            classifier = EnhancedNewsClassifier(params)
            results = classifier.run_classification(
                data_path=os.path.join(DATA_DIR, "processed_news_dataset.csv"),
                output_dir="/tmp"
            )
            
            # Extract metrics from classification report
            report = results["report"]
            
            # Log main metrics
            metrics = {
                "accuracy": float(report["accuracy"]),
                "weighted_precision": float(report["weighted avg"]["precision"]),
                "weighted_recall": float(report["weighted avg"]["recall"]),
                "weighted_f1": float(report["weighted avg"]["f1-score"]),
                "macro_f1": float(report["macro avg"]["f1-score"]),
                "training_time": float(time.time() - start_time)
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Save confusion matrix as JSON for MLflow
            conf_matrix_dict = {
                "matrix": results["confusion_matrix"].tolist(),
                "shape": results["confusion_matrix"].shape
            }
            with open("/tmp/confusion_matrix.json", "w") as f:
                json.dump(conf_matrix_dict, f)
            mlflow.log_artifact("/tmp/confusion_matrix.json")
            
            # Save complete classification report
            with open("/tmp/classification_report.json", "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact("/tmp/classification_report.json")
            
            # Save class distribution
            with open("/tmp/class_distribution.json", "w") as f:
                json.dump(results["class_distribution"], f, indent=2)
            mlflow.log_artifact("/tmp/class_distribution.json")
            
            # Create detailed results summary
            summary = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "parameters": params,
                "metrics": metrics,
                "training_details": {
                    "dataset_size": len(results["test_data"][0]),
                    "test_size": len(results["test_data"][1]),
                    "num_categories": len(report) - 3  # excluding avg rows
                }
            }
            
            # Save summary to MLflow
            with open("/tmp/experiment_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            mlflow.log_artifact("/tmp/experiment_summary.json")
            
            # Clean up memory
            gc.collect()
            
            # Print results
            print("\nExperiment Results:")
            print(f"Run ID: {run_id}")
            print(f"\nParameters:")
            print(f"  n_topics: {params['n_topics']}")
            print(f"  n_estimators: {params['n_estimators']}")
            print(f"  learning_offset: {params['learning_offset']}")
            print(f"  criterion: {params['criterion']}")
            print(f"  max_samples: {params['max_samples']}")
            print(f"\nMetrics:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")
            print(f"  Macro F1: {metrics['macro_f1']:.3f}")
            print(f"  Training Time: {metrics['training_time']:.2f} seconds")
            
            return metrics, run_id
            
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        if mlflow.active_run():
            mlflow.log_param("error", str(e))
        return None
    finally:
        # Ensure MLflow run is ended
        if mlflow.active_run():
            mlflow.end_run()
        
        # Clean up temporary files
        for f in os.listdir('/tmp'):
            if f.startswith('temp_') or f.endswith('.json'):
                try:
                    os.remove(os.path.join('/tmp', f))
                except:
                    pass

if __name__ == "__main__":
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    # Single experiment parameters - modify these manually between runs
    params = {
        "n_topics": 20,
        "n_estimators": 500,
        "learning_offset": 100.0,
        "criterion": "entropy",
        "max_samples": 0.8,
        "class_balance": "balanced",
        "max_ratio": 3,
        "min_samples": 1990
    }
    
    # Run experiment
    logger.info("Starting experiment...")
    result = run_experiment(params)
    
    if result:
        metrics, run_id = result
        print(f"\nExperiment completed successfully.")
        print(f"Run ID: {run_id}")
        print(f"View results in MLflow UI at: http://localhost:5000")