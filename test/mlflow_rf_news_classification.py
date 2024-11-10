#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:22:20 2024

@author: Equipo 16_DSA
"""

# Import required libraries
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from datetime import datetime

# Import custom modules
from rf_news_classifier import NewsTopicClassifier
from topic_modeling_prep import TopicModelingPrep

# MLflow tracking URI setup
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "rf_news-classification"
DATA_DIR = "/data"  # EC2 data directory


def load_and_prepare_data():
    """Load and prepare the news dataset."""
    data_path = os.path.join(DATA_DIR, "processed_news_dataset.csv")

    # Load data with null handling
    df = pd.read_csv(
        data_path, encoding="utf-8-sig", na_values=["", "NA", "null", "NULL", "NaN"]
    )

    # Drop rows with null values
    df = df.dropna(subset=["processed_text"])

    # Prepare data using TopicModelingPrep
    prep = TopicModelingPrep()
    topic_df, artifacts = prep.prepare_data(df)

    return topic_df, artifacts


def run_experiment(params, run_name=None):
    """Execute a single MLflow experiment with given parameters."""

    # Load and prepare data
    topic_df, artifacts = load_and_prepare_data()

    # Set up MLflow experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log data preparation metrics
        mlflow.log_param("dataset_initial_size", len(topic_df))

        # Log parameters
        mlflow.log_params(params)

        # Initialize and train model
        classifier = NewsTopicClassifier(**params)

        # Fit LDA and get topic distributions
        topic_distributions, top_words = classifier.fit_transform_lda(
            topic_df, artifacts
        )

        # Train classifier and get evaluation results
        evaluation_results = classifier.train_classifier(topic_df, topic_distributions)

        # Extract metrics
        metrics = {
            "accuracy": evaluation_results["classification_report"]["accuracy"],
            "weighted_f1": evaluation_results["classification_report"]["weighted avg"][
                "f1-score"
            ],
            "macro_f1": evaluation_results["classification_report"]["macro avg"][
                "f1-score"
            ],
        }

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(classifier, "model")

        # Print results
        print("\nExperiment Results:")
        print(f"Run ID: {run.info.run_id}")
        print(f"Parameters: {params}")
        print(f"Metrics: {metrics}")

        return metrics, run.info.run_id


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    # Example parameters (modify as needed for each experiment)
    params = {
        "n_topics": 20,
        "n_estimators": 200,
        "learning_offset": 25.0,
        "criterion": "entropy",
        "max_samples": 0.8,
        "class_balance": "balanced_subsample",
    }

    # Run single experiment
    run_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics, run_id = run_experiment(params, run_name)
