# src/news_classifier.py
import os
import sys
import logging
import time
import json
import psutil
import joblib
from joblib import parallel_backend, Memory
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import gc
import tempfile
import shutil
import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced
from collections import Counter


# Suppress warnings
warnings.filterwarnings("ignore")

# Configure for c6i.8xlarge (32 vCPU, 64GB RAM)
N_JOBS = 32  # Match vCPU count
MAX_RAM_GB = 60  # Leave some RAM for system
BATCH_SIZE = 10000  # Larger batch size for faster processing


def get_news_sampling_strategy(y, max_ratio=5, min_samples=2000):
    """
    Creates a balanced sampling strategy for news categories that:
    - Ensures minimum samples per category
    - Reduces extreme imbalance while preserving category importance
    - Maintains reasonable dataset size
    """
    class_counts = Counter(y)

    # Ensure minimum samples
    baseline = max(min(class_counts.values()), min_samples)
    # Calculate maximum samples allowed
    max_samples = baseline * max_ratio

    strategy = {}
    for category, count in class_counts.items():
        if count <= max_samples:
            # For smaller classes, keep all samples if above minimum
            strategy[category] = max(count, min_samples)
        else:
            # For larger classes, reduce to max_samples
            strategy[category] = max_samples

    return strategy


def save_category_mappings(categories, output_dir):
    """
    Save category mapping to files

    Args:
        categories: List of unique category names
        output_dir: Directory path where the mapping files will be saved
    """
    try:
        # Create category mapping
        category_mapping = {i: str(cat) for i, cat in enumerate(sorted(categories))}

        # Save as joblib
        mapping_path = Path(output_dir) / "category_mapping.joblib"
        joblib.dump(category_mapping, mapping_path)

        # Save as JSON for readability
        json_path = Path(output_dir) / "category_mapping.json"
        with open(json_path, "w") as f:
            json.dump(category_mapping, f, indent=2)

        print(f"Saved category mapping with {len(category_mapping)} categories")

    except Exception as e:
        print(f"Error saving category mappings: {str(e)}")
        raise


class EnhancedNewsClassifier:
    def __init__(self, experiment_params):
        self.params = experiment_params
        self.setup_logging()
        self.logger.info(
            f"Initializing EnhancedNewsClassifier with parameters: {experiment_params}"
        )

        self.count_vectorizer = CountVectorizer(
            max_features=21000,
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=3,
            stop_words="english",
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=21000,
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=3,
            stop_words="english",
            norm="l2",
            sublinear_tf=True,
        )

        # Initialize undersampler with custom strategy
        self.max_ratio = self.params.get("max_ratio", 3)
        self.min_samples = self.params.get("min_samples", 1990)

        self.undersampler = RandomUnderSampler(
            sampling_strategy=self.params.get("max_ratio", "auto"), random_state=42
        )

        # Initialize dimensionality reduction
        self.svd = TruncatedSVD(n_components=650, random_state=42)

        self.lda = LatentDirichletAllocation(
            n_components=self.params["n_topics"],
            max_iter=25,
            learning_method="online",
            learning_offset=self.params["learning_offset"],
            random_state=42,
            batch_size=BATCH_SIZE,
            evaluate_every=5,
            n_jobs=N_JOBS,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )

        # Initialize RF with OneVsRest
        base_rf = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            criterion=self.params["criterion"],
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features="log2",
            bootstrap=True,
            class_weight=self.params["class_balance"],
            max_samples=self.params["max_samples"],
            random_state=42,
            n_jobs=N_JOBS,
            verbose=1,
        )

        self.rf = OneVsRestClassifier(base_rf)
        self.scaler = StandardScaler()

        # Initialize feature selector with smaller number of features
        self.feature_selector = SelectKBest(chi2, k=7500)

    def process_data(self, df):
        """Enhanced data processing with multiple feature sets"""
        self.logger.info("Processing data with multiple feature extraction methods")
        try:
            # Get unique categories and create mapping
            unique_categories = sorted(
                df["category"].unique()
            )  # Sort to ensure consistent mapping
            self.category_mapping = {i: cat for i, cat in enumerate(unique_categories)}
            self.reverse_category_mapping = {
                cat: i for i, cat in enumerate(unique_categories)
            }

            # Log category distribution
            category_counts = df["category"].value_counts()
            self.logger.info("\nCategory distribution:")
            for cat, count in category_counts.items():
                self.logger.info(f"{cat}: {count}")

            # Convert categories to numeric indices
            df["category_idx"] = df["category"].map(self.reverse_category_mapping)

            # Count vectors
            self.logger.info("Generating count vectors")
            count_features = self.count_vectorizer.fit_transform(df["processed_text"])

            # TF-IDF features
            self.logger.info("Generating TF-IDF features")
            tfidf_features = self.tfidf_vectorizer.fit_transform(df["processed_text"])

            # Reduce TF-IDF dimensionality with SVD
            self.logger.info("Applying SVD")
            tfidf_svd = self.svd.fit_transform(tfidf_features)

            # Generate LDA topics
            self.logger.info("Generating LDA topics")
            lda_features = self.lda.fit_transform(count_features)

            # Combine all features
            self.logger.info("Combining features")

            # Convert sparse matrices to dense if needed
            if isinstance(count_features, np.ndarray):
                count_dense = count_features
            else:
                count_dense = count_features.toarray()

            # Select top features from count vectors without scaling
            top_count_features = count_dense[:, :1000]  # Use top 1000 count features

            # Combine features - all these should be non-negative now
            combined_features = np.hstack(
                [
                    tfidf_svd,  # SVD features can be negative, but we'll handle them separately
                    lda_features,  # LDA features are non-negative
                    top_count_features,  # Count features are non-negative
                ]
            )

            # Initialize feature masks for different types
            n_svd = tfidf_svd.shape[1]
            n_lda = lda_features.shape[1]
            n_count = top_count_features.shape[1]

            # Create feature selector for non-negative features only
            self.logger.info("Performing feature selection on non-negative features")

            # Extract only the non-negative features for chi2 selection
            non_negative_features = np.hstack([lda_features, top_count_features])

            # Apply chi2 to non-negative features
            selected_features = self.feature_selector.fit_transform(
                non_negative_features, df["category"]
            )

            # Combine SVD features (without selection) with selected features
            final_features = np.hstack(
                [
                    tfidf_svd,  # Keep all SVD features
                    selected_features,  # Selected non-negative features
                ]
            )

            # Scale the final features
            self.logger.info("Scaling final features")
            scaled_features = self.scaler.fit_transform(final_features)

            # Prepare target variable
            y = pd.Categorical(df["category"]).codes
            y = df["category_idx"].values

            return scaled_features, y, unique_categories

        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            raise
        finally:
            gc.collect()

    def train_model(self, X, y, categories):
        """Enhanced training with undersampling, stratified split and error analysis"""
        self.logger.info("Training Random Forest model with enhanced parameters")
        try:
            # First perform stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create sampling strategy based on training data
            if isinstance(self.max_ratio, (int, float)):
                sampling_strategy = get_news_sampling_strategy(
                    y_train, max_ratio=self.max_ratio, min_samples=self.min_samples
                )
                self.logger.info(
                    f"Created custom sampling strategy: {sampling_strategy}"
                )
            else:
                sampling_strategy = "auto"
                self.logger.info("Using automatic sampling strategy")

            # Initialize undersampler with computed strategy
            undersampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=42
            )

            # Apply undersampling to training data only
            self.logger.info("Applying undersampling to training data")
            X_train_resampled, y_train_resampled = undersampler.fit_resample(
                X_train, y_train
            )

            # Log class distributions
            train_dist = pd.Series(y_train).value_counts()
            resampled_dist = pd.Series(y_train_resampled).value_counts()
            test_dist = pd.Series(y_test).value_counts()

            self.logger.info("\nClass distribution summary:")
            self.logger.info("\nOriginal training set distribution:")
            self.logger.info(train_dist)
            self.logger.info("\nResampled training set distribution:")
            self.logger.info(resampled_dist)
            self.logger.info("\nReduction ratios:")
            self.logger.info(resampled_dist / train_dist)

            # Train model with progress bar using resampled data
            with tqdm(total=100, desc="Training RF") as pbar:
                self.rf.fit(X_train_resampled, y_train_resampled)
                pbar.update(100)

            # Evaluate on original test set
            y_pred = self.rf.predict(X_test)
            y_prob = self.rf.predict_proba(X_test)

            # Detailed evaluation
            report = classification_report(
                y_test, y_pred, target_names=categories, output_dict=True
            )

            conf_matrix = confusion_matrix(y_test, y_pred)
            confidence_scores = np.max(y_prob, axis=1)

            # Error analysis
            error_indices = np.where(y_test != y_pred)[0]
            error_analysis = {
                "error_indices": error_indices,
                "true_labels": y_test[error_indices],
                "predicted_labels": y_pred[error_indices],
                "confidence_scores": confidence_scores[error_indices],
            }

            # Add class distribution information
            class_distribution = {
                "original_train": np.bincount(y_train).tolist(),
                "resampled_train": np.bincount(y_train_resampled).tolist(),
                "test": np.bincount(y_test).tolist(),
            }

            return {
                "report": report,
                "confusion_matrix": conf_matrix,
                "confidence_scores": confidence_scores,
                "error_analysis": error_analysis,
                "class_distribution": class_distribution,
                "test_data": (X_test, y_test, y_pred),
            }

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def setup_logging(self):
        """Enhanced logging setup"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = (
            f"{self.params['n_topics']}topics_{self.params['n_estimators']}trees"
        )
        log_file = log_dir / f"classification_{experiment_name}_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def run_classification(self, data_path, output_dir):
        """Run the complete classification process"""
        start_time = time.time()

        try:
            # Monitor memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

            # Load data
            self.logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            df = df.dropna(subset=["processed_text"])
            self.logger.info(f"Loaded {len(df)} records")

            # Process data
            X, y, categories = self.process_data(df)

            # Save category mapping
            save_category_mappings(categories, output_dir)

            # Clear dataframe to free memory
            del df
            gc.collect()

            # Train and evaluate
            results = self.train_model(X, y, categories)

            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Create experiment-specific directory
            experiment_name = (
                f"{self.params['n_topics']}topics_{self.params['n_estimators']}trees"
            )
            experiment_dir = output_path / experiment_name
            experiment_dir.mkdir(exist_ok=True)

            # Save models
            self.logger.info("Saving models and results...")
            joblib.dump(
                self.count_vectorizer, experiment_dir / "count_vectorizer.joblib"
            )
            joblib.dump(
                self.tfidf_vectorizer, experiment_dir / "tfidf_vectorizer.joblib"
            )
            joblib.dump(self.svd, experiment_dir / "svd_model.joblib")
            joblib.dump(self.lda, experiment_dir / "lda_model.joblib")
            joblib.dump(self.rf, experiment_dir / "rf_model.joblib")
            joblib.dump(self.scaler, experiment_dir / "scaler.joblib")

            # Save results
            with open(experiment_dir / "classification_report.json", "w") as f:
                json.dump(results["report"], f, indent=4)

            np.save(
                experiment_dir / "confusion_matrix.npy", results["confusion_matrix"]
            )
            np.save(
                experiment_dir / "confidence_scores.npy", results["confidence_scores"]
            )

            # Save error analysis
            np.save(experiment_dir / "error_analysis.npy", results["error_analysis"])

            # Save parameters
            with open(experiment_dir / "parameters.json", "w") as f:
                json.dump(self.params, f, indent=4)

            # Save class distribution
            with open(experiment_dir / "class_distribution.json", "w") as f:
                json.dump(results["class_distribution"], f, indent=4)

            # Log completion
            end_time = time.time()
            final_memory = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Final memory usage: {final_memory:.2f} MB")
            self.logger.info(
                f"Total processing time: {end_time - start_time:.2f} seconds"
            )

            return results

        except Exception as e:
            self.logger.error(f"Error in classification pipeline: {str(e)}")
            raise
        finally:
            gc.collect()


def run_optimized_experiments():
    """Run experiments with optimized parameters for c6i.8xlarge"""
    parameter_combinations = [
        {
            "n_topics": 23,
            "n_estimators": 700,
            "learning_offset": 170.0,
            "criterion": "entropy",
            "max_samples": 0.9,
            "class_balance": "balanced",
            "max_ratio": 3,
            "min_samples": 1992,
        },
    ]

    results = []
    for params in parameter_combinations:
        print(f"\nRunning experiment with parameters: {params}")
        classifier = EnhancedNewsClassifier(params)

        try:
            experiment_results = classifier.run_classification(
                data_path=str(Path.home() / "data/processed_news_dataset.csv"),
                output_dir=str(Path.home() / "models"),
            )
            results.append({"parameters": params, "results": experiment_results})
        finally:
            gc.collect()

    return results


if __name__ == "__main__":
    # Set memory limit to 60GB (leaving 4GB for system)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM_GB * 1024 * 1024 * 1024, -1))
    except Exception as e:
        print(f"Warning: Could not set memory limit: {e}")

    try:
        results = run_optimized_experiments()

        print("\nExperiment Results Summary:")
        for idx, exp in enumerate(results, 1):
            print(f"\nExperiment {idx}:")
            print(f"Parameters: {exp['parameters']}")
            print(f"Accuracy: {exp['results']['report']['accuracy']:.3f}")
            print(
                f"Weighted F1: {exp['results']['report']['weighted avg']['f1-score']:.3f}"
            )

            # Add class distribution summary
            print("\nClass Distribution Summary:")
            print(
                "Original training set:",
                exp["results"]["class_distribution"]["original_train"],
            )
            print(
                "Resampled training set:",
                exp["results"]["class_distribution"]["resampled_train"],
            )
            print("Test set:", exp["results"]["class_distribution"]["test"])

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        gc.collect()
