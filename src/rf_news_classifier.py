#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:22:20 2024

@author: Equipo 16_DSA
"""

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import warnings

warnings.filterwarnings("ignore")


class NewsTopicClassifier:
    def __init__(
        self,
        n_topics=20,
        n_estimators=200,
        learning_offset=25.0,
        criterion="gini",
        max_samples=0.8,
        class_balance="balanced_subsample",
    ):
        # LDA parameters
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=25,
            learning_method="online",
            learning_offset=learning_offset,
            random_state=42,
            batch_size=128,
            evaluate_every=5,
            n_jobs=-1,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
        )

        # Base Random Forest Classifier
        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight=class_balance,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        # Wrap RF with OneVsRestClassifier
        self.rf = OneVsRestClassifier(base_rf)

        self.vocabulary = None
        self.scaler = StandardScaler()

    def fit_transform_lda(self, topic_df, artifacts):
        """
        Fit LDA model and transform document-term matrix
        """
        print("Fitting LDA model...")
        doc_term_matrix = topic_df["doc_term_matrix"].iloc[0]
        self.vocabulary = artifacts["tfidf_vocabulary"]

        # Get feature names from vocabulary
        feature_names = [
            word for word, _ in sorted(self.vocabulary.items(), key=lambda x: x[1])
        ]

        # Fit and transform using LDA
        topic_distributions = self.lda.fit_transform(doc_term_matrix)

        # Get topic words
        top_words = self._get_top_words_per_topic(feature_names)

        return topic_distributions, top_words

    def _get_top_words_per_topic(self, terms, n_words=10):
        """Get top words for each topic"""
        components = self.lda.components_
        top_words = {}

        for idx, topic in enumerate(components):
            # Get indices of top words for this topic
            top_indices = topic.argsort()[: -n_words - 1 : -1]

            # Convert indices to words, ensuring we don't exceed terms length
            topic_words = []
            for i in top_indices:
                if i < len(terms):
                    topic_words.append(terms[i])
                if len(topic_words) == n_words:
                    break

            top_words[f"Topic {idx}"] = topic_words

        return top_words

    def train_classifier(self, topic_df, topic_distributions):
        """
        Train Random Forest classifier using OneVsRest
        """
        print("\nTraining Random Forest classifier...")

        # Prepare feature matrix
        tfidf_matrix = topic_df["tfidf_matrix"].iloc[0]
        tfidf_array = (
            tfidf_matrix.toarray() if sparse.issparse(tfidf_matrix) else tfidf_matrix
        )

        # Scale features
        tfidf_scaled = self.scaler.fit_transform(tfidf_array)
        topic_scaled = self.scaler.fit_transform(topic_distributions)

        # Combine scaled features
        X = np.hstack([tfidf_scaled, topic_scaled])
        y = topic_df["category_id"].values

        # Split data
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train classifier
        self.rf.fit(self.X_train, y_train)

        # Make predictions
        y_pred = self.rf.predict(self.X_test)

        return self._evaluate_classifier(y_test, y_pred, topic_df)

    def predict(self, X):
        """
        Make predictions on new data
        """
        X_scaled = self.scaler.transform(X)
        return self.rf.predict(X_scaled)

    def get_model_parameters(self):
        """
        Get current model parameters
        """
        return {
            "lda_params": self.lda.get_params(),
            "rf_params": self.rf.estimator.get_params(),
        }

    def _evaluate_classifier(self, y_test, y_pred, topic_df):
        """Evaluation with metrics"""
        categories = sorted(topic_df["category"].unique())

        # Basic metrics
        basic_report = classification_report(
            y_test, y_pred, target_names=categories, output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Get probabilities and confidence scores
        try:
            y_prob = self.rf.predict_proba(self.X_test)
            confidence_scores = np.max(y_prob, axis=1)
        except:
            confidence_scores = np.ones(len(y_test))

        # Model parameters
        model_params = self.get_model_parameters()

        return {
            "classification_report": basic_report,
            "confusion_matrix": conf_matrix,
            "confidence_scores": confidence_scores,
            "model_parameters": model_params,
        }

    def analyze_results(self, top_words, evaluation_results):
        """
        Analysis of results
        """
        print("\nModel Parameters:")
        print("-----------------")
        params = evaluation_results["model_parameters"]
        print("LDA topics:", params["lda_params"]["n_components"])

        rf_params = params["rf_params"]
        print("RF estimators:", rf_params["n_estimators"])
        print("RF criterion:", rf_params["criterion"])
        print("RF max_samples:", rf_params["max_samples"])
        print("RF class_weight:", rf_params["class_weight"])

        print("\nTopic Modeling Results:")
        print("-----------------------")
        for topic, words in top_words.items():
            print(f"{topic}: {', '.join(words)}")

        print("\nClassification Results:")
        print("----------------------")
        report = evaluation_results["classification_report"]
        print(f"Overall Accuracy: {report['accuracy']:.3f}")
        print(f"Weighted F1-score: {report['weighted avg']['f1-score']:.3f}")

        print("\nConfidence Analysis:")
        confidence_scores = evaluation_results["confidence_scores"]
        print(f"Average prediction confidence: {np.mean(confidence_scores):.3f}")
        print(f"Min confidence: {np.min(confidence_scores):.3f}")
        print(f"Max confidence: {np.max(confidence_scores):.3f}")


def main(topic_df, artifacts):
    """
    Main function with parameter setting
    """
    try:
        # Initialize classifier with specific parameters
        classifier = NewsTopicClassifier(
            n_topics=20,
            n_estimators=200,
            learning_offset=25.0,
            criterion="entropy",
            max_samples=0.8,
            class_balance="balanced_subsample",
        )

        topic_distributions, top_words = classifier.fit_transform_lda(
            topic_df, artifacts
        )
        evaluation_results = classifier.train_classifier(topic_df, topic_distributions)
        classifier.analyze_results(top_words, evaluation_results)

        return classifier, topic_distributions, evaluation_results

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("\nDataFrame info:")
        print(topic_df.info())
        print("\nArtifacts keys:", artifacts.keys())
        raise
