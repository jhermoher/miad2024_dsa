#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:22:20 2024

@author: Equipo 16_DSA
"""

# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class TopicModelingPrep:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=17500,
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=2,
            stop_words="english",
        )
        self.count_vectorizer = CountVectorizer(
            max_features=17500,
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=2,
            stop_words="english",
        )

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare data for topic modeling and sentiment analysis

        Args:
            df: DataFrame containing news articles with columns:
                - processed_text: preprocessed text content
                - category: article category
                - date: publication date

        Returns:
            Tuple containing:
                - Prepared DataFrame
                - Dictionary of artifacts (vocabularies, mappings etc.)
        """
        # Initialize the topic modeling DataFrame
        topic_df = pd.DataFrame()

        # Copy essential columns
        topic_df["text"] = df["processed_text"]
        topic_df["category"] = df["category"]

        # Process dates
        topic_df["date"] = pd.to_datetime(df["date"])
        topic_df["year"] = topic_df["date"].dt.year
        topic_df["month"] = topic_df["date"].dt.month

        # Create text features
        topic_df["text_length"] = topic_df["text"].str.len()
        topic_df["word_count"] = topic_df["text"].str.split().str.len()
        topic_df["unique_words"] = topic_df["text"].apply(lambda x: len(set(x.split())))

        # Generate TF-IDF features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(topic_df["text"])

        # Generate document-term matrix
        doc_term_matrix = self.count_vectorizer.fit_transform(topic_df["text"])

        # Create category mappings
        category_mapping = dict(enumerate(topic_df["category"].unique()))
        category_inverse = {v: k for k, v in category_mapping.items()}
        topic_df["category_id"] = topic_df["category"].map(category_inverse)

        # Calculate category statistics
        category_stats = self._calculate_category_stats(topic_df)
        topic_df["category_frequency"] = topic_df["category"].map(
            category_stats["frequency"]
        )

        # Store matrices as sparse matrices
        topic_df["tfidf_matrix"] = pd.Series([tfidf_matrix] * len(topic_df))
        topic_df["doc_term_matrix"] = pd.Series([doc_term_matrix] * len(topic_df))

        # Create artifacts dictionary
        artifacts = {
            "tfidf_vocabulary": self.tfidf_vectorizer.vocabulary_,
            "count_vocabulary": self.count_vectorizer.vocabulary_,
            "feature_names": self.tfidf_vectorizer.get_feature_names_out(),
            "category_mapping": category_mapping,
            "category_inverse_mapping": category_inverse,
            "category_stats": category_stats,
        }

        print("\nResumen de validación:")
        self._validate_data(topic_df)

        return topic_df, artifacts

    def _calculate_category_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for each category"""
        stats = {}

        # Category frequencies
        counts = df["category"].value_counts()
        stats["frequency"] = (counts / len(df)).to_dict()
        stats["counts"] = counts.to_dict()

        # Average text length per category
        stats["avg_length"] = df.groupby("category")["text_length"].mean().to_dict()

        # Temporal distribution
        stats["temporal_dist"] = (
            df.groupby(["category", "year"]).size().unstack(fill_value=0).to_dict()
        )

        return stats

    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the prepared DataFrame"""
        print(f"Total de artículos: {len(df):,}")
        print(f"Número de categorías: {df['category'].nunique():,}")
        print(f"Longitud promedio del texto: {df['text_length'].mean():.1f} caracteres")
        print(f"Cantidad promedio de palabras: {df['word_count'].mean():.1f} palabras")

        # Check for missing values
        nulls = df.isnull().sum()
        if nulls.any():
            print(
                "\nWarning: Valores nulos encontrados en :",
                nulls[nulls > 0].index.tolist(),
            )

        print("\nTop 5 de categorias por frecuencia:")
        print(df["category"].value_counts().head())
