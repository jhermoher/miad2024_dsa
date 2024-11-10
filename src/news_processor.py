#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:22:20 2024

@author: Equipo 16_DSA
"""

# Import required libraries
import pandas as pd
from multiprocessing import Pool, cpu_count
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")

# Download required NLTK resources
try:
    resources = [
        "punkt_tab",  # for tokenization
        "averaged_perceptron_tagger_eng",  # for POS tagging
        "maxent_ne_chunker_tab",  # for named entity recognition
        "words",  # for word lists
        "stopwords",  # for stopwords
        "wordnet",  # for lemmatization
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)
    print("Fuentes NLTK cargadas exitosamente!")

except Exception as e:
    print(f"Error cargando las fuentes de NLTK: {str(e)}")


def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag for better lemmatization"""
    tag_dict = {
        "J": nltk.corpus.wordnet.ADJ,
        "N": nltk.corpus.wordnet.NOUN,
        "V": nltk.corpus.wordnet.VERB,
        "R": nltk.corpus.wordnet.ADV,
    }
    return tag_dict.get(tag[0], nltk.corpus.wordnet.NOUN)


def extract_news_patterns(pos_tags):
    """Extract news-specific patterns from POS tags"""
    patterns = {
        "named_entities": [],  # Proper nouns sequences (NNP+)
        "action_phrases": [],  # Verb + Object patterns (VB* + NN*)
        "descriptive_phrases": [],  # Adjective + Noun patterns (JJ + NN*)
        "quotes": [],  # Text within quotation marks
        "temporal_expressions": [],  # Time-related expressions
        "location_expressions": [],  # Location-related expressions
    }

    # Named Entity Recognition for better entity extraction
    ne_tree = nltk.ne_chunk(pos_tags)

    # Extract patterns
    i = 0
    while i < len(pos_tags):
        word, tag = pos_tags[i]

        # Named entities (people, organizations, locations)
        if tag.startswith("NNP"):
            entity = [word]
            j = i + 1
            while j < len(pos_tags) and pos_tags[j][1].startswith("NNP"):
                entity.append(pos_tags[j][0])
                j += 1
            patterns["named_entities"].append(" ".join(entity))
            i = j
            continue

        # Action phrases (e.g., "announced plans", "issued statement")
        if tag.startswith("VB") and i + 1 < len(pos_tags):
            next_word, next_tag = pos_tags[i + 1]
            if next_tag.startswith("NN"):
                patterns["action_phrases"].append(f"{word} {next_word}")

        # Descriptive phrases (e.g., "major breakthrough", "controversial decision")
        if tag.startswith("JJ") and i + 1 < len(pos_tags):
            next_word, next_tag = pos_tags[i + 1]
            if next_tag.startswith("NN"):
                patterns["descriptive_phrases"].append(f"{word} {next_word}")

        i += 1
    return patterns


def preprocess_text(text, purpose="both"):
    """Preprocess text with news-specific POS patterns"""
    if not isinstance(text, str):
        return {"topic": "", "sentiment": "", "pos_tags": [], "news_patterns": {}}

    # Initial cleaning
    text = text.lower()

    # Handle contractions
    contractions = {
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'ve": " have",
        "'m": " am",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Preserve quotes for potential speech attribution
    quoted_text = re.findall(r'"([^"]*)"', text)

    # Basic cleaning
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(
        r"[^a-zA-Z0-9\'\s\.\,\!\?\-\"]", "", text
    )  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

    # Tokenization and POS Tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract news-specific patterns
    news_patterns = extract_news_patterns(pos_tags)
    news_patterns["quotes"] = quoted_text

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get stopwords except negations and important news words
    stop_words = set(stopwords.words("english"))
    keep_words = {
        "no",
        "not",
        "nor",
        "none",
        "never",
        "neither",
        "who",
        "what",
        "where",
        "when",
        "why",
        "how",
        "says",
        "said",
        "reports",
        "announces",
        "claims",
    }
    stop_words = stop_words - keep_words

    # Process for topic modeling and sentiment analysis
    topic_tokens = []
    sentiment_tokens = []

    if purpose in ["topic", "both"]:
        topic_tokens = [
            lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag))
            for word, tag in pos_tags
            if (
                tag.startswith(("NN", "NNP", "VB", "JJ"))
                and word.lower() not in stop_words
                and len(word) > 2
            )
        ]

    if purpose in ["sentiment", "both"]:
        sentiment_tokens = [
            (
                lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag))
                if tag not in [".", ",", "!", "?"]
                else word
            )
            for word, tag in pos_tags
            if (
                word.lower() in keep_words
                or tag.startswith(("JJ", "RB", "VB"))
                or tag in [".", ",", "!", "?"]
                or (tag.startswith("NN") and word.lower() not in stop_words)
            )
        ]

    # Calculate POS-based features
    pos_features = {
        "adj_count": len([tag for _, tag in pos_tags if tag.startswith("JJ")]),
        "verb_count": len([tag for _, tag in pos_tags if tag.startswith("VB")]),
        "noun_count": len([tag for _, tag in pos_tags if tag.startswith("NN")]),
        "proper_noun_count": len([tag for _, tag in pos_tags if tag.startswith("NNP")]),
        "adv_count": len([tag for _, tag in pos_tags if tag.startswith("RB")]),
        "quote_count": len(quoted_text),
    }

    return {
        "topic": " ".join(topic_tokens),
        "sentiment": " ".join(sentiment_tokens),
        "pos_tags": pos_features,
        "news_patterns": news_patterns,
    }


def process_row(row):
    """Process a single row of the dataset"""
    try:
        # Handle empty strings and NaN values properly
        headline = row["headline"] if pd.notna(row["headline"]) else ""
        short_description = (
            row["short_description"] if pd.notna(row["short_description"]) else ""
        )

        # Combine texts and check if the result is empty
        full_text = f"{headline} . {short_description}".strip()

        # If full_text is empty or just contains the dot separator
        if not full_text or full_text == ".":
            return {
                "category": row["category"],
                "date": row["date"],
                "processed_text": "",  # Empty string instead of nan
                "sentiment_text": "",  # Empty string instead of nan
                "pos_features": {
                    "adj_count": 0,
                    "verb_count": 0,
                    "noun_count": 0,
                    "proper_noun_count": 0,
                    "adv_count": 0,
                    "quote_count": 0,
                },
                "news_patterns": {
                    "named_entities": [],
                    "action_phrases": [],
                    "descriptive_phrases": [],
                    "quotes": [],
                    "temporal_expressions": [],
                    "location_expressions": [],
                },
            }

        processed = preprocess_text(full_text, purpose="both")

        return {
            "category": row["category"],
            "date": row["date"],
            "processed_text": processed["topic"] if processed["topic"] else "",
            "sentiment_text": processed["sentiment"] if processed["sentiment"] else "",
            "pos_features": processed["pos_tags"],
            "news_patterns": processed["news_patterns"],
        }
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def chunk_data(df, n_chunks):
    """Split dataframe into n chunks for parallel processing"""
    chunk_size = len(df) // n_chunks
    return [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


def process_chunk(chunk):
    """Process a chunk of the dataframe"""
    results = []
    for _, row in chunk.iterrows():
        result = process_row(row)
        if result:
            results.append(result)
    return results


def main(file, sample_size=None, random_seed=42):
    # Read the dataset
    data_news_df = pd.read_csv(file)

    # Sample the dataset if sample_size is provided
    if sample_size:
        if isinstance(sample_size, float) and 0 < sample_size < 1:
            # If sample_size is a float between 0 and 1, treat it as a fraction
            n_samples = int(len(data_news_df) * sample_size)
        else:
            # Otherwise, treat it as an absolute number
            n_samples = min(int(sample_size), len(data_news_df))

        data_news_df = data_news_df.sample(n=n_samples, random_state=random_seed)
        print(
            f"Sampled {n_samples} rows ({sample_size if isinstance(sample_size, float) else n_samples/len(data_news_df):.2%} of data)"
        )

    # Calculate optimal number of processes (use 75% of available CPUs)
    n_processes = max(1, int(cpu_count() * 0.75))
    print(f"Usando {n_processes} núcleos para procesamiento en paralelo")

    # Split data into chunks
    chunks = chunk_data(data_news_df, n_processes)

    # Process chunks in parallel
    with Pool(processes=n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc="Procesando fragmentos",
            )
        )

    # Flatten results
    all_results = [item for sublist in results for item in sublist]

    # Convert to DataFrame
    processed_df = pd.DataFrame(all_results)

    # Extract POS features into separate columns
    pos_features_df = pd.DataFrame([row["pos_features"] for row in all_results])

    # Extract news patterns into separate columns
    news_patterns_df = pd.DataFrame(
        [
            {
                f"pattern_{k}_count": len(v) if isinstance(v, list) else 0
                for k, v in row["news_patterns"].items()
            }
            for row in all_results
        ]
    )

    # Combine all features
    final_df = pd.concat(
        [
            processed_df.drop(["pos_features", "news_patterns"], axis=1),
            pos_features_df,
            news_patterns_df,
        ],
        axis=1,
    )

    # Save processed dataset
    output_path = f'./data/processed_news_dataset{"_sample" if sample_size else ""}.csv'
    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",  # Includes BOM for Excel compatibility
        escapechar="\\",  # Handle special characters
        doublequote=True,  # Handle quotes within text
    )
    print(f"Dataset procesado guardado en {output_path}")

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesamiento de articulos de noticias con muestreo opcional"
    )
    parser.add_argument(
        "--sample-size",
        type=float,
        help="Número de de muestras o fracción(entre 0 y 1)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Semilla para muestreo aleatorio"
    )
    args = parser.parse_args()

    processed_df = main(sample_size=args.sample_size, random_seed=args.random_seed)
