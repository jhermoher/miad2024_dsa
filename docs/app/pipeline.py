import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ssl
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logger = logging.getLogger(__name__)

class NewsPipeline:
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all models with proper error handling"""
        try:
            logger.info(f"Loading models from {self.model_dir}")
            self._load_models()
            self._initialize_nltk()
            self.vader = SentimentIntensityAnalyzer()
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words("english")) - {
                'no', 'not', 'nor', 'none', 'never', 'neither',
                'who', 'what', 'where', 'when', 'why', 'how',
                'says', 'said', 'reports', 'announces', 'claims'
            }
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
            
    def _load_models(self):
        """Load all required models with error checking"""
        required_models = [
            "count_vectorizer.joblib",
            "tfidf_vectorizer.joblib",
            "svd_model.joblib",
            "lda_model.joblib",
            "rf_model.joblib",
            "scaler.joblib",  
            "category_mapping.joblib",
        ]
        
        for model_file in required_models:
            model_path = self.model_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        self.count_vectorizer = joblib.load(self.model_dir / "count_vectorizer.joblib")
        self.tfidf_vectorizer = joblib.load(self.model_dir / "tfidf_vectorizer.joblib")
        self.svd = joblib.load(self.model_dir / "svd_model.joblib")
        self.lda = joblib.load(self.model_dir / "lda_model.joblib")
        self.rf = joblib.load(self.model_dir / "rf_model.joblib")
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
    
        # Load category mapping
        try:
            self.category_mapping = joblib.load(self.model_dir / "category_mapping.joblib")
            logger.info(f"Loaded category mapping with {len(self.category_mapping)} categories")
        except Exception as e:
            logger.error(f"Error loading category mapping: {str(e)}")
            raise
    
    def _initialize_nltk(self):
        """Initialize NLTK resources with error handling"""
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logger.error(f"Failed to download NLTK resources: {str(e)}")
            raise RuntimeError(f"NLTK initialization failed: {str(e)}")
    
    def get_wordnet_pos(self, tag: str) -> str:
        """Map POS tag to WordNet POS tag"""
        tag_dict = {
            "J": nltk.corpus.wordnet.ADJ,
            "N": nltk.corpus.wordnet.NOUN,
            "V": nltk.corpus.wordnet.VERB,
            "R": nltk.corpus.wordnet.ADV,
        }
        return tag_dict.get(tag[0], nltk.corpus.wordnet.NOUN)
    
    def extract_news_patterns(self, pos_tags) -> Dict[str, list]:
        """Extract news-specific patterns from POS tags"""
        patterns = {
            "named_entities": [],
            "action_phrases": [],
            "descriptive_phrases": [],
            "quotes": [],
            "temporal_expressions": [],
            "location_expressions": [],
        }
        
        ne_tree = nltk.ne_chunk(pos_tags)
        
        i = 0
        while i < len(pos_tags):
            word, tag = pos_tags[i]
            
            if tag.startswith("NNP"):
                entity = [word]
                j = i + 1
                while j < len(pos_tags) and pos_tags[j][1].startswith("NNP"):
                    entity.append(pos_tags[j][0])
                    j += 1
                patterns["named_entities"].append(" ".join(entity))
                i = j
                continue
            
            if tag.startswith("VB") and i + 1 < len(pos_tags):
                next_word, next_tag = pos_tags[i + 1]
                if next_tag.startswith("NN"):
                    patterns["action_phrases"].append(f"{word} {next_word}")
            
            if tag.startswith("JJ") and i + 1 < len(pos_tags):
                next_word, next_tag = pos_tags[i + 1]
                if next_tag.startswith("NN"):
                    patterns["descriptive_phrases"].append(f"{word} {next_word}")
            
            i += 1
            
        return patterns
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """Preprocess text with news-specific patterns"""
        if not isinstance(text, str):
            return {
                "topic": "",
                "sentiment": "",
                "pos_tags": [],
                "news_patterns": {}
            }
        
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
        
        quoted_text = re.findall(r'"([^"]*)"', text)
        
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9\'\s\.\,\!\?\-\"]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        news_patterns = self.extract_news_patterns(pos_tags)
        news_patterns["quotes"] = quoted_text
        
        topic_tokens = [
            self.lemmatizer.lemmatize(word.lower(), self.get_wordnet_pos(tag))
            for word, tag in pos_tags
            if (tag.startswith(("NN", "NNP", "VB", "JJ")) and 
                word.lower() not in self.stop_words and 
                len(word) > 2)
        ]
        
        sentiment_tokens = [
            (self.lemmatizer.lemmatize(word.lower(), self.get_wordnet_pos(tag))
             if tag not in [".", ",", "!", "?"]
             else word)
            for word, tag in pos_tags
            if (word.lower() not in self.stop_words or
                tag.startswith(("JJ", "RB", "VB")) or
                tag in [".", ",", "!", "?"])
        ]
        
        pos_features = {
            "adj_count": len([tag for _, tag in pos_tags if tag.startswith("JJ")]),
            "verb_count": len([tag for _, tag in pos_tags if tag.startswith("VB")]),
            "noun_count": len([tag for _, tag in pos_tags if tag.startswith("NN")]),
            "proper_noun_count": len([tag for _, tag in pos_tags if tag.startswith("NNP")]),
            "adv_count": len([tag for _, tag in pos_tags if tag.startswith("RB")]),
            "quote_count": len(quoted_text)
        }
        
        return {
            "topic": " ".join(topic_tokens),
            "sentiment": " ".join(sentiment_tokens),
            "pos_tags": pos_features,
            "news_patterns": news_patterns
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        }

    def extract_features(self, processed_text: str) -> np.ndarray:
        """Extract features using the trained vectorizers and models"""
        count_features = self.count_vectorizer.transform([processed_text])
        tfidf_features = self.tfidf_vectorizer.transform([processed_text])
        tfidf_svd = self.svd.transform(tfidf_features)
        lda_features = self.lda.transform(count_features)
        count_dense = count_features.toarray()
        top_count_features = count_dense[:, :1000]
        
        combined_features = np.hstack([
            tfidf_svd,
            lda_features,
            top_count_features
        ])
        
        return self.scaler.transform(combined_features)

    def predict_category(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict category and confidence score"""
        try:
            probabilities = self.rf.predict_proba(features)
            predicted_class_idx = np.argmax(probabilities, axis=1)[0]
            confidence = probabilities[0][predicted_class_idx]
            
            # Convert numeric prediction to category name
            category = self.category_mapping.get(int(predicted_class_idx), "UNKNOWN")
            
            logger.info(f"Predicted category index: {predicted_class_idx}, mapped to: {category}")
            
            return category, float(confidence)
            
        except Exception as e:
            logger.error(f"Error in category prediction: {str(e)}")
            raise

    def predict(self, headline: str, short_description: str, date: Optional[str] = None) -> Dict[str, Any]:
        try:
            logger.info(f"Processing headline: {headline[:50]}...")
            
            # Input validation
            if not isinstance(headline, str) or not isinstance(short_description, str):
                raise ValueError("Headline and short_description must be strings")
            
            if not headline.strip() or not short_description.strip():
                raise ValueError("Headline and short_description cannot be empty")
            
            # Combine texts
            full_text = f"{headline.strip()} . {short_description.strip()}"
            logger.info(f"Combined text length: {len(full_text)}")
            
            # Process text
            processed_data = self.preprocess_text(full_text)
            if not processed_data or not processed_data.get("topic"):
                raise ValueError("Text processing failed to produce valid output")
                
            # Extract features
            features = self.extract_features(processed_data["topic"])
            
            # Make predictions
            category, confidence = self.predict_category(features)
            sentiment_info = self.analyze_sentiment(full_text)
            
            # Ensure all lists in news_patterns are initialized
            news_patterns = processed_data["news_patterns"]
            for key in ["named_entities", "action_phrases", "descriptive_phrases", 
                    "quotes", "temporal_expressions", "location_expressions"]:
                if key not in news_patterns:
                    news_patterns[key] = []
            
            return {
                "success": True,
                "category": {
                    "predicted": str(category),
                    "confidence": float(confidence)
                },
                "sentiment": {
                    "label": sentiment_info["sentiment"],
                    "scores": {
                        "compound": float(sentiment_info["compound"]),
                        "positive": float(sentiment_info["pos"]),
                        "neutral": float(sentiment_info["neu"]),
                        "negative": float(sentiment_info["neg"])
                    }
                },
                "processed_data": {
                    "date": date or datetime.now().strftime("%Y-%m-%d"),
                    "processed_text": processed_data["topic"],
                    "pos_features": {
                        "adj_count": int(processed_data["pos_tags"]["adj_count"]),
                        "verb_count": int(processed_data["pos_tags"]["verb_count"]),
                        "noun_count": int(processed_data["pos_tags"]["noun_count"]),
                        "proper_noun_count": int(processed_data["pos_tags"]["proper_noun_count"]),
                        "adv_count": int(processed_data["pos_tags"]["adv_count"]),
                        "quote_count": int(processed_data["pos_tags"]["quote_count"])
                    },
                    "news_patterns": news_patterns
                }
            }
                
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "success": False,
                "category": None,
                "sentiment": None,
                "processed_data": None,
                "error": str(e)
            }
        