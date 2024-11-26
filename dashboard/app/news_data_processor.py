import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime


class NewsDataProcessor:
    def __init__(self, data_path: str = "data/processed_news_dataset.csv"):
        self.data_path = Path(data_path)
        self.vader = SentimentIntensityAnalyzer()
        self.df = None
        self.processed_df = None

    def load_data(self):
        """Load and validate the news dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.df["date"] = pd.to_datetime(self.df["date"])
            print(f"Loaded {len(self.df)} news articles")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def get_sentiment_label(self, compound_score):
        """Convert VADER compound score to sentiment label"""
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def process_data(self):
        """Process the news data and add sentiment analysis"""
        if self.df is None:
            print("Please load data first using load_data()")
            return False

        try:
            # Create a copy of the dataframe with required columns
            self.processed_df = self.df[["category", "date", "sentiment_text"]].copy()

            # Apply VADER sentiment analysis
            print("Analyzing sentiments...")
            sentiments = []
            for text in self.processed_df["sentiment_text"]:
                scores = self.vader.polarity_scores(str(text))
                sentiments.append(
                    {
                        "compound": scores["compound"],
                        "label": self.get_sentiment_label(scores["compound"]),
                    }
                )

            # Add sentiment results to dataframe
            sentiment_df = pd.DataFrame(sentiments)
            self.processed_df["sentiment_score"] = sentiment_df["compound"]
            self.processed_df["sentiment"] = sentiment_df["label"]

            # Group by date and category
            self.processed_df["year_month"] = self.processed_df["date"].dt.to_period(
                "M"
            )

            print("Data processing completed")
            return True

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return False

    def plot_category_distribution(self):
        """Plot category distribution over time"""
        if self.processed_df is None:
            print("Please process data first using process_data()")
            return None

        # Group by date and category, count articles
        category_time = (
            self.processed_df.groupby(["year_month", "category"])
            .size()
            .reset_index(name="count")
        )

        # Convert year_month to datetime for plotting
        category_time["year_month"] = category_time["year_month"].astype(str)

        # Create line plot
        fig = px.line(
            category_time,
            x="year_month",
            y="count",
            color="category",
            title="News Categories Distribution Over Time",
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Articles",
            legend_title="Category",
            hovermode="x unified",
        )

        return fig

    def plot_sentiment_distribution(self):
        """Plot sentiment distribution"""
        if self.processed_df is None:
            print("Please process data first using process_data()")
            return None

        # Create sentiment distribution plot
        sentiment_dist = self.processed_df["sentiment"].value_counts()

        colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

        fig = px.pie(
            values=sentiment_dist.values,
            names=sentiment_dist.index,
            title="Distribution of Sentiments",
            color=sentiment_dist.index,
            color_discrete_map=colors,
        )

        fig.update_traces(textposition="inside", textinfo="percent+label")

        return fig

    def save_processed_data(
        self, output_path: str = "data/processed_with_sentiment.csv"
    ):
        """Save processed data to CSV"""
        if self.processed_df is None:
            print("Please process data first using process_data()")
            return False

        try:
            self.processed_df.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = NewsDataProcessor()

    # Load and process data
    if processor.load_data() and processor.process_data():
        # Create plots
        category_fig = processor.plot_category_distribution()
        sentiment_fig = processor.plot_sentiment_distribution()

        # Save plots
        if category_fig:
            category_fig.write_html("category_distribution.html")
            print("Saved category distribution plot")

        if sentiment_fig:
            sentiment_fig.write_html("sentiment_distribution.html")
            print("Saved sentiment distribution plot")

        # Save processed data
        processor.save_processed_data()
    else:
        print("Failed to process data")
