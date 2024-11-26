import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime
from loguru import logger
from config import Config
from news_data_processor import NewsDataProcessor
import os

# Initialize data processor and app
data_processor = NewsDataProcessor()
if not (data_processor.load_data() and data_processor.process_data()):
    raise RuntimeError("Failed to load and process news data")

app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)
server = app.server  # Expose server for gunicorn

# Get port from environment variable
port = int(os.getenv("PORT", 8050))

# API Configuration
API_URL = Config.get_api_url()

# Layout
app.layout = html.Div(
    [
        # Header
        html.H1(
            "News Classification and Sentiment Analysis Dashboard",
            style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "20px"},
        ),
        # Filters Row
        html.Div(
            [
                # Date Range and Category Selection in one row
                html.Div(
                    [
                        # Left side - Date Range
                        html.Div(
                            [
                                html.Label(
                                    "Date Range:", style={"marginRight": "10px"}
                                ),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    start_date=data_processor.processed_df[
                                        "date"
                                    ].min(),
                                    end_date=data_processor.processed_df["date"].max(),
                                    style={"zIndex": 100},
                                ),
                            ],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                        # Right side - Categories
                        html.Div(
                            [
                                html.Label(
                                    "Categories:", style={"marginRight": "10px"}
                                ),
                                dcc.Dropdown(
                                    id="category-filter",
                                    options=[
                                        {"label": cat, "value": cat}
                                        for cat in sorted(
                                            data_processor.processed_df[
                                                "category"
                                            ].unique()
                                        )
                                    ],
                                    value=[],
                                    multi=True,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={
                                "width": "50%",
                                "display": "inline-block",
                                "verticalAlign": "top",
                            },
                        ),
                    ],
                    style={"marginBottom": "20px", "padding": "10px"},
                )
            ]
        ),
        # Distribution Plot Section
        html.Div(
            [
                html.H3(
                    "Category Distribution Over Time",
                    style={
                        "color": "#7f8c8d",
                        "fontSize": "1.2em",
                        "marginBottom": "10px",
                    },
                ),
                dcc.Graph(id="category-timeline"),
            ],
            style={"marginBottom": "30px"},
        ),
        # Two Column Layout for Analysis and Sentiment
        html.Div(
            [
                # Left Column - News Analysis
                html.Div(
                    [
                        html.H2("News Analysis", style={"color": "#34495e"}),
                        # Input Form
                        html.Div(
                            [
                                html.Label("News Title"),
                                dcc.Input(
                                    id="news-title",
                                    type="text",
                                    placeholder="Enter news title...",
                                    style={"width": "100%", "marginBottom": "10px"},
                                ),
                                html.Label("News Description"),
                                dcc.Textarea(
                                    id="news-description",
                                    placeholder="Enter news description...",
                                    style={
                                        "width": "100%",
                                        "height": 100,
                                        "marginBottom": "10px",
                                    },
                                ),
                                html.Label("Publication Date"),
                                dcc.DatePickerSingle(
                                    id="news-date",
                                    date=datetime.now().date(),
                                    style={"marginBottom": "20px"},
                                ),
                                html.Button(
                                    "Analyze",
                                    id="submit-button",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#3498db",
                                        "color": "white",
                                        "padding": "10px 20px",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "marginTop": "10px",
                                    },
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        # Results Section
                        html.Div(
                            [
                                html.H3(
                                    "Analysis Results",
                                    style={"color": "#7f8c8d", "fontSize": "1.2em"},
                                ),
                                html.Div(
                                    id="analysis-results",
                                    style={
                                        "padding": "20px",
                                        "border": "1px solid #bdc3c7",
                                        "borderRadius": "5px",
                                    },
                                ),
                            ]
                        ),
                    ],
                    className="six columns",
                    style={"padding": "20px"},
                ),
                # Right Column - Sentiment Distribution for Predicted Category
                html.Div(
                    [
                        html.H2("Sentiment Analysis", style={"color": "#34495e"}),
                        html.Div(
                            [
                                html.H3(
                                    "Sentiment Distribution for Predicted Category",
                                    style={"color": "#7f8c8d", "fontSize": "1.2em"},
                                ),
                                dcc.Graph(id="sentiment-distribution"),
                            ],
                            style={"marginTop": "20px"},
                        ),
                    ],
                    className="six columns",
                    style={"padding": "20px"},
                ),
            ],
            className="row",
        ),
    ],
    style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"},
)


@app.callback(
    Output("category-timeline", "figure"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("category-filter", "value"),
    ],
)
def update_timeline(start_date, end_date, selected_categories):
    df = data_processor.processed_df.copy()

    # Filter by date
    if start_date and end_date:
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # Group by date and category
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    category_time = (
        df.groupby(["year_month", "category"]).size().reset_index(name="count")
    )
    category_time["year_month"] = category_time["year_month"].astype(str)

    # Create figure
    fig = px.line(
        category_time,
        x="year_month",
        y="count",
        color="category",
        title="News Categories Distribution Over Time",
    )

    # Highlight selected categories if any
    if selected_categories:
        for trace in fig["data"]:
            if trace["name"] not in selected_categories:
                trace["line"]["color"] = "lightgrey"
                trace["line"]["width"] = 1
                trace["opacity"] = 0.3

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Articles",
        legend_title="Category",
        hovermode="x unified",
    )

    return fig


@app.callback(
    [
        Output("sentiment-distribution", "figure"),
        Output("analysis-results", "children"),
    ],
    [Input("submit-button", "n_clicks")],
    [
        State("news-title", "value"),
        State("news-description", "value"),
        State("news-date", "date"),
    ],
)
def update_analysis(n_clicks, title, description, date):
    if n_clicks == 0:
        # Return empty figures for initial load
        empty_sentiment_fig = px.pie(
            title="Sentiment Distribution (Submit news to analyze)"
        )
        return (
            empty_sentiment_fig,
            "Enter news details and click 'Analyze' to see results",
        )

    if not title or not description:
        empty_sentiment_fig = px.pie(title="Sentiment Distribution (No data)")
        return empty_sentiment_fig, "Please enter both title and description"

    try:
        # Make API request
        payload = {"headline": title, "short_description": description, "date": date}
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            # Extract results
            category = data["category"]["predicted"]
            confidence = data["category"]["confidence"]
            sentiment = data["sentiment"]["label"]
            sentiment_scores = data["sentiment"]["scores"]

            # Create sentiment distribution figure for predicted category
            df_sentiment = data_processor.processed_df[
                data_processor.processed_df["category"] == category
            ]
            sentiment_fig = px.pie(
                values=df_sentiment["sentiment"].value_counts(),
                names=df_sentiment["sentiment"].value_counts().index,
                title=f"Sentiment Distribution for {category}",
                color_discrete_map={
                    "positive": "#2ecc71",
                    "neutral": "#95a5a6",
                    "negative": "#e74c3c",
                },
            )

            # Format analysis results
            results = html.Div(
                [
                    html.Div(
                        [
                            html.H4(
                                "Category Prediction",
                                style={"color": "#2c3e50", "marginBottom": "10px"},
                            ),
                            html.P(
                                f"Predicted Category: {category}",
                                style={"fontWeight": "bold"},
                            ),
                            html.P(f"Confidence: {confidence:.2%}"),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            html.H4(
                                "Sentiment Analysis",
                                style={"color": "#2c3e50", "marginBottom": "10px"},
                            ),
                            html.P(
                                f"Overall Sentiment: {sentiment}",
                                style={"fontWeight": "bold"},
                            ),
                            html.P("Sentiment Scores:"),
                            html.Ul(
                                [
                                    html.Li(
                                        f"Positive: {sentiment_scores['positive']:.2%}"
                                    ),
                                    html.Li(
                                        f"Neutral: {sentiment_scores['neutral']:.2%}"
                                    ),
                                    html.Li(
                                        f"Negative: {sentiment_scores['negative']:.2%}"
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            )

            return sentiment_fig, results

        else:
            empty_fig = px.pie(title="Error in Analysis")
            return empty_fig, "Error analyzing news article"

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        empty_fig = px.pie(title="Error")
        return empty_fig, f"An error occurred: {str(e)}"


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=port)
