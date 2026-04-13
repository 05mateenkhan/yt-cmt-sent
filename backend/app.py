# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import pandas as pd
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

import mlflow
from mlflow.tracking import MlflowClient
import dagshub
from dotenv import load_dotenv
import os
import pickle
from pydantic import BaseModel
from typing import List, Optional
load_dotenv()


os.environ["MLFLOW_TRACKING_USERNAME"] = "05mateenkhan"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# dagshub.init(repo_owner='05mateenkhan', repo_name='comments-analyzer', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/05mateenkhan/comments-analyzer.mlflow")


# Pydantic models for request/response
class CommentItem(BaseModel):
    text: str
    timestamp: Optional[str] = None


class CommentsRequest(BaseModel):
    comments: List[str]


class CommentsWithTimestampsRequest(BaseModel):
    comments: List[CommentItem]


class SentimentResponse(BaseModel):
    comment: str
    sentiment: str


class SentimentWithTimestampResponse(BaseModel):
    comment: str
    sentiment: str
    timestamp: Optional[str] = None


class SentimentCounts(BaseModel):
    positive: Optional[int] = 0
    neutral: Optional[int] = 0
    negative: Optional[int] = 0


class GenerateChartRequest(BaseModel):
    sentiment_counts: dict


class GenerateWordcloudRequest(BaseModel):
    comments: List[str]


class SentimentDataItem(BaseModel):
    timestamp: str
    sentiment: int


class GenerateTrendGraphRequest(BaseModel):
    sentiment_data: List[SentimentDataItem]


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model1(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise


def load_model(model_name: str, model_version: str):

    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.././'))

    return model


# model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed
model = load_model("yt_chrome_plugin_model", "15")  # Update paths and versions as needed


@app.get("/")
def home():
    return {"message": "Welcome to our FastAPI"}


@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: CommentsWithTimestampsRequest):
    comments_data = request.comments

    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Make predictions
        df = pd.DataFrame(preprocessed_comments, columns=["clean_comment"])
        predictions = model.predict(df).tolist()

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return response


@app.post("/predict")
def predict(request: CommentsRequest):
    comments = request.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Make predictions
        print(preprocessed_comments)
        df = pd.DataFrame(preprocessed_comments, columns=["clean_comment"])
        predictions = model.predict(df).tolist()

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Return the response with original comments and predicted sentiments
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return response


@app.post("/generate_chart")
def generate_chart(request: GenerateChartRequest):
    try:
        sentiment_counts = request.sentiment_counts

        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post("/generate_wordcloud")
def generate_wordcloud(request: GenerateWordcloudRequest):
    try:
        comments = request.comments

        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph")
def generate_trend_graph(request: GenerateTrendGraphRequest):
    try:
        sentiment_data = request.sentiment_data

        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame([item.model_dump() for item in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)