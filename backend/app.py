# app.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
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
load_dotenv()


os.environ["MLFLOW_TRACKING_USERNAME"] = "05mateenkhan"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


dagshub.init(repo_owner='05mateenkhan', repo_name='comments-analyzer', mlflow=True)


def preprocess_comment(comment):
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


# def load_model(model_name, model_version):
    
#     model_uri = f'models:/{model_name}/{model_version}'
#     model = mlflow.pyfunc.load_model(model_uri)
#     return model


# Load the model and vectorizer from the model registry and local storage
# def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
#     return model, vectorizer


def load_model1(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        raise



def load_model(model_name, model_version):
    
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.././'))
    model1 = load_model1(os.path.join(root_dir, 'lgbm_model.pkl'))

    return model


# model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")  # Update paths and versions as needed
model = load_model("yt_chrome_plugin_model", "5")  # Update paths and versions as needed


@app.route('/')
def home():
    return "Welcome to our flask api"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        # transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        print(preprocessed_comments)
        predictions = model.predict(preprocessed_comments).tolist()  # Convert to list
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)