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


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.././'))

root_dir = os.path.join(root_dir, 'lgbm_model.pkl')


with open(root_dir, 'rb') as file:
    model = pickle.load(file)

comments = ["This video is great", "I absolutely dont hate this video"]

preprocessed_comments = [preprocess_comment(comment) for comment in comments]

predictions = model.predict(preprocessed_comments).tolist()

print(predictions)
print(type(model))