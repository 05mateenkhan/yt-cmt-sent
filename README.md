# YouTube Comment Sentiment Analysis

A machine learning-powered API that analyzes sentiment in YouTube video comments. The system classifies comments as Positive, Neutral, or Negative and provides visualization tools including pie charts, word clouds, and trend graphs.

## Project Overview

This project is a sentiment analysis API for YouTube comments that:

- **Accepts** YouTube comments (with optional timestamps)
- **Analyzes** sentiment using a trained ML model (LightGBM)
- **Returns** sentiment predictions with confidence
- **Visualizes** results through charts, word clouds, and trend graphs

The model is stored in MLflow Model Registry on DagsHub and is loaded dynamically for predictions.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│   MLflow    │
│  (Frontend) │     │   Backend   │     │   Model     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │   Pie    │  │  Word    │  │  Trend   │
        │  Chart   │  │  Cloud   │  │  Graph   │
        └──────────┘  └──────────┘  └──────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend API | FastAPI |
| ML Framework | LightGBM |
| Model Registry | MLflow (DagsHub) |
| Data Versioning | DVC |
| Visualization | Matplotlib, WordCloud |
| Text Processing | NLTK (stopwords, lemmatization) |
| Deployment | Docker, AWS EC2 |
| CI/CD | GitHub Actions |

---

## API Endpoints

### 1. Home
```http
GET /
```
Returns a welcome message.

**Response:**
```json
{"message": "Welcome to our FastAPI"}
```

---

### 2. Predict Sentiment (Simple)
```http
POST /predict
```

Predicts sentiment for a list of comments.

**Request Body:**
```json
{
  "comments": ["Great video!", "I don't like this content", "It's okay"]
}
```

**Response:**
```json
[
  {"comment": "Great video!", "sentiment": "1"},
  {"comment": "I don't like this content", "sentiment": "-1"},
  {"comment": "It's okay", "sentiment": "0"}
]
```

**Sentiment Values:**
- `1` = Positive
- `0` = Neutral
- `-1` = Negative

---

### 3. Predict with Timestamps
```http
POST /predict_with_timestamps
```

Predicts sentiment for comments with timestamps (useful for trend analysis).

**Request Body:**
```json
{
  "comments": [
    {"text": "Great video!", "timestamp": "2024-01-01T10:00:00Z"},
    {"text": "Not bad", "timestamp": "2024-01-02T10:00:00Z"}
  ]
}
```

**Response:**
```json
[
  {"comment": "Great video!", "sentiment": "1", "timestamp": "2024-01-01T10:00:00Z"},
  {"comment": "Not bad", "sentiment": "0", "timestamp": "2024-01-02T10:00:00Z"}
]
```

---

### 4. Generate Pie Chart
```http
POST /generate_chart
```

Generates a pie chart showing sentiment distribution.

**Request Body:**
```json
{
  "sentiment_counts": {
    "1": 50,
    "0": 30,
    "-1": 20
  }
}
```

**Response:** PNG image file

---

### 5. Generate Word Cloud
```http
POST /generate_wordcloud
```

Generates a word cloud from comment text.

**Request Body:**
```json
{
  "comments": ["Great video loved it", "amazing content", "best tutorial ever"]
}
```

**Response:** PNG image file

---

### 6. Generate Trend Graph
```http
POST /generate_trend_graph
```

Generates a line graph showing sentiment trends over time.

**Request Body:**
```json
{
  "sentiment_data": [
    {"timestamp": "2024-01-01", "sentiment": 1},
    {"timestamp": "2024-02-01", "sentiment": 0},
    {"timestamp": "2024-03-01", "sentiment": -1}
  ]
}
```

**Response:** PNG image file

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/05mateenkhan/yt-cmt-sent-analysis.git
cd yt-cmt-sent-analysis
```

### 2. Create Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `backend` directory:
```env
DAGSHUB_TOKEN=your_dagshub_token_here
MLFLOW_TRACKING_USERNAME=your_username
```

### 5. Run the API
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### Alternative: Run with Uvicorn
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

---

## Docker Deployment

### Build Docker Image
```bash
docker build -t yt-cmt-sent .
```

### Run Container
```bash
docker run -d -p 5000:5000 -e DAGSHUB_TOKEN=your_token yt-cmt-sent
```

### Or Pull from Docker Hub
```bash
docker run -d -p 5000:5000 -e DAGSHUB_TOKEN=your_token mateenkhan14/yt-cmt-sent:latest
```

---

## Text Preprocessing Pipeline

The API uses the following preprocessing steps:

1. **Lowercase Conversion** - All text converted to lowercase
2. **Whitespace Cleaning** - Remove leading/trailing whitespace
3. **Newline Removal** - Replace `\n` with spaces
4. **Special Character Removal** - Keep only alphanumeric and `!?.,`
5. **Stopword Removal** - Remove English stopwords except: `not, but, however, no, yet`
6. **Lemmatization** - Reduce words to base form using WordNet

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

1. **Code Checkout** - Pull latest code
2. **Setup Python** - Configure Python 3.11
3. **Install Dependencies** - Install required packages
4. **Run DVC Pipeline** - Execute ML pipeline with `dvc repro`
5. **Push Data** - Push DVC-tracked files to remote storage
6. **Run Tests** - Execute model loading and signature tests
7. **Promote Model** - Move model to production stage
8. **Build Docker** - Create Docker image
9. **Deploy to EC2** - Deploy container to AWS EC2

---

## Project Structure

```
yt-cmt-sent-analysis/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── requirements.txt    # Backend dependencies
│   └── test.py            # Model testing script
├── data/
│   ├── external/          # Third-party data
│   ├── interim/           # Intermediate processed data
│   ├── processed/         # Final processed data
│   └── raw/               # Original data
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks
├── references/            # Documentation
├── reports/              # Generated reports
├── src/                   # Source code
│   ├── data/             # Data ingestion
│   └── visualization/    # Visualization scripts
├── .github/workflows/    # CI/CD configuration
├── requirements.txt      # Project dependencies
├── setup.py              # Project setup
└── README.md             # This file
```

---

## Model Information

- **Model**: LightGBM Classifier
- **Model Name**: `yt_chrome_plugin_model`
- **Model Version**: 15
- **MLflow Tracking URI**: `https://dagshub.com/05mateenkhan/comments-analyzer.mlflow`
- **Input**: Preprocessed text comments
- **Output**: Sentiment classification (-1, 0, 1)

---

## Testing

Run the backend tests:
```bash
cd backend
python test.py
```

This will load the model and test predictions on sample comments.

---

## Deployment to Production

The CI/CD pipeline automatically deploys to AWS EC2 on successful builds:

1. Docker image is pushed to Docker Hub
2. EC2 instance pulls the latest image
3. Container runs on port 5000 (mapped to port 80)

To deploy manually:
```bash
docker pull mateenkhan14/yt-cmt-sent:latest
docker run -d -p 80:5000 -e DAGSHUB_TOKEN=your_token mateenkhan14/yt-cmt-sent:latest
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DAGSHUB_TOKEN` | Authentication token for DagsHub | Yes |
| `MLFLOW_TRACKING_USERNAME` | MLflow username | Yes |

---

## License

MIT License - See LICENSE file for details.