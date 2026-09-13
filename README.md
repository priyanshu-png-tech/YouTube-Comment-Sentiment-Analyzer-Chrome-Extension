# 🎬 YouTube Comment Sentiment Analyzer — Chrome Extension

An end-to-end, production-style ML system that analyzes the sentiment of YouTube comments in real time through a Chrome extension — built with a full MLOps stack (DVC, MLflow, Docker, GitHub Actions, AWS).

Instead of scrolling through thousands of comments, a creator can open any YouTube video, click the extension, and instantly see how their audience feels — positive, neutral, or negative — along with trends, word clouds, and engagement metrics.

---

## 📌 Problem Statement

Popular creators receive thousands of comments per video and simply can't read them all — yet comments are one of the richest, most immediate sources of audience feedback (what worked, what didn't, what to make next). Neither YouTube nor Instagram provide a proper way to analyze comment sentiment at scale.

This project solves that gap with a Chrome extension that:
- Fetches all comments on the currently open YouTube video
- Classifies each comment's sentiment (Positive / Neutral / Negative) using a trained ML model
- Surfaces the results as an easy-to-read visual summary, directly on top of the YouTube page

---

## ✨ Features

- **Real-time sentiment classification** for every comment on the current video
- **Sentiment distribution** shown as a pie chart (% positive / neutral / negative)
- **Sentiment trend over time** — a line chart tracking how sentiment has evolved since the video was published
- **Word cloud** generated from all comments, highlighting the most frequent terms
- **Engagement metrics dashboard** — total comments, unique commenters, average comment length, and an overall sentiment score (out of 10)
- **Top comments view** — top comments displayed individually with their predicted sentiment

---

## 🏗️ Architecture

```
┌─────────────────────────┐        ┌──────────────────────┐        ┌────────────────────────┐
│   Chrome Extension       │  HTTP  │      Flask API        │  loads │   MLflow Model Registry │
│  (HTML + CSS + JS)       │ ─────► │   (backend, Dockerized)│ ─────► │   (backed by AWS S3)   │
│  YouTube Data API calls  │ ◄───── │  /predict, /generate.. │ ◄───── │   LightGBM model        │
└─────────────────────────┘  JSON  └──────────────────────┘  model  └────────────────────────┘
                                             │
                                             ▼
                                  Deployed on AWS EC2 (Auto Scaling
                                  Group + Load Balancer) via
                                  Docker + AWS ECR + CodeDeploy
```

**Flow:** The extension detects the active YouTube video → fetches its comments via the YouTube Data API → sends them to the Flask backend → the backend loads the registered model from the MLflow Model Registry → returns predictions and chart images → the extension renders everything inline.

---

## 🧠 ML Pipeline

| Stage | Details |
|---|---|
| **Dataset** | ~37K labeled Reddit comments (3-class: Positive / Neutral / Negative) |
| **Preprocessing** | Missing value & duplicate removal, lowercasing, URL/special-character cleanup, stopword removal (sentiment-bearing words like *not*, *but* retained), lemmatization |
| **Feature Engineering** | TF-IDF (trigram, `max_features=10000`) — benchmarked against Bag-of-Words and Word2Vec |
| **Class Imbalance** | Benchmarked SMOTE, ADASYN, and undersampling; final model uses LightGBM's built-in balanced class-weighting |
| **Model Selection** | Benchmarked 7 algorithms (XGBoost, LightGBM, Random Forest, SVM, Logistic Regression, Naive Bayes, KNN) |
| **Hyperparameter Tuning** | Bayesian optimization via **Optuna** (tuned `learning_rate`, `n_estimators`, `max_depth`) |
| **Best Model** | **LightGBM** — ~87% accuracy, ~86% F1-score on held-out test data |
| **Also Explored** | Word2Vec embeddings, custom linguistic features (POS ratios, lexical diversity), stacking ensembles, and a BERT baseline (not used in production — kept within classical ML scope) |

---

## ⚙️ MLOps Stack

- **Experiment Tracking:** MLflow tracking server hosted on AWS EC2, with artifacts stored on AWS S3
- **Pipeline Orchestration:** DVC pipeline (`data_ingestion → data_preprocessing → model_building → model_evaluation → register_model`), fully parameterized via `params.yaml`
- **Model Registry:** MLflow Model Registry with a Staging → Production promotion workflow
- **CI/CD:** GitHub Actions pipeline that, on every push:
  1. Runs the DVC pipeline end-to-end and logs the run to MLflow
  2. Runs automated `pytest` suites — model loading test, model signature test, and threshold-based performance test
  3. Promotes the model from Staging to Production if all tests pass
  4. Runs API tests against the Flask endpoints
  5. Builds a Docker image and pushes it to AWS ECR
  6. Triggers deployment via AWS CodeDeploy
- **Deployment:** Dockerized Flask API deployed to an AWS Auto Scaling Group (EC2) behind an Application Load Balancer, using CodeDeploy for rolling (one-at-a-time) zero-downtime updates

---

## 🛠️ Tech Stack

**Machine Learning:** Python, scikit-learn, LightGBM, XGBoost, NLTK, Optuna
**MLOps:** DVC, MLflow, GitHub Actions, pytest
**Backend:** Flask, Flask-CORS, Matplotlib, Seaborn, WordCloud
**Frontend:** HTML, CSS, JavaScript (Chrome Extension — Manifest V3), YouTube Data API
**Infra & DevOps:** Docker, AWS (EC2, S3, ECR, Auto Scaling Groups, CodeDeploy, IAM, Application Load Balancer)
**Tooling:** Git, Postman, VS Code

---

## 📂 Repository Structure

```
├── .github/workflows/          # CI/CD pipeline (GitHub Actions)
│   └── cicd.yaml
├── data/                       # DVC-tracked data (raw → interim)
├── src/
│   ├── data/                   # data_ingestion.py, data_preprocessing.py
│   ├── models/                 # model_building.py, model_evaluation.py, register_model.py
├── flask_app/                  # Flask backend (API + serving logic)
│   └── app.py
├── scripts/                    # Test scripts (model loading, signature, performance, API)
├── deploy/                     # CodeDeploy scripts (appspec.yml, install/start scripts)
├── Dockerfile
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Centralized pipeline parameters
├── requirements.txt
└── README.md
```

*(Frontend/Chrome extension files — `manifest.json`, `popup.html`, `popup.js` — live in a companion folder/repo.)*

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up the backend
```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
python flask_app/app.py
```

### 3. Reproduce the ML pipeline (optional)
```bash
dvc repro
```

### 4. Load the Chrome extension
1. Go to `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the extension folder
4. Open any YouTube video and click the extension icon

---

## 📊 Results

| Metric | Baseline (Random Forest + BoW) | Final Model (LightGBM + TF-IDF, tuned) |
|---|---|---|
| Accuracy | ~64% | **~87%** |
| F1-score | ~48% | **~86%** |

---

## 🔮 Future Improvements

- Per-sentiment comment tabs (click "Positive" to filter only positive comments)
- Automated model retraining triggered by monitoring/drift detection
- Deep learning (BERT-based) sentiment model for improved accuracy
- Automated content summarization and theme tagging of comments (feedback / suggestion / spam)
- Spam/bot comment detection
- Export analysis as PDF/CSV

---

## 📄 License

This project is available under the [MIT License](LICENSE).
