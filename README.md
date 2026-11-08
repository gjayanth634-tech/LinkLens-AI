# 🛡️ LinkLens AI

**An Explainable Machine Learning-Based Phishing URL Detection System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> MCA Semester 3 Mini Project

LinkLens AI is a cybersecurity web application that uses machine learning to identify potentially phishing URLs and distinguish them from legitimate-looking URLs. The system analyzes URL-string characteristics **without ever visiting or loading the submitted website**.

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Testing](#-testing)
- [Security](#-security)
- [Screenshots](#-screenshots)
- [Future Scope (Semester 4)](#-future-scope-semester-4)

---

## ✨ Features

- **ML-Powered Detection**: CatBoost, Random Forest, and XGBoost classifiers trained on URL-string features
- **Safe Analysis**: URLs are analyzed without visiting them — no HTTP requests, no browser automation
- **Explainable AI**: SHAP (SHapley Additive exPlanations) shows which features influenced the prediction
- **Three-Category Classification**: Potential Phishing, Legitimate-Looking, or Uncertain
- **User Authentication**: Secure registration and login with password hashing
- **Scan History**: Complete audit trail of all URL scans with search and filtering
- **Professional Dashboard**: Real-time statistics with Chart.js visualizations
- **Dark Cybersecurity Theme**: Modern, responsive UI designed for desktop, tablet, and mobile

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5, Chart.js |
| **Backend** | Python, Flask, Flask-WTF (CSRF) |
| **ML/AI** | Scikit-learn, CatBoost, XGBoost, SHAP, NumPy, Pandas |
| **Database** | MongoDB, PyMongo |
| **Security** | Werkzeug (password hashing), Flask sessions |

---

## 📁 Project Structure

```
LinkLens-AI/
│
├── app.py                          # Main Flask application (all routes)
├── train_model.py                  # ML training pipeline
├── config.py                       # Application configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
│
├── dataset/
│   └── phishing_urls.csv           # Training dataset (generated or provided)
│
├── models/
│   ├── random_forest.pkl           # Trained Random Forest model
│   ├── xgboost_model.pkl           # Trained XGBoost model
│   ├── catboost_model.pkl          # Trained CatBoost model
│   ├── feature_names.pkl           # Feature names for consistency
│   └── evaluation_results.json     # Model evaluation metrics
│
├── modules/
│   ├── feature_extraction.py       # 14 URL-string feature extraction
│   ├── prediction.py               # ML prediction wrapper
│   ├── explanation.py              # SHAP explanation generator
│   ├── model_evaluation.py         # Model evaluation metrics
│   └── url_validation.py           # URL input validation
│
├── database/
│   └── mongodb.py                  # MongoDB connection and helpers
│
├── templates/
│   ├── base.html                   # Base template with navigation
│   ├── index.html                  # Landing page
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── dashboard.html              # User dashboard
│   ├── scanner.html                # URL scanner
│   ├── result.html                 # Scan result with SHAP
│   ├── history.html                # Scan history
│   ├── scan_details.html           # Individual scan details
│   └── profile.html                # User profile settings
│
├── static/
│   ├── css/style.css               # Complete dark theme stylesheet
│   ├── js/main.js                  # General utilities
│   ├── js/scanner.js               # Scanner page logic
│   ├── js/dashboard.js             # Dashboard chart logic
│   └── images/logo.png             # Application logo
│
└── tests/
    ├── test_features.py            # Feature extraction tests
    ├── test_prediction.py          # Prediction module tests
    └── test_authentication.py      # Auth and validation tests
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** — [Download Python](https://python.org/downloads)
- **MongoDB** — [Download MongoDB Community](https://www.mongodb.com/try/download/community) or use [MongoDB Atlas](https://www.mongodb.com/atlas)
- **Git** (optional) — [Download Git](https://git-scm.com)

### Step-by-Step Setup

1. **Clone or download the project:**
   ```bash
   cd "LinkLens AI"
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Windows (CMD):**
   ```cmd
   venv\Scripts\activate.bat
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables:**
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and set:
   - `SECRET_KEY` — a random secret string
   - `MONGO_URI` — your MongoDB connection string (default: `mongodb://localhost:27017/`)

6. **Start MongoDB:**
   - If using local MongoDB, ensure the `mongod` service is running
   - If using MongoDB Atlas, update `MONGO_URI` in `.env` with your Atlas connection string

7. **Train the ML models:**
   ```bash
   python train_model.py
   ```
   This will:
   - Generate a synthetic training dataset (10,000 URLs)
   - Extract 14 URL-string features
   - Train Random Forest, XGBoost, and CatBoost classifiers on the same feature split
   - Save models to the `models/` directory
   - Print evaluation metrics

8. **Run the application:**
   ```bash
   python app.py
   ```

9. **Open in browser:**
   ```
   http://127.0.0.1:5000
   ```

---

## 📊 Dataset

The training pipeline generates a **synthetic dataset** of 10,000 URLs:
- **5,000 legitimate URLs**: Patterns from common domains (Google, Amazon, GitHub, etc.) with proper structure
- **5,000 phishing URLs**: Patterns including IP addresses, suspicious subdomains, typosquatting, @ symbols, excessive hyphens, and suspicious keywords

The dataset is saved to `dataset/phishing_urls.csv` with columns: `url`, `label` (0=legitimate, 1=phishing).

### Using a Real Dataset

To use a real-world dataset instead:

1. Download a phishing URL dataset (e.g., from [PhiUSIIL](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) or [Kaggle](https://www.kaggle.com/datasets))
2. Format it as CSV with columns: `url`, `label`
3. Place it at `dataset/phishing_urls.csv`
4. Re-run `python train_model.py`

---

## 🤖 Model Training

### Algorithms

| Model | Configuration |
|-------|--------------|
| **Random Forest** | 100 trees, max_depth=15, min_samples_split=5 |
| **XGBoost** | 100 estimators, max_depth=6, learning_rate=0.1 |
| **CatBoost** | 500 iterations, depth=6, learning_rate=0.05, Accuracy metric |

The scanner defaults to CatBoost. Select Random Forest or XGBoost in the model menu,
or choose **Compare all models** to display all three model predictions and class probabilities.

### Features Extracted (14 total)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Total number of characters in the URL |
| 2 | `hostname_length` | Length of the hostname |
| 3 | `domain_length` | Length of the registered domain |
| 4 | `path_length` | Length of the URL path |
| 5 | `num_dots` | Number of dots in the URL |
| 6 | `num_hyphens` | Number of hyphens in the URL |
| 7 | `num_digits` | Number of digits in the URL |
| 8 | `num_special_chars` | Number of special characters |
| 9 | `num_subdomains` | Number of subdomain levels |
| 10 | `num_params` | Number of query parameters |
| 11 | `has_https` | Whether HTTPS is used (1/0) |
| 12 | `has_ip_address` | Whether hostname is an IP address (1/0) |
| 13 | `has_at_symbol` | Whether URL contains @ symbol (1/0) |
| 14 | `has_suspicious_patterns` | Whether URL contains suspicious keywords (1/0) |

### Evaluation Metrics

Both models are evaluated on a held-out 20% test set using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

Results are saved to `models/evaluation_results.json`.

---

## 🖥️ Usage Guide

### Workflow

1. **Register** an account on the registration page
2. **Login** with your credentials
3. **Dashboard** shows your scan statistics and recent activity
4. **Scanner** — paste any URL and click "Scan URL"
5. **Result** — view the prediction (Phishing/Legitimate/Uncertain) with SHAP explanation
6. **History** — browse, search, and filter all past scans
7. **Profile** — update your name or change password

### Prediction Categories

| Category | Score Range | Meaning |
|----------|-----------|---------|
| 🔴 **Potential Phishing** | ≥ 0.6 | Suspicious characteristics detected |
| 🟢 **Legitimate-Looking** | < 0.4 | URL appears to have legitimate characteristics |
| 🟡 **Uncertain** | 0.4 – 0.6 | Model lacks confidence — verify independently |

> ⚠️ **Disclaimer**: This system analyzes URL structure only. It does not guarantee that a URL is safe or dangerous. Never rely solely on this tool for security decisions.

---

## 🧪 Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Or run individual test files:
```bash
python -m pytest tests/test_features.py -v
python -m pytest tests/test_prediction.py -v
python -m pytest tests/test_authentication.py -v
```

Using unittest:
```bash
python -m unittest discover tests -v
```

---

## 🔒 Security

- **Password Hashing**: Werkzeug `generate_password_hash` / `check_password_hash`
- **Session Security**: HTTP-only, SameSite=Lax cookies
- **CSRF Protection**: Flask-WTF CSRFProtect on all forms
- **Input Validation**: Server-side URL and form validation
- **User Isolation**: Scan history is scoped to authenticated users only
- **No URL Visiting**: Submitted URLs are never opened, loaded, or connected to
- **Safe Output**: Jinja2 auto-escaping prevents XSS

---

## 🔮 Future Scope (Semester 4)

The modular architecture supports these planned enhancements:

- Cross-dataset generalization testing
- Probability calibration for better confidence scores
- Advanced dataset quality analysis
- Robustness testing on adversarial URL patterns
- SHAP explanation consistency analysis
- Model efficiency comparison
- PDF report generation
- Advanced analytics dashboard
- Real-time URL monitoring

---

## 👨‍💻 Authors

- MCA Semester 3 Mini Project

## 📄 License

This project is developed for academic purposes as part of an MCA program.

---

*Built with ❤️ using Python, Flask, and Machine Learning*
