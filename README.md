
# LANYERO-FORTUNATE 23/U/3718/GCS/PS
# This is the first asignment and the second asignment
# 🔐 Phishing URL Detection System

A machine learning-based web application that detects whether a URL is **phishing (malicious)** or **legitimate (safe)** using advanced feature extraction and classification techniques.

---

## 🚀 Project Overview

This project uses:
- **Machine Learning (Logistic Regression)**
- **TF-IDF text features + URL structural features**
- **Flask Web Application**
- **Interactive Dashboard with Chart.js**

Users can input a URL and instantly get:
- ✅ Phishing probability (%)
- ✅ Legitimate probability (%)
- ✅ Visual graph representation



## 🧠 How It Works

1. URL is entered by the user
2. System extracts:
   - Text-based features (TF-IDF)
   - Structural features (length, symbols, domain patterns, etc.)
3. Model predicts:
   - `spam` → phishing
   - `not_spam` → legitimate
4. Results are displayed on a dashboard



## 🏗️ Project Structure
machine_learning/
│
├── App/
│ ├── app.py # Flask web application
│ ├── model.py # ML model + prediction logic
│
├── train.py # Training script
├── model.pkl # Trained model (generated)
├── metrics.json # Model performance
├── phishing_site_urls.csv # Dataset (NOT included in repo)
├── requirements.txt
├── .gitignore
└── README.md

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

2. Create virtual environment
python -m venv myenv

Activate it:

Windows:

myenv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Add dataset

Place your dataset in:

C:/machine_learning/phishing_site_urls.csv

⚠️ Dataset is not included due to size.

5. Train the model
python train.py

This will generate:

model.pkl
metrics.json
6. Run the application
python App/app.py

Open in browser:

http://127.0.0.1:5000

