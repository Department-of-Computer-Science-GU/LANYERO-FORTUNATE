import json
import os
import re
from typing import Dict, Tuple, List
from urllib.parse import urlparse
import random

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV


DATA_PATH = "C:/machine_learning/combined_phishing_dataset.csv"
MODEL_PATH = "C:/machine_learning/model.pkl"
METRICS_PATH = "C:/machine_learning/metrics.json"

# Fixed number of structural features
N_STRUCTURAL_FEATURES = 20  # IMPORTANT: Keep this consistent

# Common legitimate domains to add as training examples
LEGITIMATE_DOMAINS = [
    # Search engines
    "google.com", "google.co.ug", "bing.com", "yahoo.com", "duckduckgo.com",
    # Social media
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "reddit.com",
    "youtube.com", "tiktok.com", "pinterest.com", "whatsapp.com", "telegram.org",
    # Tech/Git
    "github.com", "gitlab.com", "bitbucket.org", "stackoverflow.com",
    # E-commerce
    "amazon.com", "ebay.com", "alibaba.com", "walmart.com",
    # Education
    "harvard.edu", "stanford.edu", "mit.edu", "ox.ac.uk",
    "gulu.ac.ug", "mak.ac.ug", "kyu.ac.ug", "utamu.ac.ug", "must.ac.ug",
    # Government
    "usa.gov", "gov.uk", "europa.eu", "nasa.gov", "nih.gov",
    # Banking
    "paypal.com", "stripe.com", "chase.com", "bankofamerica.com",
    # Cloud/SaaS
    "microsoft.com", "apple.com", "dropbox.com", "slack.com", "zoom.us",
    # Wikipedia
    "wikipedia.org", "wikimedia.org",
    # Entertainment
    "netflix.com", "spotify.com", "youtube.com",
    # Ugandan sites
    "monitor.co.ug", "newvision.co.ug", "mtn.co.ug", "airtel.co.ug"
]


def generate_legitimate_urls(domains: List[str], num_per_domain: int = 3) -> List[Tuple[str, str]]:
    """Generate legitimate URL examples with common paths"""
    legitimate_examples = []
    
    common_paths = [
        "", "/", "/home", "/index", "/about", "/contact", "/help", "/faq",
        "/search", "/login", "/signin", "/register", "/signup", "/account",
        "/profile", "/settings", "/dashboard", "/products", "/services",
        "/blog", "/news", "/articles", "/events", "/support", "/terms",
        "/privacy", "/careers", "/jobs", "/docs", "/documentation", "/api"
    ]
    
    for domain in domains:
        for _ in range(num_per_domain):
            path = random.choice(common_paths) if random.random() > 0.3 else ""
            protocol = random.choice(["https://", "http://"])
            
            # Sometimes add subdomain
            if random.random() > 0.8:
                subdomain = random.choice(["www", "mail", "blog", "shop", "news", "app", "api"])
                url = f"{protocol}{subdomain}.{domain}{path}"
            else:
                url = f"{protocol}{domain}{path}"
            
            legitimate_examples.append((url, "not_spam"))
    
    return legitimate_examples


class URLTokenizer:
    """Custom tokenizer for URLs - lightweight version"""
    
    def __call__(self, text: str) -> List[str]:
        text = text.lower()
        tokens = []
        
        # Split on URL delimiters
        parts = re.split(r'[./?=&%#\-_]', text)
        
        for part in parts:
            if part and len(part) > 0:
                tokens.append(part)
        
        # Add domain TLD as feature
        tld_match = re.search(r'\.([a-z]{2,})(?:/|$)', text)
        if tld_match:
            tokens.append(f"tld_{tld_match.group(1)}")
        
        # Add protocol indicator
        if text.startswith('https://'):
            tokens.append('proto_https')
        elif text.startswith('http://'):
            tokens.append('proto_http')
        
        return tokens


class URLStructuralFeatures(BaseEstimator, TransformerMixin):
    """
    Extract numerical structural features from URLs.
    IMPORTANT: Always returns exactly N_STRUCTURAL_FEATURES (20) features.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def get_features(self, url: str) -> List[float]:
        """Extract exactly 20 features from a URL"""
        features = []
        url_lower = url.lower()
        
        if '://' not in url_lower:
            url_lower = f'http://{url_lower}'
        
        try:
            parsed = urlparse(url_lower)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            full_url = url_lower
        except:
            domain = url_lower.split('/')[0] if '/' in url_lower else url_lower
            path = ''
            query = ''
            full_url = url_lower
        
        # 1. Total URL length (normalized)
        features.append(min(len(full_url), 500) / 500.0)
        
        # 2. Domain length (normalized)
        features.append(min(len(domain), 100) / 100.0)
        
        # 3. Path length (normalized)
        features.append(min(len(path), 200) / 200.0)
        
        # 4. Query length (normalized)
        features.append(min(len(query), 200) / 200.0)
        
        # 5. Number of dots in domain (normalized)
        features.append(min(domain.count('.'), 10) / 10.0)
        
        # 6. Number of hyphens in domain (normalized)
        features.append(min(domain.count('-'), 5) / 5.0)
        
        # 7. Digit ratio in domain
        digit_count = sum(c.isdigit() for c in domain)
        features.append(digit_count / max(len(domain), 1))
        
        # 8. Special character ratio in domain
        special_chars = len(re.findall(r'[^a-zA-Z0-9.-]', domain))
        features.append(special_chars / max(len(domain), 1))
        
        # 9. Vowel ratio in domain
        vowels = len(re.findall(r'[aeiou]', domain))
        features.append(vowels / max(len(domain), 1))
        
        # 10. Uses HTTPS
        features.append(1.0 if full_url.startswith('https://') else 0.0)
        
        # 11. Has IP address instead of domain
        features.append(1.0 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0.0)
        
        # 12. Has @ symbol (phishing indicator)
        features.append(1.0 if '@' in full_url else 0.0)
        
        # 13. Has double slash in path
        features.append(1.0 if '//' in path else 0.0)
        
        # 14. Has suspicious TLD
        suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work'}
        has_suspicious_tld = any(domain.endswith(tld) for tld in suspicious_tlds)
        features.append(1.0 if has_suspicious_tld else 0.0)
        
        # 15. Path depth (normalized)
        path_depth = len([p for p in path.split('/') if p])
        features.append(min(path_depth, 10) / 10.0)
        
        # 16. Has file extension
        has_extension = 1.0 if re.search(r'\.(php|html?|asp|jsp|do|cgi|pl|py|exe)$', path) else 0.0
        features.append(has_extension)
        
        # 17. Number of query parameters (normalized)
        num_params = len([p for p in query.split('&') if '=' in p])
        features.append(min(num_params, 10) / 10.0)
        
        # 18. Has suspicious parameters
        suspicious_params = ['redirect', 'url=', 'goto=', 'return=', 'login', 'account', 'verify']
        has_suspicious_param = any(param in query.lower() for param in suspicious_params)
        features.append(1.0 if has_suspicious_param else 0.0)
        
        # 19. Domain has numbers (binary)
        features.append(1.0 if re.search(r'\d', domain) else 0.0)
        
        # 20. URL contains common legitimate words
        legitimate_words = ['about', 'contact', 'help', 'support', 'faq', 'blog', 'news', 
                           'privacy', 'terms', 'docs', 'api', 'home', 'index']
        has_legitimate_word = any(word in url_lower for word in legitimate_words)
        features.append(1.0 if has_legitimate_word else 0.0)
        
        # Ensure exactly 20 features
        assert len(features) == N_STRUCTURAL_FEATURES, f"Expected {N_STRUCTURAL_FEATURES} features, got {len(features)}"
        
        return features
    
    def transform(self, X):
        return np.vstack([self.get_features(url) for url in X])


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detect URL and label columns"""
    columns = list(df.columns)
    
    # Try to find URL column
    url_candidates = ['URL', 'url', 'link', 'links', 'text', 'message', 'content']
    url_col = None
    for candidate in url_candidates:
        if candidate in columns:
            url_col = candidate
            break
    
    # Try to find label column
    label_candidates = ['Label', 'label', 'class', 'target', 'is_spam', 'spam', 'category']
    label_col = None
    for candidate in label_candidates:
        if candidate in columns:
            label_col = candidate
            break
    
    if url_col is None:
        url_col = columns[0]
    if label_col is None:
        label_col = columns[1] if len(columns) > 1 else None
    
    if url_col is None or label_col is None:
        raise ValueError("Could not detect URL/label columns")
    
    return url_col, label_col


def normalize_label(value) -> str:
    """Normalize label to 'spam' or 'not_spam'"""
    if pd.isna(value):
        return "unknown"
    
    text = str(value).strip().lower()
    
    spam_indicators = {"spam", "phishing", "malicious", "bad", "1", "true", "yes"}
    not_spam_indicators = {"ham", "legitimate", "benign", "safe", "good", "0", "false", "no", "not_spam"}
    
    if text in spam_indicators:
        return "spam"
    if text in not_spam_indicators:
        return "not_spam"
    
    # Try numeric
    try:
        num = float(text)
        return "spam" if int(num) == 1 else "not_spam"
    except:
        pass
    
    return "unknown"


def prepare_data(df: pd.DataFrame, url_col: str, label_col: str) -> pd.DataFrame:
    """Clean and prepare data"""
    work_df = df[[url_col, label_col]].copy()
    work_df = work_df.dropna()
    
    work_df[url_col] = work_df[url_col].astype(str).str.strip()
    work_df[label_col] = work_df[label_col].apply(normalize_label)
    
    work_df = work_df[work_df[url_col] != ""]
    work_df = work_df[work_df[label_col].isin(["spam", "not_spam"])]
    
    return work_df
# ================================
# LOAD MODEL FOR PREDICTION
# ================================
def load_model():
    """Load the saved model safely, including old models saved from __main__."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not trained yet. Expected model file at: {MODEL_PATH}"
        )

    # Compatibility fix: if the model was previously saved by running
    # model.py directly, pickle may look for these classes inside __main__.
    import sys
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "URLTokenizer", URLTokenizer)
        setattr(main_module, "URLStructuralFeatures", URLStructuralFeatures)

    return joblib.load(MODEL_PATH)


# ================================
# PREDICT FUNCTION USED BY app.py
# ================================
def predict_phishing_percentage(url: str):
    """
    Return the 4 values expected by app.py:
    phishing_percent, legit_percent, prediction, accuracy
    """
    model = load_model()

    proba = model.predict_proba([url])[0]
    prediction = model.predict([url])[0]

    classes = list(model.classes_)
    spam_index = classes.index("spam")
    not_spam_index = classes.index("not_spam")

    phishing_percent = round(proba[spam_index] * 100, 2)
    legit_percent = round(proba[not_spam_index] * 100, 2)

    accuracy = 0
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            accuracy = metrics.get("accuracy", 0)

    return phishing_percent, legit_percent, prediction, accuracy


def train_and_save(data_path: str = DATA_PATH) -> Dict:
    """Train the model"""
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    
    url_col, label_col = detect_columns(raw_df)
    print(f"Using columns: URL='{url_col}', Label='{label_col}'")
    
    clean_df = prepare_data(raw_df, url_col, label_col)
    
    if clean_df.empty:
        raise ValueError("No valid rows after cleaning")
    
    print(f"\nOriginal dataset: {len(clean_df)} rows")
    print(f"Spam: {sum(clean_df[label_col] == 'spam')}")
    print(f"Not spam: {sum(clean_df[label_col] == 'not_spam')}")
    
    # Augment with legitimate URLs
    print("\n=== Augmenting with Legitimate URLs ===")
    legitimate_examples = generate_legitimate_urls(LEGITIMATE_DOMAINS, num_per_domain=4)
    legit_df = pd.DataFrame(legitimate_examples, columns=[url_col, label_col])
    print(f"Added {len(legitimate_examples)} legitimate URL examples")
    
    # Balance the dataset
    spam_df = clean_df[clean_df[label_col] == "spam"]
    not_spam_df = clean_df[clean_df[label_col] == "not_spam"]
    
    # Sample to reasonable sizes
    spam_sample = spam_df.sample(n=min(len(spam_df), 30000), random_state=42)
    not_spam_sample = not_spam_df.sample(n=min(len(not_spam_df), 30000), random_state=42)
    
    # Combine all data
    final_df = pd.concat([spam_sample, not_spam_sample, legit_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal dataset: {len(final_df)} rows")
    print(f"Spam: {sum(final_df[label_col] == 'spam')}")
    print(f"Not spam: {sum(final_df[label_col] == 'not_spam')}")
    
    # Prepare features
    X = final_df[url_col].astype(str).tolist()
    y = final_df[label_col].astype(str).tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create feature union
    combined_features = FeatureUnion([
        ('tfidf', TfidfVectorizer(
            tokenizer=URLTokenizer(),
            lowercase=False,
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ('structural', Pipeline([
            ('extractor', URLStructuralFeatures()),
            ('scaler', StandardScaler())
        ]))
    ])
    
    # Train model
    print("\nTraining model...")
    model = Pipeline([
        ('features', combined_features),
        ('classifier', LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            C=1.0,
            solver='liblinear'
        ))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Test on legitimate URLs
    print("\n=== Testing on Common Legitimate URLs ===")
    test_urls = [
        "https://google.com",
        "https://github.com",
        "https://gulu.ac.ug",
        "https://stackoverflow.com",
        "https://wikipedia.org",
        "https://facebook.com",
        "https://amazon.com",
    ]
    
    for url in test_urls:
        pred = model.predict([url])[0]
        proba = model.predict_proba([url])[0]
        spam_prob = proba[list(model.classes_).index('spam')]
        status = "✅" if pred == "not_spam" else "❌"
        print(f"{status} {url:40} -> {pred:8} (spam: {spam_prob:.3f})")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    
    metrics = {
        "accuracy": float(accuracy),
        "url_column": url_col,
        "label_column": label_col,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": N_STRUCTURAL_FEATURES
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Metrics saved to {METRICS_PATH}")
    
    return metrics


if __name__ == "__main__":
    train_and_save()