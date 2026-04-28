import csv
import math
import random
import pickle
import os

DATA_PATH = "C:/machine_learning/combined_phishing_dataset.csv"
MODEL_PATH = "C:/machine_learning/from_scratch_ensemble.pkl"


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(url):
    url = url.lower()

    return [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("/"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        sum(c.isdigit() for c in url),
        1 if "https" in url else 0,
        1 if "login" in url else 0,
        1 if "verify" in url else 0,
        1 if "secure" in url else 0,
        1 if "account" in url else 0,
    ]


def normalize_features(X):
    cols = len(X[0])
    mins = [min(row[i] for row in X) for i in range(cols)]
    maxs = [max(row[i] for row in X) for i in range(cols)]

    new_X = []
    for row in X:
        new_row = []
        for i in range(cols):
            if maxs[i] == mins[i]:
                new_row.append(0)
            else:
                new_row.append((row[i] - mins[i]) / (maxs[i] - mins[i]))
        new_X.append(new_row)

    return new_X, mins, maxs


def apply_normalization(row, mins, maxs):
    new_row = []
    for i in range(len(row)):
        if maxs[i] == mins[i]:
            new_row.append(0)
        else:
            new_row.append((row[i] - mins[i]) / (maxs[i] - mins[i]))
    return new_row


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    X = []
    y = []

    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            url = row.get("URL") or row.get("url") or row.get("domain")
            label = row.get("Label") or row.get("label")

            if not url or label is None:
                continue

            label = str(label).lower().strip()

            if label in ["bad", "spam", "phishing", "1"]:
                y.append(1)
            elif label in ["good", "not_spam", "safe", "0"]:
                y.append(0)
            else:
                continue

            X.append(extract_features(url))

    return X, y


# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
def train_test_split(X, y, test_size=0.2):
    data = list(zip(X, y))
    random.shuffle(data)

    split = int(len(data) * (1 - test_size))

    train = data[:split]
    test = data[split:]

    X_train = [item[0] for item in train]
    y_train = [item[1] for item in train]

    X_test = [item[0] for item in test]
    y_test = [item[1] for item in test]

    return X_train, X_test, y_train, y_test


# -----------------------------
# MODEL 1: PERCEPTRON
# -----------------------------
class Perceptron:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def train(self, X, y):
        self.weights = [0] * len(X[0])

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction

                for i in range(len(self.weights)):
                    self.weights[i] += self.lr * error * xi[i]

                self.bias += self.lr * error

    def predict(self, x):
        total = sum(w * v for w, v in zip(self.weights, x)) + self.bias
        return 1 if total >= 0 else 0

    def predict_proba(self, x):
        total = sum(w * v for w, v in zip(self.weights, x)) + self.bias
        prob = 1 / (1 + math.exp(-total))
        return prob


# -----------------------------
# MODEL 2: LOGISTIC REGRESSION
# -----------------------------
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def sigmoid(self, z):
        if z < -500:
            return 0
        if z > 500:
            return 1
        return 1 / (1 + math.exp(-z))

    def train(self, X, y):
        self.weights = [0] * len(X[0])

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                linear = sum(w * v for w, v in zip(self.weights, xi)) + self.bias
                prediction = self.sigmoid(linear)

                error = prediction - target

                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * error * xi[i]

                self.bias -= self.lr * error

    def predict_proba(self, x):
        linear = sum(w * v for w, v in zip(self.weights, x)) + self.bias
        return self.sigmoid(linear)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


# -----------------------------
# MODEL 3: NAIVE BAYES
# -----------------------------
class NaiveBayesScratch:
    def train(self, X, y):
        self.classes = [0, 1]
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.prior[c] = len(X_c) / len(X)

            self.mean[c] = []
            self.var[c] = []

            for j in range(len(X[0])):
                values = [row[j] for row in X_c]
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                self.mean[c].append(mean)
                self.var[c].append(variance + 1e-6)

    def gaussian_probability(self, x, mean, var):
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    def predict_proba(self, x):
        probs = {}

        for c in self.classes:
            prob = math.log(self.prior[c])

            for i in range(len(x)):
                p = self.gaussian_probability(x[i], self.mean[c][i], self.var[c][i])
                prob += math.log(p + 1e-9)

            probs[c] = prob

        spam_score = probs[1]
        safe_score = probs[0]

        max_score = max(spam_score, safe_score)
        spam_exp = math.exp(spam_score - max_score)
        safe_exp = math.exp(safe_score - max_score)

        return spam_exp / (spam_exp + safe_exp)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


# -----------------------------
# ENSEMBLE MODEL
# -----------------------------
class EnsembleModel:
    def __init__(self):
        self.models = [
            Perceptron(),
            LogisticRegressionScratch(),
            NaiveBayesScratch()
        ]

    def train(self, X, y):
        for model in self.models:
            model.train(X, y)

    def predict(self, x):
        votes = [model.predict(x) for model in self.models]
        return 1 if votes.count(1) >= 2 else 0

    def predict_proba(self, x):
        probabilities = [model.predict_proba(x) for model in self.models]
        return sum(probabilities) / len(probabilities)


# -----------------------------
# TRAIN AND SAVE
# -----------------------------
def train_and_save():
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    print("Total records:", len(X))

    X, mins, maxs = normalize_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = EnsembleModel()

    print("Training ensemble model...")
    model.train(X_train, y_train)

    correct = 0
    for xi, yi in zip(X_test, y_test):
        if model.predict(xi) == yi:
            correct += 1

    accuracy = correct / len(y_test)

    package = {
        "model": model,
        "mins": mins,
        "maxs": maxs,
        "accuracy": accuracy
    }

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(package, file)

    print("Model saved successfully!")
    print("Accuracy:", round(accuracy * 100, 2), "%")


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first.")

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# -----------------------------
# PREDICT FOR FLASK APP
# -----------------------------
def predict_phishing_percentage(url):
    package = load_model()

    model = package["model"]
    mins = package["mins"]
    maxs = package["maxs"]
    accuracy = package["accuracy"]

    features = extract_features(url)
    features = apply_normalization(features, mins, maxs)

    spam_probability = model.predict_proba(features)

    phishing_percent = round(spam_probability * 100, 2)
    legit_percent = round((1 - spam_probability) * 100, 2)

    prediction = "spam" if model.predict(features) == 1 else "not_spam"

    return phishing_percent, legit_percent, prediction, accuracy


if __name__ == "__main__":
    train_and_save()