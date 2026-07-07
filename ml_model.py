"""
ml_model.py - Machine Learning Prediction Module
Trains a Random Forest classifier on a synthetic nutrition dataset.
Predicts health risk level: Low / Medium / High based on blood parameters.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH  = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

FEATURES = [
    "hemoglobin", "glucose", "cholesterol",
    "vitamin_d", "iron", "calcium",
    "vitamin_b12", "triglycerides"
]


# ── Synthetic Dataset Generator ──────────────────────────────────────────────

def generate_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generates a synthetic labeled dataset for training.
    Risk labels are derived from parameter deviations from normal ranges.

    Algorithm:
      1. Sample each parameter from a realistic distribution.
      2. Count how many parameters are out of normal range.
      3. Assign risk label: 0 out = Low, 1-2 = Medium, 3+ = High.
    """
    np.random.seed(42)

    # Sampling distributions (mean, std) tuned to produce realistic values
    distributions = {
        "hemoglobin":    (13.5, 2.5),
        "glucose":       (95,   30),
        "cholesterol":   (185,  40),
        "vitamin_d":     (28,   15),
        "iron":          (100,  35),
        "calcium":       (9.5,  1.0),
        "vitamin_b12":   (450,  250),
        "triglycerides": (140,  60),
    }

    # Normal ranges (min, max)
    normal_ranges = {
        "hemoglobin":    (12.0, 17.5),
        "glucose":       (70,   100),
        "cholesterol":   (0,    200),
        "vitamin_d":     (20,   50),
        "iron":          (60,   170),
        "calcium":       (8.5,  10.5),
        "vitamin_b12":   (200,  900),
        "triglycerides": (0,    150),
    }

    data = {}
    for param, (mu, sigma) in distributions.items():
        lo, hi = normal_ranges[param]
        # Clip to physiologically plausible range
        values = np.clip(np.random.normal(mu, sigma, n_samples), lo * 0.3, hi * 2.0)
        data[param] = values

    df = pd.DataFrame(data)

    # ── Label Generation ──────────────────────────────────────────────────────
    def assign_risk(row):
        out_of_range = 0
        for param, (lo, hi) in normal_ranges.items():
            val = row[param]
            if val < lo or val > hi:
                out_of_range += 1
        if out_of_range == 0:
            return "Low"
        elif out_of_range <= 2:
            return "Medium"
        return "High"

    df["risk_level"] = df.apply(assign_risk, axis=1)
    return df


# ── Model Training ───────────────────────────────────────────────────────────

def train_model(save: bool = True):
    """
    Trains a Random Forest classifier on the synthetic dataset.
    Saves model, scaler, and label encoder to disk.
    Returns: (model, scaler, encoder, accuracy, report)
    """
    os.makedirs("models", exist_ok=True)

    print("[ML] Generating training data...")
    df = generate_training_data(n_samples=3000)

    X = df[FEATURES].values
    y = df["risk_level"].values

    # Encode labels
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)  # High=0, Low=1, Medium=2 (alphabetical)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # ── Model: Random Forest ──────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=encoder.classes_)

    print(f"[ML] Training complete. Accuracy: {accuracy:.2%}")
    print(report)

    if save:
        joblib.dump(model,   MODEL_PATH)
        joblib.dump(scaler,  SCALER_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        print(f"[ML] Model saved to {MODEL_PATH}")

    return model, scaler, encoder, accuracy, report


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    """
    Loads the trained model from disk.
    If model files don't exist, trains a new one automatically.
    """
    if not os.path.exists(MODEL_PATH):
        print("[ML] No saved model found — training now...")
        train_model(save=True)

    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


# ── Prediction ───────────────────────────────────────────────────────────────

def predict_risk(values: dict) -> dict:
    """
    Predicts health risk level from a dict of blood parameter values.
    Missing parameters are filled with the population mean.

    Returns:
      {
        "risk_level": "Low" | "Medium" | "High",
        "probabilities": {"Low": 0.7, "Medium": 0.2, "High": 0.1},
        "confidence": 0.7
      }
    """
    # Population means for fallback imputation
    MEANS = {
        "hemoglobin": 13.5, "glucose": 95, "cholesterol": 185,
        "vitamin_d": 28, "iron": 100, "calcium": 9.5,
        "vitamin_b12": 450, "triglycerides": 140,
    }

    model, scaler, encoder = load_model()

    feature_vector = [[values.get(f, MEANS[f]) for f in FEATURES]]
    X_scaled = scaler.transform(feature_vector)

    pred_enc   = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0]
    risk_label = encoder.inverse_transform([pred_enc])[0]

    prob_dict = {cls: round(float(p), 3) for cls, p in zip(encoder.classes_, proba)}
    confidence = round(float(max(proba)), 3)

    return {
        "risk_level":    risk_label,
        "probabilities": prob_dict,
        "confidence":    confidence,
    }


# ── Feature Importance ───────────────────────────────────────────────────────

def get_feature_importance() -> pd.DataFrame:
    """Returns a DataFrame of feature importances for the trained model."""
    model, _, _ = load_model()
    importance = model.feature_importances_
    df = pd.DataFrame({"Feature": FEATURES, "Importance": importance})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    df["Feature"] = df["Feature"].str.replace("_", " ").str.title()
    return df


# ── Entry point: run to train ─────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
