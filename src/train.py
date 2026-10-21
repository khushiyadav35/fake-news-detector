"""
Model Training & Benchmarking Pipeline
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from src.features import get_tfidf_vectorizer
from src.preprocessing import clean_text

def load_data(data_dir: str):
    """Loads processed train and test sets."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Preprocess text
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    test_df['clean_text'] = test_df['text'].apply(clean_text)
    
    return train_df, test_df

def train_and_benchmark(data_dir: str, models_dir: str):
    """Trains multiple classifiers and selects the top performer."""
    os.makedirs(models_dir, exist_ok=True)
    
    train_df, test_df = load_data(data_dir)
    
    X_train = train_df['clean_text']
    y_train = train_df['label']
    
    X_test = test_df['clean_text']
    y_test = test_df['label']
    
    candidate_classifiers = {
        "Logistic Regression": LogisticRegression(C=1.5, max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=42, max_iter=2000, dual='auto'),
            method='sigmoid',
            cv=3
        ),
        "Passive-Aggressive (SGD)": SGDClassifier(
            loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42
        )
    }
    
    results = {}
    best_f1 = -1.0
    best_pipeline = None
    best_name = ""
    
    print("=" * 70)
    print(" BENCHMARKING NLP CLASSIFIERS FOR FAKE NEWS DETECTION")
    print("=" * 70)
    
    for name, clf in candidate_classifiers.items():
        vectorizer = get_tfidf_vectorizer()
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('clf', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }
        
        print(f"[{name}]")
        print(f"  Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f} | F1-Score:  {f1:.4f} | ROC-AUC: {auc:.4f}")
        print("-" * 70)
        
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_name = name

    print(f"\nTop Selected Model: {best_name} (F1-Score: {best_f1:.4f})")
    
    # Save best pipeline
    model_path = os.path.join(models_dir, "model_pipeline.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to: {model_path}")
    
    # Save benchmark metrics
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "benchmark_results": results,
            "classes": ["Reliable", "Suspicious"]
        }, f, indent=4)
    print(f"Saved benchmark metrics to: {metrics_path}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    
    train_and_benchmark(data_dir, models_dir)

if __name__ == "__main__":
    main()
