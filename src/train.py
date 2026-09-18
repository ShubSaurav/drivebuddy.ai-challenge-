"""
Model training, cross-validation, leakage experimentation, and evaluation pipeline
for DrivebuddyAI Pav Bhaji Challenge.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from src.data_loader import extract_dataset_if_needed, build_dataset, get_data_quality_report
from src.preprocessing import preprocess_text
from src.features import build_word_tfidf

logger = logging.getLogger(__name__)


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a stratified train/test split (80/20)."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )
    logger.info(f"Train split: {len(train_df)} samples ({dict(train_df['label'].value_counts())})")
    logger.info(f"Test split:  {len(test_df)} samples ({dict(test_df['label'].value_counts())})")
    return train_df.copy(), test_df.copy()


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """Define candidate classical ML models."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=random_state, max_iter=2000),
            cv=3
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Complement Naive Bayes": ComplementNB(alpha=0.5)
    }


def evaluate_model_cv(
    model: Any,
    X_train: List[str],
    y_train: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """Run Stratified 5-Fold Cross-Validation on training data."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    pipeline = Pipeline([
        ("tfidf", build_word_tfidf(ngram_range=(1, 2), min_df=2, max_features=3000)),
        ("classifier", model)
    ])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_precision_mean": float(np.mean(scores["test_precision"])),
        "cv_precision_std": float(np.std(scores["test_precision"])),
        "cv_recall_mean": float(np.mean(scores["test_recall"])),
        "cv_recall_std": float(np.std(scores["test_recall"])),
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"]))
    }


def train_and_evaluate_test(
    model: Any,
    X_train: List[str],
    y_train: np.ndarray,
    X_test: List[str],
    y_test: np.ndarray
) -> Tuple[Pipeline, Dict[str, Any], np.ndarray, np.ndarray]:
    """Train pipeline on full training set and evaluate strictly on test set."""
    pipeline = Pipeline([
        ("tfidf", build_word_tfidf(ngram_range=(1, 2), min_df=2, max_features=3000)),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    # Predict test
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
        d = pipeline.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-d))
    else:
        y_prob = y_pred.astype(float)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=["Not Pav Bhaji", "Pav Bhaji"], output_dict=True)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "classification_report": report
    }

    return pipeline, metrics, y_pred, y_prob


def run_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """Run both Experiment 1 (Raw Text) and Experiment 2 (Leakage-Controlled Text)."""
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    results = {"experiment_1_raw": {}, "experiment_2_leakage_controlled": {}}

    experiments = [
        ("experiment_1_raw", False),
        ("experiment_2_leakage_controlled", True)
    ]

    for exp_name, remove_leakage in experiments:
        logger.info(f"\n{'='*20} Running {exp_name} {'='*20}")
        # Preprocess text according to experiment
        X_train = [preprocess_text(t, remove_leakage=remove_leakage) for t in train_df["combined_text"]]
        y_train = train_df["label"].values.astype(int)

        X_test = [preprocess_text(t, remove_leakage=remove_leakage) for t in test_df["combined_text"]]
        y_test = test_df["label"].values.astype(int)

        models = get_models()
        exp_results = {}

        for m_name, model in models.items():
            logger.info(f"Cross-validating and training {m_name}...")
            cv_metrics = evaluate_model_cv(model, X_train, y_train)
            pipeline, test_metrics, y_pred, y_prob = train_and_evaluate_test(
                model, X_train, y_train, X_test, y_test
            )

            exp_results[m_name] = {
                "cv": cv_metrics,
                "test": test_metrics
            }

        results[exp_name] = exp_results

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics", "model_comparison.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved model comparison metrics to {metrics_path}")

    return results


def train_champion_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    remove_leakage: bool = True,
    models_dir: str = "models",
    output_dir: str = "outputs"
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train and persist the champion model pipeline.
    Uses Experiment 2 (Leakage-Controlled) by default for authentic semantic generalization,
    while also saving full test predictions.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    X_train = [preprocess_text(t, remove_leakage=remove_leakage) for t in train_df["combined_text"]]
    y_train = train_df["label"].values.astype(int)

    X_test = [preprocess_text(t, remove_leakage=remove_leakage) for t in test_df["combined_text"]]
    y_test = test_df["label"].values.astype(int)

    # Champion model: Logistic Regression (balanced, interpretable, robust)
    champion = LogisticRegression(C=1.5, class_weight="balanced", max_iter=1000, random_state=42)
    pipeline = Pipeline([
        ("tfidf", build_word_tfidf(ngram_range=(1, 2), min_df=2, max_features=3000)),
        ("classifier", champion)
    ])

    pipeline.fit(X_train, y_train)

    # Test evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Save pipeline
    model_path = os.path.join(models_dir, "pavbhaji_classifier.joblib")
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved champion pipeline to {model_path}")

    # Generate predictions DataFrame
    pred_df = test_df.copy()
    pred_df["preprocessed_text"] = X_test
    pred_df["predicted_label"] = y_pred
    pred_df["pavbhaji_probability"] = np.round(y_prob, 4)
    pred_df["confidence_percentage"] = np.round(np.where(y_pred == 1, y_prob, 1 - y_prob) * 100, 2)
    pred_df["is_correct"] = pred_df["label"] == pred_df["predicted_label"]

    predictions_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(predictions_path, index=False)
    logger.info(f"Saved test predictions to {predictions_path}")

    return pipeline, pred_df


def extract_feature_importance(pipeline: Pipeline, top_n: int = 20) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Extract top positive and negative vocabulary features from champion Logistic Regression."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    top_pos = [(feature_names[i], float(coefs[i])) for i in top_pos_idx]
    top_neg = [(feature_names[i], float(coefs[i])) for i in top_neg_idx]

    return top_pos, top_neg
