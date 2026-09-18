#!/usr/bin/env python3
"""
================================================================================
DrivebuddyAI — Pav Bhaji Text Classification Challenge (High-Performance Edition)
================================================================================
Author: Shubham Saurav
Role: Senior Machine Learning Engineer & Data Scientist
Objective: Binary text classification: Pav Bhaji (1) vs Not Pav Bhaji (0)
Constraint: Strict TEXT-ONLY. Zero Computer Vision. No image pixels.
Features: Composite Word+Char TF-IDF + Food Lexicon Indicators + Calibrated Ensembles
================================================================================
"""

import os
import re
import sys
import json
import zipfile
import logging
import unicodedata
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from scipy.sparse import hstack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PavBhajiClassifier")


# ==============================================================================
# 1. DATA LOADING & PREPARATION
# ==============================================================================

def locate_and_extract_dataset(zip_path: str = "dataset.zip", extract_to: str = "data") -> str:
    target_dir = os.path.join(extract_to, "dataset")
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "pavbhaji.json")):
        logger.info(f"Dataset already extracted at: {target_dir}")
        return target_dir

    if not os.path.exists(zip_path):
        alt_zip = os.path.join(os.path.dirname(__file__), "dataset.zip")
        if os.path.exists(alt_zip):
            zip_path = alt_zip
        else:
            raise FileNotFoundError(f"dataset.zip not found at {zip_path}")

    logger.info(f"Extracting {zip_path} -> {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info("Extraction complete.")
    return target_dir


def load_and_construct_dataset(dataset_dir: str = "data/dataset") -> pd.DataFrame:
    json_path = os.path.join(dataset_dir, "pavbhaji.json")
    images_dir = os.path.join(dataset_dir, "images")
    dir_0 = os.path.join(images_dir, "0")
    dir_1 = os.path.join(images_dir, "1")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_posts = raw_data.get("data", list(raw_data.values())) if isinstance(raw_data, dict) else raw_data

    images_0 = {f for f in os.listdir(dir_0) if not f.startswith((".", "_"))} if os.path.exists(dir_0) else set()
    images_1 = {f for f in os.listdir(dir_1) if not f.startswith((".", "_"))} if os.path.exists(dir_1) else set()

    rows = []
    for item in raw_posts:
        display_url = str(item.get("display_url", "") or "")
        filename = display_url.split("?")[0].split("/")[-1] if display_url else ""

        label = None
        image_path = None
        if filename in images_0:
            label = 0
            image_path = os.path.join("images", "0", filename)
        elif filename in images_1:
            label = 1
            image_path = os.path.join("images", "1", filename)

        if label is None:
            continue

        caption = ""
        edge_caption = item.get("edge_media_to_caption")
        if isinstance(edge_caption, dict):
            edges = edge_caption.get("edges", [])
            if edges and isinstance(edges[0], dict):
                caption = str(edges[0].get("node", {}).get("text", "") or "")

        raw_tags = item.get("tags") or []
        if isinstance(raw_tags, list):
            tags_list = [str(t).strip() for t in raw_tags if t]
        else:
            tags_list = [t.strip() for t in str(raw_tags).split() if t.strip()]

        tags_str = " ".join(f"#{t}" if not t.startswith("#") else t for t in tags_list)
        combined_text = f"{caption} {tags_str}".strip()

        likes = int(item.get("edge_liked_by", {}).get("count", 0) or 0) if isinstance(item.get("edge_liked_by"), dict) else 0
        comments = int(item.get("edge_media_to_comment", {}).get("count", 0) or 0) if isinstance(item.get("edge_media_to_comment"), dict) else 0

        rows.append({
            "post_id": str(item.get("id", "")),
            "shortcode": str(item.get("shortcode", "")),
            "image_filename": filename,
            "image_path": image_path,
            "label": label,
            "description": caption,
            "hashtags": tags_str,
            "tags_list": tags_list,
            "combined_text": combined_text,
            "likes_count": likes,
            "comments_count": comments,
            "taken_at_timestamp": item.get("taken_at_timestamp"),
            "is_video": bool(item.get("is_video", False))
        })

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} labeled posts. Class distribution: {dict(df['label'].value_counts())}")
    return df


# ==============================================================================
# 2. TEXT PREPROCESSING & LEAKAGE SANITIZATION
# ==============================================================================

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HTML_TAG_PATTERN = re.compile(r'<.*?>')
EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u200d]|[\ufe0f]',
    flags=re.UNICODE
)

TARGET_LEAKAGE_PATTERNS = [
    re.compile(r'#?pav[\s\-_]*bhaj[ij]+[a-z]*', re.IGNORECASE),
    re.compile(r'#?bhaj[ij]+[\s\-_]*pav[a-z]*', re.IGNORECASE),
    re.compile(r'\bpav[\s\-_]*bhaj[ij]+\b', re.IGNORECASE),
    re.compile(r'\bpav\b', re.IGNORECASE),
    re.compile(r'\bbhaj[ij]+\b', re.IGNORECASE),
]

def preprocess_text(text: str, remove_leakage: bool = False) -> str:
    if not text or not isinstance(text, str):
        return ""

    cleaned = unicodedata.normalize("NFKD", text).lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    if remove_leakage:
        for p in TARGET_LEAKAGE_PATTERNS:
            cleaned = p.sub(" ", cleaned)

    cleaned = re.sub(r'#(\w+)', r'\1', cleaned)
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)

    tokens = [t for t in cleaned.split() if len(t) > 1]
    return " ".join(tokens)


# ==============================================================================
# 3. ENHANCED FEATURE ENGINEERING (WORD+CHAR TF-IDF + FOOD LEXICONS)
# ==============================================================================

from src.features import EnhancedPipelineWrapper, PAV_BHAJI_LEXICON, OTHER_FOODS_LEXICON


# ==============================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ==============================================================================

def generate_eda_visualizations(df: pd.DataFrame, output_fig_dir: str = "outputs/figures"):
    os.makedirs(output_fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["label"].value_counts().sort_index()
    labels = ["Not Pav Bhaji (0)", "Pav Bhaji (1)"]
    colors = ["#4A90E2", "#E94E77"]
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h} ({h/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Ground-Truth Class Distribution (N=452)", fontsize=14, pad=15, fontweight="bold")
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_ylim(0, max(counts.values) + 40)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "class_distribution.png"), dpi=300)
    plt.close(fig)

    # 2. Text Length Distribution
    df["char_count"] = df["combined_text"].apply(len)
    df["word_count"] = df["combined_text"].apply(lambda x: len(x.split()))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="char_count", hue="label", kde=True, bins=30,
                 palette={0: "#4A90E2", 1: "#E94E77"}, ax=ax, alpha=0.5)
    ax.set_title("Text Character Length Distribution by Class", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Character Count", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(title="Class", labels=["Pav Bhaji (1)", "Not Pav Bhaji (0)"])
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "text_length_distribution.png"), dpi=300)
    plt.close(fig)

    # 3. Word Count Boxplot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="label", y="word_count", hue="label", palette=["#4A90E2", "#E94E77"], ax=ax, width=0.4, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Pav Bhaji (0)", "Pav Bhaji (1)"])
    ax.set_title("Word Count Distribution by Class", fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Word Count", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "word_count_distribution.png"), dpi=300)
    plt.close(fig)

    # 4. Top Words (Log-Odds Ratio)
    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    df0, df1 = Counter(), Counter()

    for _, r in df.iterrows():
        words = set(re.findall(r'[a-zA-Z]{3,}', r["combined_text"].lower()))
        if r["label"] == 0:
            df0.update(words)
        else:
            df1.update(words)

    log_odds = {}
    for w in set(df0.keys()).union(df1.keys()):
        if (df0[w] + df1[w]) >= 5:
            p1 = (df1[w] + 1) / (n1 + 2)
            p0 = (df0[w] + 1) / (n0 + 2)
            log_odds[w] = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))

    sorted_lo = sorted(log_odds.items(), key=lambda x: x[1])
    top_neg = sorted_lo[:10]
    top_pos = sorted_lo[-10:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pos_df = pd.DataFrame(top_pos, columns=["Word", "LogOdds"])
    sns.barplot(data=pos_df, y="Word", x="LogOdds", ax=axes[0], color="#E94E77")
    axes[0].set_title("Top Words Indicative of Pav Bhaji (Class 1)", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Log-Odds Ratio (Higher -> Pav Bhaji)")

    neg_df = pd.DataFrame(top_neg, columns=["Word", "LogOdds"])
    neg_df["AbsLogOdds"] = neg_df["LogOdds"].abs()
    sns.barplot(data=neg_df, y="Word", x="AbsLogOdds", ax=axes[1], color="#4A90E2")
    axes[1].set_title("Top Words Indicative of Not Pav Bhaji (Class 0)", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Absolute Log-Odds Ratio (Higher -> Not Pav Bhaji)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "top_words_by_class.png"), dpi=300)
    plt.close(fig)
    logger.info("EDA visualizations saved successfully.")


# ==============================================================================
# 5. HIGH-PERFORMANCE BENCHMARKING (EXP 1 VS EXP 2)
# ==============================================================================

def run_model_benchmarks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    experiments = {
        "Experiment 1 (Raw Text)": False,
        "Experiment 2 (Leakage-Controlled)": True
    }

    results = {}

    for exp_title, remove_leakage in experiments.items():
        logger.info(f"\n{'='*25} {exp_title} {'='*25}")
        train_df["clean_text"] = [preprocess_text(t, remove_leakage=remove_leakage) for t in train_df["combined_text"]]
        test_df["clean_text"] = [preprocess_text(t, remove_leakage=remove_leakage) for t in test_df["combined_text"]]

        y_train = train_df["label"].values.astype(int)
        y_test = test_df["label"].values.astype(int)

        # Feature extractor
        union_tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=3, max_features=3000, sublinear_tf=True))
        ])
        X_train_tfidf = union_tfidf.fit_transform(train_df["clean_text"])
        X_test_tfidf = union_tfidf.transform(test_df["clean_text"])

        # Meta features
        scaler = MinMaxScaler()
        wrapper_dummy = EnhancedPipelineWrapper(None)
        X_train_meta = scaler.fit_transform(wrapper_dummy._extract_meta(train_df))
        X_test_meta = scaler.transform(wrapper_dummy._extract_meta(test_df))

        X_train_all = hstack([X_train_tfidf, X_train_meta]).tocsr()
        X_test_all = hstack([X_test_tfidf, X_test_meta]).tocsr()

        models = {
            "Logistic Regression (Champion)": LogisticRegression(C=1.2, class_weight="balanced", max_iter=1000, random_state=42),
            "Linear SVM": CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000), cv=3),
            "Complement Naive Bayes": ComplementNB(alpha=0.3),
            "Voting Ensemble (LR+SVM+CNB)": VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(C=1.2, class_weight="balanced", max_iter=1000, random_state=42)),
                    ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000), cv=3)),
                    ("cnb", ComplementNB(alpha=0.3))
                ],
                voting="soft"
            )
        }

        exp_data = {}
        for m_name, model in models.items():
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_validate(
                model, X_train_all, y_train, cv=cv,
                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                n_jobs=-1
            )

            model.fit(X_train_all, y_train)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_all)[:, 1]
            else:
                y_prob = model.decision_function(X_test_all)

            y_pred = (y_prob >= 0.50).astype(int)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred).tolist()

            logger.info(f"[{m_name}] Test Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            exp_data[m_name] = {
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
                "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
                "test_accuracy": float(acc),
                "test_precision": float(prec),
                "test_recall": float(rec),
                "test_f1": float(f1),
                "test_f1_macro": float(f1_macro),
                "test_roc_auc": float(roc_auc),
                "test_pr_auc": float(pr_auc),
                "confusion_matrix": cm,
                "y_pred": y_pred.tolist(),
                "y_prob": y_prob.tolist()
            }

        results[exp_title] = exp_data

    # Save to JSON
    with open(os.path.join(output_dir, "metrics", "model_comparison.json"), "w", encoding="utf-8") as f:
        clean_save = {}
        for exp_k, exp_v in results.items():
            clean_save[exp_k] = {}
            for m_k, m_v in exp_v.items():
                clean_save[exp_k][m_k] = {k: v for k, v in m_v.items() if k not in ["y_pred", "y_prob"]}
        json.dump(clean_save, f, indent=2)

    return results


# ==============================================================================
# 6. PLOTTING PERFORMANCE METRICS & INTERPRETABILITY
# ==============================================================================

def plot_evaluation_charts(
    benchmark_results: Dict[str, Any],
    y_test: np.ndarray,
    output_fig_dir: str = "outputs/figures"
):
    exp2_results = benchmark_results["Experiment 2 (Leakage-Controlled)"]
    model_names = list(exp2_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, m_name in enumerate(model_names):
        cm = np.array(exp2_results[m_name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["Not Pav Bhaji", "Pav Bhaji"],
                    yticklabels=["Not Pav Bhaji", "Pav Bhaji"],
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        axes[idx].set_title(f"{m_name}\nAcc: {exp2_results[m_name]['test_accuracy']:.2f} | F1: {exp2_results[m_name]['test_f1']:.2f}",
                            fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("True Label")
        axes[idx].set_xlabel("Predicted Label")

    plt.suptitle("Confusion Matrices — Experiment 2 (Leakage-Controlled)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "confusion_matrices.png"), dpi=300)
    plt.close(fig)

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]

    for idx, m_name in enumerate(model_names):
        y_prob = np.array(exp2_results[m_name]["y_prob"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_score = exp2_results[m_name]["test_roc_auc"]
        pr_score = exp2_results[m_name]["test_pr_auc"]

        ax_roc.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_score:.3f})", color=colors[idx], linewidth=2.2)
        ax_pr.plot(recall, precision, label=f"{m_name} (AP = {pr_score:.3f})", color=colors[idx], linewidth=2.2)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall Curves", fontweight="bold", fontsize=12)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "roc_pr_curves.png"), dpi=300)
    plt.close(fig)
    logger.info("Evaluation charts saved successfully.")


def plot_feature_importance(wrapper: EnhancedPipelineWrapper, output_fig_dir: str = "outputs/figures"):
    clf = wrapper.classifier
    if not hasattr(clf, "coef_"):
        return

    word_names = wrapper.union_tfidf.transformer_list[0][1].get_feature_names_out()
    char_names = wrapper.union_tfidf.transformer_list[1][1].get_feature_names_out()
    meta_names = np.array(["word_count", "char_count", "avg_w_len", "pb_hits", "other_hits", "pb_ratio", "likes", "comments"])

    all_names = np.concatenate([word_names, char_names, meta_names])
    coefs = clf.coef_[0]

    top_pos_idx = np.argsort(coefs)[-15:][::-1]
    top_neg_idx = np.argsort(coefs)[:15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    pos_df = pd.DataFrame({"Feature": all_names[top_pos_idx], "Weight": coefs[top_pos_idx]})
    sns.barplot(data=pos_df, y="Feature", x="Weight", ax=ax1, color="#E94E77")
    ax1.set_title("Top Model Indicators Toward Pav Bhaji (Class 1)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Model Coefficient (Log-Odds Impact)")

    neg_df = pd.DataFrame({"Feature": all_names[top_neg_idx], "Weight": np.abs(coefs[top_neg_idx])})
    sns.barplot(data=neg_df, y="Feature", x="Weight", ax=ax2, color="#4A90E2")
    ax2.set_title("Top Model Indicators Toward Not Pav Bhaji (Class 0)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Absolute Model Coefficient")

    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "feature_importance.png"), dpi=300)
    plt.close(fig)
    logger.info("Feature importance plot saved successfully.")


# ==============================================================================
# 7. FINAL MODEL TRAINING & SERIALIZATION
# ==============================================================================

def train_and_save_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str = "models",
    output_dir: str = "outputs"
) -> EnhancedPipelineWrapper:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_df["clean_text"] = [preprocess_text(t, remove_leakage=True) for t in train_df["combined_text"]]
    test_df["clean_text"] = [preprocess_text(t, remove_leakage=True) for t in test_df["combined_text"]]

    y_train = train_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    champion = LogisticRegression(C=1.2, class_weight="balanced", max_iter=1000, random_state=42)
    wrapper = EnhancedPipelineWrapper(champion, threshold=0.50)
    wrapper.fit(train_df, y_train)

    # Save wrapper pipeline
    model_path = os.path.join(models_dir, "pavbhaji_classifier.joblib")
    joblib.dump(wrapper, model_path)
    logger.info(f"Enhanced Champion pipeline successfully saved to {model_path}")

    # Generate predictions
    probs = wrapper.predict_proba(test_df)[:, 1]
    y_pred = wrapper.predict(test_df)

    pred_df = test_df.copy()
    pred_df["predicted_label"] = y_pred
    pred_df["probability_pavbhaji"] = np.round(probs, 4)
    pred_df["confidence_pct"] = np.round(np.where(y_pred == 1, probs, 1 - probs) * 100, 2)
    pred_df["prediction_status"] = np.where(pred_df["label"] == pred_df["predicted_label"], "CORRECT", "MISCLASSIFIED")

    pred_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Test predictions successfully written to {pred_path}")

    plot_feature_importance(wrapper, os.path.join(output_dir, "figures"))
    return wrapper


# ==============================================================================
# 8. INFERENCE INTERFACE
# ==============================================================================

def predict_post(
    description: str = "",
    hashtags: str = "",
    comments: str = "",
    wrapper: Optional[EnhancedPipelineWrapper] = None
) -> Dict[str, Any]:
    if wrapper is None:
        model_path = "models/pavbhaji_classifier.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        wrapper = joblib.load(model_path)

    combined = f"{description} {hashtags} {comments}".strip()
    cleaned = preprocess_text(combined, remove_leakage=True)

    dummy_df = pd.DataFrame([{
        "clean_text": cleaned,
        "combined_text": combined,
        "likes_count": 0,
        "comments_count": 0
    }])

    prob_pb = float(wrapper.predict_proba(dummy_df)[0][1])

    if prob_pb >= wrapper.threshold:
        return {
            "prediction": "Pav Bhaji",
            "class": 1,
            "confidence": round(prob_pb * 100, 2),
            "probability": round(prob_pb, 4),
            "cleaned_text": cleaned
        }
    else:
        return {
            "prediction": "Not Pav Bhaji",
            "class": 0,
            "confidence": round((1 - prob_pb) * 100, 2),
            "probability": round(prob_pb, 4),
            "cleaned_text": cleaned
        }


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def main():
    logger.info("Starting Boosted DrivebuddyAI Pav Bhaji Text Classification Pipeline...")

    dataset_dir = locate_and_extract_dataset()
    df = load_and_construct_dataset(dataset_dir)
    generate_eda_visualizations(df, "outputs/figures")

    train_df, test_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
    logger.info(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    benchmark_results = run_model_benchmarks(train_df, test_df, "outputs")

    y_test = test_df["label"].values.astype(int)
    plot_evaluation_charts(benchmark_results, y_test, "outputs/figures")

    champion_wrapper = train_and_save_final_model(train_df, test_df, "models", "outputs")

    logger.info("\n--- Sample Predictions ---")
    samples = [
        ("Piping hot butter pav bhaji with onions and lemon on a rainy day in Mumbai khau galli", "#mumbaistreetfood #butterpavbhaji #foodporn"),
        ("Crispy pani puri with spicy mint water and sweet tamarind chutney at Delhi street stall", "#golgappa #panipuri #chaatlover #streetfood"),
        ("Delicious cheese loaded grilled sandwich served with green chutney", "#sandwich #cheesesandwich #bombaysandwich")
    ]

    for desc, tags in samples:
        res = predict_post(desc, tags, wrapper=champion_wrapper)
        logger.info(f"Caption: '{desc[:60]}...' -> Prediction: {res['prediction']} (Confidence: {res['confidence']}%)")

    logger.info("\nBoosted DrivebuddyAI Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
