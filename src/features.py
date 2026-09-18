"""
Feature engineering and extraction module for DrivebuddyAI Pav Bhaji Challenge.
Builds composite Word-level (1, 2 n-grams) + Character-level (3, 5 n-grams) TF-IDF pipelines,
and domain-specific food semantic indicators + engagement metadata features.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

# Domain-specific food lexicons
PAV_BHAJI_LEXICON = [
    "butter", "buttery", "lemon", "onion", "kanda", "limbu", "garam",
    "bhukkad", "khau", "galli", "tava", "tawa", "amul", "cheesepavbhaji",
    "spicy", "masala", "mashed", "streetstyle", "veg", "vegetarian"
]

OTHER_FOODS_LEXICON = [
    "panipuri", "golgappa", "puchka", "vadapav", "dosa", "idli", "bhelpuri",
    "sevpuri", "dahipuri", "samosa", "kachori", "tikka", "kebab", "chicken",
    "biryani", "burger", "pizza", "sandwich", "pasta", "fries", "kolkata",
    "delhi", "mutton", "egg", "nonveg", "tandoori"
]


class EnhancedPipelineWrapper(BaseEstimator, TransformerMixin):
    """Production Wrapper integrating Word+Char TF-IDF, metadata scaling, and probability inference."""

    def __init__(self, classifier=None, threshold: float = 0.50):
        self.classifier = classifier
        self.threshold = threshold
        self.union_tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=3, max_features=3000, sublinear_tf=True))
        ])
        self.scaler = MinMaxScaler()

    def _extract_meta(self, df_in):
        feats = []
        for _, r in df_in.iterrows():
            t = str(r.get("clean_text", "") or r.get("combined_text", "") or "").lower()
            words = t.split()
            word_count = len(words)
            char_count = len(t)
            avg_w = char_count / (word_count + 1)

            pb_hits = sum(1 for kw in PAV_BHAJI_LEXICON if kw in t)
            other_hits = sum(1 for kw in OTHER_FOODS_LEXICON if kw in t)
            pb_ratio = (pb_hits + 1) / (other_hits + 1)

            likes = np.log1p(float(r.get("likes_count", 0) or 0))
            comments = np.log1p(float(r.get("comments_count", 0) or 0))

            feats.append([word_count, char_count, avg_w, pb_hits, other_hits, pb_ratio, likes, comments])
        return np.array(feats, dtype=float)

    def fit(self, df_train, y_train):
        tfidf_mat = self.union_tfidf.fit_transform(df_train["clean_text"])
        meta_mat = self.scaler.fit_transform(self._extract_meta(df_train))
        X_all = hstack([tfidf_mat, meta_mat]).tocsr()
        self.classifier.fit(X_all, y_train)
        return self

    def predict_proba(self, df_test):
        tfidf_mat = self.union_tfidf.transform(df_test["clean_text"])
        meta_mat = self.scaler.transform(self._extract_meta(df_test))
        X_all = hstack([tfidf_mat, meta_mat]).tocsr()
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X_all)
        else:
            d = self.classifier.decision_function(X_all)
            p1 = 1 / (1 + np.exp(-d))
            return np.vstack([1 - p1, p1]).T

    def predict(self, df_test):
        probs = self.predict_proba(df_test)[:, 1]
        return (probs >= self.threshold).astype(int)
