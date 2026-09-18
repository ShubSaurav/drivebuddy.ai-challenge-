"""
Feature engineering and extraction module for DrivebuddyAI Pav Bhaji Challenge.
Builds composite Word-level (1, 2 n-grams) + Character-level (3, 5 n-grams) TF-IDF pipelines,
expanded domain culinary lexicons, engagement metadata, and post-calibration logic.
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

# Expanded domain-specific food lexicons
PAV_BHAJI_LEXICON = [
    "butter", "buttery", "lemon", "onion", "kanda", "limbu", "garam",
    "bhukkad", "khau", "galli", "khaugalli", "tava", "tawa", "amul", "amulbutter",
    "cheesepavbhaji", "spicy", "masala", "mashed", "streetstyle", "veg", "vegetarian",
    "makhan", "makhani", "sardar", "cannon", "maruti", "juhu", "chowpatty",
    "streetfoodindia", "coriander", "capsicum", "peas", "greenpeas", "tomato",
    "tomatoes", "potatoes", "aloo", "matar", "bread", "buns", "gravy"
]

OTHER_FOODS_LEXICON = [
    "panipuri", "golgappa", "puchka", "vadapav", "dosa", "idli", "bhelpuri",
    "sevpuri", "dahipuri", "samosa", "kachori", "tikka", "kebab", "chicken",
    "biryani", "burger", "pizza", "sandwich", "pasta", "fries", "kolkata",
    "delhi", "mutton", "egg", "nonveg", "tandoori", "momos", "momo", "noodles",
    "manchurian", "chinese", "shawarma", "roll", "kathi", "frankie", "misal",
    "misalpav", "dalbaati", "chole", "bhature", "kulcha", "naan", "paneer",
    "butterchicken", "curry", "waffle", "pancake", "sweet", "mithai", "icecream",
    "kulfi", "falooda", "gulabjamun", "jalebi", "rasgulla"
]

# Hard negative competitors that never appear as Pav Bhaji
HARD_COMPETITORS = {
    'momos', 'momo', 'chicken', 'tikka', 'kebab', 'tandoori', 'shawarma', 'biryani',
    'fish', 'mutton', 'egg', 'noodles', 'manchurian', 'chinese', 'dosa', 'idli', 'vada',
    'sambhar', 'uttapam', 'panipuri', 'golgappa', 'puchka', 'vadapav', 'misalpav',
    'bhelpuri', 'sevpuri', 'dahipuri', 'kachori', 'samosa', 'burger', 'pizza', 'pasta',
    'waffle', 'pancake', 'icecream', 'kulfi', 'falooda', 'rasgulla', 'gulabjamun', 'jalebi'
}

# Strong positive qualifiers that strongly indicate authentic Pav Bhaji
STRONG_PB_SIGNALS = {
    'butter', 'buttery', 'amul', 'amulbutter', 'tava', 'tawa', 'khaugalli', 'khau galli',
    'garam', 'lemon', 'onion', 'onions', 'bhaji'
}


class EnhancedPipelineWrapper(BaseEstimator, TransformerMixin):
    """Production Wrapper integrating Word+Char TF-IDF, metadata scaling, culinary calibration, and probability inference."""

    def __init__(self, classifier=None, threshold: float = 0.50, penalty: float = 0.10, boost: float = 0.10):
        self.classifier = classifier
        self.threshold = threshold
        self.penalty = penalty
        self.boost = boost
        self.union_tfidf = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=3500, sublinear_tf=True)),
            ("char", TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=3, max_df=0.9, max_features=5000, sublinear_tf=True))
        ])
        self.scaler = MinMaxScaler()

    def _extract_meta(self, df_in):
        feats = []
        for _, r in df_in.iterrows():
            t = str(r.get("clean_text", "") or r.get("combined_text", "") or "").lower()
            words = t.split()
            word_count = max(len(words), 1)
            char_count = len(t)
            avg_w = char_count / word_count

            pb_hits = sum(1 for kw in PAV_BHAJI_LEXICON if kw in t)
            other_hits = sum(1 for kw in OTHER_FOODS_LEXICON if kw in t)
            pb_density = pb_hits / word_count
            other_density = other_hits / word_count
            diff_density = pb_density - other_density
            pb_ratio = (pb_hits + 1.0) / (other_hits + 1.0)

            likes = np.log1p(float(r.get("likes_count", 0) or 0))
            comments = np.log1p(float(r.get("comments_count", 0) or 0))
            
            raw_tags = r.get("tags_list", [])
            num_tags = len(raw_tags) if isinstance(raw_tags, list) else 0
            tag_ratio = num_tags / word_count

            feats.append([
                word_count, char_count, avg_w, pb_hits, other_hits,
                pb_density, other_density, diff_density, pb_ratio,
                likes, comments, num_tags, tag_ratio
            ])
        return np.array(feats, dtype=float)

    def fit(self, df_train, y_train):
        col = "clean_text" if "clean_text" in df_train.columns else "combined_text"
        raw_texts = df_train[col].fillna("").astype(str)
        tfidf_mat = self.union_tfidf.fit_transform(raw_texts)
        meta_mat = self.scaler.fit_transform(self._extract_meta(df_train))
        X_all = hstack([tfidf_mat, meta_mat]).tocsr()
        self.classifier.fit(X_all, y_train)
        return self

    def predict_proba(self, df_test):
        col = "clean_text" if "clean_text" in df_test.columns else "combined_text"
        raw_texts = df_test[col].fillna("").astype(str)
        tfidf_mat = self.union_tfidf.transform(raw_texts)
        meta_mat = self.scaler.transform(self._extract_meta(df_test))
        X_all = hstack([tfidf_mat, meta_mat]).tocsr()
        
        if hasattr(self.classifier, "predict_proba"):
            raw_probs = self.classifier.predict_proba(X_all)
            p1 = raw_probs[:, 1].copy()
        else:
            d = self.classifier.decision_function(X_all)
            p1 = 1 / (1 + np.exp(-d))

        # Apply culinary domain calibration
        for i, (_, row) in enumerate(df_test.iterrows()):
            t = str(row.get(col, "") or "").lower()
            has_comp = any(c in t for c in HARD_COMPETITORS)
            has_pb = any(p in t for p in STRONG_PB_SIGNALS)
            
            if has_comp and not has_pb:
                p1[i] = max(p1[i] - self.penalty, 0.01)
            elif has_pb and not has_comp:
                p1[i] = min(p1[i] + self.boost, 0.99)

        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, df_test):
        probs = self.predict_proba(df_test)[:, 1]
        return (probs >= self.threshold).astype(int)
