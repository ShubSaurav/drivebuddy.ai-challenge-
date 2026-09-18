"""
Inference module for DrivebuddyAI Pav Bhaji Challenge.
Loads the enhanced champion pipeline (Word+Char TF-IDF + Food Lexicons + Calibrated Linear Model)
and provides a clean, robust prediction interface.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional
from src.preprocessing import preprocess_text


class PavBhajiPredictor:
    """Predictor class encapsulating preprocessing, composite feature extraction, and model inference."""

    def __init__(self, model_path: str = "models/pavbhaji_classifier.joblib"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        self.wrapper = joblib.load(model_path)

    def predict_post(
        self,
        description: str = "",
        hashtags: str = "",
        comments: str = "",
        remove_leakage: bool = True
    ) -> Dict[str, Any]:
        """
        Predict whether an Instagram post corresponds to Pav Bhaji based on text metadata.
        """
        combined = f"{description} {hashtags} {comments}".strip()
        cleaned_text = preprocess_text(combined, remove_leakage=remove_leakage)

        dummy_df = pd.DataFrame([{
            "clean_text": cleaned_text,
            "combined_text": combined,
            "likes_count": 0,
            "comments_count": 0
        }])

        prob_pb = float(self.wrapper.predict_proba(dummy_df)[0][1])
        prob_non_pb = 1.0 - prob_pb

        threshold = getattr(self.wrapper, "threshold", 0.50)

        if prob_pb >= threshold:
            pred_class = 1
            pred_label = "Pav Bhaji"
            confidence = prob_pb * 100
        else:
            pred_class = 0
            pred_label = "Not Pav Bhaji"
            confidence = prob_non_pb * 100

        # Extract tokens found in lexicons or vocabulary
        tokens = cleaned_text.split()
        return {
            "prediction": pred_label,
            "class_label": pred_class,
            "confidence": round(confidence, 2),
            "probability_pavbhaji": round(prob_pb, 4),
            "probability_not_pavbhaji": round(prob_non_pb, 4),
            "cleaned_text": cleaned_text,
            "matched_tokens": tokens[:10]
        }


_DEFAULT_PREDICTOR: Optional[PavBhajiPredictor] = None

def predict_post(
    description: str = "",
    hashtags: str = "",
    comments: str = "",
    remove_leakage: bool = True,
    model_path: str = "models/pavbhaji_classifier.joblib"
) -> Dict[str, Any]:
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or not os.path.exists(model_path):
        _DEFAULT_PREDICTOR = PavBhajiPredictor(model_path=model_path)
    return _DEFAULT_PREDICTOR.predict_post(
        description=description,
        hashtags=hashtags,
        comments=comments,
        remove_leakage=remove_leakage
    )
