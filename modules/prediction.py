"""
Model prediction module for LinkLens AI.
Loads trained machine learning models (Random Forest, XGBoost, CatBoost) and produces URL risk classifications.
"""

import os
from typing import Dict, Any, Optional
import joblib
import numpy as np

from modules.feature_extraction import features_to_array, get_feature_names


class PhishingPredictor:
    """Predicts URL phishing status using a trained machine learning model."""

    def __init__(self, model_path: str, model_name: str = "Random Forest"):
        """
        Initialize the predictor with model path and human-readable name.

        Args:
            model_path: Filepath to the serialized model (.pkl).
            model_name: Name of the model algorithm.
        """
        self.model_path: str = model_path
        self.model_name: str = model_name
        self.model: Optional[Any] = None
        self.is_loaded: bool = False

    def load_model(self) -> bool:
        """
        Load model from model_path using joblib.load().

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"[Warning] Model file not found at: {self.model_path}")
                self.is_loaded = False
                self.model = None
                return False

            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"[Info] Successfully loaded '{self.model_name}' model from '{self.model_path}'.")
            return True
        except FileNotFoundError as fnf_err:
            print(f"[Warning] Model file not found at '{self.model_path}': {fnf_err}")
            self.is_loaded = False
            self.model = None
            return False
        except Exception as exc:
            print(f"[Warning] Error loading model from '{self.model_path}': {exc}")
            self.is_loaded = False
            self.model = None
            return False

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate phishing prediction, risk probability score, and classification category.

        Prediction categories:
        - score >= 0.6: 'Potential Phishing'
        - score < 0.4: 'Legitimate-Looking'
        - 0.4 <= score < 0.6: 'Uncertain'

        Args:
            features: Dictionary of extracted URL features.

        Returns:
            dict: {
                'prediction': category,
                'score': round(float(phishing_score), 4),
                'label': int(predicted_label),
                'model_name': self.model_name
            }
        """
        if not self.is_loaded or self.model is None:
            # Attempt to lazy-load if not already loaded
            if not self.load_model():
                raise RuntimeError(
                    f"Model '{self.model_name}' is not loaded and could not be found at '{self.model_path}'."
                )

        # Convert feature dictionary to ordered list
        feature_vals = features_to_array(features)
        X = np.array(feature_vals, dtype=float).reshape(1, -1)

        # Obtain probability estimates
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            classes = list(getattr(self.model, "classes_", range(len(probabilities))))
            phishing_idx = classes.index(1) if 1 in classes else (1 if len(probabilities) > 1 else 0)
            legitimate_idx = classes.index(0) if 0 in classes else (0 if phishing_idx != 0 else None)
            phishing_score = float(probabilities[phishing_idx])
            legitimate_score = float(probabilities[legitimate_idx]) if legitimate_idx is not None else 1.0 - phishing_score
        else:
            # Fallback for models without predict_proba
            raw_pred = self.model.predict(X)[0]
            phishing_score = float(raw_pred)
            legitimate_score = 1.0 - phishing_score

        # Get binary predicted label
        if hasattr(self.model, "predict"):
            predicted_label = int(self.model.predict(X)[0])
        else:
            predicted_label = 1 if phishing_score >= 0.5 else 0

        # Determine risk category
        if phishing_score >= 0.6:
            category = "Potential Phishing"
        elif phishing_score < 0.4:
            category = "Legitimate-Looking"
        else:
            category = "Uncertain"

        return {
            "prediction": category,
            "score": round(float(phishing_score), 4),
            "phishing_probability": round(phishing_score, 4),
            "legitimate_probability": round(legitimate_score, 4),
            "label": int(predicted_label),
            "model_name": self.model_name,
        }

    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Return legitimate and phishing probabilities for extracted URL features."""
        if not self.is_loaded or self.model is None:
            if not self.load_model():
                raise RuntimeError(
                    f"Model '{self.model_name}' is not loaded and could not be found at '{self.model_path}'."
                )
        if not hasattr(self.model, "predict_proba"):
            raise TypeError(f"Model '{self.model_name}' does not support probability predictions.")

        feature_vals = features_to_array(features)
        X = np.array(feature_vals, dtype=float).reshape(1, -1)
        probabilities = self.model.predict_proba(X)[0]
        classes = list(getattr(self.model, "classes_", range(len(probabilities))))
        phishing_idx = classes.index(1) if 1 in classes else (1 if len(probabilities) > 1 else 0)
        legitimate_idx = classes.index(0) if 0 in classes else (0 if phishing_idx != 0 else None)
        phishing_score = float(probabilities[phishing_idx])
        legitimate_score = float(probabilities[legitimate_idx]) if legitimate_idx is not None else 1.0 - phishing_score
        return {
            "legitimate_probability": round(legitimate_score, 4),
            "phishing_probability": round(phishing_score, 4),
        }

    def get_model(self) -> Optional[Any]:
        """Return the underlying trained model object."""
        return self.model
