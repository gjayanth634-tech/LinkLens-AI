"""
SHAP explanation module for LinkLens AI.
Computes Shapley additive explanations (TreeExplainer) to interpret model predictions
and highlight which URL characteristics drove the risk classification.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import shap

from modules.feature_extraction import get_feature_names, get_feature_description


class SHAPExplainer:
    """Interprets tree-based model predictions using SHAP values."""

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize TreeExplainer for the specified model and feature set.

        Args:
            model: Trained scikit-learn or XGBoost tree model.
            feature_names: List of feature names matching model inputs.
        """
        self.model: Any = model
        self.feature_names: List[str] = feature_names if feature_names is not None else get_feature_names()
        self.explainer: Optional[shap.TreeExplainer] = None

        if model is not None:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as exc:
                print(f"[Warning] Failed to initialize SHAP TreeExplainer: {exc}")
                self.explainer = None

    def explain(self, features_array: np.ndarray) -> Dict[str, Any]:
        """
        Calculate SHAP feature contributions for a single URL feature vector.

        Args:
            features_array: 1D or 2D array of feature values.

        Returns:
            dict: {
                'shap_values': list of floats,
                'base_value': float,
                'feature_contributions': list of dicts sorted by absolute contribution
            }
        """
        try:
            X = np.asarray(features_array, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            if self.explainer is None and self.model is not None:
                self.explainer = shap.TreeExplainer(self.model)

            if self.explainer is None:
                raise ValueError("SHAP TreeExplainer could not be initialized.")

            raw_shap = self.explainer.shap_values(X)

            # Determine phishing class shap values
            # Sklearn RandomForest: list of 2 arrays [class_0, class_1]
            if isinstance(raw_shap, list):
                if len(raw_shap) > 1:
                    sv_phishing = raw_shap[1]
                else:
                    sv_phishing = raw_shap[0]
            elif isinstance(raw_shap, np.ndarray):
                if raw_shap.ndim == 3 and raw_shap.shape[2] >= 2:
                    sv_phishing = raw_shap[0, :, 1]
                elif raw_shap.ndim == 2:
                    sv_phishing = raw_shap[0]
                elif raw_shap.ndim == 1:
                    sv_phishing = raw_shap
                else:
                    sv_phishing = raw_shap.flatten()
            else:
                sv_phishing = np.array(raw_shap).flatten()

            sv_flat = np.asarray(sv_phishing).flatten()

            # Extract base / expected value for class 1
            expected_val = self.explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                if len(expected_val) > 1:
                    base_value = float(expected_val[1])
                else:
                    base_value = float(expected_val[0])
            elif expected_val is not None:
                base_value = float(expected_val)
            else:
                base_value = 0.5

            values_flat = X.flatten()
            contributions: List[Dict[str, Any]] = []

            for i, name in enumerate(self.feature_names):
                val = float(values_flat[i]) if i < len(values_flat) else 0.0
                sv = float(sv_flat[i]) if i < len(sv_flat) else 0.0
                direction = "phishing" if sv > 0 else "legitimate"
                desc = get_feature_description(name)

                contributions.append({
                    "feature": name,
                    "value": float(val),
                    "shap_value": float(sv),
                    "direction": direction,
                    "description": desc,
                })

            # Sort descending by absolute SHAP value impact
            contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

            return {
                "shap_values": [round(float(v), 6) for v in sv_flat],
                "base_value": round(float(base_value), 6),
                "feature_contributions": contributions,
            }

        except Exception as exc:
            print(f"[Warning] Error during SHAP computation: {exc}")
            # Fallback representation
            fallback_contributions = []
            values_flat = (
                np.asarray(features_array).flatten()
                if features_array is not None
                else [0.0] * len(self.feature_names)
            )
            for i, name in enumerate(self.feature_names):
                val = float(values_flat[i]) if i < len(values_flat) else 0.0
                fallback_contributions.append({
                    "feature": name,
                    "value": float(val),
                    "shap_value": 0.0,
                    "direction": "legitimate",
                    "description": get_feature_description(name),
                })

            return {
                "shap_values": [0.0] * len(self.feature_names),
                "base_value": 0.5,
                "feature_contributions": fallback_contributions,
                "error": str(exc),
            }

    def generate_explanation_text(self, contributions: List[Dict[str, Any]]) -> str:
        """
        Generate a human-readable explanation from the top 5 contributing features.

        Args:
            contributions: List of feature contribution dicts sorted by absolute impact.

        Returns:
            str: Human-readable narrative explaining why the URL was classified as it was.
        """
        if not contributions:
            return "No feature contributions available to generate explanation."

        top_5 = contributions[:5]
        phishing_factors = [c for c in top_5 if c.get("direction") == "phishing"]
        legit_factors = [c for c in top_5 if c.get("direction") == "legitimate"]

        parts: List[str] = []

        if phishing_factors:
            phishing_reasons: List[str] = []
            for item in phishing_factors:
                feat = item["feature"]
                val = item["value"]
                if feat == "has_suspicious_patterns" and val > 0:
                    phishing_reasons.append("presence of high-risk keywords (e.g. login/verify/account)")
                elif feat == "has_ip_address" and val > 0:
                    phishing_reasons.append("use of raw IP address in place of legitimate domain")
                elif feat == "has_at_symbol" and val > 0:
                    phishing_reasons.append("presence of '@' symbol used to mislead browser navigation")
                elif feat == "has_https" and val == 0:
                    phishing_reasons.append("unencrypted connection (HTTP instead of HTTPS)")
                elif feat == "url_length":
                    phishing_reasons.append(f"unusually long URL length ({int(val)} characters)")
                elif feat == "num_subdomains" and val > 0:
                    phishing_reasons.append(f"abnormal subdomain depth ({int(val)} subdomains)")
                elif feat == "num_dots":
                    phishing_reasons.append(f"high number of dots ({int(val)})")
                elif feat == "num_hyphens":
                    phishing_reasons.append(f"frequent hyphens in domain or path ({int(val)})")
                elif feat == "num_special_chars":
                    phishing_reasons.append(f"elevated count of special characters ({int(val)})")
                else:
                    phishing_reasons.append(f"{feat.replace('_', ' ')} ({val})")
            parts.append(f"Risk indicators detected: {'; '.join(phishing_reasons)}.")

        if legit_factors:
            legit_reasons: List[str] = []
            for item in legit_factors:
                feat = item["feature"]
                val = item["value"]
                if feat == "has_https" and val > 0:
                    legit_reasons.append("secure HTTPS encryption protocol")
                elif feat == "has_ip_address" and val == 0:
                    legit_reasons.append("standard domain name instead of raw IP")
                elif feat == "has_suspicious_patterns" and val == 0:
                    legit_reasons.append("no credential-harvesting patterns found")
                elif feat == "has_at_symbol" and val == 0:
                    legit_reasons.append("standard URL syntax without '@' redirection")
                elif feat == "url_length":
                    legit_reasons.append(f"compact URL length ({int(val)} characters)")
                elif feat == "num_subdomains" and val == 0:
                    legit_reasons.append("clean domain hierarchy without excessive subdomains")
                else:
                    legit_reasons.append(f"healthy {feat.replace('_', ' ')} ({val})")
            parts.append(f"Trust indicators detected: {'; '.join(legit_reasons)}.")

        return " ".join(parts)
