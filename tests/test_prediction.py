"""
Tests for the prediction module.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prediction import PhishingPredictor
from modules.feature_extraction import extract_features


class TestPhishingPredictor(unittest.TestCase):
    """Test the PhishingPredictor class."""

    def test_init(self):
        """Test predictor initialization."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        self.assertEqual(predictor.model_name, 'Random Forest')
        self.assertFalse(predictor.is_loaded)

    def test_load_nonexistent_model(self):
        """Loading a nonexistent model should return False."""
        predictor = PhishingPredictor('nonexistent_model.pkl')
        result = predictor.load_model()
        self.assertFalse(result)
        self.assertFalse(predictor.is_loaded)

    def test_load_real_model(self):
        """Load the actual trained model if available."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        loaded = predictor.load_model()
        if loaded:
            self.assertTrue(predictor.is_loaded)
            self.assertIsNotNone(predictor.get_model())

    def test_predict_with_model(self):
        """Test prediction with a loaded model."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        if not predictor.load_model():
            self.skipTest("Model file not found - run train_model.py first")

        features = extract_features('https://www.google.com')
        result = predictor.predict(features)

        self.assertIn('prediction', result)
        self.assertIn('score', result)
        self.assertIn('label', result)
        self.assertIn('model_name', result)

        self.assertIn(result['prediction'], [
            'Potential Phishing', 'Legitimate-Looking', 'Uncertain'
        ])
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)
        self.assertIn(result['label'], [0, 1])

    def test_predict_phishing_url(self):
        """Test prediction on a suspicious-looking URL."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        if not predictor.load_model():
            self.skipTest("Model file not found - run train_model.py first")

        features = extract_features(
            'http://192.168.1.1/secure-login-verify-account/update?user=admin@bank.com'
        )
        result = predictor.predict(features)
        # We expect this to be flagged, but don't assert exact category
        self.assertIsNotNone(result['prediction'])

    def test_predict_legitimate_url(self):
        """Test prediction on a legitimate-looking URL."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        if not predictor.load_model():
            self.skipTest("Model file not found - run train_model.py first")

        features = extract_features('https://www.github.com/user/repository')
        result = predictor.predict(features)
        self.assertIsNotNone(result['prediction'])

    def test_prediction_score_thresholds(self):
        """Verify prediction categories match score thresholds."""
        predictor = PhishingPredictor('models/random_forest.pkl', 'Random Forest')
        if not predictor.load_model():
            self.skipTest("Model file not found - run train_model.py first")

        features = extract_features('https://example.com')
        result = predictor.predict(features)

        score = result['score']
        prediction = result['prediction']

        if score >= 0.6:
            self.assertEqual(prediction, 'Potential Phishing')
        elif score < 0.4:
            self.assertEqual(prediction, 'Legitimate-Looking')
        else:
            self.assertEqual(prediction, 'Uncertain')


class TestXGBoostPredictor(unittest.TestCase):
    """Test the XGBoost model predictor."""

    def test_xgboost_prediction(self):
        """Test XGBoost model prediction."""
        predictor = PhishingPredictor('models/xgboost_model.pkl', 'XGBoost')
        if not predictor.load_model():
            self.skipTest("XGBoost model not found - run train_model.py first")

        features = extract_features('https://www.example.com/page')
        result = predictor.predict(features)

        self.assertIn('prediction', result)
        self.assertEqual(result['model_name'], 'XGBoost')


class TestCatBoostPredictor(unittest.TestCase):
    """Test the CatBoost model prediction and probability output."""

    def test_catboost_prediction_probabilities(self):
        predictor = PhishingPredictor('models/catboost_model.pkl', 'CatBoost')
        if not predictor.load_model():
            self.skipTest("CatBoost model not found - run train_model.py first")

        features = extract_features('https://www.example.com/page')
        result = predictor.predict(features)
        probabilities = predictor.predict_proba(features)

        self.assertIn('prediction', result)
        self.assertEqual(result['model_name'], 'CatBoost')
        self.assertGreaterEqual(probabilities['legitimate_probability'], 0.0)
        self.assertGreaterEqual(probabilities['phishing_probability'], 0.0)
        self.assertAlmostEqual(
            probabilities['legitimate_probability'] + probabilities['phishing_probability'],
            1.0,
            places=3,
        )


if __name__ == '__main__':
    unittest.main()
