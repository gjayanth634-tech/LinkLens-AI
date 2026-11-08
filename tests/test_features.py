"""
Tests for the URL feature extraction module.
"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.feature_extraction import (
    extract_features,
    get_feature_names,
    features_to_array,
    get_feature_description,
    FEATURE_NAMES,
    SUSPICIOUS_PATTERNS
)


class TestFeatureNames(unittest.TestCase):
    """Test feature name consistency."""

    def test_feature_names_count(self):
        """Should have exactly 14 features."""
        self.assertEqual(len(FEATURE_NAMES), 14)

    def test_get_feature_names_returns_copy(self):
        """get_feature_names() should return a copy, not the original list."""
        names = get_feature_names()
        names.append('extra')
        self.assertEqual(len(FEATURE_NAMES), 14)

    def test_feature_names_order(self):
        """Feature names must be in the specified order."""
        expected = [
            'url_length', 'hostname_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_digits', 'num_special_chars',
            'num_subdomains', 'num_params', 'has_https', 'has_ip_address',
            'has_at_symbol', 'has_suspicious_patterns'
        ]
        self.assertEqual(get_feature_names(), expected)


class TestExtractFeatures(unittest.TestCase):
    """Test feature extraction from URLs."""

    def test_basic_https_url(self):
        """Test extraction from a simple HTTPS URL."""
        features = extract_features('https://www.google.com/search?q=test')
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), 14)
        self.assertEqual(features['has_https'], 1)
        self.assertEqual(features['has_ip_address'], 0)
        self.assertEqual(features['has_at_symbol'], 0)

    def test_http_url(self):
        """Test that HTTP is correctly detected."""
        features = extract_features('http://example.com')
        self.assertEqual(features['has_https'], 0)

    def test_ip_address_url(self):
        """Test IP address detection in URL."""
        features = extract_features('http://192.168.1.1/login')
        self.assertEqual(features['has_ip_address'], 1)
        self.assertEqual(features['num_subdomains'], 0)

    def test_at_symbol(self):
        """Test @ symbol detection."""
        features = extract_features('http://google.com@evil.com/phishing')
        self.assertEqual(features['has_at_symbol'], 1)

    def test_suspicious_patterns(self):
        """Test suspicious pattern detection."""
        features = extract_features('https://secure-login-verify.example.com/account')
        self.assertEqual(features['has_suspicious_patterns'], 1)

    def test_no_suspicious_patterns(self):
        """Test clean URL without suspicious patterns."""
        features = extract_features('https://www.github.com/user/repo')
        self.assertEqual(features['has_suspicious_patterns'], 0)

    def test_url_length(self):
        """Test URL length calculation."""
        url = 'https://example.com'
        features = extract_features(url)
        self.assertEqual(features['url_length'], len(url))

    def test_subdomain_count(self):
        """Test subdomain counting."""
        features = extract_features('https://sub1.sub2.example.com/path')
        self.assertEqual(features['num_subdomains'], 2)

    def test_query_parameters(self):
        """Test query parameter counting."""
        features = extract_features('https://example.com/search?q=test&page=1&lang=en')
        self.assertEqual(features['num_params'], 3)

    def test_dots_count(self):
        """Test dot counting."""
        url = 'https://www.sub.example.com'
        features = extract_features(url)
        self.assertGreaterEqual(features['num_dots'], 3)

    def test_hyphens_count(self):
        """Test hyphen counting."""
        url = 'https://my-secure-site.example.com/my-page'
        features = extract_features(url)
        self.assertGreaterEqual(features['num_hyphens'], 3)

    def test_digits_count(self):
        """Test digit counting."""
        url = 'https://example123.com/page456'
        features = extract_features(url)
        self.assertEqual(features['num_digits'], 6)

    def test_all_features_are_numeric(self):
        """All feature values must be numeric (int or float)."""
        features = extract_features('https://www.example.com/path?q=1')
        for name, value in features.items():
            self.assertIsInstance(value, (int, float),
                                 f"Feature '{name}' is {type(value)}, expected numeric")


class TestFeaturesToArray(unittest.TestCase):
    """Test feature dict to array conversion."""

    def test_correct_order(self):
        """features_to_array should maintain FEATURE_NAMES order."""
        features = extract_features('https://example.com')
        array = features_to_array(features)
        self.assertEqual(len(array), 14)
        for i, name in enumerate(FEATURE_NAMES):
            self.assertEqual(array[i], features[name])

    def test_returns_list(self):
        """Should return a list."""
        features = extract_features('https://example.com')
        result = features_to_array(features)
        self.assertIsInstance(result, list)


class TestFeatureDescriptions(unittest.TestCase):
    """Test feature description helper."""

    def test_all_features_have_descriptions(self):
        """Every feature should have a human-readable description."""
        for name in FEATURE_NAMES:
            desc = get_feature_description(name)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 0, f"Feature '{name}' has empty description")

    def test_unknown_feature(self):
        """Unknown feature should return a fallback description."""
        desc = get_feature_description('nonexistent_feature')
        self.assertIsInstance(desc, str)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual URLs."""

    def test_very_long_url(self):
        """Test with a very long URL."""
        url = 'https://example.com/' + 'a' * 500
        features = extract_features(url)
        self.assertGreater(features['url_length'], 500)

    def test_url_with_many_special_chars(self):
        """Test URL with many special characters."""
        url = 'https://example.com/path?a=1&b=2#section!@extra'
        features = extract_features(url)
        self.assertGreater(features['num_special_chars'], 0)

    def test_minimal_url(self):
        """Test minimal valid URL."""
        features = extract_features('http://a.co')
        self.assertEqual(len(features), 14)


if __name__ == '__main__':
    unittest.main()
