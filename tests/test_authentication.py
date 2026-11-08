"""
Tests for authentication and URL validation.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from werkzeug.security import generate_password_hash, check_password_hash
from modules.url_validation import is_valid_url, sanitize_url


class TestPasswordHashing(unittest.TestCase):
    """Test Werkzeug password hashing used in authentication."""

    def test_hash_password(self):
        """Password hashing should produce a hash, not plain text."""
        password = 'SecurePass123!'
        hashed = generate_password_hash(password)
        self.assertNotEqual(password, hashed)
        self.assertGreater(len(hashed), len(password))

    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        password = 'SecurePass123!'
        hashed = generate_password_hash(password)
        self.assertTrue(check_password_hash(hashed, password))

    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hashed = generate_password_hash('CorrectPassword')
        self.assertFalse(check_password_hash(hashed, 'WrongPassword'))

    def test_different_hashes_same_password(self):
        """Same password should produce different hashes (salting)."""
        password = 'TestPassword'
        hash1 = generate_password_hash(password)
        hash2 = generate_password_hash(password)
        self.assertNotEqual(hash1, hash2)
        # Both should still verify
        self.assertTrue(check_password_hash(hash1, password))
        self.assertTrue(check_password_hash(hash2, password))


class TestURLValidation(unittest.TestCase):
    """Test URL validation functionality."""

    def test_valid_https_url(self):
        """Valid HTTPS URL should pass."""
        valid, error = is_valid_url('https://www.google.com')
        self.assertTrue(valid)
        self.assertEqual(error, '')

    def test_valid_http_url(self):
        """Valid HTTP URL should pass."""
        valid, error = is_valid_url('http://example.com/page')
        self.assertTrue(valid)

    def test_empty_url(self):
        """Empty URL should fail."""
        valid, error = is_valid_url('')
        self.assertFalse(valid)
        self.assertGreater(len(error), 0)

    def test_no_scheme(self):
        """URL without http/https should fail."""
        valid, error = is_valid_url('www.google.com')
        self.assertFalse(valid)

    def test_ftp_scheme(self):
        """FTP scheme should fail."""
        valid, error = is_valid_url('ftp://files.example.com')
        self.assertFalse(valid)

    def test_javascript_scheme(self):
        """JavaScript scheme should fail (security)."""
        valid, error = is_valid_url('javascript:alert(1)')
        self.assertFalse(valid)

    def test_very_long_url(self):
        """URL over 2048 characters should fail."""
        url = 'https://example.com/' + 'a' * 2048
        valid, error = is_valid_url(url)
        self.assertFalse(valid)

    def test_url_with_query(self):
        """URL with query parameters should pass."""
        valid, error = is_valid_url('https://example.com/search?q=test&page=1')
        self.assertTrue(valid)

    def test_url_with_fragment(self):
        """URL with fragment should pass."""
        valid, error = is_valid_url('https://example.com/page#section')
        self.assertTrue(valid)

    def test_url_with_port(self):
        """URL with port number should pass."""
        valid, error = is_valid_url('https://example.com:8080/api')
        self.assertTrue(valid)

    def test_ip_address_url(self):
        """URL with IP address should pass validation (it's valid, just suspicious)."""
        valid, error = is_valid_url('http://192.168.1.1/login')
        self.assertTrue(valid)


class TestURLSanitization(unittest.TestCase):
    """Test URL sanitization."""

    def test_strip_whitespace(self):
        """Whitespace should be stripped."""
        result = sanitize_url('  https://example.com  ')
        self.assertEqual(result, 'https://example.com')

    def test_preserve_valid_url(self):
        """Valid URL should remain unchanged."""
        url = 'https://www.example.com/path?q=1'
        result = sanitize_url(url)
        self.assertEqual(result, url)

    def test_returns_string(self):
        """Sanitize should always return a string."""
        result = sanitize_url('https://example.com')
        self.assertIsInstance(result, str)


class TestURLValidationSecurity(unittest.TestCase):
    """Test security aspects of URL validation."""

    def test_data_uri(self):
        """Data URI should fail."""
        valid, error = is_valid_url('data:text/html,<h1>test</h1>')
        self.assertFalse(valid)

    def test_none_input(self):
        """None input should be handled gracefully."""
        valid, error = is_valid_url(None)
        self.assertFalse(valid)

    def test_whitespace_only(self):
        """Whitespace-only input should fail."""
        valid, error = is_valid_url('   ')
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
