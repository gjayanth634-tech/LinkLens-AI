"""
Feature extraction module for LinkLens AI URL analysis.
Extracts 14 lexical, structural, and semantic features from URLs for machine learning classification.
"""

import re
import string
import urllib.parse
from typing import Dict, List, Any

# Ordered list of feature names used by models and explainers
FEATURE_NAMES: List[str] = [
    "url_length",
    "hostname_length",
    "domain_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "num_special_chars",
    "num_subdomains",
    "num_params",
    "has_https",
    "has_ip_address",
    "has_at_symbol",
    "has_suspicious_patterns",
]

# Common keywords targeted by phishing attackers
SUSPICIOUS_PATTERNS: List[str] = [
    "login",
    "signin",
    "verify",
    "account",
    "update",
    "secure",
    "banking",
    "confirm",
    "password",
    "suspend",
    "unusual",
    "authenticate",
    "wallet",
    "alert",
    "expire",
    "restore",
    "unlock",
]

# Descriptions for user-friendly display and SHAP explanations
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "url_length": "Total number of characters in the URL",
    "hostname_length": "Number of characters in the hostname portion",
    "domain_length": "Length of the registered domain name (e.g., example.com)",
    "path_length": "Number of characters in the URL path segment",
    "num_dots": "Total count of dot (.) characters across the URL",
    "num_hyphens": "Total count of hyphen (-) characters across the URL",
    "num_digits": "Total count of numeric digits (0-9) in the URL",
    "num_special_chars": "Count of special characters (excluding protocol colons, dots, slashes, hyphens, and underscores)",
    "num_subdomains": "Number of subdomain hierarchy levels in the hostname",
    "num_params": "Number of unique query parameter keys",
    "has_https": "Indicates whether the URL uses secure HTTPS protocol (1) or HTTP (0)",
    "has_ip_address": "Indicates whether the hostname is a direct IPv4 address (1) or domain name (0)",
    "has_at_symbol": "Indicates presence of '@' symbol which can disguise target destinations (1 or 0)",
    "has_suspicious_patterns": "Indicates presence of sensitive security/banking keywords (1 or 0)",
}


def extract_features(url: str) -> Dict[str, Any]:
    """
    Extract all 14 lexical and structural features from a URL string.

    Args:
        url: The input URL string to analyze.

    Returns:
        dict: Mapping of each feature name in FEATURE_NAMES to its calculated value.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""

    # Check for IPv4 address format
    is_ip = bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname))

    # 1. url_length
    url_length = len(url)

    # 2. hostname_length
    hostname_length = len(hostname)

    # 3. domain_length
    if is_ip or not hostname:
        domain_length = hostname_length
    else:
        parts = hostname.split(".")
        if len(parts) >= 2:
            registered_domain = ".".join(parts[-2:])
            domain_length = len(registered_domain)
        else:
            domain_length = hostname_length

    # 4. path_length
    path_length = len(parsed.path) if parsed.path else 0

    # 5. num_dots
    num_dots = url.count(".")

    # 6. num_hyphens
    num_hyphens = url.count("-")

    # 7. num_digits
    num_digits = sum(c.isdigit() for c in url)

    # 8. num_special_chars: count in '@!#$%^&*()+=[]{}|;:,<>?~`'
    # excluding protocol colon, dots, hyphens, slashes, or underscores
    url_without_protocol = re.sub(r"^[a-zA-Z]+://", "", url)
    special_char_set = set("@!#$%^&*()+=[]{}|;:,<>?~`")
    num_special_chars = sum(1 for c in url_without_protocol if c in special_char_set)

    # 9. num_subdomains: split hostname by '.', subtract 2 (min 0), 0 if IP
    if is_ip or not hostname:
        num_subdomains = 0
    else:
        parts = hostname.split(".")
        num_subdomains = max(0, len(parts) - 2)

    # 10. num_params: unique query parameter keys
    try:
        query_params = urllib.parse.parse_qs(parsed.query)
        num_params = len(query_params)
    except Exception:
        num_params = 0

    # 11. has_https
    has_https = 1 if parsed.scheme == "https" else 0

    # 12. has_ip_address
    has_ip_address = 1 if is_ip else 0

    # 13. has_at_symbol
    has_at_symbol = 1 if "@" in url else 0

    # 14. has_suspicious_patterns
    url_lower = url.lower()
    has_suspicious_patterns = 1 if any(pat in url_lower for pat in SUSPICIOUS_PATTERNS) else 0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "num_subdomains": num_subdomains,
        "num_params": num_params,
        "has_https": has_https,
        "has_ip_address": has_ip_address,
        "has_at_symbol": has_at_symbol,
        "has_suspicious_patterns": has_suspicious_patterns,
    }


def get_feature_names() -> List[str]:
    """Return a copy of the ordered list of feature names."""
    return FEATURE_NAMES.copy()


def features_to_array(features: Dict[str, Any]) -> List[Any]:
    """
    Convert a feature dictionary into an ordered list matching FEATURE_NAMES.

    Args:
        features: Dictionary containing extracted feature values.

    Returns:
        list: Ordered list of feature values.
    """
    return [features[name] for name in FEATURE_NAMES]


def get_feature_description(feature_name: str) -> str:
    """
    Return human-readable explanation for a feature name.

    Args:
        feature_name: Name of the feature.

    Returns:
        str: Human-readable description.
    """
    return FEATURE_DESCRIPTIONS.get(feature_name, f"Feature metric: {feature_name}")
