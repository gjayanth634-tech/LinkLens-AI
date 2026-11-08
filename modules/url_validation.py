"""
URL validation and sanitization utilities for LinkLens AI.
"""

import re
import urllib.parse
from typing import Tuple


def is_valid_url(url: str) -> Tuple[bool, str]:
    """
    Validate whether a URL meets syntax and format requirements for scanning.

    Args:
        url: The URL string to validate.

    Returns:
        tuple (bool, str): (is_valid, error_message).
        is_valid is True if valid, False otherwise.
        error_message is empty string if valid, or a descriptive error if invalid.
    """
    if not isinstance(url, str):
        return False, "URL must be a string."

    trimmed = url.strip()

    if not trimmed:
        return False, "URL cannot be empty."

    if len(trimmed) >= 2048:
        return False, "URL exceeds maximum length of 2048 characters."

    # Must start with http:// or https://
    if not (trimmed.startswith("http://") or trimmed.startswith("https://")):
        return False, "URL must start with http:// or https://."

    try:
        parsed = urllib.parse.urlparse(trimmed)
    except Exception as exc:
        return False, f"Failed to parse URL: {str(exc)}"

    if not parsed.scheme or parsed.scheme not in ("http", "https"):
        return False, "URL scheme must be http or https."

    if not parsed.hostname or not parsed.hostname.strip():
        return False, "URL must include a valid hostname."

    return True, ""


def sanitize_url(url: str) -> str:
    """
    Sanitize URL by stripping whitespace, normalizing slashes if path is just '/',
    and performing basic cleanup without visiting the URL.

    Args:
        url: The input URL string.

    Returns:
        Sanitized URL string.
    """
    if not url or not isinstance(url, str):
        return ""

    sanitized = url.strip()

    try:
        parsed = urllib.parse.urlparse(sanitized)
        if parsed.scheme and parsed.netloc:
            # If path consists solely of slashes (e.g. '/'), remove it
            path = parsed.path
            if path and path.strip("/") == "":
                path = ""

            sanitized = urllib.parse.urlunparse((
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
    except Exception:
        # Fallback to basic string cleaning if parsing fails
        if sanitized.endswith("/") and not sanitized.endswith("://"):
            sanitized = sanitized.rstrip("/")

    return sanitized
