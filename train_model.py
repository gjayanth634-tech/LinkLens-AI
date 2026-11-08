"""
LinkLens AI - Machine Learning Model Training Pipeline
Trains Random Forest, XGBoost, and CatBoost classifiers for phishing URL detection.
"""

import os
import sys
import json
import re
import random
import urllib.parse
import ipaddress
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Critical Feature Names Specification
FEATURE_NAMES = [
    'url_length',
    'hostname_length',
    'domain_length',
    'path_length',
    'num_dots',
    'num_hyphens',
    'num_digits',
    'num_special_chars',
    'num_subdomains',
    'num_params',
    'has_https',
    'has_ip_address',
    'has_at_symbol',
    'has_suspicious_patterns'
]

# Try importing feature extraction module
try:
    from modules.feature_extraction import (
        extract_features as mod_extract_features,
        get_feature_names as mod_get_feature_names,
        features_to_array as mod_features_to_array
    )
    HAS_FEATURE_MODULE = True
except ImportError:
    HAS_FEATURE_MODULE = False
    mod_extract_features = None
    mod_get_feature_names = None
    mod_features_to_array = None

# Try importing model evaluation module
try:
    from modules.model_evaluation import evaluate_model as mod_evaluate_model, compare_models as mod_compare_models
    HAS_EVAL_MODULE = True
except ImportError:
    HAS_EVAL_MODULE = False
    mod_evaluate_model = None
    mod_compare_models = None

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier


# ---------------------------------------------------------------------------
# Fallback / Built-in Feature Extraction
# ---------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'verification', 'account', 'update',
    'secure', 'banking', 'confirm', 'confirmation', 'password', 'security',
    'auth', 'authenticate', 'validate', 'validation', 'unlock', 'wallet',
    'credential', 'billing', 'support', 'alert', 'recover', 'session',
    'webscr', 'cmd=_login', 'cgi-bin', 'ebayisapi', 'paypal', 'appleid',
    'authorize', 'service', 'client', 'payment', 'protect'
]

IPV4_REGEX = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)

def fallback_extract_features(url: str) -> dict:
    """Extract 14 lexical and structural features from a URL."""
    url_str = str(url).strip()

    # Prepend scheme if missing for robust URL parsing
    parsed_target = url_str
    if not (url_str.startswith('http://') or url_str.startswith('https://') or url_str.startswith('ftp://')):
        parsed_target = 'http://' + url_str

    try:
        parsed = urllib.parse.urlparse(parsed_target)
        netloc = parsed.netloc or ''
        host = netloc.split(':')[0].strip().lower()
        path = parsed.path or ''
        query = parsed.query or ''
    except Exception:
        host = ''
        path = ''
        query = ''

    # 1. url_length
    url_length = len(url_str)

    # 2. hostname_length
    hostname_length = len(host)

    # 3. domain_length
    if host:
        host_parts = host.split('.')
        if len(host_parts) >= 2:
            domain_name = host_parts[-2] if len(host_parts[-1]) <= 3 and len(host_parts) > 2 else host_parts[-2] + '.' + host_parts[-1]
            domain_length = len(domain_name)
        else:
            domain_length = len(host)
    else:
        domain_length = 0

    # 4. path_length
    path_length = len(path)

    # 5. num_dots
    num_dots = url_str.count('.')

    # 6. num_hyphens
    num_hyphens = url_str.count('-')

    # 7. num_digits
    num_digits = sum(1 for c in url_str if c.isdigit())

    # 8. num_special_chars
    special_chars_set = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
    num_special_chars = sum(1 for c in url_str if c in special_chars_set)

    # 9. num_subdomains
    if host:
        clean_host = host.lower()
        if clean_host.startswith('www.'):
            clean_host = clean_host[4:]
        parts = clean_host.split('.')
        num_subdomains = max(0, len(parts) - 2)
    else:
        num_subdomains = 0

    # 10. num_params
    num_params = len([p for p in query.split('&') if p]) if query else 0

    # 11. has_https
    has_https = 1 if url_str.lower().startswith('https://') else 0

    # 12. has_ip_address
    clean_ip_host = host.split(':')[0]
    has_ip_address = 1 if IPV4_REGEX.match(clean_ip_host) else 0

    # 13. has_at_symbol
    has_at_symbol = 1 if '@' in url_str else 0

    # 14. has_suspicious_patterns
    url_lower = url_str.lower()
    has_suspicious_patterns = 1 if any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS) else 0

    return {
        'url_length': url_length,
        'hostname_length': hostname_length,
        'domain_length': domain_length,
        'path_length': path_length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_digits': num_digits,
        'num_special_chars': num_special_chars,
        'num_subdomains': num_subdomains,
        'num_params': num_params,
        'has_https': has_https,
        'has_ip_address': has_ip_address,
        'has_at_symbol': has_at_symbol,
        'has_suspicious_patterns': has_suspicious_patterns
    }

def fallback_features_to_array(features_dict: dict) -> np.ndarray:
    """Convert extracted features dictionary into ordered float array."""
    if isinstance(features_dict, np.ndarray):
        return features_dict.astype(np.float32)
    if isinstance(features_dict, list):
        return np.array(features_dict, dtype=np.float32)
    return np.array([features_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

def fallback_get_feature_names() -> list:
    """Return the ordered list of feature names."""
    return list(FEATURE_NAMES)


def extract_features(url: str) -> dict:
    """Extract features using modules.feature_extraction or fallback."""
    if HAS_FEATURE_MODULE and mod_extract_features is not None:
        try:
            return mod_extract_features(url)
        except Exception:
            return fallback_extract_features(url)
    return fallback_extract_features(url)

def features_to_array(features_dict: dict) -> np.ndarray:
    """Convert features to array using modules.feature_extraction or fallback."""
    if HAS_FEATURE_MODULE and mod_features_to_array is not None:
        try:
            return mod_features_to_array(features_dict)
        except Exception:
            return fallback_features_to_array(features_dict)
    return fallback_features_to_array(features_dict)

def get_feature_names() -> list:
    """Get feature names list using modules.feature_extraction or fallback."""
    if HAS_FEATURE_MODULE and mod_get_feature_names is not None:
        try:
            return mod_get_feature_names()
        except Exception:
            return fallback_get_feature_names()
    return fallback_get_feature_names()


# ---------------------------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_legitimate: int = 5000, n_phishing: int = 5000) -> list:
    """
    Generate diverse and realistic synthetic training dataset.
    Returns a list of (url, label) tuples where 0 = legitimate, 1 = phishing.
    """
    random.seed(42)

    # Legitimate generation building blocks
    legit_domains = [
        'google.com', 'amazon.com', 'microsoft.com', 'apple.com', 'github.com',
        'stackoverflow.com', 'wikipedia.org', 'youtube.com', 'linkedin.com',
        'twitter.com', 'facebook.com', 'reddit.com', 'netflix.com', 'medium.com',
        'wordpress.com', 'dropbox.com', 'spotify.com', 'adobe.com', 'salesforce.com',
        'zoom.us', 'paypal.com', 'ebay.com', 'craigslist.org', 'pinterest.com',
        'tumblr.com', 'imgur.com', 'gitlab.com', 'bitbucket.org', 'npmjs.com',
        'pypi.org', 'coursera.org', 'edx.org', 'udemy.com', 'duckduckgo.com',
        'canva.com', 'notion.so', 'figma.com', 'slack.com', 'stripe.com',
        'shopify.com', 'cloudflare.com', 'digitalocean.com', 'vimeo.com',
        'nytimes.com', 'cnn.com', 'bbc.com', 'theguardian.com', 'reuters.com',
        'bloomberg.com', 'forbes.com', 'wired.com', 'techcrunch.com', 'theverge.com',
        'python.org', 'rust-lang.org', 'apache.org', 'mozilla.org', 'w3.org',
        'mit.edu', 'stanford.edu', 'harvard.edu', 'berkeley.edu', 'usa.gov',
        'nih.gov', 'cdc.gov', 'who.int', 'chase.com', 'bankofamerica.com',
        'wellsfargo.com', 'citi.com', 'fidelity.com', 'vanguard.com'
    ]

    legit_subdomains = [
        '', '', '', 'www', 'www', 'www', 'api', 'docs', 'dev', 'support',
        'mail', 'blog', 'app', 'portal', 'status', 'en', 'help', 'accounts',
        'community', 'forum', 'cdn', 'download', 'mobile', 'web', 'connect', 'store'
    ]

    legit_paths = [
        '/search', '/about', '/about-us', '/contact', '/pricing', '/features',
        '/explore', '/faq', '/help', '/terms', '/privacy', '/careers',
        '/security', '/settings', '/profile', '/dashboard', '/overview',
        '/documentation', '/resources', '/blog', '/news', '/press'
    ]

    legit_categories = [
        'technology', 'science', 'business', 'education', 'health', 'travel',
        'lifestyle', 'design', 'engineering', 'analytics', 'marketing', 'finance',
        'guides', 'tutorials', 'developer', 'hardware', 'software', 'mobile'
    ]

    legit_slugs = [
        'machine-learning-guide', 'getting-started-tutorial', 'api-reference-v2',
        'annual-financial-report', 'privacy-policy-updates', 'community-guidelines',
        'developer-portal-setup', 'cloud-infrastructure-best-practices',
        'cybersecurity-essentials', 'user-onboarding-workflow', 'product-roadmap-2025',
        'how-to-configure-dns', 'data-science-fundamentals', 'quantum-computing-intro',
        'open-source-contributions', 'enterprise-architecture', 'mobile-ui-patterns',
        'web-performance-tuning', 'system-status-incident-report', 'release-notes-v4'
    ]

    legit_usernames = [
        'alex_dev', 'sarah99', 'tech_guru', 'code_master', 'john_doe',
        'emma_w', 'david_k', 'cloud_architect', 'security_pro', 'data_nerd'
    ]

    legit_search_queries = [
        'python+asyncio+tutorial', 'best+mechanical+keyboard', 'docker+compose+examples',
        'machine+learning+algorithms', 'react+state+management', 'cloud+security+tips',
        'home+workout+routines', 'healthy+dinner+recipes', 'git+interactive+rebase'
    ]

    # Phishing generation building blocks
    phish_tlds = [
        'tk', 'ml', 'ga', 'cf', 'gq', 'top', 'xyz', 'buzz', 'work', 'icu',
        'site', 'vip', 'click', 'monster', 'support', 'rest', 'cam', 'kim',
        'link', 'cc', 'ws', 'info', 'club', 'online', 'space', 'website'
    ]

    target_brands = [
        'paypal', 'google', 'apple', 'microsoft', 'amazon', 'netflix', 'chase',
        'wellsfargo', 'bankofamerica', 'facebook', 'instagram', 'twitter',
        'dropbox', 'ebay', 'linkedin', 'yahoo', 'binance', 'coinbase', 'citibank'
    ]

    typosquat_domains = [
        'g00gle.com', 'amaz0n.com', 'paypa1.com', 'micros0ft.com', 'faceb00k.com',
        'netf1ix.com', 'app1e.com', 'yah00.com', 'tw1tter.com', 'linkediin.com',
        'instargam.com', 'wellsfarg0.com', 'chase-on1ine.com', 'bofa-secur1ty.com',
        'walm4rt.com', 'eb4y.com', 'netfllx.com', 'pay-pal-security.com',
        'apple-id-verify.com', 'blockchain-wal1et.com', 'coinb4se.com', 'biinance.com',
        'g00gle.org', 'amaz0n-prime.net', 'paypa1-update.com', 'micros0ft-live.com'
    ]

    suspicious_words = [
        'login', 'signin', 'sign-in', 'verify', 'verification', 'account', 'update',
        'secure', 'banking', 'confirm', 'confirmation', 'password', 'security',
        'auth', 'authenticate', 'validate', 'validation', 'unlock', 'wallet',
        'credential', 'billing', 'support', 'alert', 'recover', 'session',
        'webscr', 'cmd-login', 'cgi-bin', 'authorize', 'protection', 'identity'
    ]

    phish_file_paths = [
        '/login.php', '/signin.html', '/verify.php', '/account/update.php',
        '/webscr/cmd.php', '/auth/login.aspx', '/online/banking/verify.html',
        '/wp-admin/login.php', '/secure/account_verification.html',
        '/billing/update-info.cgi', '/security/resolve-lockout.php',
        '/session/auth/index.html', '/wallet/restore-access.php',
        '/identity/confirm-identity.html', '/customer/signon.aspx'
    ]

    # Generate Legitimate URLs
    legitimate_urls = set()
    while len(legitimate_urls) < n_legitimate:
        scheme = 'https://' if random.random() < 0.94 else 'http://'
        domain = random.choice(legit_domains)
        sub = random.choice(legit_subdomains)
        host = f"{sub}.{domain}" if sub else domain

        pattern_choice = random.randint(1, 8)
        if pattern_choice == 1:
            # Clean domain or simple page
            path = random.choice(legit_paths) if random.random() < 0.6 else ''
            url = f"{scheme}{host}{path}"
        elif pattern_choice == 2:
            # Category and article/product slug
            cat = random.choice(legit_categories)
            slug = random.choice(legit_slugs)
            url = f"{scheme}{host}/{cat}/{slug}"
        elif pattern_choice == 3:
            # User profile or repository
            user = random.choice(legit_usernames)
            slug = random.choice(legit_slugs)
            url = f"{scheme}{host}/user/{user}/projects/{slug}"
        elif pattern_choice == 4:
            # Search query or structured params
            q = random.choice(legit_search_queries)
            page = random.randint(1, 15)
            url = f"{scheme}{host}/search?q={q}&page={page}&sort=relevance"
        elif pattern_choice == 5:
            # Documentation / versioned path
            ver = f"v{random.randint(1, 5)}.{random.randint(0, 9)}"
            topic = random.choice(legit_slugs)
            url = f"{scheme}{host}/docs/{ver}/{topic}.html"
        elif pattern_choice == 6:
            # Product view with query parameters
            item_id = random.randint(10000, 999999)
            ref_source = random.choice(['homepage', 'nav_menu', 'search_result', 'sidebar'])
            url = f"{scheme}{host}/products/item-{item_id}?ref={ref_source}&currency=USD"
        elif pattern_choice == 7:
            # Media / API endpoint
            res_id = random.randint(1000, 99999)
            url = f"{scheme}{host}/api/v2/items/{res_id}?format=json&include_metadata=true"
        else:
            # Legitimate authentication portal (Nuanced: legitimate with auth keywords)
            if 'google' in domain:
                url = f"https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn&hl=en"
            elif 'github' in domain:
                url = f"https://github.com/login?return_to=%2Fsettings%2Fsecurity"
            elif 'amazon' in domain:
                url = f"https://www.amazon.com/ap/signin?openid.assoc_handle=usflex&pageId=usflex"
            elif 'microsoft' in domain:
                url = f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=58392"
            else:
                url = f"{scheme}{host}/login?redirect_to=/dashboard&session_check=1"

        legitimate_urls.add(url)

    # Generate Phishing URLs
    phishing_urls = set()
    while len(phishing_urls) < n_phishing:
        # Mix protocols: ~65% http, ~35% https (reflecting modern phishing)
        scheme = 'https://' if random.random() < 0.35 else 'http://'
        archetype = random.randint(1, 9)

        if archetype == 1:
            # 1. IP address as host
            ip = f"{random.randint(11, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            port = f":{random.choice([8080, 8443, 8888, 8000])}" if random.random() < 0.25 else ""
            fpath = random.choice(phish_file_paths)
            token = f"{random.randint(100000, 999999)}"
            url = f"http://{ip}{port}{fpath}?session_id={token}"

        elif archetype == 2:
            # 2. Deceptive Subdomain Stacking / Spoofing
            brand = random.choice(target_brands)
            w1 = random.choice(suspicious_words)
            w2 = random.choice(suspicious_words)
            tld = random.choice(phish_tlds)
            fpath = random.choice(phish_file_paths)
            url = f"{scheme}{brand}.com.{w1}.{w2}.portal-update.{tld}{fpath}"

        elif archetype == 3:
            # 3. Typosquatting / Misspelled domains
            tdomain = random.choice(typosquat_domains)
            fpath = random.choice(phish_file_paths)
            p_id = random.randint(1000, 99999)
            url = f"{scheme}{tdomain}{fpath}?user_id={p_id}&verify=true"

        elif archetype == 4:
            # 4. Hyphen-stuffed suspicious domains
            w1 = random.choice(['secure', 'verify', 'update', 'online', 'auth'])
            w2 = random.choice(['login', 'account', 'banking', 'signon', 'portal'])
            w3 = random.choice(['alert', 'security', 'service', 'center', 'confirm'])
            brand = random.choice(target_brands)
            tld = random.choice(['com', 'net', 'org', 'info', 'xyz'])
            fpath = random.choice(phish_file_paths)
            url = f"{scheme}{w1}-{brand}-{w2}-{w3}.{tld}{fpath}"

        elif archetype == 5:
            # 5. Abuse-heavy / Free TLDs with suspicious brand keywords
            brand = random.choice(target_brands)
            action = random.choice(['security-update', 'verification-login', 'account-restore', 'billing-confirm'])
            tld = random.choice(phish_tlds)
            fpath = random.choice(phish_file_paths)
            url = f"{scheme}{brand}-{action}.{tld}{fpath}"

        elif archetype == 6:
            # 6. Userinfo spoofing with @ symbol in URL
            brand = random.choice(target_brands)
            phish_host = f"secure-auth-check-{random.randint(10, 99)}.{random.choice(phish_tlds)}"
            fpath = random.choice(phish_file_paths)
            url = f"{scheme}{brand}.com:security-check@{phish_host}{fpath}"

        elif archetype == 7:
            # 7. Digit-heavy domain and path
            digits_host = f"{random.choice(target_brands)}{random.randint(10000, 999999)}"
            tld = random.choice(['com', 'net', 'xyz', 'top', 'info'])
            digits_path = f"{random.randint(10000, 99999)}/{random.randint(100, 999)}"
            url = f"{scheme}{digits_host}.{tld}/{digits_path}/login.php?client={random.randint(1000000, 9999999)}"

        elif archetype == 8:
            # 8. Very long URL (150+ chars with nested paths and long tokens)
            brand = random.choice(target_brands)
            w1 = random.choice(suspicious_words)
            w2 = random.choice(suspicious_words)
            long_token = ''.join(random.choices('0123456789abcdef', k=48))
            long_hash = ''.join(random.choices('0123456789abcdef', k=32))
            url = (
                f"{scheme}{brand}-security-resolution-verification-center.net/"
                f"cgi-bin/webscr/dispatch/{w1}/{w2}/login.php?"
                f"session_token={long_token}&redirect_url=https://{brand}.com/secure&auth_signature={long_hash}"
            )

        else:
            # 9. Nuanced / Sneaky phishing (looks relatively clean, uses HTTPS)
            brand = random.choice(target_brands)
            url = f"https://{brand}-security-service.com/signin?ref=account_alert"

        phishing_urls.add(url)

    # Assemble dataset with labels: 0 = legitimate, 1 = phishing
    dataset = [(url, 0) for url in legitimate_urls] + [(url, 1) for url in phishing_urls]
    random.shuffle(dataset)

    return dataset


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def safe_evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    """Evaluate a trained model and return standardized performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else 0.0
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

    # Attempt calling modules.model_evaluation.evaluate_model if available
    if HAS_EVAL_MODULE and mod_evaluate_model is not None:
        try:
            external_res = mod_evaluate_model(model, X_test, y_test, model_name=model_name)
            if isinstance(external_res, dict):
                metrics.update(external_res)
        except Exception:
            try:
                external_res = mod_evaluate_model(model, X_test, y_test)
                if isinstance(external_res, dict):
                    metrics.update(external_res)
            except Exception:
                pass

    return metrics


def print_evaluation_report(metrics: dict, model_name: str) -> None:
    """Print formatted evaluation metrics and confusion matrix."""
    print(f"\n{'='*20} {model_name.upper()} RESULTS {'='*20}")
    print(f"  Accuracy:         {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Precision:        {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:           {metrics.get('recall', 0.0):.4f}")
    print(f"  F1 Score:         {metrics.get('f1', metrics.get('f1_score', 0.0)):.4f}")
    print(f"  ROC-AUC:          {metrics.get('roc_auc', 0.0):.4f}")

    cm = metrics.get('confusion_matrix')
    print("\n  Confusion Matrix:")
    if isinstance(cm, (list, np.ndarray)) and len(cm) == 2 and len(cm[0]) == 2:
        print(f"    True Negative (Legit -> Legit):     {cm[0][0]:>5}")
        print(f"    False Positive (Legit -> Phish):    {cm[0][1]:>5}")
        print(f"    False Negative (Phish -> Legit):    {cm[1][0]:>5}")
        print(f"    True Positive (Phish -> Phish):     {cm[1][1]:>5}")
    else:
        print(f"    {cm}")


def safe_compare_models(rf_metrics: dict, xgb_metrics: dict, catboost_metrics: dict = None) -> None:
    """Print side-by-side comparison of trained models."""
    print(f"\n{'='*22} MODEL COMPARISON {'='*22}")
    model_metrics = [
        ("Random Forest", rf_metrics),
        ("XGBoost", xgb_metrics),
    ]
    if catboost_metrics is not None:
        model_metrics.append(("CatBoost", catboost_metrics))
    
    # Try calling compare_models from modules if available
    if HAS_EVAL_MODULE and mod_compare_models is not None:
        try:
            mod_compare_models([metrics for _, metrics in model_metrics])
        except Exception:
            try:
                mod_compare_models(rf_metrics, xgb_metrics)
            except Exception:
                pass

    # Print clean comparison table
    model_columns = "".join(f" {name:<18}" for name, _ in model_metrics)
    print(f"{'Metric':<20}{model_columns} {'Best Model':<15}")
    print("-" * (35 + 18 * len(model_metrics)))
    metric_keys = [
        ('Accuracy', 'accuracy'),
        ('Precision', 'precision'),
        ('Recall', 'recall'),
        ('F1 Score', 'f1'),
        ('ROC-AUC', 'roc_auc')
    ]

    for label, key in metric_keys:
        values = [
            metrics.get(key, metrics.get('f1_score' if key == 'f1' else key, 0.0))
            for _, metrics in model_metrics
        ]
        best_index = max(range(len(values)), key=values.__getitem__)
        winner = model_metrics[best_index][0] if values.count(values[best_index]) == 1 else 'Tie'
        values_text = "".join(f" {value:<18.4f}" for value in values)
        print(f"{label:<20}{values_text} {winner:<15}")


def make_json_serializable(obj):
    """Recursively convert NumPy types to native Python types for JSON output."""
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("   LinkLens AI - Phishing Detection ML Training Pipeline")
    print("=" * 70)

    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    dataset_file = os.path.join(dataset_dir, "phishing_urls.csv")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Dataset Generation or Loading
    if os.path.exists(dataset_file):
        print(f"\n[1/8] Loading existing dataset from: {dataset_file}")
        df = pd.read_csv(dataset_file)
        if 'url' in df.columns and 'label' in df.columns:
            urls = df['url'].astype(str).tolist()
            labels = df['label'].astype(int).tolist()
            dataset = list(zip(urls, labels))
            print(f"      Successfully loaded {len(dataset)} records from CSV.")
        else:
            print("      Invalid CSV format. Re-generating synthetic dataset...")
            dataset = generate_synthetic_dataset(n_legitimate=5000, n_phishing=5000)
            df = pd.DataFrame(dataset, columns=['url', 'label'])
            df.to_csv(dataset_file, index=False)
            print(f"      Saved {len(dataset)} records to {dataset_file}.")
    else:
        print("\n[1/8] Generating synthetic dataset (5,000 legitimate and 5,000 phishing URLs)...")
        dataset = generate_synthetic_dataset(n_legitimate=5000, n_phishing=5000)
        df = pd.DataFrame(dataset, columns=['url', 'label'])
        df.to_csv(dataset_file, index=False)
        print(f"      Generated and saved {len(dataset)} URLs to {dataset_file}.")

    # 2. Extract Features
    print(f"\n[2/8] Extracting features for {len(dataset)} URLs...")
    feature_names = get_feature_names()
    print(f"      Using {len(feature_names)} features: {feature_names}")

    X_list = []
    y_list = []
    for idx, (url, label) in enumerate(dataset):
        features = extract_features(url)
        feature_vector = features_to_array(features)
        X_list.append(feature_vector)
        y_list.append(label)

        if (idx + 1) % 2500 == 0 or (idx + 1) == len(dataset):
            print(f"      Extracted features for {idx + 1}/{len(dataset)} URLs...")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # 3. Data Analysis
    print("\n[3/8] Performing dataset analysis...")
    print(f"      Dataset Matrix Shape: X={X.shape}, y={y.shape}")

    num_legit = int(np.sum(y == 0))
    num_phish = int(np.sum(y == 1))
    print(f"      Class Distribution: Legitimate (0) = {num_legit} ({num_legit/len(y):.1%}), "
          f"Phishing (1) = {num_phish} ({num_phish/len(y):.1%})")

    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['label'] = y

    # Missing values check
    missing_counts = features_df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"      Missing Values Check: {total_missing} missing values across all features.")

    # Duplicate URLs check
    duplicate_count = df['url'].duplicated().sum()
    print(f"      Duplicate URLs Check: {duplicate_count} duplicate URLs found.")

    # Feature statistics table
    print("\n      Feature Statistics Summary (Mean and Std Dev):")
    stats_data = []
    for f in feature_names:
        mean_all = float(features_df[f].mean())
        std_all = float(features_df[f].std())
        mean_legit = float(features_df[features_df['label'] == 0][f].mean())
        mean_phish = float(features_df[features_df['label'] == 1][f].mean())
        stats_data.append({
            'Feature': f,
            'Mean (All)': f"{mean_all:.2f}",
            'Std (All)': f"{std_all:.2f}",
            'Mean (Legit)': f"{mean_legit:.2f}",
            'Mean (Phish)': f"{mean_phish:.2f}"
        })
    stats_table = pd.DataFrame(stats_data)
    print(stats_table.to_string(index=False))

    # 4. Train / Test Split
    print("\n[4/8] Splitting data into 80% train and 20% test sets (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Testing set:  {X_test.shape[0]} samples")

    # 5. Train Random Forest Classifier
    print("\n[5/8] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("      Random Forest training completed successfully.")

    # 6. Train XGBoost Classifier
    print("\n[6/8] Training XGBoost Classifier...")
    xgb_kwargs = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    try:
        # Standard XGBClassifier parameter specified in prompt
        xgb_model = XGBClassifier(**xgb_kwargs, use_label_encoder=False)
    except TypeError:
        # Compatibility fallback for newer XGBoost versions
        xgb_model = XGBClassifier(**xgb_kwargs)

    xgb_model.fit(X_train, y_train)
    print("      XGBoost training completed successfully.")

    # 7. Train CatBoost Classifier on the same extracted features and split
    print("\n[7/8] Training CatBoost Classifier...")
    catboost_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric='Accuracy',
        verbose=0,
        random_seed=42
    )
    catboost_model.fit(X_train, y_train)
    print("      CatBoost training completed successfully.")

    # 8. Model Evaluation & Comparison
    print("\n[8/8] Evaluating trained models on test dataset...")
    rf_metrics = safe_evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
    xgb_metrics = safe_evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    catboost_metrics = safe_evaluate_model(catboost_model, X_test, y_test, model_name="CatBoost")

    print_evaluation_report(rf_metrics, "Random Forest")
    print_evaluation_report(xgb_metrics, "XGBoost")
    print_evaluation_report(catboost_metrics, "CatBoost")
    safe_compare_models(rf_metrics, xgb_metrics, catboost_metrics)

    # 8. Save Models and Metadata
    print("\n" + "=" * 70)
    print("SAVING MODELS & ARTIFACTS")
    print("=" * 70)

    rf_path = os.path.join(models_dir, "random_forest.pkl")
    xgb_path = os.path.join(models_dir, "xgboost_model.pkl")
    catboost_path = os.path.join(models_dir, "catboost_model.pkl")
    feat_path = os.path.join(models_dir, "feature_names.pkl")
    results_path = os.path.join(models_dir, "evaluation_results.json")

    joblib.dump(rf_model, rf_path)
    print(f"  [+] Random Forest model saved to: {rf_path}")

    joblib.dump(xgb_model, xgb_path)
    print(f"  [+] XGBoost model saved to:       {xgb_path}")

    joblib.dump(catboost_model, catboost_path)
    print(f"  [+] CatBoost model saved to:      {catboost_path}")

    joblib.dump(get_feature_names(), feat_path)
    print(f"  [+] Feature names saved to:       {feat_path}")

    evaluation_results = {
        'random_forest': rf_metrics,
        'xgboost': xgb_metrics,
        'catboost': catboost_metrics,
        'features': feature_names,
        'dataset_summary': {
            'total_samples': len(dataset),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_legitimate': num_legit,
            'num_phishing': num_phish,
            'split_ratio': '80/20',
            'random_state': 42
        }
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(evaluation_results), f, indent=4)
    print(f"  [+] Evaluation results saved to:  {results_path}")

    print("\nPipeline execution complete! Models are ready for inference.")
    print("=" * 70)


if __name__ == '__main__':
    main()
