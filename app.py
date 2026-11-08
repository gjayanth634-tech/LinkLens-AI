"""
LinkLens AI - Phishing URL Detection System
Main Flask Application
"""

import os
import re
from datetime import datetime
from functools import wraps

import numpy as np
from bson.objectid import ObjectId
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config
from database.mongodb import (
    get_db,
    get_scans_collection,
    get_users_collection,
    init_db,
)
from modules.explanation import SHAPExplainer
from modules.feature_extraction import (
    extract_features,
    features_to_array,
    get_feature_names,
)
from modules.prediction import PhishingPredictor
from modules.url_validation import is_valid_url, sanitize_url

# ==============================================================================
# Feature Names and Classification Thresholds
# ==============================================================================
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

# Classification logic helper
def classify_score(score: float) -> str:
    if score >= 0.6:
        return 'Potential Phishing'
    elif score < 0.4:
        return 'Legitimate-Looking'
    else:
        return 'Uncertain'

# Helper to recursively convert numpy types to native Python types for JSON & BSON
def make_json_safe(data):
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_json_safe(v) for v in data]
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return [make_json_safe(v) for v in data.tolist()]
    return data

# ==============================================================================
# Application Setup & Database Initialization
# ==============================================================================
app = Flask(__name__)
app.config.from_object(Config)
csrf = CSRFProtect(app)

# Initialize MongoDB
users_collection = None
scans_collection = None

try:
    init_db(app)
    users_collection = get_users_collection()
    scans_collection = get_scans_collection()
    print("[INFO] MongoDB initialized and collections bound successfully.")
except Exception as e:
    print(f"[WARNING] MongoDB initialization error: {e}")

def get_users():
    global users_collection
    if users_collection is None:
        users_collection = get_users_collection()
    return users_collection

def get_scans():
    global scans_collection
    if scans_collection is None:
        scans_collection = get_scans_collection()
    return scans_collection

# ==============================================================================
# Model Loading & SHAP Initialization
# ==============================================================================
rf_predictor = None
xgb_predictor = None
catboost_predictor = None
shap_explainer = None

# Load Random Forest model
try:
    rf_path = getattr(Config, 'MODEL_PATH_RF', 'models/random_forest.pkl')
    rf_predictor = PhishingPredictor(rf_path, 'Random Forest')
    rf_predictor.load_model()
    print("[INFO] Random Forest model loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not load Random Forest model: {e}")

# Load XGBoost model
try:
    xgb_path = getattr(Config, 'MODEL_PATH_XGB', 'models/xgboost_model.pkl')
    xgb_predictor = PhishingPredictor(xgb_path, 'XGBoost')
    xgb_predictor.load_model()
    print("[INFO] XGBoost model loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not load XGBoost model: {e}")

# Load CatBoost model
try:
    catboost_path = getattr(Config, 'MODEL_PATH_CATBOOST', 'models/catboost_model.pkl')
    catboost_predictor = PhishingPredictor(catboost_path, 'CatBoost')
    catboost_predictor.load_model()
    print("[INFO] CatBoost model loaded successfully.")
except Exception as e:
    print(f"[WARNING] Could not load CatBoost model: {e}")

# Initialize SHAP explainer (use RF model)
try:
    if rf_predictor and rf_predictor.is_loaded:
        feature_names_list = get_feature_names() if callable(get_feature_names) else FEATURE_NAMES
        shap_explainer = SHAPExplainer(rf_predictor.get_model(), feature_names_list)
        print("[INFO] SHAP explainer initialized successfully.")
    else:
        print("[WARNING] Random Forest model not loaded; SHAP explainer skipped.")
except Exception as e:
    print(f"[WARNING] Could not initialize SHAP explainer: {e}")

# ==============================================================================
# Authentication Decorator
# ==============================================================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==============================================================================
# Context Processor
# ==============================================================================
@app.context_processor
def inject_user():
    user = None
    if 'user_id' in session:
        user = {
            'id': session.get('user_id'),
            'name': session.get('user_name'),
            'email': session.get('user_email')
        }
    return dict(user=user)

# ==============================================================================
# Routes
# ==============================================================================

# 1. GET / -> Index Landing Page
@app.route('/')
def index():
    return render_template('index.html')


# 2. GET & POST /register -> User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validate required fields
        if not full_name or not email or not password or not confirm_password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        # Validate email format
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if not re.match(email_regex, email):
            flash('Please enter a valid email address.', 'danger')
            return redirect(url_for('register'))

        # Validate password length
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return redirect(url_for('register'))

        # Validate password match
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        users_col = get_users()
        try:
            # Check if email already exists
            existing_user = users_col.find_one({'email': email})
            if existing_user:
                flash('An account with this email already exists. Please log in.', 'warning')
                return redirect(url_for('register'))

            password_hash = generate_password_hash(password)
            user_doc = {
                'full_name': full_name,
                'email': email,
                'password_hash': password_hash,
                'created_at': datetime.utcnow()
            }
            users_col.insert_one(user_doc)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'An error occurred during registration: {str(e)}', 'danger')
            return redirect(url_for('register'))

    return render_template('register.html')


# 3. GET & POST /login -> User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash('Please enter both email and password.', 'danger')
            return redirect(url_for('login'))

        users_col = get_users()
        try:
            user = users_col.find_one({'email': email})
            if not user or not check_password_hash(user.get('password_hash', ''), password):
                flash('Invalid email or password. Please try again.', 'danger')
                return redirect(url_for('login'))

            session.permanent = True
            session['user_id'] = str(user['_id'])
            session['user_name'] = user.get('full_name', '')
            session['user_email'] = user.get('email', '')

            flash(f"Welcome back, {user.get('full_name', 'User')}!", 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash(f'An error occurred during login: {str(e)}', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


# 4. GET /logout -> User Logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))


# 5. GET /dashboard -> User Dashboard with Scan Statistics
@app.route('/dashboard')
@login_required
def dashboard():
    users_col = get_users()
    scans_col = get_scans()
    user_id_str = session['user_id']

    try:
        user_obj_id = ObjectId(user_id_str)
    except Exception:
        session.clear()
        flash('Invalid session. Please log in again.', 'warning')
        return redirect(url_for('login'))

    user = users_col.find_one({'_id': user_obj_id})
    if not user:
        session.clear()
        flash('User account not found. Please log in again.', 'warning')
        return redirect(url_for('login'))

    user['_id'] = str(user['_id'])
    user['id'] = str(user['_id'])

    # Aggregate scan counts
    total_scans = scans_col.count_documents({'user_id': user_obj_id})
    phishing_count = scans_col.count_documents({
        'user_id': user_obj_id,
        'prediction': 'Potential Phishing'
    })
    legitimate_count = scans_col.count_documents({
        'user_id': user_obj_id,
        'prediction': 'Legitimate-Looking'
    })
    uncertain_count = scans_col.count_documents({
        'user_id': user_obj_id,
        'prediction': 'Uncertain'
    })

    stats = {
        'total_scans': total_scans,
        'phishing_count': phishing_count,
        'legitimate_count': legitimate_count,
        'uncertain_count': uncertain_count
    }

    # Fetch last 5 scans
    recent_cursor = scans_col.find({'user_id': user_obj_id}).sort('scanned_at', -1).limit(5)
    recent_scans = []
    for s in recent_cursor:
        s['_id'] = str(s['_id'])
        s['id'] = str(s['_id'])
        s['user_id'] = str(s['user_id'])
        recent_scans.append(s)

    return render_template('dashboard.html', user=user, stats=stats, recent_scans=recent_scans)


# 6. GET /scanner -> Scanner Interface
@app.route('/scanner')
@login_required
def scanner():
    return render_template('scanner.html')


# 7. POST /scan -> Process URL Scan (AJAX endpoint)
@app.route('/scan', methods=['POST'])
@csrf.exempt
@login_required
def scan():
    try:
        url = None
        requested_model = None
        if request.is_json:
            json_data = request.get_json(silent=True) or {}
            url = json_data.get('url')
            requested_model = json_data.get('model')
        if not url:
            url = request.form.get('url')
        if requested_model is None:
            requested_model = request.form.get('model')

        if not url or not url.strip():
            return jsonify({'success': False, 'error': 'URL is required.'}), 400

        # Sanitize and validate URL
        sanitized_url = sanitize_url(url.strip())
        url_valid, url_error = is_valid_url(sanitized_url)
        if not url_valid:
            return jsonify({
                'success': False,
                'error': url_error or 'Invalid URL format. Please provide a valid HTTP or HTTPS URL (e.g., https://example.com).'
            }), 400

        available_predictors = {
            'catboost': catboost_predictor,
            'random_forest': rf_predictor,
            'xgboost': xgb_predictor,
        }
        model_aliases = {
            'catboost': 'catboost',
            'rf': 'random_forest',
            'random_forest': 'random_forest',
            'random forest': 'random_forest',
            'xgb': 'xgboost',
            'xgboost': 'xgboost',
        }
        selection = str(requested_model or 'catboost').strip().lower()
        compare_all = selection in ('all', 'compare')
        selected_key = model_aliases.get(selection)

        if not compare_all and selected_key is None:
            return jsonify({
                'success': False,
                'error': 'Unknown model. Choose CatBoost, Random Forest, XGBoost, or all.'
            }), 400

        if compare_all:
            unavailable = [
                model_key for model_key, model_predictor in available_predictors.items()
                if not model_predictor or not model_predictor.is_loaded
            ]
            if unavailable:
                return jsonify({
                    'success': False,
                    'error': 'Train and load all three models before requesting comparative predictions.'
                }), 503
            predictor = catboost_predictor
        else:
            predictor = available_predictors[selected_key]
            if not predictor or not predictor.is_loaded:
                # Preserve the old RF/XGBoost fallback for default scans if CatBoost
                # has not been trained yet; explicit model selections never change.
                if requested_model is None and selected_key == 'catboost':
                    predictor = rf_predictor if rf_predictor and rf_predictor.is_loaded else (
                        xgb_predictor if xgb_predictor and xgb_predictor.is_loaded else None
                    )
                if not predictor or not predictor.is_loaded:
                    return jsonify({
                        'success': False,
                        'error': 'The requested detection model is not loaded. Train the model or choose another.'
                    }), 503

        if not predictor:
            return jsonify({
                'success': False,
                'error': 'Detection model is not loaded. Please contact the administrator.'
            }), 503

        # Extract features
        features = extract_features(sanitized_url)

        comparative_predictions = {}
        if compare_all:
            comparative_predictions = {
                model_key: model_predictor.predict(features)
                for model_key, model_predictor in available_predictors.items()
            }
            result = comparative_predictions['catboost']
        else:
            result = predictor.predict(features)
        score = float(result.get('score', 0.0))
        prediction = result.get('prediction')
        if not prediction:
            prediction = classify_score(score)

        model_name = result.get('model_name', predictor.model_name if hasattr(predictor, 'model_name') else 'Random Forest')

        # Use the explainer that matches the selected model, rather than reporting
        # Random Forest feature contributions for a different model's prediction.
        explanation = {}
        active_explainer = shap_explainer
        if predictor is not rf_predictor:
            try:
                active_explainer = SHAPExplainer(predictor.get_model(), get_feature_names())
            except Exception as exp_err:
                print(f"[WARNING] Could not initialize {predictor.model_name} SHAP explainer: {exp_err}")
                active_explainer = None
        if active_explainer:
            try:
                features_array = np.array(features_to_array(features))
                explanation = active_explainer.explain(features_array)
            except Exception as exp_err:
                print(f"[WARNING] SHAP explanation failed: {exp_err}")
                explanation = {}

        # Prepare database record
        clean_features = make_json_safe(features)
        clean_explanation = make_json_safe(explanation)

        scans_col = get_scans()
        scan_doc = {
            'user_id': ObjectId(session['user_id']),
            'submitted_url': sanitized_url,
            'prediction': prediction,
            'model_name': model_name,
            'model_score': score,
            'phishing_probability': result.get('phishing_probability', score),
            'legitimate_probability': result.get('legitimate_probability', 1.0 - score),
            'comparative_predictions': comparative_predictions,
            'extracted_features': clean_features,
            'shap_explanation': clean_explanation,
            'scanned_at': datetime.utcnow()
        }

        inserted = scans_col.insert_one(scan_doc)

        return jsonify({
            'success': True,
            'scan_id': str(inserted.inserted_id),
            'model_name': model_name,
            'prediction': prediction,
            'score': score,
            'phishing_probability': scan_doc['phishing_probability'],
            'legitimate_probability': scan_doc['legitimate_probability'],
            'comparative_predictions': comparative_predictions,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# 8. GET /result/<scan_id> -> Display Scan Result Page
@app.route('/result/<scan_id>')
@login_required
def result(scan_id):
    try:
        obj_id = ObjectId(scan_id)
        user_id = ObjectId(session['user_id'])
    except Exception:
        flash('Invalid scan ID format.', 'danger')
        return redirect(url_for('scanner'))

    scans_col = get_scans()
    scan_record = scans_col.find_one({'_id': obj_id, 'user_id': user_id})

    if not scan_record:
        flash('Scan result not found or access denied.', 'danger')
        return redirect(url_for('scanner'))

    scan_record['_id'] = str(scan_record['_id'])
    scan_record['id'] = str(scan_record['_id'])
    scan_record['user_id'] = str(scan_record['user_id'])

    return render_template('result.html', scan=scan_record)


# 9. GET /history -> User Scan History Page
@app.route('/history')
@login_required
def history():
    try:
        user_id = ObjectId(session['user_id'])
    except Exception:
        flash('Invalid session. Please log in again.', 'warning')
        return redirect(url_for('login'))

    scans_col = get_scans()
    cursor = scans_col.find({'user_id': user_id}).sort('scanned_at', -1)

    scans = []
    for s in cursor:
        s['_id'] = str(s['_id'])
        s['id'] = str(s['_id'])
        s['user_id'] = str(s['user_id'])
        scans.append(s)

    return render_template('history.html', scans=scans)


# 10. GET /scan/<scan_id> -> Detailed Scan View Page
@app.route('/scan/<scan_id>')
@login_required
def scan_details(scan_id):
    try:
        obj_id = ObjectId(scan_id)
        user_id = ObjectId(session['user_id'])
    except Exception:
        flash('Invalid scan ID format.', 'danger')
        return redirect(url_for('history'))

    scans_col = get_scans()
    scan_record = scans_col.find_one({'_id': obj_id, 'user_id': user_id})

    if not scan_record:
        flash('Scan details not found or access denied.', 'danger')
        return redirect(url_for('history'))

    scan_record['_id'] = str(scan_record['_id'])
    scan_record['id'] = str(scan_record['_id'])
    scan_record['user_id'] = str(scan_record['user_id'])

    return render_template('scan_details.html', scan=scan_record)


# 11. POST /history/clear -> Clear All User's Scans
@app.route('/history/clear', methods=['POST'])
@login_required
def clear_history():
    try:
        user_id = ObjectId(session['user_id'])
        scans_col = get_scans()
        scans_col.delete_many({'user_id': user_id})
        flash('Scan history cleared successfully.', 'success')
    except Exception as e:
        flash(f'Failed to clear scan history: {str(e)}', 'danger')

    return redirect(url_for('history'))


# 12. GET /api/history -> Filtered & Paginated History API
@app.route('/api/history', methods=['GET'])
@csrf.exempt
@login_required
def api_history():
    try:
        user_id = ObjectId(session['user_id'])
    except Exception:
        return jsonify({'success': False, 'error': 'Invalid session'}), 401

    search = request.args.get('search', '').strip()
    filter_val = request.args.get('filter', 'all').strip()
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    per_page = 10

    query = {'user_id': user_id}

    if search:
        query['submitted_url'] = {'$regex': search, '$options': 'i'}

    if filter_val and filter_val.lower() != 'all':
        filter_lower = filter_val.lower()
        if filter_lower in ['phishing', 'potential phishing']:
            query['prediction'] = 'Potential Phishing'
        elif filter_lower in ['legitimate', 'legitimate-looking']:
            query['prediction'] = 'Legitimate-Looking'
        elif filter_lower in ['uncertain']:
            query['prediction'] = 'Uncertain'
        else:
            query['prediction'] = filter_val

    scans_col = get_scans()
    total_count = scans_col.count_documents(query)
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    skip = (page - 1) * per_page

    cursor = scans_col.find(query).sort('scanned_at', -1).skip(skip).limit(per_page)
    scans_list = []
    for s in cursor:
        scanned_at_val = s.get('scanned_at')
        if isinstance(scanned_at_val, datetime):
            scanned_at_str = scanned_at_val.isoformat()
        else:
            scanned_at_str = str(scanned_at_val) if scanned_at_val else ''

        scans_list.append({
            'id': str(s['_id']),
            '_id': str(s['_id']),
            'submitted_url': s.get('submitted_url', ''),
            'prediction': s.get('prediction', ''),
            'model_name': s.get('model_name', ''),
            'model_score': float(s.get('model_score', 0.0)),
            'scanned_at': scanned_at_str,
            'extracted_features': make_json_safe(s.get('extracted_features', {})),
            'shap_explanation': make_json_safe(s.get('shap_explanation', {}))
        })

    return jsonify({
        'success': True,
        'scans': scans_list,
        'total': total_count,
        'page': page,
        'per_page': per_page,
        'total_pages': total_pages
    })


# 13. GET /profile -> User Profile View
@app.route('/profile')
@login_required
def profile():
    try:
        user_id = ObjectId(session['user_id'])
    except Exception:
        session.clear()
        flash('Invalid session. Please log in again.', 'warning')
        return redirect(url_for('login'))

    users_col = get_users()
    scans_col = get_scans()

    user = users_col.find_one({'_id': user_id})
    if not user:
        session.clear()
        flash('User account not found. Please log in again.', 'warning')
        return redirect(url_for('login'))

    user['_id'] = str(user['_id'])
    user['id'] = str(user['_id'])

    total_scans = scans_col.count_documents({'user_id': user_id})
    stats = {
        'total_scans': total_scans,
        'member_since': user.get('created_at')
    }

    return render_template('profile.html', user=user, stats=stats)


# 14. POST /profile/update -> Update User Profile Details
@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    full_name = request.form.get('full_name', '').strip()
    if not full_name:
        flash('Full name cannot be empty.', 'danger')
        return redirect(url_for('profile'))

    try:
        user_id = ObjectId(session['user_id'])
        users_col = get_users()
        users_col.update_one({'_id': user_id}, {'$set': {'full_name': full_name}})
        session['user_name'] = full_name
        flash('Profile updated successfully.', 'success')
    except Exception as e:
        flash(f'Failed to update profile: {str(e)}', 'danger')

    return redirect(url_for('profile'))


# 15. POST /profile/password -> Change User Password
@app.route('/profile/password', methods=['POST'])
@login_required
def profile_password():
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')

    if not current_password or not new_password or not confirm_password:
        flash('All password fields are required.', 'danger')
        return redirect(url_for('profile'))

    try:
        user_id = ObjectId(session['user_id'])
        users_col = get_users()
        user = users_col.find_one({'_id': user_id})

        if not user or not check_password_hash(user.get('password_hash', ''), current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('profile'))

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('profile'))

        if len(new_password) < 8:
            flash('New password must be at least 8 characters long.', 'danger')
            return redirect(url_for('profile'))

        new_hash = generate_password_hash(new_password)
        users_col.update_one({'_id': user_id}, {'$set': {'password_hash': new_hash}})
        flash('Password updated successfully.', 'success')
    except Exception as e:
        flash(f'Failed to change password: {str(e)}', 'danger')

    return redirect(url_for('profile'))


# ==============================================================================
# Error Handlers
# ==============================================================================
@app.errorhandler(404)
def page_not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Endpoint not found'}), 404
    try:
        return render_template('404.html'), 404
    except Exception:
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>404 - Page Not Found | LinkLens AI</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
                .container { text-align: center; max-width: 500px; padding: 2rem; }
                h1 { font-size: 4rem; margin-bottom: 0.5rem; color: #38bdf8; }
                p { font-size: 1.125rem; color: #94a3b8; margin-bottom: 1.5rem; }
                a { display: inline-block; padding: 0.75rem 1.5rem; background-color: #0284c7; color: white; border-radius: 0.5rem; text-decoration: none; font-weight: 500; }
                a:hover { background-color: #0369a1; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>404</h1>
                <h2>Page Not Found</h2>
                <p>The page you are looking for doesn't exist or has been moved.</p>
                <a href="/">Return Home</a>
            </div>
        </body>
        </html>
        """, 404


@app.errorhandler(500)
def internal_server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    try:
        return render_template('500.html'), 500
    except Exception:
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>500 - Server Error | LinkLens AI</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
                .container { text-align: center; max-width: 500px; padding: 2rem; }
                h1 { font-size: 4rem; margin-bottom: 0.5rem; color: #f43f5e; }
                p { font-size: 1.125rem; color: #94a3b8; margin-bottom: 1.5rem; }
                a { display: inline-block; padding: 0.75rem 1.5rem; background-color: #0284c7; color: white; border-radius: 0.5rem; text-decoration: none; font-weight: 500; }
                a:hover { background-color: #0369a1; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>500</h1>
                <h2>Internal Server Error</h2>
                <p>Something went wrong on our end. Please try again later.</p>
                <a href="/">Return Home</a>
            </div>
        </body>
        </html>
        """, 500


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
