"""Integration test script for LinkLens AI."""
import sys
import re
try:
    import requests
except ImportError:
    print("Installing requests...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

base = 'http://127.0.0.1:5000'
s = requests.Session()

def get_csrf(resp_text):
    """Extract CSRF token from HTML page."""
    match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)', resp_text)
    if match:
        return match.group(1)
    match = re.search(r'name=["\']csrf-token["\'][^>]*content=["\']([^"\']+)', resp_text)
    if match:
        return match.group(1)
    match = re.search(r'content=["\']([^"\']+)["\'][^>]*name=["\']csrf-token["\']', resp_text)
    if match:
        return match.group(1)
    return ''

# 1. Register
print('=== REGISTER ===')
# First GET the register page to get CSRF token
resp = s.get(base + '/register')
csrf = get_csrf(resp.text)
print('Got CSRF token: {}'.format('yes' if csrf else 'no'))

resp = s.post(base + '/register', data={
    'full_name': 'Test User',
    'email': 'testuser@example.com',
    'password': 'TestPass123!',
    'confirm_password': 'TestPass123!',
    'csrf_token': csrf
}, allow_redirects=False)
loc = resp.headers.get('Location', '')
print('Register: {} -> {}'.format(resp.status_code, loc))

# 2. Login
print('=== LOGIN ===')
resp = s.get(base + '/login')
csrf = get_csrf(resp.text)

resp = s.post(base + '/login', data={
    'email': 'testuser@example.com',
    'password': 'TestPass123!',
    'csrf_token': csrf
}, allow_redirects=False)
loc = resp.headers.get('Location', '')
print('Login: {} -> {}'.format(resp.status_code, loc))

# Follow redirect to dashboard
resp = s.get(base + '/dashboard')
print('Dashboard: status={}, length={}'.format(resp.status_code, len(resp.text)))
assert resp.status_code == 200, 'Dashboard failed!'
has_welcome = 'Welcome' in resp.text or 'welcome' in resp.text
print('Dashboard has welcome: {}'.format(has_welcome))

# 3. Scan a legitimate-looking URL
print('\n=== SCAN LEGITIMATE URL ===')
resp = s.post(base + '/scan', json={'url': 'https://www.google.com/search?q=test'})
print('Scan response: {}'.format(resp.status_code))
data = resp.json()
print('Scan data: success={}, scan_id={}'.format(data.get('success'), data.get('scan_id')))
assert data['success'] == True, 'Scan failed!'
scan_id = data['scan_id']

# 4. View result page
print('\n=== VIEW RESULT ===')
resp = s.get(base + '/result/' + scan_id)
print('Result page: status={}, length={}'.format(resp.status_code, len(resp.text)))
assert resp.status_code == 200, 'Result page failed!'
has_pred = 'Legitimate' in resp.text or 'Phishing' in resp.text or 'Uncertain' in resp.text
has_shap = 'SHAP' in resp.text or 'Why This' in resp.text
print('Has prediction display: {}'.format(has_pred))
print('Has SHAP explanation: {}'.format(has_shap))

# 5. Scan a suspicious-looking URL
print('\n=== SCAN SUSPICIOUS URL ===')
resp = s.post(base + '/scan', json={'url': 'http://192.168.1.1/secure-login-verify/account/update?user=admin'})
data = resp.json()
print('Suspicious scan: success={}, scan_id={}'.format(data.get('success'), data.get('scan_id')))
assert data['success'] == True, 'Suspicious scan failed!'

# 6. Check history
print('\n=== HISTORY ===')
resp = s.get(base + '/history')
print('History page: status={}, length={}'.format(resp.status_code, len(resp.text)))
assert resp.status_code == 200, 'History page failed!'

# 7. Check dashboard stats updated
print('\n=== DASHBOARD STATS ===')
resp = s.get(base + '/dashboard')
print('Dashboard: status={}'.format(resp.status_code))
assert resp.status_code == 200
has_stats = 'Total Scans' in resp.text
print('Has stats cards: {}'.format(has_stats))

# 8. Profile page
print('\n=== PROFILE ===')
resp = s.get(base + '/profile')
print('Profile: status={}'.format(resp.status_code))
assert resp.status_code == 200, 'Profile page failed!'
has_info = 'testuser@example.com' in resp.text or 'Test User' in resp.text
print('Has user info: {}'.format(has_info))

# 9. Scanner page
print('\n=== SCANNER ===')
resp = s.get(base + '/scanner')
print('Scanner: status={}'.format(resp.status_code))
assert resp.status_code == 200, 'Scanner page failed!'

# 10. Test URL validation (invalid URL)
print('\n=== URL VALIDATION ===')
resp = s.post(base + '/scan', json={'url': 'not-a-url'})
data = resp.json()
print('Invalid URL response: success={}, error={}'.format(data.get('success'), data.get('error', '')[:60]))
assert data['success'] == False, 'Invalid URL should fail!'

# 11. Logout
print('\n=== LOGOUT ===')
resp = s.get(base + '/logout', allow_redirects=False)
loc = resp.headers.get('Location', '')
print('Logout: {} -> {}'.format(resp.status_code, loc))

# Verify cant access dashboard after logout
resp = s.get(base + '/dashboard', allow_redirects=False)
print('Dashboard after logout: {} (should be 302)'.format(resp.status_code))
assert resp.status_code == 302, 'Should redirect after logout!'

print('\n' + '=' * 50)
print('  ALL INTEGRATION TESTS PASSED SUCCESSFULLY!')
print('=' * 50)
