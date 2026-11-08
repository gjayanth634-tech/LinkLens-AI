/**
 * LinkLens AI - Scanner Page JavaScript
 * Handles URL submission, client-side validation, loading animation, and redirection.
 */

/**
 * Retrieves CSRF token from the meta tag
 * @returns {string} The CSRF token value or empty string
 */
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

document.addEventListener('DOMContentLoaded', function () {
    const urlInput = document.getElementById('urlInput');
    const modelSelect = document.getElementById('modelSelect');
    const scanBtn = document.getElementById('scanBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const errorArea = document.getElementById('errorArea');
    const errorMessage = document.getElementById('errorMessage');
    const scannerForm = document.getElementById('scannerForm');

    if (!urlInput || !scanBtn) return;

    /**
     * Display error banner with the provided message
     * @param {string} msg
     */
    function showError(msg) {
        if (!errorArea || !errorMessage) return;
        errorMessage.textContent = msg;
        errorArea.classList.remove('d-none');
        errorArea.classList.add('d-flex');
        errorArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    /**
     * Hide error banner
     */
    function hideError() {
        if (!errorArea || !errorMessage) return;
        errorArea.classList.add('d-none');
        errorArea.classList.remove('d-flex');
        errorMessage.textContent = '';
    }

    /**
     * Display modal loading overlay
     */
    function showLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
            loadingOverlay.classList.add('d-flex');
        }
    }

    /**
     * Hide modal loading overlay
     */
    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
            loadingOverlay.classList.remove('d-flex');
        }
    }

    /**
     * Handle clear button click
     */
    if (clearBtn) {
        clearBtn.addEventListener('click', function () {
            urlInput.value = '';
            hideError();
            urlInput.focus();
        });
    }

    /**
     * Perform the URL scan request
     */
    async function handleScan() {
        hideError();
        const urlValue = urlInput.value.trim();

        // 1. Basic client-side validation
        if (!urlValue) {
            showError('Please enter a URL to scan.');
            urlInput.focus();
            return;
        }

        if (!urlValue.startsWith('http://') && !urlValue.startsWith('https://')) {
            showError('Invalid URL protocol. The URL must begin with http:// or https:// (e.g. https://example.com)');
            urlInput.focus();
            return;
        }

        // Additional sanity check for valid URL syntax
        try {
            new URL(urlValue);
        } catch (e) {
            showError('Invalid URL format. Please enter a valid web address.');
            urlInput.focus();
            return;
        }

        // 2. Show loading overlay and disable scan button
        showLoading();
        scanBtn.disabled = true;
        const originalBtnHtml = scanBtn.innerHTML;
        scanBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Analyzing...';

        try {
            // 3. Send POST request to backend
            const response = await fetch('/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    url: urlValue,
                    model: modelSelect ? modelSelect.value : 'catboost'
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Determine redirect path
                const scanId = data.scan_id || data.id || (data.scan && data.scan._id);
                const redirectUrl = data.redirect || data.redirect_url || (scanId ? `/result/${scanId}` : '/scanner');
                window.location.href = redirectUrl;
            } else {
                hideLoading();
                const errMsg = data.error || data.message || 'An error occurred while analyzing the URL. Please try again.';
                showError(errMsg);
            }
        } catch (err) {
            console.error('Scan request failed:', err);
            hideLoading();
            showError('Unable to connect to the scanning service. Please check your internet connection and try again.');
        } finally {
            scanBtn.disabled = false;
            scanBtn.innerHTML = originalBtnHtml;
        }
    }

    // Attach click listener to scan button
    scanBtn.addEventListener('click', handleScan);

    // Allow Enter key press in input field to submit
    urlInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            handleScan();
        }
    });

    // Form submit listener if wrapped in form
    if (scannerForm) {
        scannerForm.addEventListener('submit', function (e) {
            e.preventDefault();
            handleScan();
        });
    }
});
