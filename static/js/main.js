/**
 * LinkLens AI - Main Application JavaScript
 * Global utilities, flash message auto-dismiss, active navigation, and helpers.
 */

/**
 * Retrieves CSRF token from the meta tag
 * @returns {string} The CSRF token value or empty string
 */
function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

/**
 * Truncates a URL for display purposes
 * @param {string} url - The URL string to truncate
 * @param {number} [maxLength=60] - Maximum allowed length
 * @returns {string}
 */
function truncateUrl(url, maxLength = 60) {
    if (!url) return '';
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength) + '...';
}

/**
 * Formats a date string or timestamp into a readable date string
 * @param {string|Date} dateInput - The date to format
 * @param {boolean} [includeTime=true] - Whether to include time
 * @returns {string}
 */
function formatDate(dateInput, includeTime = true) {
    if (!dateInput) return '';
    const date = (dateInput instanceof Date) ? dateInput : new Date(dateInput);
    if (isNaN(date.getTime())) return String(dateInput);

    const options = {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    };

    if (includeTime) {
        options.hour = '2-digit';
        options.minute = '2-digit';
    }

    return date.toLocaleDateString(undefined, options);
}

document.addEventListener('DOMContentLoaded', function () {
    // 1. Auto-dismiss flash messages after 5 seconds
    const flashAlerts = document.querySelectorAll('.alert:not(.alert-permanent):not(#errorArea)');
    if (flashAlerts.length > 0) {
        setTimeout(function () {
            flashAlerts.forEach(function (alert) {
                // If Bootstrap Alert JS is available
                if (window.bootstrap && bootstrap.Alert) {
                    try {
                        const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                        bsAlert.close();
                        return;
                    } catch (e) {
                        // Fallback below
                    }
                }
                // Fallback smooth transition
                alert.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                alert.style.opacity = '0';
                alert.style.transform = 'translateY(-10px)';
                setTimeout(function () {
                    alert.remove();
                }, 600);
            });
        }, 5000);
    }

    // 2. Navbar active link highlighting based on current path
    const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link, .nav .nav-link');

    navLinks.forEach(function (link) {
        const href = link.getAttribute('href');
        if (!href || href.startsWith('#') || href.startsWith('javascript:')) return;

        try {
            const linkPath = new URL(link.href, window.location.origin).pathname.replace(/\/+$/, '') || '/';
            if (currentPath === linkPath || (linkPath !== '/' && currentPath.startsWith(linkPath))) {
                link.classList.add('active');
                link.setAttribute('aria-current', 'page');
            } else {
                link.classList.remove('active');
                link.removeAttribute('aria-current');
            }
        } catch (e) {
            // Ignore parsing errors
        }
    });

    // 3. Smooth scroll for internal anchor links
    document.querySelectorAll('a[href^="#"]:not([href="#"]):not([data-bs-toggle])').forEach(function (anchor) {
        anchor.addEventListener('click', function (e) {
            const targetId = this.getAttribute('href');
            if (targetId && targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });

    // 4. Form validation helper for forms marked with .needs-validation
    const forms = document.querySelectorAll('.needs-validation');
    Array.from(forms).forEach(function (form) {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // 5. Global confirmation dialog for dangerous actions via data-confirm attribute
    document.addEventListener('click', function (e) {
        const confirmEl = e.target.closest('[data-confirm]');
        if (confirmEl) {
            const message = confirmEl.getAttribute('data-confirm') || 'Are you sure you want to perform this action?';
            if (!window.confirm(message)) {
                e.preventDefault();
                e.stopImmediatePropagation();
                return false;
            }
        }
    }, true);
});

// Expose utilities on window.LinkLens
window.LinkLens = {
    getCsrfToken: getCsrfToken,
    truncateUrl: truncateUrl,
    formatDate: formatDate
};
