/**
 * LinkLens AI - Dashboard JavaScript
 * Renders Chart.js doughnut chart for scan predictions distribution.
 */

document.addEventListener('DOMContentLoaded', function () {
    const chartCanvas = document.getElementById('scanChart');
    if (!chartCanvas) return;

    // Check if scanStats variable exists (set globally or on window)
    const stats = (typeof window.scanStats !== 'undefined') ? window.scanStats : (typeof scanStats !== 'undefined' ? scanStats : null);

    // Helper to render 'No data' message in the chart container
    function renderEmptyChart(canvas) {
        const parent = canvas.parentElement;
        if (parent) {
            canvas.style.display = 'none';
            // Avoid duplicate message
            if (!parent.querySelector('.no-data-chart-message')) {
                const emptyDiv = document.createElement('div');
                emptyDiv.className = 'no-data-chart-message d-flex flex-column align-items-center justify-content-center text-secondary py-5 h-100';
                emptyDiv.innerHTML = `
                    <i class="fa-solid fa-chart-pie fa-3x mb-3 text-secondary opacity-50"></i>
                    <h6 class="text-light fw-semibold mb-1">No scan data yet</h6>
                    <p class="small text-secondary mb-0">Scan URLs to visualize prediction breakdown</p>
                `;
                parent.appendChild(emptyDiv);
            }
        }
    }

    if (!stats) {
        renderEmptyChart(chartCanvas);
        return;
    }

    // Extract statistics counts with flexible key fallbacks
    const phishing = Number(stats.phishing ?? stats.phishing_count ?? 0);
    const legitimate = Number(stats.legitimate ?? stats.legitimate_count ?? 0);
    const uncertain = Number(stats.uncertain ?? stats.uncertain_count ?? 0);
    const total = (typeof stats.total !== 'undefined') 
        ? Number(stats.total) 
        : (typeof stats.total_scans !== 'undefined' ? Number(stats.total_scans) : (phishing + legitimate + uncertain));

    if (total <= 0) {
        renderEmptyChart(chartCanvas);
        return;
    }

    if (typeof Chart === 'undefined') {
        console.warn('Chart.js library is not loaded. Unable to render scan distribution chart.');
        return;
    }

    // Ensure canvas is visible
    chartCanvas.style.display = 'block';

    // Create Chart.js Doughnut Chart
    try {
        new Chart(chartCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Potential Phishing', 'Legitimate-Looking', 'Uncertain'],
                datasets: [{
                    data: [phishing, legitimate, uncertain],
                    backgroundColor: ['#ef4444', '#22c55e', '#f59e0b'],
                    borderColor: '#1a2332',
                    borderWidth: 3,
                    hoverOffset: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '72%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                family: "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
                                size: 13,
                                weight: '500'
                            },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle'
                        }
                    },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#ffffff',
                        bodyColor: '#e5e7eb',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function (context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return ` ${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    } catch (e) {
        console.error('Error creating scanChart:', e);
    }
});
