

// ✅ Chart.js plugin registration (GLOBAL FIX)
if (typeof Chart !== "undefined") {
    if (typeof ChartDataLabels !== "undefined") {
        Chart.register(ChartDataLabels);
    }

    // ✅ Center Text plugin used in donut charts (history.html)
    Chart.register({
        id: "centerText",
        afterDraw(chart) {
            if (!chart || chart.config.type !== "doughnut") return;

            const ctx = chart.ctx;
            ctx.save();
            ctx.font = "bold 16px Poppins";
            ctx.fillStyle = "#000";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const centerX = chart.width / 2;
            const centerY = chart.height / 2;

            ctx.fillText("Overall", centerX, centerY - 10);
            ctx.fillText("Performance", centerX, centerY + 12);
            ctx.restore();
        }
    });
}


document.addEventListener('DOMContentLoaded', () => {

    // ✅ Practice page elements (unchanged)
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusOverlay = document.getElementById('status-overlay');
    const webcamFeed = document.getElementById('webcam-feed');

    // ✅ Prevent errors on history.html (buttons may not exist)
    if (startBtn && stopBtn) {
        startBtn.addEventListener('click', () => {
            statusOverlay.textContent = 'Analyzing...';
            statusOverlay.classList.remove('bg-dark', 'bg-danger');
            statusOverlay.classList.add('bg-success');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        });

        stopBtn.addEventListener('click', () => {
            statusOverlay.textContent = 'Ready';
            statusOverlay.classList.remove('bg-success');
            statusOverlay.classList.add('bg-dark');
            startBtn.disabled = false;
            stopBtn.disabled = true;
        });
    }

    // ✅ Emotion chart ONLY if it exists (practice page)
    const emotionCanvas = document.getElementById('emotion-chart');
    let emotionChart = null;

    if (emotionCanvas && typeof Chart !== "undefined") {
        const ctx = emotionCanvas.getContext('2d');

        emotionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Neutral', 'Happy', 'Surprised', 'Angry', 'Sad'],
                datasets: [{
                    label: 'Emotion Intensity (%)',
                    data: [20, 20, 20, 20, 20],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderColor: 'rgba(0, 0, 0, 0.1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // ✅ Feedback update function (unchanged, just safe)
    window.updateFeedback = function (emotion, vocal, suggestion, emotionData) {
        const e = document.getElementById('emotion-feedback');
        const v = document.getElementById('vocal-feedback');
        const s = document.getElementById('suggestion-feedback');

        if (e) e.textContent = emotion;
        if (v) v.textContent = vocal;
        if (s) s.textContent = suggestion;

        if (emotionChart && emotionData) {
            emotionChart.data.datasets[0].data = emotionData;
            emotionChart.update();
        }
    };

});
