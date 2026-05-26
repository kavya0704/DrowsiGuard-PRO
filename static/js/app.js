/**
 * DrowsiGuard PRO — AI Driver Drowsiness Detection System
 * Frontend Dashboard Logic — Hybrid Desktop/Mobile
 * 
 * PRO Features:
 * - Real-time EAR/MAR chart (canvas-based, 60s rolling window)
 * - Session summary stats (duration, alerts, blink rate, avg fatigue)
 * - Alert history log (last 8 events with timestamps)
 * - Dark/Light theme toggle
 * - Enhanced micro-animations
 */

// ─── Device Detection ───────────────────────────────────────────
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) 
    || (navigator.maxTouchPoints > 1 && window.innerWidth < 1024);

// ─── Constants ──────────────────────────────────────────────────
const POLL_INTERVAL = 200;
const CHART_HISTORY_SECONDS = 60;
const CHART_SAMPLE_MS = 200;
const MAX_ALERT_HISTORY = 8;
const API = {
    STATUS: '/status',
    START: '/start',
    STOP: '/stop',
    TOGGLE_SOUND: '/toggle_sound',
    TOGGLE_PRIVACY: '/toggle_privacy',
    UPDATE_THRESHOLDS: '/update_thresholds',
    VIDEO_FEED: '/video_feed'
};

// Landmark indices (same as config.py)
const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
const LEFT_EYE = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78];
const MOUTH_TOP = 13, MOUTH_BOTTOM = 14, MOUTH_LEFT = 78, MOUTH_RIGHT = 308;
const MOUTH_TOP_INNER = [81, 311], MOUTH_BOTTOM_INNER = [178, 402];

// Thresholds
let EAR_THRESHOLD = 0.22;
let MAR_THRESHOLD = 0.65;
let EAR_CONSEC_FRAMES = 15;
let MAR_CONSEC_FRAMES = 10;

// ─── DOM References ─────────────────────────────────────────────
const dom = {
    videoFeed: document.getElementById('videoFeed'),
    mobileVideo: document.getElementById('mobileVideo'),
    mobileCanvas: document.getElementById('mobileCanvas'),
    videoPlaceholder: document.getElementById('videoPlaceholder'),
    placeholderText: document.getElementById('placeholderText'),
    videoContainer: document.getElementById('videoContainer'),
    liveBadge: document.getElementById('liveBadge'),
    modeBadge: document.getElementById('modeBadge'),
    modeIcon: document.getElementById('modeIcon'),
    modeText: document.getElementById('modeText'),
    cameraUrl: document.getElementById('cameraUrl'),
    cameraUrlGroup: document.getElementById('cameraUrlGroup'),
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    pasteBtn: document.getElementById('pasteBtn'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    fpsValue: document.getElementById('fpsValue'),
    uptimeValue: document.getElementById('uptimeValue'),
    eyeCard: document.getElementById('eyeCard'),
    eyeIndicator: document.getElementById('eyeIndicator'),
    eyeStatusText: document.getElementById('eyeStatusText'),
    earValue: document.getElementById('earValue'),
    earBar: document.getElementById('earBar'),
    blinkCount: document.getElementById('blinkCount'),
    mouthCard: document.getElementById('mouthCard'),
    mouthIndicator: document.getElementById('mouthIndicator'),
    mouthStatusText: document.getElementById('mouthStatusText'),
    marValue: document.getElementById('marValue'),
    marBar: document.getElementById('marBar'),
    yawnCount: document.getElementById('yawnCount'),
    fatigueCard: document.getElementById('fatigueCard'),
    ringFill: document.getElementById('ringFill'),
    fatiguePercent: document.getElementById('fatiguePercent'),
    fatigueLabel: document.getElementById('fatigueLabel'),
    soundToggle: document.getElementById('soundToggle'),
    privacyToggle: document.getElementById('privacyToggle'),
    faceStatus: document.getElementById('faceStatus'),
    earThreshold: document.getElementById('earThreshold'),
    marThreshold: document.getElementById('marThreshold'),
    earFrames: document.getElementById('earFrames'),
    marFrames: document.getElementById('marFrames'),
    earThresholdVal: document.getElementById('earThresholdVal'),
    marThresholdVal: document.getElementById('marThresholdVal'),
    earFramesVal: document.getElementById('earFramesVal'),
    marFramesVal: document.getElementById('marFramesVal'),
    applyThresholds: document.getElementById('applyThresholds'),
    alertOverlay: document.getElementById('alertOverlay'),
    // PRO elements
    themeToggle: document.getElementById('themeToggle'),
    realtimeChart: document.getElementById('realtimeChart'),
    sessionDuration: document.getElementById('sessionDuration'),
    sessionAlerts: document.getElementById('sessionAlerts'),
    sessionBlinkRate: document.getElementById('sessionBlinkRate'),
    sessionAvgFatigue: document.getElementById('sessionAvgFatigue'),
    alertHistoryList: document.getElementById('alertHistoryList'),
    alertHistoryEmpty: document.getElementById('alertHistoryEmpty'),
    // Copilot elements
    copilotTtsToggle: document.getElementById('copilotTtsToggle'),
    chatMessages: document.getElementById('chatMessages'),
    copilotInput: document.getElementById('copilotInput'),
    copilotSendBtn: document.getElementById('copilotSendBtn'),
    copilotApiKey: document.getElementById('copilotApiKey'),
    copilotModel: document.getElementById('copilotModel'),
    copilotBaseUrl: document.getElementById('copilotBaseUrl'),
    saveCopilotSettings: document.getElementById('saveCopilotSettings'),
    copilotChatContainer: document.getElementById('copilotChatContainer')
};

// ─── State ──────────────────────────────────────────────────────
let pollingTimer = null;
let isRunning = false;
let mobileStream = null;
let faceLandmarker = null;
let mobileAnimFrame = null;
let mobileAlarmAudio = null;

// Mobile detection state
let closedFrames = 0;
let yawnFrames = 0;
let blinkCount = 0;
let yawnCount = 0;
let startTime = null;
let fpsCounter = 0;
let lastFpsTime = 0;
let currentFps = 0;

// ─── PRO State ──────────────────────────────────────────────────
let chartData = { ear: [], mar: [], timestamps: [] };
let alertHistory = [];
let sessionAlertCount = 0;
let fatigueAccumulator = [];
let lastDrowsyState = false;
let lastYawnState = false;
let chartAnimFrame = null;

// AI Copilot state
let copilotMessages = [];
let lastAutoAlertTime = 0;

// ─── API Helpers ────────────────────────────────────────────────
async function apiGet(url) {
    try {
        const res = await fetch(url);
        return await res.json();
    } catch (e) { return null; }
}

async function apiPost(url, data = {}) {
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return await res.json();
    } catch (e) { return null; }
}

// ─── Utilities ──────────────────────────────────────────────────
function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatAlertTime(date) {
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function euclideanDist(p1, p2) {
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}

function calcEAR(landmarks, indices) {
    const pts = indices.map(i => landmarks[i]);
    const v1 = euclideanDist(pts[1], pts[5]);
    const v2 = euclideanDist(pts[2], pts[4]);
    const h = euclideanDist(pts[0], pts[3]);
    if (h === 0) return 0;
    return (v1 + v2) / (2.0 * h);
}

function calcMAR(landmarks) {
    const top = landmarks[MOUTH_TOP];
    const bottom = landmarks[MOUTH_BOTTOM];
    const left = landmarks[MOUTH_LEFT];
    const right = landmarks[MOUTH_RIGHT];
    const it = MOUTH_TOP_INNER.map(i => landmarks[i]);
    const ib = MOUTH_BOTTOM_INNER.map(i => landmarks[i]);
    const v1 = euclideanDist(top, bottom);
    const v2 = euclideanDist(it[0], ib[0]);
    const v3 = euclideanDist(it[1], ib[1]);
    const h = euclideanDist(left, right);
    if (h === 0) return 0;
    return (v1 + v2 + v3) / (3.0 * h);
}

// ═══════════════════════════════════════════════════════════════
//  PRO: Real-Time Chart Engine
// ═══════════════════════════════════════════════════════════════
function addChartDataPoint(ear, mar) {
    const now = Date.now();
    chartData.ear.push(ear);
    chartData.mar.push(mar);
    chartData.timestamps.push(now);

    // Trim to last 60 seconds
    const cutoff = now - (CHART_HISTORY_SECONDS * 1000);
    while (chartData.timestamps.length > 0 && chartData.timestamps[0] < cutoff) {
        chartData.ear.shift();
        chartData.mar.shift();
        chartData.timestamps.shift();
    }
}

function drawRealtimeChart() {
    const canvas = dom.realtimeChart;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background grid
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    const gridColor = isDark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.05)';
    const textColor = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.25)';

    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = textColor;
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    const yLabels = ['1.0', '0.75', '0.5', '0.25', '0'];
    for (let i = 0; i <= 4; i++) {
        ctx.fillText(yLabels[i], 2, (h / 4) * i + 10);
    }

    if (chartData.ear.length < 2) return;

    // EAR threshold line
    const earThLineY = h - (EAR_THRESHOLD / 1.0) * h;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255, 211, 42, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, earThLineY);
    ctx.lineTo(w, earThLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    // MAR threshold line
    const marThLineY = h - (MAR_THRESHOLD / 1.0) * h;
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(255, 211, 42, 0.25)';
    ctx.beginPath();
    ctx.moveTo(0, marThLineY);
    ctx.lineTo(w, marThLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    const len = chartData.ear.length;
    const stepX = w / Math.max(len - 1, 1);

    // Draw EAR line
    drawSmoothLine(ctx, chartData.ear, stepX, w, h, '#00d4ff', 'rgba(0, 212, 255, 0.08)');
    // Draw MAR line
    drawSmoothLine(ctx, chartData.mar, stepX, w, h, '#ff9f43', 'rgba(255, 159, 67, 0.06)');
}

function drawSmoothLine(ctx, data, stepX, w, h, color, fillColor) {
    if (data.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    for (let i = 0; i < data.length; i++) {
        const x = i * stepX;
        const y = h - (Math.min(data[i], 1.0) / 1.0) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill gradient under line
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, fillColor);
    grad.addColorStop(1, 'transparent');

    ctx.lineTo((data.length - 1) * stepX, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
}

function startChartAnimation() {
    function animate() {
        drawRealtimeChart();
        chartAnimFrame = requestAnimationFrame(animate);
    }
    animate();
}

function stopChartAnimation() {
    if (chartAnimFrame) {
        cancelAnimationFrame(chartAnimFrame);
        chartAnimFrame = null;
    }
}

// ═══════════════════════════════════════════════════════════════
//  PRO: Session Summary Tracker
// ═══════════════════════════════════════════════════════════════
function updateSessionSummary(data) {
    if (!data) return;

    // Duration
    const uptime = data.uptime || 0;
    dom.sessionDuration.textContent = formatUptime(uptime);

    // Alert count
    dom.sessionAlerts.textContent = sessionAlertCount;

    // Blink rate (blinks per minute)
    const minutes = uptime / 60;
    const blinkRate = minutes > 0 ? ((data.blink_count || 0) / minutes).toFixed(1) : '0';
    dom.sessionBlinkRate.textContent = blinkRate + ' /min';

    // Average fatigue
    if (data.fatigue_level !== undefined) {
        fatigueAccumulator.push(data.fatigue_level);
        // Keep max 300 samples (60 seconds at 5Hz)
        if (fatigueAccumulator.length > 300) fatigueAccumulator.shift();
    }
    const avgFatigue = fatigueAccumulator.length > 0
        ? Math.round(fatigueAccumulator.reduce((a, b) => a + b, 0) / fatigueAccumulator.length)
        : 0;
    dom.sessionAvgFatigue.textContent = avgFatigue + '%';
}

// ═══════════════════════════════════════════════════════════════
//  PRO: Alert History Manager
// ═══════════════════════════════════════════════════════════════
function checkAndLogAlerts(data) {
    if (!data) return;

    // Detect rising edge of drowsy alert
    if (data.drowsy_alert && !lastDrowsyState) {
        addAlertToHistory('drowsy', '😴 Drowsiness Detected');
        sessionAlertCount++;
    }

    // Detect rising edge of yawn alert
    if (data.yawn_alert && !lastYawnState) {
        addAlertToHistory('yawn', '🥱 Yawning Detected');
        sessionAlertCount++;
    }

    lastDrowsyState = data.drowsy_alert || false;
    lastYawnState = data.yawn_alert || false;
}

function addAlertToHistory(type, message) {
    const now = new Date();
    alertHistory.unshift({ type, message, time: now });
    if (alertHistory.length > MAX_ALERT_HISTORY) alertHistory.pop();
    renderAlertHistory();
}

function renderAlertHistory() {
    if (!dom.alertHistoryList) return;

    if (alertHistory.length === 0) {
        dom.alertHistoryList.innerHTML = '<div class="alert-history-empty">No alerts yet</div>';
        return;
    }

    dom.alertHistoryList.innerHTML = alertHistory.map(a => `
        <div class="alert-history-item ${a.type === 'yawn' ? 'yawn' : ''}">
            <span class="alert-time">${formatAlertTime(a.time)}</span>
            <span class="alert-msg">${a.message}</span>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════════════════════
//  PRO: Theme Toggle
// ═══════════════════════════════════════════════════════════════
function initTheme() {
    const saved = localStorage.getItem('drowsiguard-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('drowsiguard-theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    if (dom.themeToggle) {
        dom.themeToggle.textContent = theme === 'dark' ? '🌙' : '☀️';
    }
}

// ─── Update UI from Data ────────────────────────────────────────
function updateUI(data) {
    if (!data) return;

    // Header Status
    if (data.running) {
        if (data.alert_active) {
            dom.statusDot.className = 'status-dot alert';
            dom.statusText.textContent = 'ALERT!';
            dom.statusText.style.color = 'var(--accent-red)';
        } else {
            dom.statusDot.className = 'status-dot online';
            dom.statusText.textContent = 'Monitoring';
            dom.statusText.style.color = 'var(--accent-green)';
        }
    } else {
        dom.statusDot.className = 'status-dot offline';
        dom.statusText.textContent = 'Offline';
        dom.statusText.style.color = 'var(--text-muted)';
    }

    dom.fpsValue.textContent = data.fps || 0;
    dom.uptimeValue.textContent = formatUptime(data.uptime || 0);

    // Eye
    dom.earValue.textContent = (data.ear || 0).toFixed(3);
    dom.blinkCount.textContent = data.blink_count || 0;
    dom.earBar.style.width = Math.min(100, ((data.ear || 0) / 0.5) * 100) + '%';

    if (data.eye_status === 'OPEN') {
        dom.eyeIndicator.className = 'status-indicator open';
        dom.eyeStatusText.textContent = 'OPEN';
        dom.eyeStatusText.style.color = 'var(--accent-green)';
        dom.eyeCard.classList.remove('alert');
    } else if (data.eye_status === 'CLOSED') {
        dom.eyeIndicator.className = 'status-indicator closed';
        dom.eyeStatusText.textContent = 'CLOSED';
        dom.eyeStatusText.style.color = 'var(--accent-red)';
        if (data.drowsy_alert) dom.eyeCard.classList.add('alert');
    } else {
        dom.eyeIndicator.className = 'status-indicator inactive';
        dom.eyeStatusText.textContent = '\u2014';
        dom.eyeStatusText.style.color = 'var(--text-muted)';
        dom.eyeCard.classList.remove('alert');
    }

    // Mouth
    dom.marValue.textContent = (data.mar || 0).toFixed(3);
    dom.yawnCount.textContent = data.yawn_count || 0;
    dom.marBar.style.width = Math.min(100, ((data.mar || 0) / 1.0) * 100) + '%';

    if (data.mouth_status === 'YAWNING') {
        dom.mouthIndicator.className = 'status-indicator yawn';
        dom.mouthStatusText.textContent = 'YAWNING';
        dom.mouthStatusText.style.color = 'var(--accent-orange)';
        if (data.yawn_alert) dom.mouthCard.classList.add('warning');
    } else if (data.mouth_status === 'CLOSED' || data.mouth_status === 'NORMAL') {
        dom.mouthIndicator.className = 'status-indicator open';
        dom.mouthStatusText.textContent = 'NORMAL';
        dom.mouthStatusText.style.color = 'var(--accent-green)';
        dom.mouthCard.classList.remove('warning');
    } else {
        dom.mouthIndicator.className = 'status-indicator inactive';
        dom.mouthStatusText.textContent = '\u2014';
        dom.mouthStatusText.style.color = 'var(--text-muted)';
        dom.mouthCard.classList.remove('warning');
    }

    // Fatigue
    const fatigue = data.fatigue_level || 0;
    const circumference = 2 * Math.PI * 52;
    const offset = circumference * (1 - fatigue / 100);
    dom.ringFill.style.strokeDashoffset = offset;
    dom.fatiguePercent.textContent = fatigue + '%';

    let fatigueColor, fatigueText;
    if (fatigue < 30) { fatigueColor = 'var(--accent-green)'; fatigueText = 'ALERT'; }
    else if (fatigue < 60) { fatigueColor = 'var(--accent-yellow)'; fatigueText = 'MODERATE'; }
    else if (fatigue < 80) { fatigueColor = 'var(--accent-orange)'; fatigueText = 'DROWSY'; }
    else { fatigueColor = 'var(--accent-red)'; fatigueText = 'DANGER!'; }

    dom.ringFill.style.stroke = fatigueColor;
    dom.fatiguePercent.style.color = fatigueColor;
    dom.fatigueLabel.style.color = fatigueColor;
    dom.fatigueLabel.textContent = fatigueText;
    dom.fatigueCard.classList.toggle('alert', fatigue >= 60);

    // Face detect
    if (data.face_detected) {
        dom.faceStatus.textContent = 'Yes \u2713';
        dom.faceStatus.className = 'face-status detected';
    } else {
        dom.faceStatus.textContent = 'No \u2717';
        dom.faceStatus.className = 'face-status not-detected';
    }

    // Alert overlay
    if (data.drowsy_alert) {
        dom.alertOverlay.classList.remove('hidden');
        dom.videoContainer.classList.add('alert-active');
    } else {
        dom.alertOverlay.classList.add('hidden');
        dom.videoContainer.classList.remove('alert-active');
    }

    // PRO: Chart data
    addChartDataPoint(data.ear || 0, data.mar || 0);

    // PRO: Alert history
    checkAndLogAlerts(data);

    // PRO: Session summary
    updateSessionSummary(data);

    // Auto warning if fatigue is high
    if (data.fatigue_level >= 70) {
        triggerCopilotAutoWarning(data.fatigue_level);
    }
}

// ═══════════════════════════════════════════════════════════════
//  DESKTOP MODE — Server-side processing (existing behavior)
// ═══════════════════════════════════════════════════════════════
function startDesktopPolling() {
    if (pollingTimer) return;
    async function poll() {
        const data = await apiGet(API.STATUS);
        updateUI(data);
        if (data && !data.running && isRunning) setStoppedState();
    }
    poll();
    pollingTimer = setInterval(poll, POLL_INTERVAL);
}

function stopDesktopPolling() {
    if (pollingTimer) { clearInterval(pollingTimer); pollingTimer = null; }
}

async function startDesktop() {
    const cameraSource = dom.cameraUrl.value.trim();
    const result = await apiPost(API.START, { camera_source: cameraSource });
    if (result) {
        dom.videoFeed.classList.remove('hidden');
        dom.videoFeed.src = API.VIDEO_FEED + '?t=' + Date.now();
        setRunningState();
        startDesktopPolling();
    } else {
        alert('Failed to start detection. Check the server console.');
        dom.startBtn.disabled = false;
        dom.startBtn.innerHTML = '<span class="btn-icon-left">\u25b6</span> Start Detection';
    }
}

async function stopDesktop() {
    await apiPost(API.STOP);
    stopDesktopPolling();
}

// ═══════════════════════════════════════════════════════════════
//  MOBILE MODE — Client-side camera + MediaPipe JS
// ═══════════════════════════════════════════════════════════════

async function initMobileMediaPipe() {
    try {
        const { FaceLandmarker, FilesetResolver } = await import(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
        );

        const filesetResolver = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        console.log('[Mobile] MediaPipe FaceLandmarker initialized');
        return true;
    } catch (err) {
        console.error('[Mobile] MediaPipe init failed:', err);
        // Fallback: try CPU delegate
        try {
            const { FaceLandmarker, FilesetResolver } = await import(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
            );
            const filesetResolver = await FilesetResolver.forVisionTasks(
                'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
            );
            faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
                baseOptions: {
                    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                    delegate: 'CPU'
                },
                runningMode: 'VIDEO',
                numFaces: 1
            });
            console.log('[Mobile] MediaPipe FaceLandmarker initialized (CPU fallback)');
            return true;
        } catch (err2) {
            console.error('[Mobile] MediaPipe CPU fallback also failed:', err2);
            return false;
        }
    }
}

async function startMobileCamera() {
    try {
        mobileStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        dom.mobileVideo.srcObject = mobileStream;
        await dom.mobileVideo.play();
        console.log('[Mobile] Camera started');
        return true;
    } catch (err) {
        console.error('[Mobile] Camera access denied:', err);
        alert('Camera access denied. Please allow camera access and try again.');
        return false;
    }
}

function stopMobileCamera() {
    if (mobileAnimFrame) { cancelAnimationFrame(mobileAnimFrame); mobileAnimFrame = null; }
    if (mobileStream) {
        mobileStream.getTracks().forEach(t => t.stop());
        mobileStream = null;
    }
    dom.mobileVideo.srcObject = null;
    stopMobileAlarm();
}

// Mobile alarm using Web Audio API
let audioCtx = null;
let alarmOsc = null;
let alarmGain = null;
let mobileAlarmPlaying = false;

function playMobileAlarm() {
    if (mobileAlarmPlaying || !dom.soundToggle.checked) return;
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        alarmOsc = audioCtx.createOscillator();
        alarmGain = audioCtx.createGain();
        alarmOsc.type = 'square';
        alarmOsc.frequency.value = 880;
        alarmGain.gain.value = 0.3;
        alarmOsc.connect(alarmGain);
        alarmGain.connect(audioCtx.destination);
        alarmOsc.start();
        mobileAlarmPlaying = true;

        // Alternate frequency for urgency
        let high = true;
        window._alarmInterval = setInterval(() => {
            if (alarmOsc) {
                alarmOsc.frequency.value = high ? 660 : 880;
                high = !high;
            }
        }, 150);

        // Vibrate on mobile
        if (navigator.vibrate) navigator.vibrate([300, 100, 300, 100, 300]);
    } catch (e) {
        console.error('Mobile alarm error:', e);
    }
}

function stopMobileAlarm() {
    if (window._alarmInterval) { clearInterval(window._alarmInterval); window._alarmInterval = null; }
    if (alarmOsc) { try { alarmOsc.stop(); } catch(e){} alarmOsc = null; }
    alarmGain = null;
    mobileAlarmPlaying = false;
    if (navigator.vibrate) navigator.vibrate(0);
}

function mobileDetectionLoop() {
    if (!isRunning || !faceLandmarker) return;

    const video = dom.mobileVideo;
    const canvas = dom.mobileCanvas;
    const ctx = canvas.getContext('2d');

    if (video.readyState < 2) {
        mobileAnimFrame = requestAnimationFrame(mobileDetectionLoop);
        return;
    }

    // Match canvas to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const now = performance.now();
    const results = faceLandmarker.detectForVideo(video, now);

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mirror canvas to match mirrored video
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    let ear = 0, mar = 0;
    let eyeStatus = 'N/A', mouthStatus = 'N/A';
    let faceDetected = false;
    let drowsyAlert = false, yawnAlert = false;

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        faceDetected = true;
        const lm = results.faceLandmarks[0];

        // Calculate EAR
        const leftEar = calcEAR(lm, LEFT_EYE);
        const rightEar = calcEAR(lm, RIGHT_EYE);
        ear = (leftEar + rightEar) / 2;

        // Calculate MAR
        mar = calcMAR(lm);

        // Eye state
        if (ear < EAR_THRESHOLD) {
            closedFrames++;
            eyeStatus = 'CLOSED';
        } else {
            if (closedFrames > 2) blinkCount++;
            closedFrames = 0;
            eyeStatus = 'OPEN';
        }

        // Yawn state
        if (mar > MAR_THRESHOLD) {
            yawnFrames++;
            mouthStatus = 'YAWNING';
        } else {
            if (yawnFrames > MAR_CONSEC_FRAMES) yawnCount++;
            yawnFrames = 0;
            mouthStatus = 'NORMAL';
        }

        drowsyAlert = closedFrames >= EAR_CONSEC_FRAMES;
        yawnAlert = yawnFrames >= MAR_CONSEC_FRAMES;

        // Fatigue
        const eyeFatigue = Math.min(100, (closedFrames / EAR_CONSEC_FRAMES) * 60);
        const yawnFatigue = Math.min(40, yawnCount * 10);
        const fatigueLevel = Math.min(100, Math.floor(eyeFatigue + yawnFatigue));

        // Alarm
        if (drowsyAlert || yawnAlert) {
            playMobileAlarm();
        } else {
            stopMobileAlarm();
        }

        // Draw landmarks on canvas
        const eyeColor = eyeStatus === 'OPEN' ? '#00ff88' : '#ff3366';
        const mouthColor = mouthStatus === 'NORMAL' ? '#00ff88' : '#ff9f43';

        // Draw eyes
        drawLandmarkContour(ctx, lm, RIGHT_EYE_FULL, eyeColor, canvas.width, canvas.height);
        drawLandmarkContour(ctx, lm, LEFT_EYE_FULL, eyeColor, canvas.width, canvas.height);
        drawLandmarkContour(ctx, lm, LIPS, mouthColor, canvas.width, canvas.height);

        // Eye center dots
        for (const eyeIdx of [RIGHT_EYE, LEFT_EYE]) {
            const cx = eyeIdx.reduce((s, i) => s + lm[i].x, 0) / eyeIdx.length * canvas.width;
            const cy = eyeIdx.reduce((s, i) => s + lm[i].y, 0) / eyeIdx.length * canvas.height;
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
            ctx.fillStyle = eyeColor;
            ctx.fill();
        }

        // FPS
        fpsCounter++;
        if (now - lastFpsTime > 1000) {
            currentFps = Math.round(fpsCounter * 1000 / (now - lastFpsTime));
            fpsCounter = 0;
            lastFpsTime = now;
        }

        // Update UI
        const uptime = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
        updateUI({
            running: true,
            face_detected: true,
            ear, mar,
            eye_status: eyeStatus,
            mouth_status: mouthStatus,
            fatigue_level: fatigueLevel,
            alert_active: drowsyAlert || yawnAlert,
            drowsy_alert: drowsyAlert,
            yawn_alert: yawnAlert,
            blink_count: blinkCount,
            yawn_count: yawnCount,
            fps: currentFps,
            uptime
        });
    } else {
        // No face
        closedFrames = 0;
        yawnFrames = 0;
        stopMobileAlarm();

        fpsCounter++;
        if (now - lastFpsTime > 1000) {
            currentFps = Math.round(fpsCounter * 1000 / (now - lastFpsTime));
            fpsCounter = 0;
            lastFpsTime = now;
        }

        const uptime = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
        updateUI({
            running: true,
            face_detected: false,
            ear: 0, mar: 0,
            eye_status: 'N/A', mouth_status: 'N/A',
            fatigue_level: 0,
            alert_active: false, drowsy_alert: false, yawn_alert: false,
            blink_count: blinkCount, yawn_count: yawnCount,
            fps: currentFps, uptime
        });
    }

    ctx.restore();
    mobileAnimFrame = requestAnimationFrame(mobileDetectionLoop);
}

function drawLandmarkContour(ctx, landmarks, indices, color, w, h) {
    if (indices.length < 2) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    const first = landmarks[indices[0]];
    ctx.moveTo(first.x * w, first.y * h);
    for (let i = 1; i < indices.length; i++) {
        const p = landmarks[indices[i]];
        ctx.lineTo(p.x * w, p.y * h);
    }
    ctx.closePath();
    ctx.stroke();
}

async function startMobile() {
    dom.startBtn.innerHTML = '\u23f3 Loading AI model...';

    // Init MediaPipe
    const mpReady = await initMobileMediaPipe();
    if (!mpReady) {
        alert('Failed to load AI model. Please check your internet connection.');
        dom.startBtn.disabled = false;
        dom.startBtn.innerHTML = '<span class="btn-icon-left">\u25b6</span> Start Detection';
        return;
    }

    // Start camera
    dom.startBtn.innerHTML = '\u23f3 Opening camera...';
    const camReady = await startMobileCamera();
    if (!camReady) {
        dom.startBtn.disabled = false;
        dom.startBtn.innerHTML = '<span class="btn-icon-left">\u25b6</span> Start Detection';
        return;
    }

    // Show mobile video + canvas, hide server feed
    dom.videoFeed.classList.add('hidden');
    dom.mobileVideo.classList.remove('hidden');
    dom.mobileCanvas.classList.remove('hidden');

    // Reset counters
    closedFrames = 0;
    yawnFrames = 0;
    blinkCount = 0;
    yawnCount = 0;
    startTime = Date.now();
    lastFpsTime = performance.now();
    fpsCounter = 0;

    setRunningState();

    // Start detection loop
    mobileDetectionLoop();
}

function stopMobile() {
    stopMobileCamera();
    stopMobileAlarm();
    dom.mobileVideo.classList.add('hidden');
    dom.mobileCanvas.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════
//  PRO: AI Copilot Assistant Engine
// ═══════════════════════════════════════════════════════════════

// Default configs
const COPILOT_DEFAULTS = {
    apiKey: "",
    model: "openai/gpt-oss-120b",
    baseUrl: "https://api.groq.com/openai/v1"
};

function initCopilot() {
    // Load settings from local storage or use defaults
    const key = localStorage.getItem('drowsiguard-copilot-key') || COPILOT_DEFAULTS.apiKey;
    const model = localStorage.getItem('drowsiguard-copilot-model') || COPILOT_DEFAULTS.model;
    const url = localStorage.getItem('drowsiguard-copilot-url') || COPILOT_DEFAULTS.baseUrl;
    const tts = localStorage.getItem('drowsiguard-copilot-tts') !== 'false';

    if (dom.copilotApiKey) dom.copilotApiKey.value = key;
    if (dom.copilotModel) dom.copilotModel.value = model;
    if (dom.copilotBaseUrl) dom.copilotBaseUrl.value = url;
    if (dom.copilotTtsToggle) dom.copilotTtsToggle.checked = tts;

    // Bind event listeners
    if (dom.copilotSendBtn) {
        dom.copilotSendBtn.addEventListener('click', sendCopilotMessage);
    }
    if (dom.copilotInput) {
        dom.copilotInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') sendCopilotMessage();
        });
    }
    if (dom.saveCopilotSettings) {
        dom.saveCopilotSettings.addEventListener('click', saveCopilotSettingsFn);
    }
    if (dom.copilotTtsToggle) {
        dom.copilotTtsToggle.addEventListener('change', () => {
            localStorage.setItem('drowsiguard-copilot-tts', dom.copilotTtsToggle.checked);
        });
    }
}

function speakText(text) {
    if (!dom.copilotTtsToggle || !dom.copilotTtsToggle.checked) return;
    if (!window.speechSynthesis) return;

    // Cancel currently speaking
    window.speechSynthesis.cancel();

    // Remove emojis or special symbols for cleaner TTS
    const cleanText = text.replace(/[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF]/g, "")
                          .replace(/[⚠️🥱😴🤖👄👁️⚡📊⚙️🔧🔊📢]/g, "");

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

function addChatMessage(sender, text, type) {
    if (!dom.chatMessages) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-message ${type}`;
    msgDiv.innerHTML = `
        <span class="message-sender">${sender}</span>
        <p class="message-text">${text}</p>
    `;
    dom.chatMessages.appendChild(msgDiv);

    // Auto-scroll
    if (dom.copilotChatContainer) {
        dom.copilotChatContainer.scrollTop = dom.copilotChatContainer.scrollHeight;
    }
}

async function sendCopilotMessage() {
    const input = dom.copilotInput;
    if (!input) return;

    const text = input.value.trim();
    if (!text) return;

    input.value = '';
    addChatMessage('👤 Driver', text, 'user');

    // Add user message to history
    copilotMessages.push({ role: 'user', content: text });

    // Show thinking indicator
    const thinkingId = 'thinking_' + Date.now();
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'chat-message ai thinking';
    thinkingDiv.id = thinkingId;
    thinkingDiv.innerHTML = `
        <span class="message-sender">🤖 Copilot</span>
        <p class="message-text">Thinking...</p>
    `;
    dom.chatMessages.appendChild(thinkingDiv);
    if (dom.copilotChatContainer) {
        dom.copilotChatContainer.scrollTop = dom.copilotChatContainer.scrollHeight;
    }

    // Disable input
    input.disabled = true;
    if (dom.copilotSendBtn) dom.copilotSendBtn.disabled = true;

    // Fetch config
    const key = dom.copilotApiKey ? dom.copilotApiKey.value : '';
    const model = dom.copilotModel ? dom.copilotModel.value : '';
    const baseUrl = dom.copilotBaseUrl ? dom.copilotBaseUrl.value : '';

    try {
        let aiText = '';
        
        // If not isMobile (i.e. Desktop/Flask mode), call our python backend proxy
        if (!isMobile) {
            const res = await apiPost('/api/chat', {
                messages: copilotMessages,
                api_key: key,
                model: model,
                base_url: baseUrl
            });
            if (res && res.choices && res.choices.length > 0) {
                aiText = res.choices[0].message.content;
            } else if (res && res.error) {
                throw new Error(res.error);
            } else {
                throw new Error('Received empty response from Copilot server.');
            }
        } else {
            // Mobile mode: call Groq API directly from browser
            const headers = {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${key}`
            };
            
            const systemPrompt = "You are 'DrowsiGuard AI Copilot', an interactive AI assistant integrated into a Driver Drowsiness Detection System. Your primary goal is to keep the driver awake, engaged, and safe. Keep your responses relatively short, highly conversational, and energetic. You can offer to play a trivia game, tell a joke, share an interesting fact, or talk about a topic of interest. If the user tells you they are tired or if a drowsiness alert is triggered, show concern, advise them to pull over if they are too tired, and offer to engage them in conversation to help keep them awake.";
            const fullMsgs = [{ role: 'system', content: systemPrompt }, ...copilotMessages];

            const res = await fetch(`${baseUrl.replace(/\/$/, '')}/chat/completions`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({
                    model: model,
                    messages: fullMsgs,
                    max_tokens: 1000
                })
            });

            if (!res.ok) {
                const errText = await res.text();
                let errMsg = `HTTP Error ${res.status}`;
                try {
                    const errJson = JSON.parse(errText);
                    errMsg = errJson.error.message || errMsg;
                } catch(e){}
                throw new Error(errMsg);
            }

            const data = await res.json();
            if (data && data.choices && data.choices.length > 0) {
                aiText = data.choices[0].message.content;
            } else {
                throw new Error('Received empty response from Groq API.');
            }
        }

        // Remove thinking message
        const thinkingElement = document.getElementById(thinkingId);
        if (thinkingElement) thinkingElement.remove();

        // Add assistant message to history & UI
        copilotMessages.push({ role: 'assistant', content: aiText });
        addChatMessage('🤖 Copilot', aiText, 'ai');
        
        // Speak AI text
        speakText(aiText);

    } catch (err) {
        console.error('Copilot error:', err);
        const thinkingElement = document.getElementById(thinkingId);
        if (thinkingElement) thinkingElement.remove();
        addChatMessage('⚠️ System', `Error: ${err.message}`, 'error');
    } finally {
        input.disabled = false;
        if (dom.copilotSendBtn) dom.copilotSendBtn.disabled = false;
        input.focus();
    }
}

function saveCopilotSettingsFn() {
    const key = dom.copilotApiKey ? dom.copilotApiKey.value.trim() : '';
    const model = dom.copilotModel ? dom.copilotModel.value.trim() : '';
    const url = dom.copilotBaseUrl ? dom.copilotBaseUrl.value.trim() : '';

    localStorage.setItem('drowsiguard-copilot-key', key);
    localStorage.setItem('drowsiguard-copilot-model', model);
    localStorage.setItem('drowsiguard-copilot-url', url);

    if (dom.saveCopilotSettings) {
        dom.saveCopilotSettings.textContent = '✓ Saved!';
        setTimeout(() => { dom.saveCopilotSettings.textContent = 'Save Settings'; }, 2000);
    }
}

function triggerCopilotAutoWarning(fatigueLevel) {
    const now = Date.now();
    if (now - lastAutoAlertTime < 45000) return; // 45s cooldown
    lastAutoAlertTime = now;

    const warningText = `Warning! High fatigue detected (${fatigueLevel}%). Please pull over to rest safely. I am here to talk if you need to stay alert. would you like to hear a joke or play a quick game?`;
    
    // Add to history
    copilotMessages.push({ role: 'assistant', content: warningText });
    addChatMessage('🤖 Copilot', `⚠️ High Fatigue warning triggered (${fatigueLevel}%). Let's chat to stay alert!`, 'system');
    
    speakText(warningText);
}

// ═══════════════════════════════════════════════════════════════
//  Common UI State Management
// ═══════════════════════════════════════════════════════════════
function setRunningState() {
    isRunning = true;
    dom.startBtn.disabled = true;
    dom.startBtn.innerHTML = '<span class="btn-icon-left">\u25b6</span> Start Detection';
    dom.stopBtn.disabled = false;
    dom.cameraUrl.disabled = true;
    dom.videoPlaceholder.classList.add('hidden');
    dom.liveBadge.classList.remove('hidden');
    dom.modeBadge.classList.remove('hidden');

    // PRO: Reset session stats
    sessionAlertCount = 0;
    fatigueAccumulator = [];
    alertHistory = [];
    chartData = { ear: [], mar: [], timestamps: [] };
    lastDrowsyState = false;
    lastYawnState = false;
    renderAlertHistory();

    // PRO: Start chart
    startChartAnimation();
}

function setStoppedState() {
    isRunning = false;
    dom.startBtn.disabled = false;
    dom.startBtn.innerHTML = '<span class="btn-icon-left">\u25b6</span> Start Detection';
    dom.stopBtn.disabled = true;
    dom.cameraUrl.disabled = false;
    dom.liveBadge.classList.add('hidden');
    dom.alertOverlay.classList.add('hidden');
    dom.videoContainer.classList.remove('alert-active');

    dom.statusDot.className = 'status-dot offline';
    dom.statusText.textContent = 'Offline';
    dom.statusText.style.color = 'var(--text-muted)';
    dom.eyeIndicator.className = 'status-indicator inactive';
    dom.eyeStatusText.textContent = '\u2014';
    dom.mouthIndicator.className = 'status-indicator inactive';
    dom.mouthStatusText.textContent = '\u2014';

    // PRO: Stop chart
    stopChartAnimation();
}

// ─── Event Listeners ────────────────────────────────────────────
dom.startBtn.addEventListener('click', async () => {
    dom.startBtn.disabled = true;
    if (isMobile) {
        await startMobile();
    } else {
        dom.startBtn.textContent = '\u23f3 Starting...';
        await startDesktop();
    }
});

dom.stopBtn.addEventListener('click', async () => {
    dom.stopBtn.disabled = true;
    if (isMobile) {
        stopMobile();
    } else {
        await stopDesktop();
    }
    setStoppedState();
});

dom.pasteBtn.addEventListener('click', async () => {
    try {
        const text = await navigator.clipboard.readText();
        dom.cameraUrl.value = text;
    } catch (e) {}
});

dom.soundToggle.addEventListener('change', async () => {
    if (!isMobile) await apiPost(API.TOGGLE_SOUND);
    if (!dom.soundToggle.checked) stopMobileAlarm();
});

dom.privacyToggle.addEventListener('change', async () => {
    if (!isMobile) await apiPost(API.TOGGLE_PRIVACY);
});

// PRO: Theme toggle
if (dom.themeToggle) {
    dom.themeToggle.addEventListener('click', toggleTheme);
}

// Threshold sliders
dom.earThreshold.addEventListener('input', e => { dom.earThresholdVal.textContent = parseFloat(e.target.value).toFixed(2); });
dom.marThreshold.addEventListener('input', e => { dom.marThresholdVal.textContent = parseFloat(e.target.value).toFixed(2); });
dom.earFrames.addEventListener('input', e => { dom.earFramesVal.textContent = e.target.value; });
dom.marFrames.addEventListener('input', e => { dom.marFramesVal.textContent = e.target.value; });

dom.applyThresholds.addEventListener('click', async () => {
    EAR_THRESHOLD = parseFloat(dom.earThreshold.value);
    MAR_THRESHOLD = parseFloat(dom.marThreshold.value);
    EAR_CONSEC_FRAMES = parseInt(dom.earFrames.value);
    MAR_CONSEC_FRAMES = parseInt(dom.marFrames.value);

    if (!isMobile) {
        await apiPost(API.UPDATE_THRESHOLDS, {
            ear_threshold: EAR_THRESHOLD,
            mar_threshold: MAR_THRESHOLD,
            ear_consec_frames: EAR_CONSEC_FRAMES,
            mar_consec_frames: MAR_CONSEC_FRAMES
        });
    }

    dom.applyThresholds.textContent = '\u2713 Applied!';
    setTimeout(() => { dom.applyThresholds.textContent = 'Apply Changes'; }, 2000);
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 's' || e.key === 'S') { if (!isRunning) dom.startBtn.click(); }
    if (e.key === 'x' || e.key === 'X') { if (isRunning) dom.stopBtn.click(); }
    if (e.key === 'm' || e.key === 'M') {
        dom.soundToggle.checked = !dom.soundToggle.checked;
        dom.soundToggle.dispatchEvent(new Event('change'));
    }
    if (e.key === 'p' || e.key === 'P') {
        dom.privacyToggle.checked = !dom.privacyToggle.checked;
        dom.privacyToggle.dispatchEvent(new Event('change'));
    }
    if (e.key === 't' || e.key === 'T') {
        toggleTheme();
    }
});

// ─── Initialize ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // PRO: Init theme
    initTheme();

    // AI Copilot initialization
    initCopilot();

    if (isMobile) {
        // Mobile mode: hide camera URL input, set mode badge
        dom.cameraUrlGroup.style.display = 'none';
        dom.modeBadge.classList.add('mobile');
        dom.modeIcon.textContent = '\ud83d\udcf1';
        dom.modeText.textContent = 'Phone Camera';
        dom.placeholderText.textContent = 'Tap Start to open your camera and begin detection';
        console.log('\ud83d\udcf1 DrowsiGuard PRO: Mobile mode — using phone camera + client-side AI');
    } else {
        // Desktop mode
        dom.modeIcon.textContent = '\ud83d\udcbb';
        dom.modeText.textContent = 'Desktop';
        dom.placeholderText.textContent = 'Enter IP Webcam URL or leave empty for laptop webcam';
        console.log('\ud83d\udcbb DrowsiGuard PRO: Desktop mode — using server-side AI');

        // Check if already running on page load
        apiGet(API.STATUS).then(data => {
            if (data && data.running) {
                dom.cameraUrl.value = data.camera_source || '';
                dom.videoFeed.classList.remove('hidden');
                setRunningState();
                startDesktopPolling();
            }
        });
    }

    console.log('⌨️ Keyboard shortcuts: S=Start, X=Stop, M=Mute, P=Privacy, T=Theme');
});
