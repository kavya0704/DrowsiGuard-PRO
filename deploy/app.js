import { FaceLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

// ═══════════════════════════════════════════════════════════
// Landmark Indices (unchanged)
// ═══════════════════════════════════════════════════════════
const R_EYE = [33, 160, 158, 133, 153, 144], L_EYE = [362, 385, 387, 263, 373, 380];
const R_EYE_F = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const L_EYE_F = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398];
const LIPS = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,185,40,39,37,0,267,269,270,409,415,310,311,312,13,82,81,42,183,78];
const MT=13, MB=14, ML=78, MR=308, MTI=[81,311], MBI=[178,402];
// Face oval for heatmap/enhanced mode
const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];

// ═══════════════════════════════════════════════════════════
// DOM References (new HUD layout)
// ═══════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
const dom = {
    video: $('mobileVideo'), canvas: $('mobileCanvas'), placeholder: $('videoPlaceholder'),
    container: $('videoContainer'), liveBadge: $('liveBadge'), modeBadge: $('modeBadge'),
    startBtn: $('startBtn'), stopBtn: $('stopBtn'),
    statusChip: $('statusChip'), statusDot: $('statusDot'), statusText: $('statusText'),
    fpsValue: $('fpsValue'), uptimeValue: $('uptimeValue'),
    // Eye
    eyeCard: $('eyeCard'), eyeInd: $('eyeIndicator'), eyeText: $('eyeStatusText'),
    earVal: $('earValue'), blinkCnt: $('blinkCount'), blinkRate: $('blinkRate'),
    earGaugeFill: $('earGaugeFill'),
    // Mouth
    mouthCard: $('mouthCard'), mouthInd: $('mouthIndicator'), mouthText: $('mouthStatusText'),
    marVal: $('marValue'), yawnCnt: $('yawnCount'), yawnRate: $('yawnRate'),
    marGaugeFill: $('marGaugeFill'),
    // Fatigue
    fatigueCard: $('fatigueCard'),
    eyeFatigueBar: $('eyeFatigueBar'), eyeFatigueVal: $('eyeFatigueVal'),
    yawnFatigueBar: $('yawnFatigueBar'), yawnFatigueVal: $('yawnFatigueVal'),
    overallFatigueBar: $('overallFatigueBar'), overallFatigueVal: $('overallFatigueVal'),
    // Score
    safetyScore: $('safetyScore'), scoreRingFill: $('scoreRingFill'), safetyLabel: $('safetyLabel'),
    // Face
    faceDot: $('faceDot'), faceText: $('faceText'),
    // Camera controls
    camControls: $('camControls'), flipCamBtn: $('flipCamBtn'), fullscreenBtn: $('fullscreenBtn'),
    // Settings
    soundToggle: $('soundToggle'), vibrateToggle: $('vibrateToggle'), wakeLockToggle: $('wakeLockToggle'),
    earSlider: $('earSlider'), marSlider: $('marSlider'),
    earThVal: $('earThVal'), marThVal: $('marThVal'),
    privacyToggle: $('privacyToggle'), heatmapToggle: $('heatmapToggle'),
    //Overlay + History
    alertOverlay: $('alertOverlay'), historyBody: $('historyBody'),
    // Charts
    earChart: $('earChart'), marChart: $('marChart'),
    // Background
    bgCanvas: $('bgCanvas'),
    // HUD elements
    scanSweep: $('scanSweep'), hudCoords: $('hudCoords'),
    aiConfidence: $('aiConfidence'),
    // Settings drawer
    settingsBtn: $('settingsBtn'), settingsDrawer: $('settingsDrawer'),
    closeDrawer: $('closeDrawer'), drawerBackdrop: $('drawerBackdrop'),
    // Mode
    modeSwitch: $('modeSwitch'), modeIndicator: $('modeIndicator'),
    // Orb
    jarvisOrb: $('jarvisOrb'),
    // Copilot elements
    copilotDrawer: $('copilotDrawer'),
    closeCopilotDrawer: $('closeCopilotDrawer'),
    copilotTtsToggle: $('copilotTtsToggle'),
    copilotChatContainer: $('copilotChatContainer'),
    chatMessages: $('chatMessages'),
    copilotInput: $('copilotInput'),
    copilotSendBtn: $('copilotSendBtn'),
    copilotApiKey: $('copilotApiKey'),
    copilotModel: $('copilotModel'),
    copilotBaseUrl: $('copilotBaseUrl'),
    saveCopilotSettings: $('saveCopilotSettings')
};

// ═══════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════
let landmarker = null, stream = null, animFrame = null, running = false;
let closedFr = 0, yawnFr = 0, blinks = 0, yawns = 0, startT = 0;
let fpsCnt = 0, lastFpsT = 0, fps = 0;
let audioCtx = null, osc = null, alarmOn = false;
let wakeLock = null, useFrontCam = true, privacyMode = false, heatmapMode = false;
let EAR_TH = 0.22, MAR_TH = 0.65;
const EAR_FR = 15, MAR_FR = 10;
const earHistory = [], marHistory = [];
const MAX_HISTORY = 60;
let alertLog = [];
let currentMode = 'normal'; // normal, night, enhanced
let lastFaceConf = 0;
let scanAngle = 0;

// AI Copilot state
let copilotMessages = [];
let lastAutoAlertTime = 0;

// ═══════════════════════════════════════════════════════════
// Background System — Constellation Particles + Hex Grid
// ═══════════════════════════════════════════════════════════
function initBackground() {
    const c = dom.bgCanvas, ctx = c.getContext('2d');
    let particles = [], mouse = { x: -1000, y: -1000 };
    function resize() { c.width = window.innerWidth; c.height = window.innerHeight; }
    resize(); window.addEventListener('resize', resize);

    // Mouse parallax
    document.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });

    // Create particles
    const count = Math.min(50, Math.floor(window.innerWidth / 25));
    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * c.width, y: Math.random() * c.height,
            vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25,
            r: Math.random() * 1.8 + 0.3,
            a: Math.random() * 0.4 + 0.08,
            color: Math.random() > 0.7 ? '138, 43, 226' : '0, 245, 255'
        });
    }

    function draw() {
        ctx.clearRect(0, 0, c.width, c.height);

        particles.forEach(p => {
            // Mouse repulsion
            const dx = p.x - mouse.x, dy = p.y - mouse.y;
            const md = Math.sqrt(dx * dx + dy * dy);
            if (md < 150) {
                const force = (150 - md) / 150 * 0.02;
                p.vx += dx * force; p.vy += dy * force;
            }
            // Damping
            p.vx *= 0.99; p.vy *= 0.99;
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = c.width; if (p.x > c.width) p.x = 0;
            if (p.y < 0) p.y = c.height; if (p.y > c.height) p.y = 0;

            ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${p.color}, ${p.a})`; ctx.fill();
        });

        // Constellation lines
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x, dy = particles[i].y - particles[j].y;
                const d = Math.sqrt(dx * dx + dy * dy);
                if (d < 140) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    const a = 0.05 * (1 - d / 140);
                    ctx.strokeStyle = `rgba(0, 245, 255, ${a})`;
                    ctx.lineWidth = 0.4; ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
}

// ═══════════════════════════════════════════════════════════
// Waveform Chart Drawing (Enhanced with glow)
// ═══════════════════════════════════════════════════════════
function drawWaveChart(canvas, data, threshold, color, glowColor) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = canvas.offsetHeight * 2;
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;

    // Threshold line
    const thY = h - (threshold / 0.5) * h;
    ctx.setLineDash([3, 5]);
    ctx.strokeStyle = 'rgba(255, 0, 60, 0.3)'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(0, thY); ctx.lineTo(w, thY); ctx.stroke();
    ctx.setLineDash([]);

    // Build path
    const step = w / (MAX_HISTORY - 1);
    const startI = Math.max(0, data.length - MAX_HISTORY);
    ctx.beginPath();
    for (let i = startI; i < data.length; i++) {
        const x = (i - startI) * step;
        const y = h - (data[i] / 0.5) * h;
        if (i === startI) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }

    // Glow line
    ctx.shadowColor = glowColor; ctx.shadowBlur = 8;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.shadowBlur = 0;

    // Gradient fill
    const lastX = (data.length - 1 - startI) * step;
    ctx.lineTo(lastX, h); ctx.lineTo(0, h); ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, glowColor); grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad; ctx.globalAlpha = 0.12; ctx.fill(); ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════════════════
// Math (unchanged)
// ═══════════════════════════════════════════════════════════
function dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }
function ear(lm, idx) {
    const p = idx.map(i => lm[i]);
    const v1 = dist(p[1], p[5]), v2 = dist(p[2], p[4]), h = dist(p[0], p[3]);
    return h ? (v1 + v2) / (2 * h) : 0;
}
function mar(lm) {
    const t = lm[MT], b = lm[MB], l = lm[ML], r = lm[MR];
    const v1 = dist(t, b), v2 = dist(lm[MTI[0]], lm[MBI[0]]), v3 = dist(lm[MTI[1]], lm[MBI[1]]), h = dist(l, r);
    return h ? (v1 + v2 + v3) / (3 * h) : 0;
}

// ═══════════════════════════════════════════════════════════
// Drawing — Neon Wireframe Contours + HUD Effects
// ═══════════════════════════════════════════════════════════
function drawNeonContour(ctx, lm, idx, color, w, h) {
    if (idx.length < 2) return;
    ctx.beginPath();
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.shadowColor = color; ctx.shadowBlur = 10;
    ctx.moveTo(lm[idx[0]].x * w, lm[idx[0]].y * h);
    for (let i = 1; i < idx.length; i++) {
        ctx.lineTo(lm[idx[i]].x * w, lm[idx[i]].y * h);
    }
    ctx.closePath(); ctx.stroke();
    ctx.shadowBlur = 0;
}

function drawPulsingPoints(ctx, lm, indices, color, w, h, time) {
    const pulse = 0.5 + 0.5 * Math.sin(time * 0.005);
    const r = 2 + pulse * 1.5;
    indices.forEach(i => {
        const px = lm[i].x * w, py = lm[i].y * h;
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color; ctx.shadowBlur = 8 + pulse * 4;
        ctx.fill(); ctx.shadowBlur = 0;
    });
}

function drawFaceTargeting(ctx, lm, w, h) {
    // Find face bounding box
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    FACE_OVAL.forEach(i => {
        minX = Math.min(minX, lm[i].x); maxX = Math.max(maxX, lm[i].x);
        minY = Math.min(minY, lm[i].y); maxY = Math.max(maxY, lm[i].y);
    });
    const pad = 0.03;
    const x1 = (minX - pad) * w, y1 = (minY - pad) * h;
    const x2 = (maxX + pad) * w, y2 = (maxY + pad) * h;
    const bw = x2 - x1, bh = y2 - y1;
    const cl = Math.min(bw, bh) * 0.2;

    ctx.strokeStyle = 'rgba(0, 245, 255, 0.5)'; ctx.lineWidth = 1.5;
    ctx.shadowColor = 'rgba(0, 245, 255, 0.3)'; ctx.shadowBlur = 6;

    // Top-left
    ctx.beginPath(); ctx.moveTo(x1, y1 + cl); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cl, y1); ctx.stroke();
    // Top-right
    ctx.beginPath(); ctx.moveTo(x2 - cl, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cl); ctx.stroke();
    // Bottom-left
    ctx.beginPath(); ctx.moveTo(x1, y2 - cl); ctx.lineTo(x1, y2); ctx.lineTo(x1 + cl, y2); ctx.stroke();
    // Bottom-right
    ctx.beginPath(); ctx.moveTo(x2 - cl, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cl); ctx.stroke();

    ctx.shadowBlur = 0;

    // Update HUD coordinates
    const cx = ((minX + maxX) / 2).toFixed(3);
    const cy = ((minY + maxY) / 2).toFixed(3);
    dom.hudCoords.innerHTML = `<span>X: ${cx}</span><span>Y: ${cy}</span>`;
    dom.hudCoords.classList.add('visible');
}

function drawHeatmap(ctx, lm, w, h) {
    // Simple attention zone heatmap
    const zones = [
        { indices: R_EYE, color: 'rgba(0, 245, 255, 0.08)' },
        { indices: L_EYE, color: 'rgba(0, 245, 255, 0.08)' },
        { indices: LIPS, color: 'rgba(138, 43, 226, 0.06)' },
    ];
    zones.forEach(zone => {
        const cx = zone.indices.reduce((s, i) => s + lm[i].x, 0) / zone.indices.length * w;
        const cy = zone.indices.reduce((s, i) => s + lm[i].y, 0) / zone.indices.length * h;
        const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, 40);
        grad.addColorStop(0, zone.color);
        grad.addColorStop(1, 'transparent');
        ctx.beginPath(); ctx.arc(cx, cy, 40, 0, Math.PI * 2);
        ctx.fillStyle = grad; ctx.fill();
    });
}

function drawScanEffect(ctx, w, h, time) {
    // Rotating scan line through face area
    scanAngle += 0.015;
    const cx = w / 2, cy = h / 2;
    const len = Math.max(w, h);
    const x2 = cx + Math.cos(scanAngle) * len;
    const y2 = cy + Math.sin(scanAngle) * len;

    ctx.beginPath();
    ctx.moveTo(cx, cy); ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.06)';
    ctx.lineWidth = 1; ctx.stroke();
}

function drawEnhancedMesh(ctx, lm, w, h) {
    // Full face mesh in AI Enhanced mode
    for (const pt of lm) {
        const px = pt.x * w, py = pt.y * h;
        ctx.beginPath(); ctx.arc(px, py, 0.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 245, 255, 0.2)'; ctx.fill();
    }
    // Face oval contour
    drawNeonContour(ctx, lm, FACE_OVAL, 'rgba(138, 43, 226, 0.25)', w, h);
}

// ═══════════════════════════════════════════════════════════
// Alarm (unchanged)
// ═══════════════════════════════════════════════════════════
function playAlarm() {
    if (alarmOn || !dom.soundToggle.checked) {
        if (dom.vibrateToggle.checked && navigator.vibrate && !alarmOn) navigator.vibrate([300, 100, 300, 100, 300]);
        if (alarmOn) return;
    }
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        osc = audioCtx.createOscillator();
        const g = audioCtx.createGain();
        osc.type = 'square'; osc.frequency.value = 880; g.gain.value = 0.2;
        osc.connect(g); g.connect(audioCtx.destination); osc.start(); alarmOn = true;
        let hi = true;
        window._ai = setInterval(() => { if (osc) osc.frequency.value = hi ? 660 : 880; hi = !hi; }, 150);
    } catch (e) { }
    if (dom.vibrateToggle.checked && navigator.vibrate) navigator.vibrate([300, 100, 300, 100, 300, 100, 300]);
}
function stopAlarm() {
    if (window._ai) { clearInterval(window._ai); window._ai = null; }
    if (osc) { try { osc.stop(); } catch (e) { } osc = null; }
    alarmOn = false;
    if (navigator.vibrate) navigator.vibrate(0);
}

// ═══════════════════════════════════════════════════════════
// Wake Lock (unchanged)
// ═══════════════════════════════════════════════════════════
async function requestWakeLock() {
    if (!('wakeLock' in navigator) || !dom.wakeLockToggle.checked) return;
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch (e) { }
}
function releaseWakeLock() {
    if (wakeLock) { wakeLock.release(); wakeLock = null; }
}

// ═══════════════════════════════════════════════════════════
// Alert History (unchanged logic, updated HTML classes)
// ═══════════════════════════════════════════════════════════
let lastAlertType = null, lastAlertTime = 0;
function addAlert(type) {
    const now = Date.now();
    if (type === lastAlertType && now - lastAlertTime < 5000) return;
    lastAlertType = type; lastAlertTime = now;
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    alertLog.unshift({ type, time });
    if (alertLog.length > 20) alertLog.pop();
    renderHistory();
}
function renderHistory() {
    if (alertLog.length === 0) {
        dom.historyBody.innerHTML = '<div class="history-empty">NO ALERTS — ALL SYSTEMS NOMINAL 🛡️</div>';
        return;
    }
    dom.historyBody.innerHTML = alertLog.map(a => {
        const icon = a.type === 'drowsy' ? '😴' : '🥱';
        const text = a.type === 'drowsy' ? 'Drowsiness detected' : 'Yawning detected';
        return `<div class="history-item ${a.type}"><span class="hi-icon">${icon}</span><span class="hi-text">${text}</span><span class="hi-time">${time(a)}</span></div>`;
    }).join('');
}
function time(a) { return a.time; }

// ═══════════════════════════════════════════════════════════
// UI Update (enhanced for new HUD)
// ═══════════════════════════════════════════════════════════
function formatTime(s) {
    const m = Math.floor(s / 60), ss = s % 60;
    return `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
}

function getColor(v) {
    return v < 30 ? 'var(--neo-green)' : v < 60 ? 'var(--neo-yellow)' : v < 80 ? 'var(--neo-orange)' : 'var(--neo-red)';
}

function updateUI(d) {
    // Status chip
    if (d.running) {
        if (d.alert) {
            dom.statusChip.className = 'status-chip alert';
            dom.statusText.textContent = 'ALERT!';
            dom.statusText.style.color = 'var(--neo-red)';
            dom.jarvisOrb.classList.add('alert');
        } else {
            dom.statusChip.className = 'status-chip online';
            dom.statusText.textContent = 'ACTIVE';
            dom.statusText.style.color = 'var(--neo-green)';
            dom.jarvisOrb.classList.remove('alert');
        }
    }
    dom.fpsValue.textContent = d.fps || 0;
    dom.uptimeValue.textContent = formatTime(d.uptime || 0);
    dom.earVal.textContent = d.ear.toFixed(3); dom.blinkCnt.textContent = d.blinks;
    dom.marVal.textContent = d.mar.toFixed(3); dom.yawnCnt.textContent = d.yawns;

    // Blink/yawn rate
    const mins = Math.max(1, d.uptime / 60);
    dom.blinkRate.textContent = (d.blinks / mins).toFixed(1) + '/min';
    dom.yawnRate.textContent = (d.yawns / mins).toFixed(1) + '/min';

    // AI Confidence
    dom.aiConfidence.textContent = d.face ? Math.round(d.confidence || 85) + '%' : '—';

    // Eye status
    if (d.eye === 'OPEN') {
        dom.eyeInd.className = 'status-led open'; dom.eyeText.textContent = 'OPEN';
        dom.eyeText.style.color = 'var(--neo-green)'; dom.eyeCard.classList.remove('alert');
    } else if (d.eye === 'CLOSED') {
        dom.eyeInd.className = 'status-led closed'; dom.eyeText.textContent = 'CLOSED';
        dom.eyeText.style.color = 'var(--neo-red)'; if (d.drowsy) dom.eyeCard.classList.add('alert');
    } else {
        dom.eyeInd.className = 'status-led'; dom.eyeText.textContent = '—';
        dom.eyeText.style.color = 'var(--text-muted)'; dom.eyeCard.classList.remove('alert');
    }

    // Mouth status
    if (d.mouth === 'YAWNING') {
        dom.mouthInd.className = 'status-led yawn'; dom.mouthText.textContent = 'YAWNING';
        dom.mouthText.style.color = 'var(--neo-orange)'; if (d.yawnAlert) dom.mouthCard.classList.add('warning');
    } else if (d.mouth === 'NORMAL') {
        dom.mouthInd.className = 'status-led open'; dom.mouthText.textContent = 'NORMAL';
        dom.mouthText.style.color = 'var(--neo-green)'; dom.mouthCard.classList.remove('warning');
    } else {
        dom.mouthInd.className = 'status-led'; dom.mouthText.textContent = '—';
        dom.mouthCard.classList.remove('warning');
    }

    // Radial gauges (EAR & MAR)
    const earCirc = 2 * Math.PI * 32;
    const earPct = Math.min(1, d.ear / 0.4);
    dom.earGaugeFill.style.strokeDashoffset = earCirc * (1 - earPct);
    const marPct = Math.min(1, d.mar / 0.8);
    dom.marGaugeFill.style.strokeDashoffset = earCirc * (1 - marPct);

    // Fatigue bars
    const ef = d.eyeFatigue || 0, yf = d.yawnFatigue || 0, of_ = d.fatigue || 0;
    dom.eyeFatigueBar.style.width = ef + '%'; dom.eyeFatigueVal.textContent = Math.round(ef) + '%';
    dom.yawnFatigueBar.style.width = yf + '%'; dom.yawnFatigueVal.textContent = Math.round(yf) + '%';
    dom.overallFatigueBar.style.width = of_ + '%'; dom.overallFatigueVal.textContent = Math.round(of_) + '%';
    dom.overallFatigueBar.style.background = `linear-gradient(90deg, ${getColor(of_)}, ${getColor(Math.min(100, of_ + 20))})`;
    dom.fatigueCard.classList.toggle('alert', of_ >= 60);

    // Safety score
    const safety = Math.max(0, 100 - of_);
    dom.safetyScore.textContent = safety;
    const scoreColor = getColor(100 - safety);
    dom.safetyScore.style.color = scoreColor;
    dom.safetyScore.style.textShadow = `0 0 20px ${scoreColor.includes('green') ? 'rgba(57,255,20,0.3)' : scoreColor.includes('red') ? 'rgba(255,0,60,0.3)' : 'rgba(255,215,0,0.3)'}`;
    const circ = 2 * Math.PI * 60;
    dom.scoreRingFill.style.strokeDashoffset = circ * (1 - safety / 100);

    let sl;
    if (safety >= 80) { sl = 'EXCELLENT'; dom.safetyLabel.style.color = 'var(--neo-green)'; }
    else if (safety >= 60) { sl = 'GOOD'; dom.safetyLabel.style.color = 'var(--neo-yellow)'; }
    else if (safety >= 40) { sl = 'CAUTION'; dom.safetyLabel.style.color = 'var(--neo-orange)'; }
    else { sl = 'DANGER'; dom.safetyLabel.style.color = 'var(--neo-red)'; }
    dom.safetyLabel.textContent = sl;

    // Face status
    dom.faceText.textContent = d.face ? 'TRACKING' : 'NO FACE';
    dom.faceDot.className = 'face-dot-neo' + (d.face ? ' detected' : '');
    if (!d.face) dom.hudCoords.classList.remove('visible');

    // Alert overlay
    if (d.drowsy) { dom.alertOverlay.classList.remove('hidden'); dom.container.classList.add('alert-active'); }
    else { dom.alertOverlay.classList.add('hidden'); dom.container.classList.remove('alert-active'); }

    // Waveform charts
    drawWaveChart(dom.earChart, earHistory, EAR_TH, '#00F5FF', 'rgba(0, 245, 255, 0.4)');
    drawWaveChart(dom.marChart, marHistory, MAR_TH, '#8A2BE2', 'rgba(138, 43, 226, 0.4)');

    // Auto warning if fatigue is high
    if (d.fatigue >= 70) {
        triggerCopilotAutoWarning(d.fatigue);
    }
}

// ═══════════════════════════════════════════════════════════
// Detection Loop (enhanced drawing)
// ═══════════════════════════════════════════════════════════
function detectLoop() {
    if (!running || !landmarker) return;
    const v = dom.video, c = dom.canvas, ctx = c.getContext('2d');
    if (v.readyState < 2) { animFrame = requestAnimationFrame(detectLoop); return; }
    c.width = v.videoWidth; c.height = v.videoHeight;
    const now = performance.now();
    let res;
    try { res = landmarker.detectForVideo(v, now); } catch (e) { animFrame = requestAnimationFrame(detectLoop); return; }
    ctx.clearRect(0, 0, c.width, c.height);

    let e = 0, m = 0, eyeS = 'N/A', mouthS = 'N/A', face = false, drowsy = false, yawnA = false;
    let eyeFatigue = 0, yawnFatigue = 0, fat = 0;
    let confidence = 0;

    // Scan effect (always when running)
    if (currentMode === 'enhanced') {
        drawScanEffect(ctx, c.width, c.height, now);
    }

    if (res.faceLandmarks && res.faceLandmarks.length > 0) {
        face = true; const lm = res.faceLandmarks[0];
        confidence = (res.faceBlendshapes?.[0]?.categories?.[0]?.score || 0.85) * 100;
        lastFaceConf = confidence;

        e = (ear(lm, L_EYE) + ear(lm, R_EYE)) / 2; m = mar(lm);
        earHistory.push(e); marHistory.push(m);
        if (earHistory.length > MAX_HISTORY * 2) earHistory.splice(0, earHistory.length - MAX_HISTORY);
        if (marHistory.length > MAX_HISTORY * 2) marHistory.splice(0, marHistory.length - MAX_HISTORY);

        if (e < EAR_TH) { closedFr++; eyeS = 'CLOSED'; } else { if (closedFr > 2) blinks++; closedFr = 0; eyeS = 'OPEN'; }
        if (m > MAR_TH) { yawnFr++; mouthS = 'YAWNING'; } else { if (yawnFr > MAR_FR) yawns++; yawnFr = 0; mouthS = 'NORMAL'; }
        drowsy = closedFr >= EAR_FR; yawnA = yawnFr >= MAR_FR;
        eyeFatigue = Math.min(100, (closedFr / EAR_FR) * 60);
        yawnFatigue = Math.min(100, yawns * 15);
        fat = Math.min(100, Math.floor(eyeFatigue * 0.6 + yawnFatigue * 0.4));

        if (drowsy) addAlert('drowsy');
        if (yawnA) addAlert('yawn');
        if (drowsy || yawnA) playAlarm(); else stopAlarm();

        // === ENHANCED DRAWING ===
        const ec = eyeS === 'OPEN' ? '#00F5FF' : '#FF003C';
        const mc = mouthS === 'NORMAL' ? '#39FF14' : '#FF6B2B';

        // Neon wireframe contours
        drawNeonContour(ctx, lm, R_EYE_F, ec, c.width, c.height);
        drawNeonContour(ctx, lm, L_EYE_F, ec, c.width, c.height);
        drawNeonContour(ctx, lm, LIPS, mc, c.width, c.height);

        // Pulsating points at eye centers
        drawPulsingPoints(ctx, lm, [R_EYE[0], R_EYE[3], L_EYE[0], L_EYE[3]], ec, c.width, c.height, now);

        // Face targeting brackets
        drawFaceTargeting(ctx, lm, c.width, c.height);

        // Heatmap overlay
        if (heatmapMode) drawHeatmap(ctx, lm, c.width, c.height);

        // AI Enhanced mode — full mesh
        if (currentMode === 'enhanced' || privacyMode) {
            drawEnhancedMesh(ctx, lm, c.width, c.height);
        }

    } else {
        closedFr = 0; yawnFr = 0; stopAlarm();
        confidence = 0;
    }

    fpsCnt++;
    if (now - lastFpsT > 1000) { fps = Math.round(fpsCnt * 1000 / (now - lastFpsT)); fpsCnt = 0; lastFpsT = now; }
    const up = startT ? Math.floor((Date.now() - startT) / 1000) : 0;
    updateUI({
        running: true, face, ear: e, mar: m, eye: eyeS, mouth: mouthS,
        fatigue: fat, eyeFatigue, yawnFatigue, alert: drowsy || yawnA,
        drowsy, yawnAlert: yawnA, blinks, yawns, fps, uptime: up, confidence
    });
    animFrame = requestAnimationFrame(detectLoop);
}

// ═══════════════════════════════════════════════════════════
// Camera (unchanged)
// ═══════════════════════════════════════════════════════════
async function openCamera() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: useFrontCam ? 'user' : 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
    });
    dom.video.srcObject = stream; await dom.video.play();
}

// ═══════════════════════════════════════════════════════════
// Settings Drawer
// ═══════════════════════════════════════════════════════════
function openDrawer() {
    dom.settingsDrawer.classList.add('open');
    dom.drawerBackdrop.classList.remove('hidden');
}
function closeDrawer() {
    dom.settingsDrawer.classList.remove('open');
    dom.drawerBackdrop.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════
// Mode Switch
// ═══════════════════════════════════════════════════════════
function setMode(mode) {
    currentMode = mode;
    document.body.classList.remove('night-mode', 'enhanced-mode');
    if (mode === 'night') document.body.classList.add('night-mode');
    if (mode === 'enhanced') document.body.classList.add('enhanced-mode');

    // Update mode button states
    dom.modeSwitch.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Move indicator
    const btns = [...dom.modeSwitch.querySelectorAll('.mode-btn')];
    const idx = btns.findIndex(b => b.dataset.mode === mode);
    dom.modeIndicator.style.left = `calc(${idx * 33.33}% + 2px)`;
}

// ═══════════════════════════════════════════════════════════
// Event Listeners
// ═══════════════════════════════════════════════════════════
dom.startBtn.addEventListener('click', async () => {
    dom.startBtn.disabled = true;
    dom.startBtn.querySelector('span:last-child').textContent = 'LOADING AI...';
    try {
        const fr = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
        landmarker = await FaceLandmarker.createFromOptions(fr, {
            baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', delegate: 'GPU' },
            runningMode: 'VIDEO', numFaces: 1, minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
            outputFaceBlendshapes: true
        });
    } catch (e) {
        try {
            const fr = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
            landmarker = await FaceLandmarker.createFromOptions(fr, {
                baseOptions: { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', delegate: 'CPU' },
                runningMode: 'VIDEO', numFaces: 1,
                outputFaceBlendshapes: true
            });
        } catch (e2) {
            alert('Failed to load AI model. Check internet connection.');
            dom.startBtn.disabled = false;
            dom.startBtn.querySelector('span:last-child').textContent = 'INITIALIZE';
            return;
        }
    }
    dom.startBtn.querySelector('span:last-child').textContent = 'CAMERA...';
    try { await openCamera(); } catch (e) {
        alert('Camera access denied.');
        dom.startBtn.disabled = false;
        dom.startBtn.querySelector('span:last-child').textContent = 'INITIALIZE';
        return;
    }

    running = true; closedFr = 0; yawnFr = 0; blinks = 0; yawns = 0; startT = Date.now();
    lastFpsT = performance.now(); fpsCnt = 0; earHistory.length = 0; marHistory.length = 0;
    alertLog = []; renderHistory();
    dom.video.classList.remove('hidden'); dom.canvas.classList.remove('hidden');
    dom.placeholder.classList.add('hidden');
    dom.liveBadge.classList.remove('hidden'); dom.modeBadge.classList.remove('hidden');
    dom.camControls.classList.remove('hidden');
    dom.scanSweep.classList.add('active');
    dom.startBtn.querySelector('span:last-child').textContent = 'INITIALIZE';
    dom.stopBtn.disabled = false;
    requestWakeLock();
    detectLoop();
});

dom.stopBtn.addEventListener('click', () => {
    running = false;
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    dom.video.srcObject = null; stopAlarm(); releaseWakeLock();
    dom.video.classList.add('hidden'); dom.canvas.classList.add('hidden');
    dom.placeholder.classList.remove('hidden');
    dom.liveBadge.classList.add('hidden'); dom.alertOverlay.classList.add('hidden');
    dom.container.classList.remove('alert-active'); dom.camControls.classList.add('hidden');
    dom.scanSweep.classList.remove('active');
    dom.hudCoords.classList.remove('visible');
    dom.startBtn.disabled = false; dom.stopBtn.disabled = true;
    dom.statusChip.className = 'status-chip';
    dom.statusText.textContent = 'OFFLINE'; dom.statusText.style.color = 'var(--text-muted)';
    dom.eyeInd.className = 'status-led'; dom.eyeText.textContent = '—';
    dom.mouthInd.className = 'status-led'; dom.mouthText.textContent = '—';
    dom.jarvisOrb.classList.remove('alert');
    dom.aiConfidence.textContent = '—';
});

dom.flipCamBtn.addEventListener('click', async () => {
    useFrontCam = !useFrontCam;
    try { await openCamera(); } catch (e) { useFrontCam = !useFrontCam; }
});

dom.fullscreenBtn.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else dom.container.requestFullscreen?.();
});

dom.earSlider.addEventListener('input', (e) => { EAR_TH = parseFloat(e.target.value); dom.earThVal.textContent = EAR_TH.toFixed(2); });
dom.marSlider.addEventListener('input', (e) => { MAR_TH = parseFloat(e.target.value); dom.marThVal.textContent = MAR_TH.toFixed(2); });

dom.privacyToggle.addEventListener('change', () => {
    privacyMode = dom.privacyToggle.checked;
    dom.container.classList.toggle('privacy-mode', privacyMode);
});

dom.heatmapToggle.addEventListener('change', () => {
    heatmapMode = dom.heatmapToggle.checked;
});

// Settings drawer
dom.settingsBtn.addEventListener('click', openDrawer);
dom.closeDrawer.addEventListener('click', closeDrawer);
dom.drawerBackdrop.addEventListener('click', closeDrawer);

// Mode switch
dom.modeSwitch.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════
initBackground();
setMode('normal');

// ═══════════════════════════════════════════════════════════
// AI Copilot Assistant Engine (direct Groq fetch for static deployment)
// ═══════════════════════════════════════════════════════════

const COPILOT_DEFAULTS = {
    apiKey: "",
    model: "openai/gpt-oss-120b",
    baseUrl: "https://api.groq.com/openai/v1"
};

function openCopilotDrawer() {
    dom.copilotDrawer.classList.add('open');
    dom.drawerBackdrop.classList.remove('hidden');
}

function closeCopilotDrawer() {
    dom.copilotDrawer.classList.remove('open');
    dom.drawerBackdrop.classList.add('hidden');
}

function initCopilot() {
    const key = localStorage.getItem('drowsiguard-copilot-key') || COPILOT_DEFAULTS.apiKey;
    const model = localStorage.getItem('drowsiguard-copilot-model') || COPILOT_DEFAULTS.model;
    const url = localStorage.getItem('drowsiguard-copilot-url') || COPILOT_DEFAULTS.baseUrl;
    const tts = localStorage.getItem('drowsiguard-copilot-tts') !== 'false';

    if (dom.copilotApiKey) dom.copilotApiKey.value = key;
    if (dom.copilotModel) dom.copilotModel.value = model;
    if (dom.copilotBaseUrl) dom.copilotBaseUrl.value = url;
    if (dom.copilotTtsToggle) dom.copilotTtsToggle.checked = tts;

    // Bind event listeners
    dom.jarvisOrb.addEventListener('click', openCopilotDrawer);
    dom.closeCopilotDrawer.addEventListener('click', closeCopilotDrawer);
    dom.drawerBackdrop.addEventListener('click', closeCopilotDrawer);

    if (dom.copilotSendBtn) dom.copilotSendBtn.addEventListener('click', sendCopilotMessage);
    if (dom.copilotInput) {
        dom.copilotInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') sendCopilotMessage();
        });
    }
    if (dom.saveCopilotSettings) dom.saveCopilotSettings.addEventListener('click', saveCopilotSettingsFn);
    if (dom.copilotTtsToggle) {
        dom.copilotTtsToggle.addEventListener('change', () => {
            localStorage.setItem('drowsiguard-copilot-tts', dom.copilotTtsToggle.checked);
        });
    }
}

function speakText(text) {
    if (!dom.copilotTtsToggle || !dom.copilotTtsToggle.checked) return;
    if (!window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    
    // Clean emojis & HUD tags for voice
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
    copilotMessages.push({ role: 'user', content: text });

    const thinkingId = 'thinking_' + Date.now();
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'chat-message ai thinking';
    thinkingDiv.id = thinkingId;
    thinkingDiv.innerHTML = `
        <span class="message-sender">🤖 Copilot</span>
        <p class="message-text">Thinking...</p>
    `;
    dom.chatMessages.appendChild(thinkingDiv);
    if (dom.copilotChatContainer) dom.copilotChatContainer.scrollTop = dom.copilotChatContainer.scrollHeight;

    input.disabled = true;
    if (dom.copilotSendBtn) dom.copilotSendBtn.disabled = true;

    const key = dom.copilotApiKey ? dom.copilotApiKey.value : '';
    const model = dom.copilotModel ? dom.copilotModel.value : '';
    const baseUrl = dom.copilotBaseUrl ? dom.copilotBaseUrl.value : '';

    try {
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
        let aiText = '';
        if (data && data.choices && data.choices.length > 0) {
            aiText = data.choices[0].message.content;
        } else {
            throw new Error('Received empty response from Groq API.');
        }

        const thinkingElement = document.getElementById(thinkingId);
        if (thinkingElement) thinkingElement.remove();

        copilotMessages.push({ role: 'assistant', content: aiText });
        addChatMessage('🤖 Copilot', aiText, 'ai');
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
        dom.saveCopilotSettings.textContent = '✓ SAVED';
        setTimeout(() => { dom.saveCopilotSettings.textContent = 'SAVE PARAMETERS'; }, 2000);
    }
}

function triggerCopilotAutoWarning(fatigueLevel) {
    const now = Date.now();
    if (now - lastAutoAlertTime < 45000) return;
    lastAutoAlertTime = now;

    const warningText = `Warning! High fatigue detected (${fatigueLevel}%). Let's talk to keep you focused. would you like to hear a joke or play a quick game?`;
    
    copilotMessages.push({ role: 'assistant', content: warningText });
    addChatMessage('🤖 Copilot', `⚠️ High Fatigue warning triggered (${fatigueLevel}%). Let's chat to stay alert!`, 'system');
    
    speakText(warningText);
}

initCopilot();
