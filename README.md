# 🚗 AI Driver Drowsiness Detection System — DrowsiGuard

A **real-time AI-powered driver drowsiness detection system** that uses your phone as a camera and your laptop as the processing unit. Built with MediaPipe Face Mesh, OpenCV, and Flask.

![Status](https://img.shields.io/badge/status-demo--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- 👁️ **Eye Blink Detection (EAR)** — Detects prolonged eye closure
- 🥱 **Yawn Detection (MAR)** — Monitors mouth aspect ratio
- ⚡ **Fatigue Level Gauge** — Real-time fatigue percentage
- 🔊 **Audio Alarm System** — Instant alarm when drowsiness detected
- 📱 **Phone as Camera** — Use IP Webcam app to stream from your phone
- 🔒 **Privacy-Safe** — No video recording, no storage, local-only processing
- 🎛️ **Adjustable Thresholds** — Tune sensitivity in real-time
- 🔒 **Privacy Mode** — Show only face mesh landmarks (no real face)

---

## 📋 Prerequisites

1. **Python 3.10+** — [Download](https://python.org/downloads/)
2. **Phone Camera App** (choose one):
   - **Android**: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (free)
   - **iOS**: [iVCam](https://apps.apple.com/app/ivcam/id1164464478)
3. **Same Wi-Fi Network** — Phone and laptop must be on the same network

---

## 🚀 Quick Start

### Option 1: One-Click Launch (Windows)

```
Double-click start.bat
```

This will install dependencies, generate the alarm sound, and open the dashboard.

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate alarm sound
python generate_alarm.py

# 3. Start the server
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 📱 Phone Camera Setup

### Android (IP Webcam)

1. Install **IP Webcam** from Play Store
2. Open the app → scroll down → tap **"Start server"**
3. Note the URL shown (e.g., `http://192.168.1.100:8080`)
4. In the dashboard, enter: `http://192.168.1.100:8080/video`
5. Click **Start Detection**

### iOS (iVCam)

1. Install **iVCam** from App Store
2. Install the PC client from [iVCam website](https://www.e2esoft.com/ivcam/)
3. Both devices connect automatically on the same Wi-Fi
4. Use camera index `0` in the dashboard (leave URL field empty)

### No Phone? Use Laptop Webcam

Just leave the camera URL field empty and click **Start Detection**. The system will use your laptop's built-in webcam.

---

## 🎮 Dashboard Controls

| Control | Action |
|---------|--------|
| **Start Detection** | Begin monitoring (keyboard: `S`) |
| **Stop** | Stop monitoring (keyboard: `X`) |
| **🔊 Alarm Sound** | Toggle audio alerts |
| **🔒 Privacy Mode** | Show only face landmarks (keyboard: `P`) |
| **🎛️ Threshold Settings** | Adjust EAR/MAR sensitivity |

---

## 🎯 Demo Presentation Script

1. **Setup** (before audience arrives):
   - Run `start.bat` or `python app.py`
   - Connect phone to same Wi-Fi
   - Open IP Webcam on phone
   - Enter URL in dashboard and verify feed works

2. **Demo Flow**:
   - Show the dashboard with normal monitoring (green indicators)
   - **Slowly close your eyes** for 2-3 seconds → alarm triggers! 🔊
   - **Open eyes** → alarm stops, indicators go green
   - **Yawn widely** → yawn detection triggers
   - Toggle **Privacy Mode** to show landmark-only view
   - Show the **fatigue gauge** increasing/decreasing in real-time
   - Adjust thresholds live to demonstrate sensitivity tuning

3. **Key Talking Points**:
   - "Real-time processing with zero latency"
   - "Privacy-safe: no video is ever recorded"
   - "Works on any phone + laptop combo over local Wi-Fi"
   - "Adjustable sensitivity for different lighting conditions"

---

## 🔧 Configuration

Edit `config.py` to change defaults:

```python
EAR_THRESHOLD = 0.22      # Lower = more sensitive to eye closure
MAR_THRESHOLD = 0.65      # Lower = more sensitive to yawning
EAR_CONSEC_FRAMES = 15    # Frames of closed eyes before alarm
ALARM_VOLUME = 0.8        # 0.0 to 1.0
```

---

## 🏗️ Architecture

```
📱 Phone (IP Webcam)
    │
    │  MJPEG Stream over Wi-Fi
    ▼
💻 Laptop (Python Backend)
    ├── OpenCV: captures video frames
    ├── MediaPipe Face Mesh: extracts 468 facial landmarks
    ├── EAR Calculator: monitors eye closure
    ├── MAR Calculator: monitors yawning
    ├── Alarm System: pygame audio alerts
    └── Flask Server: serves dashboard + API
         │
         ▼
    🖥️ Browser Dashboard
         ├── Live video feed (MJPEG)
         ├── Real-time metrics (JSON polling)
         ├── Fatigue gauge
         └── Alert overlays
```

---

## 📁 Project Structure

```
AI Driver Drowsiness Detection System demo/
├── app.py                  # Main Flask backend + AI engine
├── config.py               # Configuration parameters
├── generate_alarm.py       # Alarm sound generator
├── requirements.txt        # Python dependencies
├── start.bat               # One-click Windows launcher
├── README.md               # This file
├── static/
│   ├── css/style.css       # Dashboard styling
│   ├── js/app.js           # Frontend logic
│   ├── sounds/alarm.wav    # Alert sound
│   └── img/logo.png        # App logo
└── templates/
    └── index.html          # Dashboard HTML
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera won't connect | Ensure phone & laptop are on same Wi-Fi. Try `/shot.jpg` URL instead of `/video` |
| Low FPS | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py`. Close other apps |
| False alarms | Increase `EAR_THRESHOLD` or `EAR_CONSEC_FRAMES` in Settings |
| No alarm sound | Run `python generate_alarm.py` to regenerate. Check volume isn't muted |
| MediaPipe error | Ensure Python 3.10+ is installed. Run `pip install mediapipe --upgrade` |
| Face not detected | Improve lighting. Ensure face is clearly visible to camera |

---

## 🔒 Privacy & Security

- ✅ **No video recording** — frames are processed and discarded immediately
- ✅ **No cloud services** — everything runs locally on your laptop
- ✅ **No internet required** — works on offline local network
- ✅ **Privacy mode** — optional landmark-only view hides real faces
- ✅ **No data storage** — no logs, no files, no databases

---

## 📄 License

MIT License — Free for educational and personal use.

---

Built with ❤️ for safer driving.
