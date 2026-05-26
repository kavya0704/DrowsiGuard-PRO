"""
AI Driver Drowsiness Detection System - Configuration
All tunable parameters in one place.
"""

# ─── Camera Settings ───────────────────────────────────────────────
# Default: laptop webcam (0). Change to IP Webcam URL for phone camera.
# Android IP Webcam format: "http://192.168.1.100:8080/video"
# DroidCam format: "http://192.168.1.100:4747/video"
CAMERA_SOURCE = 0  # 0 = laptop webcam, or paste IP Webcam URL string

# ─── Eye Aspect Ratio (EAR) ───────────────────────────────────────
EAR_THRESHOLD = 0.22          # Below this = eyes considered closed
EAR_CONSEC_FRAMES = 15        # Consecutive closed-eye frames to trigger alarm

# ─── Mouth Aspect Ratio (MAR) ─────────────────────────────────────
MAR_THRESHOLD = 0.65          # Above this = mouth considered open (yawn)
MAR_CONSEC_FRAMES = 10        # Consecutive yawn frames to trigger alert

# ─── Head Pose ─────────────────────────────────────────────────────
HEAD_TILT_THRESHOLD = 15      # Degrees of head tilt to detect nodding off

# ─── Alert Settings ───────────────────────────────────────────────
ALARM_SOUND_PATH = "static/sounds/alarm.wav"
ALARM_VOLUME = 0.8            # 0.0 to 1.0
ALERT_COOLDOWN = 3            # Seconds between repeated alerts

# ─── Flask Server ─────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
DEBUG_MODE = False

# ─── MediaPipe Settings ───────────────────────────────────────────
FACE_MESH_CONFIDENCE = 0.5
FACE_MESH_TRACKING = 0.5
FACE_MESH_MAX_FACES = 1

# ─── Display Settings ─────────────────────────────────────────────
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 80

# ─── MediaPipe Face Mesh Landmark Indices ──────────────────────────
# Right eye landmarks for EAR calculation
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye landmarks for EAR calculation
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR calculation
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP_INNER = [81, 311]
MOUTH_BOTTOM_INNER = [178, 402]

# Full eye contour for drawing
RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Lip contour for drawing
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

import os

# ─── Groq AI Copilot Settings ─────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

