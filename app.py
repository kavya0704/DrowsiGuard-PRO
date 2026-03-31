"""
AI Driver Drowsiness Detection System
Main Flask Application + AI Detection Engine

Uses MediaPipe Tasks API (FaceLandmarker) for face mesh detection.

Features:
- Real-time eye blink detection (EAR)
- Yawn detection (MAR)
- Audio alarm system
- Live MJPEG video stream with landmarks overlay
- RESTful API for frontend dashboard
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import math
import json
import os
from flask import Flask, render_template, Response, jsonify, request

# Import configuration
import config

# ─── Flask App ─────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Global State ──────────────────────────────────────────────────
class DetectionState:
    """Thread-safe state container for the detection system."""
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.camera_source = config.CAMERA_SOURCE
        self.cap = None
        self.landmarker = None
        
        # Detection metrics
        self.ear_value = 0.0
        self.mar_value = 0.0
        self.eye_status = "OPEN"
        self.mouth_status = "CLOSED"
        self.fatigue_level = 0       # 0-100
        self.alert_active = False
        self.drowsy_alert = False
        self.yawn_alert = False
        self.yawn_count = 0
        self.blink_count = 0
        
        # Frame counters
        self.closed_frames = 0
        self.yawn_frames = 0
        self.total_frames = 0
        self.fps = 0
        
        # Thresholds (mutable at runtime)
        self.ear_threshold = config.EAR_THRESHOLD
        self.mar_threshold = config.MAR_THRESHOLD
        self.ear_consec_frames = config.EAR_CONSEC_FRAMES
        self.mar_consec_frames = config.MAR_CONSEC_FRAMES
        
        # Timing
        self.start_time = None
        self.last_alert_time = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Alert settings
        self.sound_enabled = True
        self.privacy_mode = False
        
        # Latest frame for streaming
        self.latest_frame = None
        self.face_detected = False

        # Alarm
        self.alarm_playing = False

state = DetectionState()

# ─── Alarm System ──────────────────────────────────────────────────
def init_alarm():
    """Initialize pygame mixer for alarm sounds."""
    try:
        import pygame
        pygame.mixer.init()
        if os.path.exists(config.ALARM_SOUND_PATH):
            pygame.mixer.music.load(config.ALARM_SOUND_PATH)
            pygame.mixer.music.set_volume(config.ALARM_VOLUME)
            return True
        else:
            print(f"[WARN] Alarm sound not found: {config.ALARM_SOUND_PATH}")
            print("   Run: python generate_alarm.py")
            return False
    except Exception as e:
        print(f"[WARN] Could not initialize alarm: {e}")
        return False

alarm_initialized = False

def play_alarm():
    """Play alarm sound in a loop."""
    global alarm_initialized
    if not alarm_initialized:
        alarm_initialized = init_alarm()
    
    if not alarm_initialized or not state.sound_enabled:
        return
        
    try:
        import pygame
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)  # Loop indefinitely
            state.alarm_playing = True
    except Exception as e:
        print(f"Alarm play error: {e}")

def stop_alarm():
    """Stop alarm sound."""
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            state.alarm_playing = False
    except:
        state.alarm_playing = False


# ─── Math Utilities ────────────────────────────────────────────────
def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    Calculate Eye Aspect Ratio (EAR).
    
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    
    eye_indices = [p1, p2, p3, p4, p5, p6]
    p1, p4 = horizontal eye corners
    p2, p3 = upper eyelid
    p5, p6 = lower eyelid
    """
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * frame_w, lm.y * frame_h))
    
    # Vertical distances
    v1 = euclidean_distance(points[1], points[5])  # p2-p6
    v2 = euclidean_distance(points[2], points[4])  # p3-p5
    
    # Horizontal distance
    h = euclidean_distance(points[0], points[3])    # p1-p4
    
    if h == 0:
        return 0.0
    
    ear = (v1 + v2) / (2.0 * h)
    return ear

def calculate_mar(landmarks, frame_w, frame_h):
    """
    Calculate Mouth Aspect Ratio (MAR).
    """
    def get_point(idx):
        lm = landmarks[idx]
        return (lm.x * frame_w, lm.y * frame_h)
    
    top = get_point(config.MOUTH_TOP)
    bottom = get_point(config.MOUTH_BOTTOM)
    left = get_point(config.MOUTH_LEFT)
    right = get_point(config.MOUTH_RIGHT)
    
    inner_top = [get_point(i) for i in config.MOUTH_TOP_INNER]
    inner_bottom = [get_point(i) for i in config.MOUTH_BOTTOM_INNER]
    
    # Vertical distances
    v1 = euclidean_distance(top, bottom)
    v2 = euclidean_distance(inner_top[0], inner_bottom[0])
    v3 = euclidean_distance(inner_top[1], inner_bottom[1])
    
    # Horizontal distance
    h = euclidean_distance(left, right)
    
    if h == 0:
        return 0.0
    
    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar


# ─── Drawing Utilities ─────────────────────────────────────────────
def draw_eye_contour(frame, landmarks, eye_indices, color, frame_w, frame_h):
    """Draw eye contour on frame."""
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], True, color, 1, cv2.LINE_AA)

def draw_mouth_contour(frame, landmarks, mouth_indices, color, frame_w, frame_h):
    """Draw mouth contour on frame."""
    points = []
    for idx in mouth_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], True, color, 1, cv2.LINE_AA)

def draw_status_overlay(frame, ear, mar, eye_status, fatigue_level, drowsy_alert, yawn_alert, face_detected):
    """Draw status information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status text
    if face_detected:
        status_color = (0, 255, 136) if not drowsy_alert else (0, 80, 255)
        status_text = "MONITORING" if not drowsy_alert else "!! DROWSY!"
    else:
        status_color = (0, 200, 255)
        status_text = "NO FACE DETECTED"
    
    cv2.putText(frame, status_text, (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    
    # EAR & MAR values on top right
    ear_color = (0, 255, 136) if eye_status == "OPEN" else (0, 80, 255)
    cv2.putText(frame, f"EAR: {ear:.2f}", (w - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1, cv2.LINE_AA)
    
    mar_color = (0, 255, 136) if mar < state.mar_threshold else (0, 165, 255)
    cv2.putText(frame, f"MAR: {mar:.2f}", (w - 200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mar_color, 1, cv2.LINE_AA)
    
    # Alert overlays
    if drowsy_alert:
        # Pulsing red border
        pulse = int(127 + 128 * math.sin(time.time() * 8))
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, pulse), 4)
        
        # Warning text
        text = "DROWSINESS DETECTED!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 40
        
        # Background for text
        cv2.rectangle(frame, (text_x - 10, text_y - 35), 
                      (text_x + text_size[0] + 10, text_y + 10), (0, 0, 180), -1)
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    
    if yawn_alert:
        cv2.putText(frame, "YAWNING DETECTED!", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)


# ─── Detection Pipeline ───────────────────────────────────────────
def detection_loop():
    """Main detection loop running in a background thread."""
    global alarm_initialized
    
    print(f"[START] Starting detection with camera: {state.camera_source}")
    
    # Initialize camera
    source = state.camera_source
    if isinstance(source, str) and source.strip():
        state.cap = cv2.VideoCapture(source)
    else:
        state.cap = cv2.VideoCapture(0)
    
    if not state.cap.isOpened():
        print("[ERROR] Could not open camera!")
        state.running = False
        return
    
    # Set resolution
    state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize FaceLandmarker with new Tasks API
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("  Download it from: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        state.running = False
        return
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=config.FACE_MESH_MAX_FACES,
        min_face_detection_confidence=config.FACE_MESH_CONFIDENCE,
        min_face_presence_confidence=config.FACE_MESH_CONFIDENCE,
        min_tracking_confidence=config.FACE_MESH_TRACKING,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    
    state.landmarker = vision.FaceLandmarker.create_from_options(options)
    
    state.start_time = time.time()
    state.frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    timestamp_ms = 0
    
    print("[OK] Detection running!")
    
    while state.running:
        ret, frame = state.cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame_h, frame_w = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        timestamp_ms += 33  # ~30fps increment
        try:
            results = state.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            time.sleep(0.01)
            continue
        
        # Privacy mode: blank the face area
        display_frame = frame.copy()
        if state.privacy_mode:
            display_frame = np.zeros_like(frame)
            display_frame[:] = (20, 15, 40)  # Dark background
        
        ear = 0.0
        mar = 0.0
        face_detected = False
        
        if results.face_landmarks and len(results.face_landmarks) > 0:
            face_detected = True
            landmarks = results.face_landmarks[0]  # First face
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, config.LEFT_EYE, frame_w, frame_h)
            right_ear = calculate_ear(landmarks, config.RIGHT_EYE, frame_w, frame_h)
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR
            mar = calculate_mar(landmarks, frame_w, frame_h)
            
            # ── Eye state detection ──
            if ear < state.ear_threshold:
                state.closed_frames += 1
                eye_status = "CLOSED"
                eye_color = (0, 80, 255)   # Red
            else:
                if state.closed_frames > 2:
                    state.blink_count += 1
                state.closed_frames = 0
                eye_status = "OPEN"
                eye_color = (0, 255, 136)   # Green
            
            # ── Yawn detection ──
            if mar > state.mar_threshold:
                state.yawn_frames += 1
                mouth_status = "YAWNING"
                mouth_color = (0, 165, 255)  # Orange
            else:
                if state.yawn_frames > state.mar_consec_frames:
                    state.yawn_count += 1
                state.yawn_frames = 0
                mouth_status = "CLOSED"
                mouth_color = (0, 255, 136)  # Green
            
            # ── Drowsiness alert ──
            drowsy_alert = state.closed_frames >= state.ear_consec_frames
            yawn_alert = state.yawn_frames >= state.mar_consec_frames
            
            # ── Fatigue level calculation ──
            eye_fatigue = min(100, (state.closed_frames / state.ear_consec_frames) * 60)
            yawn_fatigue = min(40, state.yawn_count * 10)
            fatigue_level = min(100, int(eye_fatigue + yawn_fatigue))
            
            # ── Alarm control ──
            if drowsy_alert or yawn_alert:
                if not state.alarm_playing and state.sound_enabled:
                    play_alarm()
            else:
                if state.alarm_playing:
                    stop_alarm()
            
            # ── Draw landmarks on display frame ──
            draw_eye_contour(display_frame, landmarks, config.RIGHT_EYE_FULL, eye_color, frame_w, frame_h)
            draw_eye_contour(display_frame, landmarks, config.LEFT_EYE_FULL, eye_color, frame_w, frame_h)
            draw_mouth_contour(display_frame, landmarks, config.LIPS, mouth_color, frame_w, frame_h)
            
            # Draw eye center dots
            for eye_indices in [config.RIGHT_EYE, config.LEFT_EYE]:
                cx = int(sum(landmarks[i].x for i in eye_indices) / len(eye_indices) * frame_w)
                cy = int(sum(landmarks[i].y for i in eye_indices) / len(eye_indices) * frame_h)
                cv2.circle(display_frame, (cx, cy), 3, eye_color, -1)
            
            # Privacy mode: draw all face mesh landmarks
            if state.privacy_mode:
                for lm in landmarks:
                    x, y = int(lm.x * frame_w), int(lm.y * frame_h)
                    cv2.circle(display_frame, (x, y), 1, (100, 100, 140), -1)
            
            # Update state
            with state.lock:
                state.ear_value = round(ear, 3)
                state.mar_value = round(mar, 3)
                state.eye_status = eye_status
                state.mouth_status = mouth_status
                state.fatigue_level = fatigue_level
                state.drowsy_alert = drowsy_alert
                state.yawn_alert = yawn_alert
                state.alert_active = drowsy_alert or yawn_alert
                state.face_detected = True
        
        else:
            # No face detected
            with state.lock:
                state.face_detected = False
                state.eye_status = "N/A"
                state.mouth_status = "N/A"
                state.ear_value = 0.0
                state.mar_value = 0.0
                state.closed_frames = 0
                state.yawn_frames = 0
            
            if state.alarm_playing:
                stop_alarm()
            
            drowsy_alert = False
            yawn_alert = False
            face_detected = False
        
        # Draw status overlay
        draw_status_overlay(display_frame, ear, mar, 
                           state.eye_status, state.fatigue_level,
                           state.drowsy_alert, state.yawn_alert, face_detected)
        
        # FPS calculation
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            state.fps = round(fps_frame_count / elapsed, 1)
            fps_frame_count = 0
            fps_start_time = time.time()
        
        cv2.putText(display_frame, f"FPS: {state.fps}", (frame_w - 100, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', display_frame, 
                                 [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        state.latest_frame = buffer.tobytes()
        state.total_frames += 1
    
    # Cleanup
    if state.cap:
        state.cap.release()
    if state.landmarker:
        state.landmarker.close()
    stop_alarm()
    print("[STOP] Detection stopped")


# ─── Video Stream Generator ───────────────────────────────────────
def generate_frames():
    """Generator function for MJPEG video stream."""
    while True:
        if state.latest_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.latest_frame + b'\r\n')
        else:
            # Send a blank frame when no video
            blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            blank[:] = (20, 15, 40)
            cv2.putText(blank, "Waiting for camera...", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 140), 2)
            _, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


# ─── Flask Routes ─────────────────────────────────────────────────
@app.route('/')
def index():
    """Serve the dashboard page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Return current detection status as JSON."""
    with state.lock:
        uptime = 0
        if state.start_time:
            uptime = int(time.time() - state.start_time)
        
        return jsonify({
            'running': state.running,
            'face_detected': state.face_detected,
            'ear': state.ear_value,
            'mar': state.mar_value,
            'eye_status': state.eye_status,
            'mouth_status': state.mouth_status,
            'fatigue_level': state.fatigue_level,
            'alert_active': state.alert_active,
            'drowsy_alert': state.drowsy_alert,
            'yawn_alert': state.yawn_alert,
            'blink_count': state.blink_count,
            'yawn_count': state.yawn_count,
            'fps': state.fps,
            'uptime': uptime,
            'sound_enabled': state.sound_enabled,
            'privacy_mode': state.privacy_mode,
            'camera_source': str(state.camera_source)
        })

@app.route('/start', methods=['POST'])
def start_detection():
    """Start the detection system."""
    if state.running:
        return jsonify({'status': 'already_running'})
    
    # Get camera source from request
    data = request.get_json(silent=True) or {}
    camera_source = data.get('camera_source', '')
    
    if camera_source and camera_source.strip():
        state.camera_source = camera_source.strip()
    else:
        state.camera_source = 0  # Default to laptop webcam
    
    state.running = True
    state.blink_count = 0
    state.yawn_count = 0
    state.closed_frames = 0
    state.yawn_frames = 0
    state.fatigue_level = 0
    state.alert_active = False
    state.drowsy_alert = False
    state.yawn_alert = False
    state.latest_frame = None
    
    thread = threading.Thread(target=detection_loop, daemon=True)
    thread.start()
    
    return jsonify({'status': 'started', 'camera': str(state.camera_source)})

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop the detection system."""
    state.running = False
    stop_alarm()
    return jsonify({'status': 'stopped'})

@app.route('/toggle_sound', methods=['POST'])
def toggle_sound():
    """Toggle alarm sound on/off."""
    state.sound_enabled = not state.sound_enabled
    if not state.sound_enabled:
        stop_alarm()
    return jsonify({'sound_enabled': state.sound_enabled})

@app.route('/toggle_privacy', methods=['POST'])
def toggle_privacy():
    """Toggle privacy mode (landmarks only)."""
    state.privacy_mode = not state.privacy_mode
    return jsonify({'privacy_mode': state.privacy_mode})

@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    """Update detection thresholds at runtime."""
    data = request.get_json(silent=True) or {}
    
    if 'ear_threshold' in data:
        state.ear_threshold = float(data['ear_threshold'])
    if 'mar_threshold' in data:
        state.mar_threshold = float(data['mar_threshold'])
    if 'ear_consec_frames' in data:
        state.ear_consec_frames = int(data['ear_consec_frames'])
    if 'mar_consec_frames' in data:
        state.mar_consec_frames = int(data['mar_consec_frames'])
    
    return jsonify({
        'ear_threshold': state.ear_threshold,
        'mar_threshold': state.mar_threshold,
        'ear_consec_frames': state.ear_consec_frames,
        'mar_consec_frames': state.mar_consec_frames
    })


# ─── Main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  AI Driver Drowsiness Detection System")
    print("=" * 55)
    print(f"  Dashboard: http://localhost:{config.FLASK_PORT}")
    print(f"  Camera:    {config.CAMERA_SOURCE}")
    print(f"  EAR threshold: {config.EAR_THRESHOLD}")
    print(f"  MAR threshold: {config.MAR_THRESHOLD}")
    print("=" * 55)
    
    # Generate alarm if missing
    if not os.path.exists(config.ALARM_SOUND_PATH):
        print("[...] Generating alarm sound...")
        try:
            from generate_alarm import generate_alarm
            generate_alarm()
        except Exception as e:
            print(f"[WARN] Could not generate alarm: {e}")
    
    # Check model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_landmarker.task')
    if not os.path.exists(model_path):
        print("[WARN] face_landmarker.task not found!")
        print("  Downloading model file...")
        try:
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("[OK] Model downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] Could not download model: {e}")
            print("  Please download manually from:")
            print("  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG_MODE,
        threaded=True
    )
