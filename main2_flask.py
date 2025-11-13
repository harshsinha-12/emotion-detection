import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template_string, jsonify, request
from collections import deque
from datetime import datetime
import time
import json
import base64

# Configuration
MODEL_PATH = "models/emotion_densenet.keras"  # Updated path
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
EMOTION_COLORS = {
    'Anger': '#ef4444',
    'Disgust': '#10b981',
    'Fear': '#8b5cf6',
    'Happy': '#fbbf24',
    'Neutral': '#6b7280',
    'Sadness': '#3b82f6',
    'Surprise': '#ec4899'
}
EMOTION_EMOJIS = {
    'Anger': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😊',
    'Neutral': '😐',
    'Sadness': '😢',
    'Surprise': '😲'
}

app = Flask(__name__)

# Global state
class AppState:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.camera = None
        self.camera_active = False
        self.emotion_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.detection_count = 0
        self.stats = {emotion: 0 for emotion in CLASS_LABELS}
        
state = AppState()


def load_model(path: str):
    """Load the emotion detection model"""
    try:
        model = tf.keras.models.load_model(path)
        print(f"✓ Model loaded successfully from {path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def preprocess_bgr_roi(bgr_roi: np.ndarray) -> np.ndarray:
    """Preprocess face ROI for model input"""
    rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT)).astype("float32")
    rgb = tf.keras.applications.densenet.preprocess_input(rgb)
    return np.expand_dims(rgb, 0)


def initialize_resources():
    """Initialize model, face cascade, and camera"""
    if state.model is None:
        state.model = load_model(MODEL_PATH)
        if state.model is None:
            return False
    
    if state.face_cascade is None:
        try:
            from cv2 import data as cv2_data
            haar_dir = cv2_data.haarcascades
        except Exception:
            base_dir = os.path.dirname(cv2.__file__)
            haar_dir = os.path.join(base_dir, "data", "haarcascades")
        
        cascade_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
        state.face_cascade = cv2.CascadeClassifier(cascade_path)
        print(f"✓ Face cascade loaded")
    
    return True


def start_camera():
    """Start the camera"""
    if state.camera is None or not state.camera.isOpened():
        state.camera = cv2.VideoCapture(0)
        state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        state.camera.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(0.5)  # Allow camera to warm up
        
    state.camera_active = True
    print("✓ Camera started")


def stop_camera():
    """Stop the camera"""
    state.camera_active = False
    if state.camera is not None:
        state.camera.release()
        state.camera = None
    print("✓ Camera stopped")


def calculate_fps():
    """Calculate current FPS"""
    current_time = time.time()
    fps = 1 / (current_time - state.last_frame_time) if state.last_frame_time else 0
    state.last_frame_time = current_time
    state.fps_history.append(fps)
    return sum(state.fps_history) / len(state.fps_history)


def generate_frames():
    """Generate video frames with emotion detection"""
    while state.camera_active:
        if state.camera is None or not state.camera.isOpened():
            break
            
        success, frame = state.camera.read()
        if not success:
            break

        # Calculate FPS
        fps = calculate_fps()

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = state.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(30, 30)
        )

        detected_emotions = []
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract and process ROI
            roi = frame[y:y + h, x:x + w]
            x_input = preprocess_bgr_roi(roi)
            
            # Predict emotion
            probs = state.model.predict(x_input, verbose=0)[0]
            idx = int(np.argmax(probs))
            emotion = CLASS_LABELS[idx]
            confidence = probs[idx] * 100
            
            # Update statistics
            state.stats[emotion] += 1
            state.detection_count += 1
            detected_emotions.append({
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            # Draw emotion label with background
            label = f"{emotion} ({confidence:.1f}%)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x, y - text_height - 15),
                (x + text_width + 10, y),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame, 
                label, 
                (x + 5, y - 8), 
                font, 
                font_scale, 
                (0, 0, 0), 
                thickness
            )
            
            # Draw face number
            cv2.putText(
                frame,
                f"Face #{i+1}",
                (x + 5, y + h - 10),
                font,
                0.5,
                (0, 255, 0),
                1
            )

        # Add detected emotions to history
        if detected_emotions:
            state.emotion_history.extend(detected_emotions)

        # Draw FPS counter
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Draw face count
        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Encode frame
        ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Emotion Detection System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Inter', sans-serif;
        }

        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 200% 200%;
            animation: gradientShift 15s ease infinite;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .card {
            backdrop-filter: blur(16px) saturate(180%);
            background: rgba(255, 255, 255, 0.97);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .glass-card {
            backdrop-filter: blur(12px) saturate(180%);
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.4);
        }

        .pulse-dot {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }
            50% {
                opacity: 0.7;
                transform: scale(1.1);
            }
        }

        .glow {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
            animation: glow 3s ease-in-out infinite;
        }

        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
            50% { box-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
        }

        #videoFeed {
            max-width: 100%;
            height: auto;
            border-radius: 16px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            transition: all 0.3s ease;
        }

        #videoFeed:hover {
            transform: scale(1.01);
            box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.3);
        }

        .stat-card {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
        }

        .stat-card:hover {
            transform: translateY(-6px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
        }

        .btn-primary {
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .btn-primary::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .btn-primary:hover::before {
            width: 300px;
            height: 300px;
        }

        .fade-in {
            animation: fadeIn 0.6s ease-in;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .emotion-badge {
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .emotion-badge:hover {
            transform: scale(1.05);
        }

        .progress-ring {
            transform: rotate(-90deg);
        }

        .progress-ring-circle {
            transition: stroke-dashoffset 0.35s;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.7);
        }

        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: rgba(0, 0, 0, 0.9);
            color: #fff;
            text-align: center;
            border-radius: 8px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body class="gradient-bg min-h-screen">
    <div class="container mx-auto px-4 py-8 fade-in">
        <!-- Header -->
        <div class="text-center mb-10">
            <div class="inline-block mb-4">
                <h1 class="text-6xl font-extrabold text-white mb-3 tracking-tight">
                    <span class="inline-block animate-pulse">🎭</span>
                    AI Emotion Detection
                </h1>
                <div class="h-1 bg-gradient-to-r from-transparent via-white to-transparent opacity-50 rounded-full"></div>
            </div>
            <p class="text-purple-100 text-xl font-light tracking-wide">Real-time facial emotion recognition powered by Deep Learning</p>
            <p class="text-purple-200 text-sm font-medium mt-2 opacity-80">DenseNet169 • TensorFlow • OpenCV</p>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Video Feed (Left Column) -->
            <div class="lg:col-span-2 space-y-6">
                <div class="card rounded-3xl shadow-2xl p-7">
                    <div class="flex items-center justify-between mb-6">
                        <div>
                            <h2 class="text-3xl font-bold text-gray-800 mb-1">Live Camera Feed</h2>
                            <p class="text-gray-500 text-sm">Real-time emotion detection & analysis</p>
                        </div>
                        <div class="flex items-center gap-3 px-4 py-2 rounded-full glass-card">
                            <span id="statusDot" class="w-3 h-3 rounded-full bg-gray-400 shadow-lg"></span>
                            <span id="statusText" class="text-sm font-semibold text-gray-700">Inactive</span>
                        </div>
                    </div>

                    <div class="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-2xl overflow-hidden shadow-inner" style="min-height: 480px;">
                        <img id="videoFeed" src="/placeholder" alt="Video Feed" class="w-full">
                        <div id="noVideoPlaceholder" class="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
                            <div class="text-center text-gray-400">
                                <div class="inline-block p-6 bg-gray-800 rounded-full mb-6 shadow-2xl">
                                    <svg class="w-24 h-24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                                    </svg>
                                </div>
                                <p class="text-2xl font-semibold mb-2">Camera Inactive</p>
                                <p class="text-sm opacity-75">Click "Start Detection" below to begin</p>
                            </div>
                        </div>
                    </div>

                    <!-- Controls -->
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mt-7">
                        <button id="startBtn" onclick="startDetection()"
                            class="btn-primary bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 text-white font-bold py-4 px-8 rounded-xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
                            </svg>
                            <span>Start Detection</span>
                        </button>
                        <button id="stopBtn" onclick="stopDetection()" disabled
                            class="btn-primary bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 text-white font-bold py-4 px-8 rounded-xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center justify-center gap-2">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
                            </svg>
                            <span>Stop Detection</span>
                        </button>
                        <button onclick="resetStats()"
                            class="btn-primary bg-gradient-to-r from-purple-500 to-indigo-600 hover:from-purple-600 hover:to-indigo-700 text-white font-bold py-4 px-8 rounded-xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 flex items-center justify-center gap-2">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
                            </svg>
                            <span>Reset Stats</span>
                        </button>
                    </div>
                </div>

                <!-- Emotion Distribution Chart -->
                <div class="card rounded-3xl shadow-2xl p-7">
                    <div class="mb-6">
                        <h2 class="text-2xl font-bold text-gray-800 mb-1">Emotion Distribution</h2>
                        <p class="text-gray-500 text-sm">Visual breakdown of detected emotions</p>
                    </div>
                    <canvas id="emotionChart" height="80"></canvas>
                </div>
            </div>

            <!-- Stats Panel (Right Column) -->
            <div class="space-y-6">
                <!-- Real-time Stats -->
                <div class="card rounded-3xl shadow-2xl p-7">
                    <div class="mb-6">
                        <h2 class="text-2xl font-bold text-gray-800 mb-1">Live Statistics</h2>
                        <p class="text-gray-500 text-sm">Real-time performance metrics</p>
                    </div>

                    <div class="space-y-4">
                        <div class="stat-card bg-gradient-to-br from-blue-500 via-blue-600 to-indigo-600 rounded-2xl p-5 text-white shadow-lg">
                            <div class="flex items-center justify-between mb-2">
                                <div class="text-sm font-medium opacity-90">Total Detections</div>
                                <svg class="w-5 h-5 opacity-75" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
                                </svg>
                            </div>
                            <div id="totalDetections" class="text-4xl font-extrabold tracking-tight">0</div>
                            <div class="text-xs mt-2 opacity-75">detections processed</div>
                        </div>

                        <div class="stat-card bg-gradient-to-br from-emerald-500 via-green-600 to-teal-600 rounded-2xl p-5 text-white shadow-lg">
                            <div class="flex items-center justify-between mb-2">
                                <div class="text-sm font-medium opacity-90">Processing Speed</div>
                                <svg class="w-5 h-5 opacity-75" fill="currentColor" viewBox="0 0 20 20">
                                    <path fill-rule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clip-rule="evenodd"/>
                                </svg>
                            </div>
                            <div id="currentFPS" class="text-4xl font-extrabold tracking-tight">0</div>
                            <div class="text-xs mt-2 opacity-75">frames per second</div>
                        </div>

                        <div class="stat-card bg-gradient-to-br from-purple-500 via-violet-600 to-fuchsia-600 rounded-2xl p-5 text-white shadow-lg">
                            <div class="flex items-center justify-between mb-2">
                                <div class="text-sm font-medium opacity-90">Dominant Emotion</div>
                                <svg class="w-5 h-5 opacity-75" fill="currentColor" viewBox="0 0 20 20">
                                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                                </svg>
                            </div>
                            <div id="mostCommon" class="text-4xl font-extrabold tracking-tight">-</div>
                            <div class="text-xs mt-2 opacity-75">most detected emotion</div>
                        </div>
                    </div>
                </div>

                <!-- Emotion Breakdown -->
                <div class="card rounded-3xl shadow-2xl p-7">
                    <div class="mb-6">
                        <h2 class="text-2xl font-bold text-gray-800 mb-1">Emotion Breakdown</h2>
                        <p class="text-gray-500 text-sm">Detailed count by emotion type</p>
                    </div>
                    <div id="emotionBreakdown" class="space-y-3">
                        <!-- Will be populated by JS -->
                    </div>
                </div>

                <!-- Recent Detections -->
                <div class="card rounded-3xl shadow-2xl p-7">
                    <div class="mb-6">
                        <h2 class="text-2xl font-bold text-gray-800 mb-1">Recent Detections</h2>
                        <p class="text-gray-500 text-sm">Latest emotion readings</p>
                    </div>
                    <div id="recentDetections" class="space-y-2 max-h-80 overflow-y-auto pr-2">
                        <div class="text-center py-8">
                            <svg class="w-16 h-16 mx-auto text-gray-300 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
                            </svg>
                            <p class="text-gray-400 text-sm font-medium">No detections yet</p>
                            <p class="text-gray-400 text-xs mt-1">Start detection to see results</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="text-center mt-12 pb-6">
            <div class="inline-block glass-card rounded-full px-8 py-4 shadow-lg">
                <p class="text-white font-semibold text-sm mb-1">🚀 Powered by Advanced AI</p>
                <p class="text-purple-200 text-xs opacity-90">TensorFlow • DenseNet169 • OpenCV • Flask</p>
            </div>
        </div>
    </div>

    <script>
        let isActive = false;
        let chart = null;
        const emotionColors = {
            'Anger': '#ef4444',
            'Disgust': '#10b981',
            'Fear': '#8b5cf6',
            'Happy': '#f59e0b',
            'Neutral': '#6b7280',
            'Sadness': '#3b82f6',
            'Surprise': '#ec4899'
        };

        // Initialize chart with improved styling
        function initChart() {
            const ctx = document.getElementById('emotionChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise'],
                    datasets: [{
                        label: 'Detection Count',
                        data: [0, 0, 0, 0, 0, 0, 0],
                        backgroundColor: Object.values(emotionColors),
                        borderRadius: 10,
                        borderWidth: 0,
                        hoverBackgroundColor: Object.values(emotionColors).map(c => c + 'dd'),
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    animation: {
                        duration: 750,
                        easing: 'easeInOutQuart'
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.9)',
                            padding: 16,
                            titleFont: { size: 15, weight: 'bold', family: 'Inter' },
                            bodyFont: { size: 14, family: 'Inter' },
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            cornerRadius: 8,
                            displayColors: true,
                            callbacks: {
                                label: function(context) {
                                    return context.parsed.y + ' detections';
                                }
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                precision: 0,
                                font: { family: 'Inter', size: 12 },
                                color: '#6b7280'
                            },
                            grid: {
                                color: 'rgba(0, 0, 0, 0.06)',
                                drawBorder: false
                            }
                        },
                        x: {
                            ticks: {
                                font: { family: 'Inter', size: 12, weight: '500' },
                                color: '#374151'
                            },
                            grid: { display: false }
                        }
                    }
                }
            });
        }

        async function startDetection() {
            try {
                const response = await fetch('/start', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    isActive = true;
                    document.getElementById('videoFeed').src = '/video_feed?' + new Date().getTime();
                    document.getElementById('noVideoPlaceholder').style.display = 'none';
                    document.getElementById('statusDot').className = 'w-3 h-3 rounded-full bg-green-500 pulse-dot shadow-lg shadow-green-500/50';
                    document.getElementById('statusText').textContent = 'Active';
                    document.getElementById('statusText').className = 'text-sm font-bold text-green-600';
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    updateStats();

                    // Show success notification
                    showNotification('Detection started successfully!', 'success');
                }
            } catch (error) {
                console.error('Error starting detection:', error);
                showNotification('Failed to start detection. Please check console.', 'error');
            }
        }

        async function stopDetection() {
            try {
                const response = await fetch('/stop', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'stopped') {
                    isActive = false;
                    document.getElementById('videoFeed').src = '/placeholder';
                    document.getElementById('noVideoPlaceholder').style.display = 'flex';
                    document.getElementById('statusDot').className = 'w-3 h-3 rounded-full bg-gray-400 shadow-lg';
                    document.getElementById('statusText').textContent = 'Inactive';
                    document.getElementById('statusText').className = 'text-sm font-semibold text-gray-700';
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;

                    // Show success notification
                    showNotification('Detection stopped', 'info');
                }
            } catch (error) {
                console.error('Error stopping detection:', error);
                showNotification('Failed to stop detection', 'error');
            }
        }

        async function resetStats() {
            if (confirm('Are you sure you want to reset all statistics?')) {
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const data = await response.json();
                    if (data.status === 'reset') {
                        updateStats();
                        showNotification('Statistics reset successfully', 'success');
                    }
                } catch (error) {
                    console.error('Error resetting stats:', error);
                    showNotification('Failed to reset statistics', 'error');
                }
            }
        }

        // Notification system
        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            const colors = {
                success: 'from-green-500 to-emerald-600',
                error: 'from-red-500 to-rose-600',
                info: 'from-blue-500 to-indigo-600'
            };
            notification.className = `fixed top-6 right-6 bg-gradient-to-r ${colors[type]} text-white px-6 py-4 rounded-xl shadow-2xl z-50 transform translate-x-0 transition-all duration-300 flex items-center gap-3`;
            notification.innerHTML = `
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                </svg>
                <span class="font-semibold">${message}</span>
            `;
            document.body.appendChild(notification);
            setTimeout(() => {
                notification.style.transform = 'translateX(400px)';
                setTimeout(() => notification.remove(), 300);
            }, 3000);
        }

        async function updateStats() {
            if (!isActive && chart.data.datasets[0].data.every(v => v === 0)) {
                return;
            }

            try {
                const response = await fetch('/stats');
                const data = await response.json();
                
                // Update total detections
                document.getElementById('totalDetections').textContent = data.total_detections.toLocaleString();
                
                // Update FPS
                document.getElementById('currentFPS').textContent = data.fps.toFixed(1);
                
                // Update most common emotion
                const maxEmotion = Object.entries(data.emotion_counts)
                    .reduce((max, [emotion, count]) => count > max[1] ? [emotion, count] : max, ['None', 0]);
                document.getElementById('mostCommon').textContent = maxEmotion[1] > 0 ? maxEmotion[0] : '-';
                
                // Update chart
                chart.data.datasets[0].data = [
                    data.emotion_counts['Anger'],
                    data.emotion_counts['Disgust'],
                    data.emotion_counts['Fear'],
                    data.emotion_counts['Happy'],
                    data.emotion_counts['Neutral'],
                    data.emotion_counts['Sadness'],
                    data.emotion_counts['Surprise']
                ];
                chart.update('none');
                
                // Update emotion breakdown with enhanced visuals
                const breakdownHTML = Object.entries(data.emotion_counts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([emotion, count]) => {
                        const percentage = data.total_detections > 0
                            ? ((count / data.total_detections) * 100).toFixed(1)
                            : 0;
                        const barWidth = percentage;
                        return `
                            <div class="emotion-badge glass-card rounded-xl p-4 hover:shadow-lg transition-all">
                                <div class="flex items-center justify-between mb-2">
                                    <div class="flex items-center gap-3">
                                        <div class="w-4 h-4 rounded-full shadow-lg" style="background-color: ${emotionColors[emotion]}"></div>
                                        <span class="font-bold text-gray-800 text-lg">${emotion}</span>
                                    </div>
                                    <div class="text-right">
                                        <span class="font-extrabold text-gray-900 text-xl">${count}</span>
                                        <span class="text-sm text-gray-500 ml-2 font-medium">${percentage}%</span>
                                    </div>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                    <div class="h-2 rounded-full transition-all duration-500" style="width: ${barWidth}%; background-color: ${emotionColors[emotion]}"></div>
                                </div>
                            </div>
                        `;
                    }).join('');
                document.getElementById('emotionBreakdown').innerHTML = breakdownHTML;
                
                // Update recent detections with enhanced styling
                if (data.recent_detections.length > 0) {
                    const recentHTML = data.recent_detections
                        .slice(-10)
                        .reverse()
                        .map((detection, index) => {
                            const time = new Date(detection.timestamp).toLocaleTimeString();
                            return `
                                <div class="glass-card rounded-xl py-3 px-4 hover:shadow-md transition-all transform hover:scale-102 fade-in" style="animation-delay: ${index * 0.05}s">
                                    <div class="flex items-center justify-between">
                                        <div class="flex items-center gap-3 flex-1">
                                            <div class="w-3 h-3 rounded-full shadow-lg" style="background-color: ${emotionColors[detection.emotion]}"></div>
                                            <span class="font-bold text-gray-800">${detection.emotion}</span>
                                        </div>
                                        <div class="flex items-center gap-4">
                                            <span class="px-3 py-1 rounded-full text-xs font-bold text-white shadow-sm" style="background-color: ${emotionColors[detection.emotion]}">${detection.confidence.toFixed(1)}%</span>
                                            <span class="text-xs text-gray-500 font-medium">${time}</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('');
                    document.getElementById('recentDetections').innerHTML = recentHTML;
                }
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }

        // Initialize
        initChart();
        setInterval(updateStats, 1000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/start", methods=["POST"])
def start():
    """Start emotion detection"""
    try:
        if not initialize_resources():
            return jsonify({"status": "error", "message": "Failed to initialize resources"}), 500
        
        start_camera()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stop", methods=["POST"])
def stop():
    """Stop emotion detection"""
    try:
        stop_camera()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset statistics"""
    try:
        state.emotion_history.clear()
        state.stats = {emotion: 0 for emotion in CLASS_LABELS}
        state.detection_count = 0
        return jsonify({"status": "reset"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/stats")
def stats():
    """Get current statistics"""
    recent_detections = list(state.emotion_history)[-10:] if state.emotion_history else []
    
    return jsonify({
        "total_detections": state.detection_count,
        "emotion_counts": state.stats,
        "fps": sum(state.fps_history) / len(state.fps_history) if state.fps_history else 0,
        "recent_detections": recent_detections
    })


@app.route("/video_feed")
def video_feed():
    """Video streaming route"""
    if not state.camera_active:
        return Response("Camera not active", status=503)
    
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/placeholder")
def placeholder():
    """Return placeholder image when camera is off"""
    return "", 204


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎭 EMOTION DETECTION SYSTEM")
    print("="*60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Server: http://localhost:8501")
    print("="*60 + "\n")
    
    # Initialize resources on startup
    if not initialize_resources():
        print("⚠️  Warning: Failed to initialize all resources")
        print("   Make sure the model path is correct")
    
    try:
        app.run(host="0.0.0.0", port=8501, debug=False, threaded=True)
    finally:
        # Cleanup
        if state.camera is not None:
            state.camera.release()
        print("\n✓ Server shutdown complete")