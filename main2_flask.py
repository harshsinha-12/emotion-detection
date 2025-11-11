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
MODEL_PATH = "/Users/harshsinha/VS Code/Emotion Detection/models/emotion_densenet.keras"
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
EMOTION_COLORS = {
    'Anger': '#ef4444',
    'Disgust': '#10b981',
    'Fear': '#8b5cf6',
    'Happy': '#f59e0b',
    'Neutral': '#6b7280',
    'Sadness': '#3b82f6',
    'Surprise': '#ec4899'
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
    <title>Emotion Detection System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .card {
            backdrop-filter: blur(10px);
            background: rgba(255, 255, 255, 0.95);
        }
        .pulse-dot {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        #videoFeed {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        .stat-card {
            transition: transform 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-4px);
        }
    </style>
</head>
<body class="gradient-bg min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-5xl font-bold text-white mb-2">🎭 Emotion Detection System</h1>
            <p class="text-purple-200 text-lg">Real-time AI-powered facial emotion recognition</p>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Video Feed (Left Column) -->
            <div class="lg:col-span-2">
                <div class="card rounded-2xl shadow-2xl p-6">
                    <div class="flex items-center justify-between mb-4">
                        <h2 class="text-2xl font-bold text-gray-800">Live Feed</h2>
                        <div class="flex items-center gap-2">
                            <span id="statusDot" class="w-3 h-3 rounded-full bg-gray-400"></span>
                            <span id="statusText" class="text-sm font-medium text-gray-600">Inactive</span>
                        </div>
                    </div>
                    
                    <div class="relative bg-gray-900 rounded-xl overflow-hidden" style="min-height: 480px;">
                        <img id="videoFeed" src="/placeholder" alt="Video Feed" class="w-full">
                        <div id="noVideoPlaceholder" class="absolute inset-0 flex items-center justify-center bg-gray-800">
                            <div class="text-center text-gray-400">
                                <svg class="w-24 h-24 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                                </svg>
                                <p class="text-xl">Camera Inactive</p>
                                <p class="text-sm mt-2">Click "Start Detection" to begin</p>
                            </div>
                        </div>
                    </div>

                    <!-- Controls -->
                    <div class="flex gap-3 mt-6">
                        <button id="startBtn" onclick="startDetection()" 
                            class="flex-1 bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg">
                            ▶ Start Detection
                        </button>
                        <button id="stopBtn" onclick="stopDetection()" disabled
                            class="flex-1 bg-red-500 hover:bg-red-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed">
                            ⏸ Stop Detection
                        </button>
                        <button onclick="resetStats()"
                            class="bg-purple-500 hover:bg-purple-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors shadow-lg">
                            🔄 Reset Stats
                        </button>
                    </div>
                </div>

                <!-- Emotion Distribution Chart -->
                <div class="card rounded-2xl shadow-2xl p-6 mt-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Emotion Distribution</h2>
                    <canvas id="emotionChart" height="80"></canvas>
                </div>
            </div>

            <!-- Stats Panel (Right Column) -->
            <div class="space-y-6">
                <!-- Real-time Stats -->
                <div class="card rounded-2xl shadow-2xl p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Statistics</h2>
                    
                    <div class="space-y-4">
                        <div class="stat-card bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-4 text-white">
                            <div class="text-sm opacity-90">Total Detections</div>
                            <div id="totalDetections" class="text-3xl font-bold">0</div>
                        </div>
                        
                        <div class="stat-card bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-4 text-white">
                            <div class="text-sm opacity-90">Current FPS</div>
                            <div id="currentFPS" class="text-3xl font-bold">0</div>
                        </div>
                        
                        <div class="stat-card bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl p-4 text-white">
                            <div class="text-sm opacity-90">Most Common</div>
                            <div id="mostCommon" class="text-3xl font-bold">-</div>
                        </div>
                    </div>
                </div>

                <!-- Emotion Breakdown -->
                <div class="card rounded-2xl shadow-2xl p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Emotion Counts</h2>
                    <div id="emotionBreakdown" class="space-y-3">
                        <!-- Will be populated by JS -->
                    </div>
                </div>

                <!-- Recent Detections -->
                <div class="card rounded-2xl shadow-2xl p-6">
                    <h2 class="text-2xl font-bold text-gray-800 mb-4">Recent Detections</h2>
                    <div id="recentDetections" class="space-y-2 max-h-64 overflow-y-auto">
                        <p class="text-gray-500 text-sm">No detections yet</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="text-center mt-8 text-white">
            <p class="text-sm opacity-75">Powered by TensorFlow & DenseNet169 | Flask Backend</p>
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

        // Initialize chart
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
                        borderRadius: 8,
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            padding: 12,
                            titleFont: { size: 14 },
                            bodyFont: { size: 13 }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: { precision: 0 },
                            grid: { color: 'rgba(0, 0, 0, 0.05)' }
                        },
                        x: {
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
                    document.getElementById('statusDot').className = 'w-3 h-3 rounded-full bg-green-500 pulse-dot';
                    document.getElementById('statusText').textContent = 'Active';
                    document.getElementById('statusText').className = 'text-sm font-medium text-green-600';
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    updateStats();
                }
            } catch (error) {
                console.error('Error starting detection:', error);
                alert('Failed to start detection. Please check console.');
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
                    document.getElementById('statusDot').className = 'w-3 h-3 rounded-full bg-gray-400';
                    document.getElementById('statusText').textContent = 'Inactive';
                    document.getElementById('statusText').className = 'text-sm font-medium text-gray-600';
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                }
            } catch (error) {
                console.error('Error stopping detection:', error);
            }
        }

        async function resetStats() {
            if (confirm('Are you sure you want to reset all statistics?')) {
                try {
                    const response = await fetch('/reset', { method: 'POST' });
                    const data = await response.json();
                    if (data.status === 'reset') {
                        updateStats();
                    }
                } catch (error) {
                    console.error('Error resetting stats:', error);
                }
            }
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
                
                // Update emotion breakdown
                const breakdownHTML = Object.entries(data.emotion_counts)
                    .sort((a, b) => b[1] - a[1])
                    .map(([emotion, count]) => {
                        const percentage = data.total_detections > 0 
                            ? ((count / data.total_detections) * 100).toFixed(1) 
                            : 0;
                        return `
                            <div class="flex items-center justify-between">
                                <div class="flex items-center gap-2">
                                    <div class="w-3 h-3 rounded-full" style="background-color: ${emotionColors[emotion]}"></div>
                                    <span class="font-medium text-gray-700">${emotion}</span>
                                </div>
                                <div class="text-right">
                                    <span class="font-bold text-gray-800">${count}</span>
                                    <span class="text-sm text-gray-500 ml-1">(${percentage}%)</span>
                                </div>
                            </div>
                        `;
                    }).join('');
                document.getElementById('emotionBreakdown').innerHTML = breakdownHTML;
                
                // Update recent detections
                if (data.recent_detections.length > 0) {
                    const recentHTML = data.recent_detections
                        .slice(-10)
                        .reverse()
                        .map(detection => {
                            const time = new Date(detection.timestamp).toLocaleTimeString();
                            return `
                                <div class="flex items-center justify-between py-2 px-3 bg-gray-50 rounded-lg">
                                    <span class="font-medium" style="color: ${emotionColors[detection.emotion]}">${detection.emotion}</span>
                                    <span class="text-sm text-gray-600">${detection.confidence.toFixed(1)}%</span>
                                    <span class="text-xs text-gray-400">${time}</span>
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