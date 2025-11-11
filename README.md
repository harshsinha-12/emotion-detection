## Emotion Detection (Realtime Webcam)

AI-powered real-time facial emotion recognition using TensorFlow/Keras (DenseNet169), OpenCV face detection, with two ready-to-run UIs:

- Streamlit app with in-browser WebRTC webcam
- Flask server with a modern dashboard (Tailwind + Chart.js)

Supported emotions: Anger, Disgust, Fear, Happy, Neutral, Sadness, Surprise.


### Features
- Real-time webcam inference with FPS overlay
- Face detection via OpenCV Haar Cascade
- DenseNet169-based classifier with proper image preprocessing
- Streamlit WebRTC pipeline or Flask MJPEG stream
- Live charts and statistics in the Flask UI


### Project Structure
```
Emotion Detection/
├─ main2_streamlit.py         # Streamlit + streamlit-webrtc app
├─ main2_flask.py             # Flask server with live dashboard
├─ models/
│  └─ emotion_densenet.keras  # Keras model file (see Model section)
├─ requirements.txt
├─ emotion-detection.ipynb    # Notebook (training/experiments)
├─ inference_demo.ipynb       # Inference demo notebook
└─ data/                      # Optional dataset folder (if present)
```


### Requirements
- Python 3.10 or 3.11 recommended
- macOS, Linux, or Windows with a working webcam
- For Streamlit WebRTC:
  - FFmpeg is recommended for media handling
    - macOS: `brew install ffmpeg`
    - Ubuntu: `sudo apt-get install -y ffmpeg`


### Setup
1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows PowerShell
```

2) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: If `cv2` causes a pip conflict, prefer `opencv-python` only and remove the `cv2==...` line in `requirements.txt`.


### Model
Both apps expect a Keras model at:

```
models/emotion_densenet.keras
```

If your model file has a different name (e.g., `emotion_densenet169.keras`), either:
- Rename it to `emotion_densenet.keras`, or
- Update `MODEL_PATH` at the top of `main2_streamlit.py` and `main2_flask.py` to point to the correct filename.

Example path setting in code:
```python
MODEL_PATH = "models/emotion_densenet.keras"
```


### Training (from `emotion-detection.ipynb`)
This summarizes the end-to-end training pipeline implemented in the notebook.

- Dataset expected on disk in a directory layout consumable by `flow_from_directory`:

```
data/
├─ train/
│  ├─ Anger/ ...images...
│  ├─ Disgust/ ...
│  ├─ Fear/ ...
│  ├─ Happy/ ...
│  ├─ Neutral/ ...
│  ├─ Sadness/ ...
│  └─ Surprise/ ...
├─ val/
│  ├─ Anger/ ...
│  └─ ... same classes ...
└─ test/
   ├─ Anger/ ...
   └─ ... same classes ...
```

- Key constants used during training:
  - `IMG_HEIGHT = 48`, `IMG_WIDTH = 48`
  - `NUM_CLASSES = 7`
  - `CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']`

- Data input pipeline with `ImageDataGenerator`:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "data/train",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    "data/val",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    "data/test",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)
```

- Model: DenseNet169 transfer learning
  - Base: `tf.keras.applications.DenseNet169(include_top=False)`
  - Preprocessing consistent with DenseNet (`tf.keras.applications.densenet.preprocess_input` at inference)
  - Global average pooling
  - Final Dense layer: `Dense(NUM_CLASSES, activation="softmax")`

```python
import tensorflow as tf

feature_extractor = tf.keras.applications.DenseNet169(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights="imagenet"
)
feature_extractor.trainable = False  # warm-up/freeze stage

inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = feature_extractor(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="classification")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

- Callbacks used:
  - EarlyStopping on `val_loss` with patience (stops when validation loss plateaus)

```python
earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

- Typical training loop:

```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=[earlyStoppingCallback]
)
```

- Optional fine-tuning:
  - Unfreeze top DenseNet blocks and continue training with a lower learning rate (e.g., 1e-4 to 1e-5).

- Evaluation:
  - Use `test_generator` for final accuracy and confusion matrix.

```python
test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# Example confusion matrix (abbrev.)
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

y_true = test_generator.classes
y_prob = model.predict(test_generator)
y_pred = np.argmax(y_prob, axis=1)
cm = confusion_matrix(y_true, y_pred)
```

- Saving/exporting the model in native Keras format used by the apps:

```python
save_path = "models/emotion_densenet.keras"
model.save(save_path)  # native Keras .keras format
```

After saving, confirm the app `MODEL_PATH` points to `models/emotion_densenet.keras` (or update it accordingly).


### Running the Streamlit App
The Streamlit app uses WebRTC to access your webcam in the browser.

```bash
streamlit run main2_streamlit.py
```

Then open the displayed local URL in your browser, grant camera permission, and you should see real-time detections with emotion labels.

Notes:
- On macOS, you may need to grant Terminal/IDE camera access in System Settings > Privacy & Security > Camera.
- If you see a blank feed, ensure no other app is using the camera and refresh the page.


### Running the Flask Server
The Flask server serves an HTML dashboard with an MJPEG video feed and live statistics.

```bash
python main2_flask.py
```

Open `http://localhost:8501` in your browser. Click “Start Detection” to start the camera and begin inference. Use “Stop Detection” to release the camera. “Reset Stats” clears counters and recent detections.


### How It Works (High-level)
- Face detection: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`)
- Preprocessing: Convert BGR → RGB, resize to 48×48, DenseNet preprocessing
- Model: TensorFlow/Keras DenseNet169 adapted for 7-class emotion classification
- Streamlit: `streamlit-webrtc` processes frames via a `VideoProcessor`
- Flask: frames encoded to JPEG and streamed via a multipart response


### Troubleshooting
- Streamlit page loads but no video:
  - Ensure camera permissions are granted to your browser and terminal/IDE.
  - Close other apps that might be using the camera (Zoom, Teams, FaceTime).
  - Install FFmpeg (`brew install ffmpeg` on macOS).

- Flask “Camera not active”:
  - Click “Start Detection” first.
  - Check console output for model or camera errors.
  - If `/video_feed` returns 503, the camera is not started or is in use.

- Model load error:
  - Verify the file exists at `models/emotion_densenet.keras`.
  - Ensure the file is a Keras SavedModel/`.keras` compatible with your TensorFlow version.
  - If you renamed the file, update `MODEL_PATH` accordingly.

- OpenCV Haar cascade not found:
  - The code resolves the cascade path from your OpenCV install; ensure `opencv-python` is installed.

- Dependency conflicts (cv2 vs opencv-python):
  - Use only `opencv-python` in `requirements.txt`. Remove any `cv2==...` entry and reinstall.


### Performance Tips
- Good lighting improves face detection and classification accuracy.
- Lower the camera resolution if needed to improve FPS:
  - In `main2_flask.py`, adjust `CAP_PROP_FRAME_WIDTH/HEIGHT`.
- Close browser tabs or apps competing for CPU/GPU.


### Development Notes
- Streamlit entrypoint: `main2_streamlit.py` (uses `streamlit-webrtc`)
- Flask entrypoint: `main2_flask.py` (Tailwind + Chart.js inline template)
- Emotions and colors are defined in the top of `main2_flask.py`
- Edit `CLASS_LABELS` if you retrain with a different label set


### License
This repository is provided for educational and research purposes. Please review and adapt licensing for your use case.

# emotion-detection
