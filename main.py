import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "/Users/harshsinha/VS Code/Emotion Detection/models/emotion_densenet169.keras"
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']

@st.cache_resource
def load_model():
	model = tf.keras.models.load_model(MODEL_PATH)
	return model

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
	# Ensure RGB
	image = pil_image.convert("RGB")
	# Resize to training size
	image = image.resize((IMG_WIDTH, IMG_HEIGHT))
	# To numpy
	x = np.asarray(image).astype("float32")
	# Mirror training pipeline: DenseNet preprocess only (no manual rescale)
	x = tf.keras.applications.densenet.preprocess_input(x)
	# Add batch dim
	x = np.expand_dims(x, axis=0)
	return x

def predict_emotion(model, pil_image: Image.Image):
	x = preprocess_image(pil_image)
	probs = model.predict(x, verbose=0)[0]
	idx = int(np.argmax(probs))
	return CLASS_LABELS[idx], float(probs[idx]), probs

st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="centered")
st.title("Emotion Detection (Camera)")

model = load_model()

st.write("Use the camera below to capture an image. The app will predict your emotion.")
img_data = st.camera_input("Camera", label_visibility="visible")

if img_data is not None:
	image = Image.open(img_data)
	pred_label, pred_conf, probs = predict_emotion(model, image)

	st.subheader(f"Prediction: {pred_label}")
	st.write(f"Confidence: {pred_conf:.2%}")

	# Show image
	st.image(image, caption="Captured image", use_column_width=True)

	# Show probabilities per class
	st.write("Class probabilities:")
	for label, p in zip(CLASS_LABELS, probs):
		st.write(f"- {label}: {p:.2%}")