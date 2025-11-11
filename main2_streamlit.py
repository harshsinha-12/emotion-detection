import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

MODEL_PATH = "/Users/harshsinha/VS Code/Emotion Detection/models/emotion_densenet.keras"
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


@st.cache_resource
def load_model():
	model = tf.keras.models.load_model(MODEL_PATH)
	return model


def preprocess_bgr_roi(bgr_roi: np.ndarray) -> np.ndarray:
	rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT)).astype("float32")
	rgb = tf.keras.applications.densenet.preprocess_input(rgb)
	return np.expand_dims(rgb, 0)


class EmotionProcessor(VideoProcessorBase):
	def __init__(self):
		self.model = load_model()
		self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

	def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
		image = frame.to_ndarray(format="bgr24")

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
			roi = image[y : y + h, x : x + w]
			x_input = preprocess_bgr_roi(roi)
			probs = self.model.predict(x_input, verbose=0)[0]
			idx = int(np.argmax(probs))
			label = f"{CLASS_LABELS[idx]} ({probs[idx]*100:.1f}%)"
			cv2.putText(image, label, (x + 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

		return av.VideoFrame.from_ndarray(image, format="bgr24")


st.set_page_config(page_title="Emotion Detection - Realtime", page_icon="😊", layout="centered")
st.title("Emotion Detection (Realtime Webcam)")
st.write("Allow camera access and look at the preview. Press Stop to end.")

webrtc_streamer(
	key="emotion-webrtc",
	mode=WebRtcMode.SENDRECV,
	video_processor_factory=EmotionProcessor,
	media_stream_constraints={"video": True, "audio": False},
)


