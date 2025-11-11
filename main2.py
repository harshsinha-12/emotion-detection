import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "/Users/harshsinha/VS Code/Emotion Detection/models/emotion_densenet169.keras"
IMG_HEIGHT = 48
IMG_WIDTH = 48
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']


def load_model(path: str):
	model = tf.keras.models.load_model(path)
	return model


def preprocess_bgr_roi(bgr_roi: np.ndarray) -> np.ndarray:
	rgb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT)).astype("float32")
	rgb = tf.keras.applications.densenet.preprocess_input(rgb)
	return np.expand_dims(rgb, 0)


def main():
	model = load_model(MODEL_PATH)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print("Error: Could not open webcam.")
		return

	try:
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
				roi = frame[y : y + h, x : x + w]
				x_input = preprocess_bgr_roi(roi)
				probs = model.predict(x_input, verbose=0)[0]
				idx = int(np.argmax(probs))
				label = f"{CLASS_LABELS[idx]} ({probs[idx]*100:.1f}%)"
				cv2.putText(frame, label, (x + 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

			cv2.imshow("Emotion Detection (DenseNet169)", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		cap.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()


