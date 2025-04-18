from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector and emotion model
detector = MTCNN()
emotion_model = load_model("emotion_model.hdf5")  # Make sure this file exists

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_faces_emotions(frame):
    faces = detector.detect_faces(frame)
    results = []

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        face_img = frame[y:y+h, x:x+w]

        try:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized.astype("float") / 255.0
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))
            prediction = emotion_model.predict(face_reshaped, verbose=0)[0]
            label = emotion_labels[np.argmax(prediction)]
            confidence = round(np.max(prediction), 2)

            results.append({'box': (x, y, w, h), 'emotion': label, 'confidence': confidence})
        except Exception as e:
            print(f"Skipping a face due to error: {e}")

    return results
