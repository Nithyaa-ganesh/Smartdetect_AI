import streamlit as st
import cv2
import numpy as np
from yolov8_model import load_model, detect_objects
from face_emotion import detect_faces_emotions
from tracking import CentroidTracker
from utils import draw_all, log_events, check_alert


# Set page config for better layout
st.set_page_config(page_title="SmartVision Pro", layout="wide")
st.title("🚀 SmartVision Pro - Real-Time Detection & Classification")

# Sidebar for controls
with st.sidebar:
    st.header("🛠 Settings")
    run = st.checkbox("📷 Start Camera", key="camera_running")
    track = st.checkbox("🎯 Enable Object Tracking")
    face_detect = st.checkbox("😎 Detect Faces & Emotions")
    st.markdown("---")
    st.subheader("🔧 Other Options")
    st.write("You can enable or disable features like Object Tracking and Face Detection here.")

# Display camera feed
FRAME_WINDOW = st.image([])

# Load models and tracker
model = load_model()
tracker = CentroidTracker()

frame_counter = 0  # Frame counter for reducing load

# Function to process frames
def process_frame(frame):
    global frame_counter
    frame_counter += 1

    if frame_counter % 5 != 0:  # Skip every 5th frame to reduce processing
        return None

    frame = cv2.resize(frame, (640, 480))

    # Detect objects using YOLOv8
    objects = detect_objects(model, frame)
    if track:
        objects = tracker.update(objects)

    # Detect faces and emotions
    faces = []
    if face_detect:
        faces = detect_faces_emotions(frame)

    # Log all detections
    log_events(objects, faces)

    # Check for real-time alerts
    st.write("Detected objects: ", [obj[4] for obj in objects])
    st.write("Detected emotions: ", [face['emotion'] for face in faces])


    alert_message = check_alert(objects, faces, target_objects=["knife", "scissors"], target_emotions=["angry","neutral"])
    st.write("Alert message: ", alert_message)
    if alert_message:
        st.error(alert_message)
    

    # Draw everything
    final_frame = draw_all(frame, objects, faces, tracking=track)

    return final_frame

# Function to capture frames from webcam
def capture_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 Failed to access webcam.")
        return

    while run:  # While the camera is running
        ret, frame = cap.read()
        if not ret:
            st.error("🚫 Failed to read frame from webcam.")
            break

        # Process the frame in the main thread to avoid threading issues
        final_frame = process_frame(frame)

        if final_frame is not None:
            # Display the processed frame in the Streamlit app
            FRAME_WINDOW.image(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))

    cap.release()

# Start webcam capture when camera is toggled on
if run:
    capture_frames()

elif not run:
    st.info("Camera stopped.")
else:
    st.info("Toggle 'Start Camera' to begin.")
