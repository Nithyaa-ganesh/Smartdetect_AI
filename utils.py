import cv2
import datetime
import winsound

# Function to log the detected objects and faces
def log_events(objects, faces):
    # Get the current time for logging
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log objects
    print(f"LOG ({current_time}) - Objects Detected:")
    for obj in objects:
        x1, y1, x2, y2, label, conf = obj
        print(f"  Object: {label}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Log faces and emotions
    print(f"LOG ({current_time}) - Faces & Emotions Detected:")
    for face in faces:
        x, y, w, h = face['box']
        emotion = face['emotion']
        confidence = face['confidence']
        print(f"  Face Box: ({x}, {y}, {w}, {h}), Emotion: {emotion}, Confidence: {confidence:.2f}")

# Function to check for alerts (e.g., if a person is detected with angry emotion)
def check_alert(objects, faces, target_objects=["knife", "scissors"], target_emotions=["angry", "neutral"]):
    alert_message = None

    # Print debugging to verify what we're detecting
    print("Detected Objects:", [obj[4] for obj in objects])
    print("Detected Emotions:", [face['emotion'] for face in faces])

    # Extract object labels, remove " ID:x" suffix, and convert to lowercase
    detected_objects = [obj[4].split(" ID:")[0].lower() for obj in objects]
    
    # Extract emotions and convert to lowercase
    detected_emotions = [face['emotion'].lower() for face in faces]

    # Check if any target object + emotion combo exists
    for obj in target_objects:
        if obj.lower() in detected_objects:  # Check if target object is detected
            for emotion in target_emotions:
                if emotion.lower() in detected_emotions:  # Check if target emotion is detected
                    alert_message = f"⚠️ Alert: {obj} detected with emotion '{emotion}'!"
                    winsound.Beep(1000, 1000)
                    break
            if alert_message:  # If alert message is set, exit loop
                break

    return alert_message




def draw_all(frame, objects, faces, tracking=False):
    # Draw the detected objects
    for x1, y1, x2, y2, label, conf in objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw the faces and emotions
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        text = f"{face['emotion']} ({face['confidence']})"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return frame
