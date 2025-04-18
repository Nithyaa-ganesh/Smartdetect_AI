# 🚀 SmartdetectAI – Real-Time AI Surveillance System

SmartdetectAI is a real-time, AI-powered object detection and classification system built using YOLOv8, OpenCV, MTCNN, and a CNN-based emotion classifier. It runs on a sleek Streamlit interface and is deployable on Hugging Face Spaces or locally.

---

## 🧠 Features

✅ Real-time **object detection** using YOLOv8  
✅ **Face detection** using MTCNN  
✅ **Emotion recognition** using a CNN model trained on FER2013  
✅ **Object tracking** with unique IDs (Centroid Tracker)  
✅ 📝 **Event logging** (objects/emotions/timestamps) to `event_log.csv`  
✅ 🔔 **Real-time alerts** for specific objects or emotions produces beep sound (e.g., person or angry)  
✅ 📷 Live webcam feed with toggle controls  
✅ 📦 Ready for deployment on Hugging Face Spaces 

TECHSTACK USED

Core AI/ML Components
Technology	Purpose	Version
YOLOv8	Real-time object detection	Ultralytics implementation
MTCNN	Face detection	mtcnn package
CNN Emotion Model	Emotion classification (Angry, Happy, etc.)	Pre-trained Keras model
CentroidTracker	Object tracking	Custom implementation

Programming
Technology	Usage
Python	Main language	3.8-3.10
TensorFlow/Keras	Emotion model backend	TF 2.12+
OpenCV	Image processing	opencv-python
