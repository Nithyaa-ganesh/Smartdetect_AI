from ultralytics import YOLO

def load_model():
    return YOLO("yolov8n.pt")

def detect_objects(model, frame):
    results = model(frame, stream=True)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = model.names[int(box.cls[0])]
            conf = round(float(box.conf[0]), 2)
            detections.append((x1, y1, x2, y2, cls, conf))
    return detections
