import cv2
import numpy as np
import time
import urllib.request
import os

# COCO class labels
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Download model files if they don't exist
def download_model():
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Use YOLO v4 for better COCO detection
    config_path = os.path.join(model_dir, "yolov4.cfg")
    weights_path = os.path.join(model_dir, "yolov4.weights")
    names_path = os.path.join(model_dir, "coco.names")
    
    try:
        # Download YOLOv4 config
        if not os.path.exists(config_path):
            print("Downloading YOLOv4 config...")
            config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
            urllib.request.urlretrieve(config_url, config_path)
        
        # Download COCO names
        if not os.path.exists(names_path):
            print("Downloading COCO class names...")
            names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
            urllib.request.urlretrieve(names_url, names_path)
        
        # For weights, we'll use a smaller model (YOLOv4-tiny) that's easier to download
        weights_tiny_path = os.path.join(model_dir, "yolov4-tiny.weights")
        config_tiny_path = os.path.join(model_dir, "yolov4-tiny.cfg")
        
        if not os.path.exists(config_tiny_path):
            print("Downloading YOLOv4-tiny config...")
            config_tiny_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
            urllib.request.urlretrieve(config_tiny_url, config_tiny_path)
        
        if not os.path.exists(weights_tiny_path):
            print("Downloading YOLOv4-tiny weights (smaller model)...")
            weights_tiny_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4-tiny.weights"
            urllib.request.urlretrieve(weights_tiny_url, weights_tiny_path)
        
        return config_tiny_path, weights_tiny_path, names_path
    except Exception as e:
        print(f"Failed to download models: {e}")
        # Try alternative approach with OpenCV's DNN models
        return try_opencv_models(model_dir)

def try_opencv_models(model_dir):
    """Try to use OpenCV's built-in models as fallback"""
    try:
        # Use MobileNet SSD with proper URLs
        config_path = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        weights_path = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
        
        if not os.path.exists(config_path):
            print("Downloading MobileNet SSD config...")
            config_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt.txt"
            urllib.request.urlretrieve(config_url, config_path)
        
        if not os.path.exists(weights_path):
            print("Downloading MobileNet SSD weights...")
            weights_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
            urllib.request.urlretrieve(weights_url, weights_path)
        
        return config_path, weights_path, None
    except:
        return None, None, None

# Try to load or download models
config_path, weights_path, names_path = download_model()

# Load COCO class names
if names_path and os.path.exists(names_path):
    with open(names_path, 'r') as f:
        COCO_CLASSES = ['__background__'] + [line.strip() for line in f.readlines()]

# Load object detection model
model_loaded = False
net = None
is_yolo = False

if config_path and weights_path and os.path.exists(config_path) and os.path.exists(weights_path):
    try:
        if 'yolo' in config_path.lower():
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            is_yolo = True
            print("YOLO model loaded successfully")
        else:
            net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
            is_yolo = False
            print("MobileNet SSD model loaded successfully")
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")

# Fallback: Try to use OpenCV's built-in DNN with a simpler approach
if not model_loaded:
    try:
        # Use OpenCV's built-in pre-trained models
        print("Trying OpenCV's built-in object detection...")
        # This uses a simple approach with available cascade classifiers
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        if face_cascade.empty() or body_cascade.empty():
            raise Exception("Cascade classifiers not found")
        model_loaded = True
        print("Using Haar Cascade classifiers as fallback")
    except Exception as e:
        print(f"Error loading cascade classifiers: {e}")
        model_loaded = False

def detect_objects(frame):
    if not model_loaded:
        return []
    
    if net is not None:
        height, width = frame.shape[:2]
        
        if is_yolo:
            # YOLO preprocessing - improved parameters
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
            net.setInput(blob)
            
            # Get output layer names
            layer_names = net.getLayerNames()
            try:
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            except:
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
            outputs = net.forward(output_layers)
            
            boxes = []
            confidences = []
            class_ids = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Lower confidence threshold for better detection of all classes
                    if confidence > 0.2:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression with adjusted parameters
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
            
            objects = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    if class_id < len(COCO_CLASSES):
                        objects.append({
                            'class_id': class_id,
                            'class_name': COCO_CLASSES[class_id],
                            'confidence': confidence,
                            'box': (max(0, x), max(0, y), min(width, x + w), min(height, y + h))
                        })
            
            return objects
        
        else:
            # MobileNet SSD preprocessing - improved parameters
            blob = cv2.dnn.blobFromImage(frame, 0.017, (416, 416), (103.94, 116.78, 123.68), swapRB=False)
            net.setInput(blob)
            detections = net.forward()
            
            objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                # Lower confidence threshold for better multi-class detection
                if confidence > 0.15:
                    class_id = int(detections[0, 0, i, 1])
                    
                    # Ensure class_id is valid
                    if 0 <= class_id < len(COCO_CLASSES):
                        x1 = int(detections[0, 0, i, 3] * width)
                        y1 = int(detections[0, 0, i, 4] * height)
                        x2 = int(detections[0, 0, i, 5] * width)
                        y2 = int(detections[0, 0, i, 6] * height)
                        
                        # Ensure bounding box is within frame
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        # Only add if bounding box is valid
                        if x2 > x1 and y2 > y1:
                            objects.append({
                                'class_id': class_id,
                                'class_name': COCO_CLASSES[class_id],
                                'confidence': confidence,
                                'box': (x1, y1, x2, y2)
                            })
            return objects
    
    else:
        # Fallback to Haar cascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
        
        objects = []
        for (x, y, w, h) in faces:
            objects.append({
                'class_id': 1,
                'class_name': 'person',
                'confidence': 0.8,
                'box': (x, y, x+w, y+h)
            })
        
        for (x, y, w, h) in bodies:
            objects.append({
                'class_id': 1,
                'class_name': 'person',
                'confidence': 0.7,
                'box': (x, y, x+w, y+h)
            })
        
        return objects

# Use laptop camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not model_loaded:
    print("No detection model available. Exiting.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not available. Retrying in 2 seconds...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(0)
        continue

    # Run detection
    detected_objects = detect_objects(frame)
    
    # Draw detections with improved visualization
    for obj in detected_objects:
        x1, y1, x2, y2 = obj['box']
        class_name = obj['class_name']
        confidence = obj['confidence']
        
        # Skip background and invalid classes
        if class_name in ['__background__', 'N/A'] or class_name == '':
            continue
        
        # Enhanced color coding for different object categories
        if class_name == 'person':
            color = (0, 255, 0)  # Green for person
        elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'motorbike', 'train', 'boat', 'aeroplane']:
            color = (255, 0, 0)  # Blue for vehicles
        elif class_name in ['cat', 'dog', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
            color = (0, 255, 255)  # Yellow for animals
        elif class_name in ['chair', 'sofa', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'book']:
            color = (255, 0, 255)  # Magenta for furniture/electronics
        elif class_name in ['bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange']:
            color = (0, 128, 255)  # Orange for food/drink
        else:
            color = (255, 255, 0)  # Cyan for other objects
        
        # Draw bounding box with thicker lines for visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label with background for better readability
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Enhanced detection summary
    if detected_objects:
        detected_classes = {}
        for obj in detected_objects:
            if obj['class_name'] not in ['__background__', 'N/A', '']:
                class_name = obj['class_name']
                if class_name in detected_classes:
                    detected_classes[class_name] += 1
                else:
                    detected_classes[class_name] = 1
        
        summary = ', '.join([f"{name}({count})" for name, count in detected_classes.items()])
        print(f"Detected: {summary}")
        
        # Add detection count to frame
        count_text = f"Objects: {len(detected_objects)}"
        cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No objects detected in this frame.")
        cv2.putText(frame, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Patrol Robot - Multi-Object Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()