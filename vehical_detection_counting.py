import cv2
import numpy as np

# Loading YOLOv3 model and COCO class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
if net.empty():
    print("Error: Could not load YOLO model.")
else:
    print("YOLO model loaded successfully.")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initializing video capture
cap = cv2.VideoCapture("video1.mp4")

# Checking if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define vehicle classes and ROI line for counting
vehicle_classes = ["car", "bus", "truck"]
line_position = 450  
font = cv2.FONT_HERSHEY_SIMPLEX

# Vehicle count initialization
vehicle_count = {"car": 0, "bus": 0, "truck": 0}

# Vehicle tracking variables
tracker = {}
vehicle_id = 0
vehicle_crossed = set()

def draw_boxes(frame, boxes, confidences, class_ids, indexes, previous_centers):
    global vehicle_id, vehicle_count
    new_centers = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label in vehicle_classes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), font, 0.5, (255, 255, 255), 2)

                center_y = y + h // 2
                center_x = x + w // 2
                new_centers.append((center_x, center_y))

                for cid, (prev_x, prev_y) in previous_centers.items():
                    if np.linalg.norm(np.array([center_x, center_y]) - np.array([prev_x, prev_y])) < 50:
                        tracker[cid] = (center_x, center_y)
                        if center_y > line_position and (cid, label) not in vehicle_crossed:
                            vehicle_count[label] += 1
                            vehicle_crossed.add((cid, label))
                        break
                else:
                    tracker[vehicle_id] = (center_x, center_y)
                    vehicle_id += 1

    return {k: v for k, v in tracker.items() if k in [i for i, _ in enumerate(new_centers)]}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not read properly.")
        break

    height, width, _ = frame.shape
    print(f"Processing frame of size: {height}x{width}")

    # Preprocess input frame for YOLO with correct input size
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    tracker = draw_boxes(frame, boxes, confidences, class_ids, indexes, tracker)

    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)
    cv2.putText(frame, f"Cars: {vehicle_count['car']}  Buses: {vehicle_count['bus']}  Trucks: {vehicle_count['truck']}",
                (10, 50), font, 1, (0, 255, 255), 2)

    # Display the processed frame
    cv2.imshow("Vehicle Detection and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()