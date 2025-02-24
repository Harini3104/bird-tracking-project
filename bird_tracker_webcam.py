import cv2
import time
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLO model
model_path = r"C:\Users\karpa\Desktop\finalproject\yolov9c-seg.pt"
model = YOLO(model_path)

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=30, n_init=3, nn_budget=70)

# Open webcam (0 for default webcam)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

prev_time = 0  # FPS Counter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Error: Could not read frame")
        break

    # YOLO Inference
    results = model(frame, verbose=False)  

    detections = []
    for result in results:
        if hasattr(result, "boxes"):
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            # Format detections for DeepSORT
            for i in range(len(boxes)):
                detections.append(([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]], confidences[i], int(classes[i])))

    # Update DeepSORT tracker with formatted detections
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracked_objects:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get (x1, y1, x2, y2) bounding box
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("🐦 Bird Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("🔄 Program exited successfully.")
