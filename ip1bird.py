import cv2
import time
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 🎯 Load YOLO model (Ensure correct path)
model_path = r"yolov9c-seg.pt path"
model = YOLO(model_path)

# 🔍 Define the class ID for birds in your dataset
BIRD_CLASS_ID = 14  # ⚠️ Change this based on your dataset's class index for birds

# 🎥 Load video file (Replace with your video path)
video_path = r"video path"
cap = cv2.VideoCapture(video_path)

# 📏 Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
output_path = r"C:\Users\karpa\Desktop\finalproject\bird_tracking_output.avi"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 📌 Initialize DeepSORT Tracker
tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

# 🕒 Initialize FPS Counter
prev_time = 0
track_history = {}  # Store past positions for smooth tracking

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video Processing Complete.")
        break

    # 🏆 YOLO Inference
    results = model(frame, verbose=False)

    detections = []
    for result in results:
        if hasattr(result, "boxes"):
            boxes = result.boxes.xyxy.cpu().numpy()  # Get (x1, y1, x2, y2)
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            classes = result.boxes.cls.cpu().numpy()  # Class IDs

            # 🎯 Filter for birds only
            for i in range(len(boxes)):
                if int(classes[i]) == BIRD_CLASS_ID:
                    detections.append(([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]], confidences[i], int(classes[i])))

    # 🔄 Update DeepSORT tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    # 📍 Draw tracking results
    for track in tracked_objects:
        if not track.is_confirmed():
            continue  # Ignore unconfirmed tracks
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # Get (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, ltrb)

        # 🎨 Draw tracking box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bird {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ✍️ Draw past trajectory for smoother tracking
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        # Keep only the last 20 positions
        if len(track_history[track_id]) > 20:
            track_history[track_id] = track_history[track_id][-20:]

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 255), 2)

    # 🕒 FPS Calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 💾 Save processed frame to output video
    out.write(frame)

    # 🎬 Display the frame
    cv2.imshow("🦜 Bird Tracking", frame)

    # 🚪 Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔄 Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Bird tracking saved at: {output_path}")
