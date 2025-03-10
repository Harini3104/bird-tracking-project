import cv2
import time
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 🎯 Load YOLO model
model_path = r"C:\Users\karpa\Desktop\finalproject\yolov9c-seg.pt"
model = YOLO(model_path)

# 🔍 Define the class ID for birds
BIRD_CLASS_ID = 14  # Change if needed

# 🎥 Load video
video_path = r"C:\Users\karpa\Desktop\finalproject\bird3.mp4"
cap = cv2.VideoCapture(video_path)

# 📏 Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = r"C:\Users\karpa\Desktop\finalproject\bird_tracking_output.avi"

out = cv2.VideoWriter(output_path, fourcc, fps_in, (frame_width, frame_height))

# 📌 Initialize DeepSORT
tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)

# 🕒 FPS Counter
prev_time = 0
track_history = {}

# 🔑 Shrink factor (50% smaller bounding boxes)
shrink_factor = 0.5

# 🔑 Skip frames to increase speed (process fewer frames)
skip_frames = 2
frame_count = 0

# 🖥 Single display window
window_name = "Bird Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)  # Adjust as desired

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video Processing Complete.")
        break

    frame_count += 1

    # Skip frames to speed up detection
    if frame_count % skip_frames != 0:
        # No detection here. We won't show or save anything.
        continue

    # 🏆 YOLO Inference
    results = model(frame, verbose=False)

    detections = []
    for result in results:
        if hasattr(result, "boxes"):
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for i in range(len(boxes)):
                if int(classes[i]) == BIRD_CLASS_ID:
                    detections.append((
                        [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]],
                        confidences[i],
                        int(classes[i])
                    ))

    # 🔄 Update DeepSORT
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # ---- Shrink bounding box ----
        width = x2 - x1
        height = y2 - y1

        new_width = int(width * (1 - shrink_factor))
        new_height = int(height * (1 - shrink_factor))

        x_center = x1 + width // 2
        y_center = y1 + height // 2

        x1 = x_center - new_width // 2
        x2 = x_center + new_width // 2
        y1 = y_center - new_height // 2
        y2 = y_center + new_height // 2

        # Clamp to frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_width - 1, x2)
        y2 = min(frame_height - 1, y2)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Bird {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw past trajectory
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((x_center, y_center))

        if len(track_history[track_id]) > 20:
            track_history[track_id] = track_history[track_id][-20:]

        for i in range(1, len(track_history[track_id])):
            cv2.line(frame,
                     track_history[track_id][i - 1],
                     track_history[track_id][i],
                     (0, 255, 255), 2)

    # 🕒 FPS Calculation
    curr_time = time.time()
    fps_val = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps_val:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 💾 Save processed frame & show
    out.write(frame)
    cv2.imshow(window_name, frame)

    # 🚪 Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Bird tracking saved at: {output_path}")
