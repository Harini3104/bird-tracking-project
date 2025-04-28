import cv2
import time
import torch
import threading
from playsound import playsound
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def load_model(model_path):
    print("🚀 Loading YOLO model...")
    return YOLO(model_path)

def initialize_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"❌ Error: Cannot open video file at {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print("🎥 Video capture initialized.")
    return cap, out

def process_frame(frame, model, bird_class_id):
    results = model(frame, verbose=False)
    detections = []
    
    for result in results:
        if hasattr(result, "boxes"):
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            masks = result.masks.xy if hasattr(result, "masks") and result.masks is not None else None

            for i, box in enumerate(boxes):
                if int(classes[i]) == bird_class_id:
                    detections.append(([box[0], box[1], box[2], box[3]], confidences[i], int(classes[i])))
                    
                    if masks is not None:
                        mask = masks[i].astype(int)
                        cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return detections, frame

# 🔊 Play alert sound in background
def play_alert_sound(sound_path):
    threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

def main():
    # 📂 Paths
    model_path = r"C:\Users\karpa\Desktop\finalproject\yolov9c-seg.pt"
    video_path = r"C:\Users\karpa\Desktop\finalproject\istock1.jpg"
    output_path = r"D:\newone\yolov9\bird_tracking_output.avi"
    sound_path = r"C:\Users\karpa\Desktop\finalproject\buzzer.mp3"  # 🔔 Set your alert sound path here
    
    # 🎯 Bird class ID
    BIRD_CLASS_ID = 14  # 🛠️ Update according to your dataset

    # ⚙️ Initialize
    model = load_model(model_path)
    cap, out = initialize_video(video_path, output_path)
    tracker = DeepSort(max_age=50, n_init=3, nn_budget=100)
    track_history = {}
    
    # 🕒 FPS Calculation
    prev_time = time.time()
    
    print("🛠️ Starting tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video processing complete.")
            break
        
        detections, frame = process_frame(frame, model, BIRD_CLASS_ID)
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        # 🔔 Play sound if any bird is detected
        if len(tracked_objects) > 0:
            play_alert_sound(sound_path)
        
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()

            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Initialize history for new track_id
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append(center)

            # Limit trajectory length
            if len(track_history[track_id]) > 20:
                track_history[track_id] = track_history[track_id][-20:]
            
            # Draw trajectory
            for i in range(1, len(track_history[track_id])):
                cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (0, 255, 255), 2)

        # 🧮 Update FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # 🖼️ Annotate FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 💾 Save frame
        out.write(frame)
        
        # 🎬 Display
        cv2.imshow("🦜 Bird Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopping early...")
            break

    # 🔄 Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Bird tracking saved at: {output_path}")

if __name__ == "__main__":
    main()
