import cv2
from src.tracker import Tracker

def debug_detections(video_path, limit=100):
    tracker = Tracker()
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        detections = tracker.track_objects(frame)
        
        classes = [d['class'] for d in detections]
        ids = [d['id'] for d in detections]
        
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: Classes={classes}, IDs={ids}")
            
    cap.release()

if __name__ == "__main__":
    debug_detections("data/videos/VID_20251129_0805111.mp4", limit=200)
