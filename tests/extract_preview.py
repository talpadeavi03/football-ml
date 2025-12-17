import cv2
import sys
import os

def extract_preview(video_path, output_path):
    if not os.path.exists(video_path):
        print(f"Error: Video {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Go to the end? No, the file might be incomplete/growing.
    # Let's just read a few frames or try to seek to the end.
    # Since it's being written to, the header might be incomplete, but let's try reading from start and skipping.
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {frame_count}")
    
    # If frame_count is 0 or invalid (common in growing files), try reading a bit.
    # Let's try to read the last few frames.
    
    success = False
    last_frame = None
    
    # Try reading 100 frames or until end
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame
        count += 1
        # If we have a lot of frames, maybe skip?
        # But for a preview, any processed frame is good.
        # Let's just save the 100th frame or the last one we find.
        if count > 200: 
            break
            
    if last_frame is not None:
        cv2.imwrite(output_path, last_frame)
        print(f"Preview saved to {output_path}")
    else:
        print("Error: Could not read any frames.")

    cap.release()

if __name__ == "__main__":
    extract_preview('data/videos/output_stats_persistent.mp4', 'data/videos/preview_persistent.jpg')
