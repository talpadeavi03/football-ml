from ultralytics import YOLO
import cv2
import numpy as np

class Tracker:
    # FIX 1: Hardcode the correct model path to guarantee use of trained weights.
    def __init__(self):
        """
        Initializes the YOLOv8 tracker using the custom-trained weights.
        """
        model_path = 'football_mobile_project/yolov8n_mobile/weights/best.pt'
        
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"Classes: {self.class_names}")

    def track_objects(self, frame):
        """
        Runs tracking on the frame.
        """
        results = self.model.track(frame, persist=True, verbose=False)
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else -1
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names[cls_id]
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # FIX 2: Filter for the custom classes: 0: ball, 1: player, 2: referee, 3: goalkeeper
                    if cls_id in [0, 1, 2, 3]:
                        detections.append({
                            'id': track_id,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': cls_name,
                            'conf': conf,
                            'center': [int((x1+x2)/2), int((y1+y2)/2)]
                        })
                        
        return detections

    def draw_annotations(self, frame, detections, team_assignments=None):
        """
        Draws bounding boxes and IDs on the frame.
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0) # Default Green
            label = f"{det['class']} | ID {det['id']}"
            
            # Handle Ball detection
            if det['class'] == 'ball':
                color = (0, 165, 255) # Orange
                label = "Ball"
            
            # Handle Player/Referee
            elif team_assignments and det['id'] in team_assignments:
                team = team_assignments[det['id']]
                if team == 'Team A':
                    color = (255, 0, 0) # Blue
                elif team == 'Team B':
                    color = (0, 0, 255) # Red
                
                name = det.get('name', 'NA')
                label = f"{team} | {name}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated_frame