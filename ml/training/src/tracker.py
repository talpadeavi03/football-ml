# src/tracker.py

import cv2
from ultralytics import YOLO

# 🔒 EXACTLY TWO COLORS — NEVER CHANGE
TEAM_COLORS = {
    'Team A': (255, 0, 0),   # Team A (Blue)
    'Team B': (0, 0, 255)    # Team B (Red)
}


class Tracker:
    def __init__(self):
        self.model = YOLO(
            "football_mobile_project/yolov8n_mobile/weights/best.pt"
        )
        # Fallback for ball detection if custom model fails
        self.fallback_model = YOLO("yolov8n.pt")

    def track_objects(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=0.15,
            iou=0.5,
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        untracked_count = 0

        for box in boxes:
            cls = int(box.cls.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if box.id is not None:
                track_id = int(box.id.item())
            else:
                # Assign a temporary unique ID for this frame
                untracked_count += 1
                track_id = -100 - untracked_count

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = self.model.names[cls]

            detections.append({
                "id": track_id,
                "class": label,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy)
            })

        # --- FALLBACK FOR BALL ---
        has_ball = any(d["class"] == "ball" for d in detections)
        if not has_ball:
            fb_results = self.fallback_model.predict(frame, conf=0.05, verbose=False)
            if fb_results and fb_results[0].boxes:
                for box in fb_results[0].boxes:
                    cls = int(box.cls.item())
                    if cls == 32: # 'sports ball'
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append({
                            "id": -1,
                            "class": "ball",
                            "bbox": (x1, y1, x2, y2),
                            "center": ((x1+x2)//2, (y1+y2)//2)
                        })
                        break

        return detections

    def draw_annotations(self, frame, detections, team_assignments):
        """
        Draw ONLY confirmed team players.
        NO third color is possible.
        """

        annotated = frame.copy()

        for det in detections:
            if det['class'] != 'player':
                continue

            pid = det['id']

            # ❌ DO NOT DRAW if no team
            if pid not in team_assignments:
                continue

            team_id = team_assignments[pid]

            # ❌ SAFETY CHECK
            if team_id not in TEAM_COLORS:
                continue

            color = TEAM_COLORS[team_id]

            x1, y1, x2, y2 = det['bbox']

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

        return annotated
