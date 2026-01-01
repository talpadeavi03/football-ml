# src/main.py

import os
import cv2
import argparse
import numpy as np

from src.tracker import Tracker
from src.team_classifier import TeamClassifier
from src.face_processor import FaceProcessor
from src.face_detector import FaceDetector
from src.identity_manager import IdentityManager
from src.game_logic import GameLogic

# 🔒 EXACTLY TWO COLORS — NEVER CHANGE
TEAM_COLORS = {
    'Team A': (255, 0, 0),   # Team A (Blue)
    'Team B': (0, 0, 255)    # Team B (Red)
}


def main():
    # ---------------- ARGUMENTS ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # ---------------- VIDEO INPUT ----------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # ---------------- VIDEO OUTPUT ----------------
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot open VideoWriter")

    # ---------------- COMPONENTS ----------------
    tracker = Tracker()
    team_classifier = TeamClassifier(init_frames=100)
    face_processor = FaceProcessor("data/faces")
    face_detector = FaceDetector()
    identity_manager = IdentityManager()
    game_logic = GameLogic()

    frame_idx = 0
    print("▶ Processing video...")

    # =====================================================
    # ===================== MAIN LOOP =====================
    # =====================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.limit and frame_idx > args.limit:
            break

        # --------- PLAYER TRACKING ---------
        detections = tracker.track_objects(frame)

        # --------- TEAM ASSIGNMENT (ALWAYS) ---------
        team_assignments = team_classifier.assign_teams(
            detections=detections,
            frame=frame,
            frame_idx=frame_idx,
            identity_manager=identity_manager
        )

        # --------- IDENTITY & FACE LOGIC ---------
        for det in detections:
            if det["class"] != "player":
                continue

            track_id = det["id"]
            bbox = det["bbox"]
            center = det["center"]

            # 1. Update or Create Identity (ReID recovery)
            # We now create an identity for EVERY player to enable ReID even without faces
            identity_id = identity_manager.update(track_id, frame, bbox, center)
            
            if not identity_id:
                # New player, no ReID match -> create fresh identity
                identity_manager.create_identity(
                    track_id=track_id,
                    frame=frame,
                    bbox=bbox,
                    center=center,
                    name=None, # Unknown for now
                    team_id=team_assignments.get(track_id)
                )

            # 2. Periodic Face Detection (to add names to existing identities)
            if frame_idx % 15 == 0:
                face_bbox = face_detector.detect_face_in_player(frame, bbox)
                if face_bbox:
                    fx, fy, fw, fh = face_bbox
                    face_crop = frame[fy:fy+fh, fx:fx+fw]
                    
                    if face_crop.size > 0:
                        name, team_from_face = face_processor.recognize_single_face(face_crop)
                        
                        if name != "Unknown":
                            # Update identity with the recognized name
                            identity_manager.create_identity(
                                track_id=track_id,
                                frame=frame,
                                bbox=bbox,
                                center=center,
                                name=name,
                                team_id=team_from_face if team_from_face is not None else team_assignments.get(track_id)
                            )

        # --------- GAME LOGIC (POSSESSION) ---------
        game_logic.update(frame_idx, detections, team_assignments)

        # --------- VISUALIZATION ---------
        for det in detections:
            if det["class"] != "player":
                continue

            pid = det["id"]
            team_id = team_assignments.get(pid)

            if team_id not in TEAM_COLORS:
                continue

            x1, y1, x2, y2 = det["bbox"]
            color = TEAM_COLORS[team_id]

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Name/Identity if available
            identity = identity_manager.get_identity(pid)
            label = team_id # Default to "Team A" or "Team B"
            if identity and identity.get("name"):
                label = f"{team_id}: {identity['name']}"
            
            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Draw Stats Overlay
        stats = game_logic.get_stats()
        pos = stats['possession']
        act = stats['activity']
        
        if pos['Team A'] > 0 or pos['Team B'] > 0:
            text = f"Possession: A {pos['Team A']}% | B {pos['Team B']}%"
        else:
            text = f"Activity: A {act['Team A']}% | B {act['Team B']}%"
            
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # --------- WRITE FRAME ---------
        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames")

    # ---------------- CLEANUP ----------------
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"✅ Output video written to: {args.output}")
    print("Final Stats:", game_logic.get_stats())


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
