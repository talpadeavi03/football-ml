import argparse
import cv2
import sys

from face_processor import FaceProcessor
from tracker import Tracker
from team_classifier import TeamClassifier
from game_logic import GameLogic
from identity_manager import IdentityManager


def main():
    parser = argparse.ArgumentParser(description="Football Player Tracking & Stats")
    parser.add_argument('--mode', type=str, default='live',
                        choices=['live', 'upload'], help="Mode: live or upload")
    parser.add_argument('--video', type=str,
                        help="Path to video file (required for upload mode)")
    parser.add_argument('--faces', type=str, default='data/faces',
                        help="Path to faces directory")
    parser.add_argument('--output', type=str,
                        help="Path to save output video")
    parser.add_argument('--limit', type=int,
                        help="Limit number of frames to process")
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help="Path to YOLO model weights")

    args = parser.parse_args()

    # ===============================
    # Initialize Modules
    # ===============================
    face_processor = FaceProcessor(args.faces)
    tracker = Tracker()
    team_classifier = TeamClassifier()
    game_logic = GameLogic()
    identity_manager = IdentityManager()

    # ===============================
    # Video Source
    # ===============================
    if args.mode == 'upload':
        if not args.video:
            print("Error: --video argument is required for upload mode.")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
    else:
        print("Starting Live Mode (Webcam)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video Source: {frame_width}x{frame_height} @ {fps} FPS")

    # ===============================
    # Video Writer
    # ===============================
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            args.output, fourcc, fps, (frame_width, frame_height)
        )

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.limit and frame_idx > args.limit:
                print(f"Reached frame limit of {args.limit}. Stopping.")
                break

            # ===============================
            # 1. Object Tracking (YOLO)
            # ===============================
            detections = tracker.track_objects(frame)

            # ===============================
            # 2. Team Assignment (Position-based)
            # ===============================
            team_assignments = team_classifier.assign_teams(
                detections, frame_width, frame_idx
            )

            # ===============================
            # 3. Face Recognition (ONLY to bootstrap identity)
            # ===============================
            if frame_idx % 10 == 0:
                rois = []
                roi_dets = []

                for det in detections:
                    if det['class'] in ['player', 'goalkeeper']:
                        rois.append(det['bbox'])
                        roi_dets.append(det)

                if rois:
                    face_results = face_processor.recognize_faces(frame, rois)

                    for i, (coords, name) in enumerate(face_results):
                        if name != "Unknown":
                            det = roi_dets[i]

                            # 🔐 Create / confirm identity ONCE
                            identity_manager.update(
                                track_id=det['id'],
                                frame=frame,
                                bbox=det['bbox'],
                                center=det['center'],
                                face_confirmed=True,
                                name=name
                            )

                            # Optional: update team from name
                            team_classifier.update_player(det['id'], name)

            # ===============================
            # 4. Identity Update (EVERY FRAME)
            # ===============================
            for det in detections:
                if det['class'] != 'player':
                    continue

                identity_manager.update(
                    track_id=det['id'],
                    frame=frame,
                    bbox=det['bbox'],
                    center=det['center'],
                    face_confirmed=False
                )

            # ===============================
            # 5. Game Logic (Stats)
            # ===============================
            game_logic.update(frame_idx, detections, team_assignments)

            # ===============================
            # 6. Visualization
            # ===============================
            annotated_frame = tracker.draw_annotations(
                frame, detections, team_assignments
            )

            # Draw identity names (if known)
            for det in detections:
                name = identity_manager.get_identity_name(det['id'])
                if name:
                    x1, y1, _, _ = det['bbox']
                    cv2.putText(
                        annotated_frame,
                        name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            # Draw stats
            stats = game_logic.get_stats()
            stats_text = (
                f"Possession: Team A {stats.get('Team A', 0)}% | "
                f"Team B {stats.get('Team B', 0)}%"
            )
            cv2.putText(
                annotated_frame,
                stats_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if out:
                out.write(annotated_frame)

            # Uncomment for live preview
            # cv2.imshow("Football Tracker", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Processing Complete.")
        print("Final Stats:", game_logic.get_stats())


if __name__ == "__main__":
    main()
