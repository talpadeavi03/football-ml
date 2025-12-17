# Football Player Tracking & Stats - Project Documentation

## Overview
This project is an AI-powered application designed to track football players, the ball, and referees in video footage. It provides real-time statistics such as possession percentages and goal detection. The system uses **YOLOv8** for object detection and tracking, **DeepFace** for player identification, and custom logic for game state analysis.

## Directory Structure
```
pro/
├── data/
│   ├── datasets/       # Training datasets (SoccerNet Tracking)
│   └── faces/          # Reference images for face recognition
├── scripts/            # Utility scripts for training and data setup
├── src/                # Core application source code
├── tests/              # Unit tests and verification scripts
├── football.yaml       # YOLO dataset configuration
├── requirements.txt    # Python dependencies
└── yolov8n.pt          # Pre-trained YOLOv8 model weights
```

## File Descriptions

### Source Code (`src/`)

#### `src/main.py`
**Purpose:** The entry point of the application.
**Functionality:**
- Parses command-line arguments (`--mode`, `--video`, `--output`).
- Initializes the pipeline modules (`Tracker`, `FaceProcessor`, `TeamClassifier`, `GameLogic`).
- Captures video frames (from file or webcam).
- Orchestrates the processing loop:
    1.  **Tracking:** Detects objects using YOLO.
    2.  **Team Assignment:** Assigns teams based on position or face recognition.
    3.  **Face Recognition:** Periodically identifies specific players.
    4.  **Game Logic:** Updates possession and stats.
    5.  **Visualization:** Draws bounding boxes and stats on the frame.

#### `src/tracker.py`
**Purpose:** Wrapper around the YOLOv8 model.
**Functionality:**
- Loads the YOLO model (`yolov8n.pt`).
- `track_objects(frame)`: Runs object tracking on a video frame. Filters detections to only include relevant classes (Person, Sports Ball).
- `draw_annotations(...)`: Visualizes tracking results with color-coded bounding boxes (Team A vs Team B).

#### `src/game_logic.py`
**Purpose:** Manages the state of the football game.
**Functionality:**
- Tracks possession based on the proximity of players to the ball.
- Detects goals using simple heuristic logic (ball crossing frame edges).
- Calculates and returns possession statistics (`get_stats()`).

#### `src/team_classifier.py`
**Purpose:** Assigns detected players to teams.
**Functionality:**
- Uses a heuristic (left vs right side of screen) for initial team assignment during the first few seconds.
- Supports updating player identity and team based on face recognition results.
- Maintains a persistent mapping of `track_id` to `team_name`.

#### `src/face_processor.py`
**Purpose:** Handles facial recognition to identify specific players.
**Functionality:**
- `load_known_faces()`: Scans `data/faces/` for reference images and computes embeddings using **DeepFace** (Facenet512).
- `recognize_faces(...)`: Extracts faces from current frame detections and compares them against known embeddings to find matches.

### Scripts (`scripts/`)

#### `scripts/train_mobile.py`
**Purpose:** Training script to fine-tune the YOLO model.
**Functionality:**
- Loads a pre-trained YOLOv8 Nano model (optimized for mobile).
- Trains the model using the dataset defined in `football.yaml`.
- Exports the trained model to TFLite format for Android deployment.

#### `scripts/download_datasets.py`
**Purpose:** Automates the downloading of the SoccerNet Tracking dataset.
**Functionality:**
- Uses the `SoccerNet` python package to download specific dataset splits (train, test, challenge).
- Handles directory creation and package installation.

#### `scripts/convert_soccernet_to_yolo.py`
**Purpose:** Converts dataset labels from SoccerNet format to YOLO format.
**Functionality:**
- Reads SoccerNet's `gt.txt` (MOT format).
- Converts bounding box coordinates to normalized YOLO format (`class x_center y_center width height`).
- Uses a size-based heuristic to distinguish the Ball (class 0) from Players (class 1).
- Saves `.txt` label files in `data/datasets/soccernet_tracking/labels/`.

### Configuration Files

#### `football.yaml`
**Purpose:** YOLO dataset configuration file.
**Content:**
- Defines paths to training and validation image directories.
- Lists class names and IDs (0: ball, 1: player, etc.).

#### `requirements.txt`
**Purpose:** Lists all Python libraries required to run the project.
**Key Dependencies:** `ultralytics` (YOLO), `deepface` (Face Rec), `opencv-python`, `numpy`.

## Usage Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Data Setup
Download and prepare the dataset:
```bash
python3 scripts/download_datasets.py
unzip data/datasets/soccernet_tracking/tracking/train.zip -d data/datasets/soccernet_tracking/images
python3 scripts/convert_soccernet_to_yolo.py
```

### 3. Training
Train the model (this may take hours depending on hardware):
```bash
python3 scripts/train_mobile.py
```

### 4. Running Inference
Run on a video file:
```bash
python3 src/main.py --mode upload --video path/to/video.mp4 --output result.mp4
```
Run on webcam:
```bash
python3 src/main.py --mode live
```
