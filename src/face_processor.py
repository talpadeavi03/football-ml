# src/face_processor.py

import os
import cv2
import numpy as np

# 🔒 FORCE CPU MODE (VERY IMPORTANT)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from deepface import DeepFace


class FaceProcessor:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.known_faces = []   # list of {name, team_id, embedding}
        self.load_known_faces()

    # --------------------------------------------------
    # Utility helpers
    # --------------------------------------------------

    def _is_image(self, path):
        try:
            img = cv2.imread(path)
            return img is not None
        except Exception:
            return False

    def _infer_team(self, folder_name):
        name = folder_name.lower()
        if "a" in name:
            return 'Team A'
        if "b" in name:
            return 'Team B'
        return None

    # --------------------------------------------------
    # Load known player faces
    # --------------------------------------------------

    def load_known_faces(self):
        print("Loading known faces (CPU mode)...")

        if not os.path.exists(self.faces_dir):
            print("Faces directory not found:", self.faces_dir)
            return

        for root, _, files in os.walk(self.faces_dir):
            folder = os.path.basename(root)
            team_id = self._infer_team(folder)

            for file in files:
                img_path = os.path.join(root, file)

                if not self._is_image(img_path):
                    continue

                # Clean player name
                name = os.path.splitext(file)[0]
                name = name.replace("-", " ").replace("_", " ").strip()

                try:
                    rep = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=False,
                        detector_backend="opencv"
                    )

                    embedding = np.array(rep[0]["embedding"])

                    self.known_faces.append({
                        "name": name,
                        "team_id": team_id,
                        "embedding": embedding
                    })

                    print(f"✔ Loaded face: {name} (team={team_id})")

                except Exception as e:
                    print(f"✖ Failed to load {img_path}: {e}")

        print(f"Total known faces loaded: {len(self.known_faces)}")

    # --------------------------------------------------
    # 🔥 SINGLE FACE RECOGNITION (USED BY main.py)
    # --------------------------------------------------

    def recognize_single_face(self, face_img):
        """
        Recognize ONE face image.
        Returns: (name, team_id) or ("Unknown", None)
        """
        try:
            rep = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend="opencv"
            )[0]["embedding"]

            rep = np.array(rep)

            best_match = None
            best_score = 0.0

            for known in self.known_faces:
                score = np.dot(rep, known["embedding"]) / (
                    np.linalg.norm(rep) * np.linalg.norm(known["embedding"])
                )
                if score > best_score:
                    best_score = score
                    best_match = known

            if best_match and best_score > 0.75:
                return best_match["name"], best_match["team_id"]

        except Exception:
            pass

        return "Unknown", None
