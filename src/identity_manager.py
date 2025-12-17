# src/identity_manager.py

import uuid

from appearance_model import AppearanceModel
from motion_model import MotionModel
from reid_engine import ReIDEngine


class IdentityManager:
    def __init__(self):
        # identity_id -> data
        self.identity_db = {}

        # track_id -> identity_id
        self.track_to_identity = {}

        self.appearance_model = AppearanceModel()
        self.reid_engine = ReIDEngine()

    def create_identity(self, track_id, frame, bbox, name=None):
        identity_id = str(uuid.uuid4())

        appearance = self.appearance_model.extract(frame, bbox)
        motion = MotionModel()

        self.identity_db[identity_id] = {
            "name": name,          # Optional (None for now)
            "appearance": appearance,
            "motion": motion,
            "confidence": 1.0
        }

        self.track_to_identity[track_id] = identity_id
        return identity_id

    def update(self, track_id, frame, bbox, center, face_confirmed=False, name=None):
        # 1️⃣ Track already linked → update motion
        if track_id in self.track_to_identity:
            identity_id = self.track_to_identity[track_id]
            self.identity_db[identity_id]["motion"].update(center)
            return identity_id

        # 2️⃣ Try re-identification using appearance + motion
        current_appearance = self.appearance_model.extract(frame, bbox)

        best_match = None
        best_score = 0.0

        for identity_id, data in self.identity_db.items():
            appearance_sim = self.appearance_model.similarity(
                current_appearance, data["appearance"]
            )

            motion_sim = data["motion"].get_signature()
            score = self.reid_engine.match(appearance_sim, motion_sim)

            if score > best_score:
                best_score = score
                best_match = identity_id

        if best_score > 0.75:
            self.track_to_identity[track_id] = best_match
            self.identity_db[best_match]["confidence"] = min(
                1.0, self.identity_db[best_match]["confidence"] + 0.05
            )
            return best_match

        # 3️⃣ Create identity ONLY if face seen
        if face_confirmed:
            return self.create_identity(track_id, frame, bbox, name)

        return None

    def get_identity_name(self, track_id):
        identity_id = self.track_to_identity.get(track_id)
        if not identity_id:
            return None
        return self.identity_db[identity_id]["name"]
