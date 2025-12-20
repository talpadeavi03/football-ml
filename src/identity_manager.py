# src/identity_manager.py

import uuid
import numpy as np
from src.appearance_model import AppearanceModel
from src.motion_model import MotionModel
from src.reid_engine import ReIDEngine


class IdentityManager:
    def __init__(self):
        self.identity_db = {}          # identity_id → data
        self.track_to_identity = {}    # track_id → identity_id
        self.appearance_model = AppearanceModel()
        self.reid_engine = ReIDEngine()

    def create_identity(self, track_id, frame, bbox, center, name, team_id):
        # Check if this track already has an identity
        if track_id in self.track_to_identity:
            identity_id = self.track_to_identity[track_id]
            # Update existing identity if needed (e.g., name discovered)
            if name and name != "Unknown":
                self.identity_db[identity_id]["name"] = name
            if team_id is not None:
                self.identity_db[identity_id]["team_id"] = team_id
            return identity_id

        # Try to match with existing identities first (ReID)
        appearance = self.appearance_model.extract(frame, bbox)
        best_identity_id = self.match_track_to_identity(appearance)

        if best_identity_id:
            self.track_to_identity[track_id] = best_identity_id
            if name and name != "Unknown":
                self.identity_db[best_identity_id]["name"] = name
            return best_identity_id

        # Create new identity if no match
        identity_id = str(uuid.uuid4())
        self.identity_db[identity_id] = {
            "name": name,
            "team_id": team_id,
            "appearance": appearance,
            "motion": MotionModel()
        }
        self.identity_db[identity_id]["motion"].update(center)
        self.track_to_identity[track_id] = identity_id
        return identity_id

    def match_track_to_identity(self, appearance):
        if appearance is None:
            return None

        best_id = None
        best_score = 0.0

        for identity_id, data in self.identity_db.items():
            score = self.appearance_model.similarity(appearance, data["appearance"])
            if score > best_score:
                best_score = score
                best_id = identity_id

        # Threshold for ReID matching
        if best_score > 0.8:
            return best_id
        return None

    def update(self, track_id, frame, bbox, center):
        if track_id not in self.track_to_identity:
            # Try to recover identity via ReID even without a face
            appearance = self.appearance_model.extract(frame, bbox)
            identity_id = self.match_track_to_identity(appearance)
            if identity_id:
                self.track_to_identity[track_id] = identity_id
            else:
                return None

        identity_id = self.track_to_identity[track_id]
        self.identity_db[identity_id]["motion"].update(center)
        
        # Periodically update appearance to handle lighting changes
        # (Simplified: just store the latest good crop)
        new_appearance = self.appearance_model.extract(frame, bbox)
        if new_appearance:
            self.identity_db[identity_id]["appearance"] = new_appearance
            
        return identity_id

    def get_identity(self, track_id):
        identity_id = self.track_to_identity.get(track_id)
        if not identity_id:
            return None
        return self.identity_db[identity_id]
