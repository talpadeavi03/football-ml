# src/team_classifier.py

import math


class TeamClassifier:
    def __init__(self, init_frames=100):
        self.init_frames = init_frames
        self.player_team = {}   # player_id -> team_id
        self.locked = False
        self.init_positions = [] # Store (pid, x) during init

    def assign_teams(self, detections, frame, frame_idx, identity_manager=None):
        """
        Assign teams using median split during init, then centroids.
        """
        team_assignments = {}
        
        # 1. Collect positions during init
        if not self.locked:
            for det in detections:
                if det['class'] in ['player', 'goalkeeper']:
                    self.init_positions.append(det['center'][0])
            
            if frame_idx >= self.init_frames:
                self.locked = True
                if self.init_positions:
                    self.median_x = sum(self.init_positions) / len(self.init_positions)
                else:
                    self.median_x = frame.shape[1] / 2

        # 2. Collect current team centroids for post-lock assignment
        team_centers = {'Team A': [], 'Team B': []}
        for det in detections:
            pid = det['id']
            
            # Recover from ReID
            if identity_manager:
                identity = identity_manager.get_identity(pid)
                if identity and identity.get("team_id"):
                    self.player_team[pid] = identity["team_id"]

            if pid in self.player_team:
                team_id = self.player_team[pid]
                team_centers[team_id].append(det['center'])

        team_centroid = {}
        for tid, centers in team_centers.items():
            if centers:
                team_centroid[tid] = (
                    sum(c[0] for c in centers) / len(centers),
                    sum(c[1] for c in centers) / len(centers)
                )

        # 3. Assign teams to current detections
        for det in detections:
            if det['class'] not in ['player', 'goalkeeper', 'referee']:
                continue

            pid = det['id']
            if pid in self.player_team:
                team_assignments[pid] = self.player_team[pid]
                continue

            cx, cy = det['center']

            if not self.locked:
                # Use current frame's position relative to center for now
                if cx < frame.shape[1] / 2:
                    team_id = 'Team A'
                else:
                    team_id = 'Team B'
            else:
                # Use centroids if available
                if team_centroid:
                    dists = {tid: math.dist((cx, cy), c) for tid, c in team_centroid.items()}
                    team_id = min(dists, key=dists.get)
                else:
                    team_id = 'Team A' if cx < self.median_x else 'Team B'

            self.player_team[pid] = team_id
            team_assignments[pid] = team_id

        return team_assignments

    def get_team(self, player_id):
        return self.player_team.get(player_id, None)

    def update_player(self, player_id, name):
        # Names do not affect team
        return
