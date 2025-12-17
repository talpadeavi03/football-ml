import numpy as np

class GameLogic:
    def __init__(self):
        self.possession_history = []
        self.current_possession = None
        self.goals = []
        self.possession_stats = {'Team A': 0, 'Team B': 0}
        self.debug_count = 0 # Counter to limit debug prints

    def update(self, frame_idx, detections, team_assignments):
        """
        Updates game state based on current frame detections.
        """
        ball = None
        players = []
        
        # --- DEBUG 1: Check Team Assignments up to frame 150 ---
        if frame_idx == 150:
            print(f"\n--- DEBUG: Team Assignments at Frame 150 ---")
            print(f"Total Assigned Players: {len(team_assignments)}")
            print(f"Sample Assignments: {list(team_assignments.items())[:5]}")
            print(f"-----------------------------------------\n")

        for det in detections:
            if det['class'].lower() == 'ball':
                ball = det
            elif det['class'].lower() in ['player', 'goalkeeper', 'referee'] and det['id'] != -1:
                players.append(det)
                
        # --- DEBUG 2: Check Ball and Player Detection ---
        if frame_idx < 10 and self.debug_count < 5:
            print(f"Frame {frame_idx}: Ball Detected = {ball is not None}, Players Detected = {len(players)}")
            self.debug_count += 1
            
        if ball and players:
            # 1. Check Possession
            closest_player = None
            min_dist = float('inf')
            possession_threshold = 50 
            
            ball_center = np.array(ball['center'])
            
            for player in players:
                player_center = np.array(player['center'])
                dist = np.linalg.norm(ball_center - player_center)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_player = player
            
            if closest_player and min_dist < possession_threshold:
                pid = closest_player['id']
                team = team_assignments.get(pid, 'Unknown')
                
                # --- DEBUG 3: Check Closest Player and Team Assignment ---
                if frame_idx < 20:
                    if team == 'Unknown':
                         print(f"Frame {frame_idx}: **POSSESSION FAILED**: Closest Player ID {pid} is UNKNOWN in team_assignments.")
                    else:
                         print(f"Frame {frame_idx}: POSSESSION SUCCESS: Player {pid} (Team {team}) has ball (Dist: {min_dist:.1f}).")

                
                if team != 'Unknown':
                    self.current_possession = (pid, team)
                    self.possession_history.append((frame_idx, pid, team))
                    self.possession_stats[team] += 1
                    
            # 2. Check Goals (Using 1920x1080 dimensions)
            frame_width = 1920
            
            # ... (Goal logic remains the same, omitted for brevity) ...

    def get_stats(self):
        # ... (Same as before) ...
        total_frames = sum(self.possession_stats.values())
        if total_frames == 0:
            return {'Team A': 0, 'Team B': 0}
            
        return {
            'Team A': round(self.possession_stats['Team A'] / total_frames * 100, 1),
            'Team B': round(self.possession_stats['Team B'] / total_frames * 100, 1)
        }