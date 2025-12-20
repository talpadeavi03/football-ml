import numpy as np

class GameLogic:
    def __init__(self):
        self.possession_history = []
        self.current_possession = None
        self.goals = []
        self.possession_stats = {'Team A': 0, 'Team B': 0}
        self.activity_stats = {'Team A': 0, 'Team B': 0}
        self.prev_positions = {} # pid -> last_center
        self.debug_count = 0 

    def update(self, frame_idx, detections, team_assignments):
        ball = None
        players = []
        
        for det in detections:
            if det['class'].lower() == 'ball':
                ball = det
            elif det['class'].lower() in ['player', 'goalkeeper', 'referee']:
                players.append(det)
                
                # Track activity (distance moved)
                pid = det['id']
                team = team_assignments.get(pid)
                if team in self.activity_stats:
                    if pid in self.prev_positions:
                        dist = np.linalg.norm(np.array(det['center']) - np.array(self.prev_positions[pid]))
                        if dist < 100: # Filter out jumps
                            self.activity_stats[team] += dist
                    self.prev_positions[pid] = det['center']

        if ball and players:
            closest_player = None
            min_dist = float('inf')
            possession_threshold = 80 
            
            ball_center = np.array(ball['center'])
            for player in players:
                dist = np.linalg.norm(ball_center - np.array(player['center']))
                if dist < min_dist:
                    min_dist = dist
                    closest_player = player
            
            if closest_player and min_dist < possession_threshold:
                pid = closest_player['id']
                team = team_assignments.get(pid, 'Unknown')
                if team != 'Unknown':
                    self.current_possession = (pid, team)
                    self.possession_stats[team] += 1

    def get_stats(self):
        total_pos = sum(self.possession_stats.values())
        pos_a = round(self.possession_stats['Team A'] / total_pos * 100, 1) if total_pos > 0 else 0
        pos_b = round(self.possession_stats['Team B'] / total_pos * 100, 1) if total_pos > 0 else 0
        
        total_act = sum(self.activity_stats.values())
        act_a = round(self.activity_stats['Team A'] / total_act * 100, 1) if total_act > 0 else 0
        act_b = round(self.activity_stats['Team B'] / total_act * 100, 1) if total_act > 0 else 0

        return {
            'possession': {'Team A': pos_a, 'Team B': pos_b},
            'activity': {'Team A': act_a, 'Team B': act_b}
        }