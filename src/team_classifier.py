class TeamClassifier:
    def __init__(self):
        self.player_teams = {} # Map track_id -> team_name
        
        # Explicit Name Mapping
        self.name_map = {
            'amol': 'Team A',
            'Vitekar': 'Team A',
            'harshal': 'Team A',
            'shreyas': 'Team B',
            'tejas': 'Team B'
        }

    def assign_teams(self, detections, frame_width, frame_idx):
        """
        Assigns teams based on position.
        The half-line logic is used persistently to assign teams to ANY new, unassigned player.
        """
        half_line_x = frame_width / 2
        
        for det in detections:
            # Only consider players and goalkeepers for team assignment
            if det['class'] not in ['player', 'goalkeeper']:
                continue
            
            track_id = det['id']
            if track_id == -1:
                continue
                
            # --- FIX: Only assign team if the player is NOT already in the dictionary ---
            if track_id in self.player_teams:
                continue
            
            # Assign team based on position (INITIAL ASSIGNMENT for ANY new ID)
            center_x = det['center'][0]
            
            # We assume Team A starts on the left side of the screen
            if center_x < half_line_x:
                self.player_teams[track_id] = 'Team A'
            else:
                self.player_teams[track_id] = 'Team B'
                
        return self.player_teams

    def update_player(self, track_id, name):
        """
        Updates the team of a player based on their recognized name.
        Overrides existing assignment.
        """
        for key, team in self.name_map.items():
            if key.lower() in name.lower():
                self.player_teams[track_id] = team
                print(f"Updated Player {track_id} ({name}) to {team}")
                return

    def get_team(self, track_id):
        return self.player_teams.get(track_id, "Unknown")