class ReIDEngine:
    def match(self, appearance_sim, motion_sim):
        return (
            0.7 * appearance_sim +
            0.3 * motion_sim
        )
