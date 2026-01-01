import math

class MotionModel:
    def __init__(self):
        self.positions = []

    def update(self, center):
        self.positions.append(center)
        if len(self.positions) > 10:
            self.positions.pop(0)

    def get_signature(self):
        if len(self.positions) < 2:
            return 0.0

        dist = 0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            dist += math.sqrt(dx*dx + dy*dy)

        return dist / len(self.positions)
