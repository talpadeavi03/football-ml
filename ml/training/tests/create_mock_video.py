import cv2
import numpy as np

def create_mock_video(filename='data/videos/mock_match.mp4', duration=5, fps=30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Mock Players: [x, y, dx, dy, color]
    players = [
        {'pos': [300, 360], 'vel': [2, 0], 'color': (255, 0, 0)},   # Team A
        {'pos': [900, 360], 'vel': [-2, 0], 'color': (0, 0, 255)},  # Team B
        {'pos': [600, 200], 'vel': [0, 2], 'color': (255, 0, 0)},   # Team A
    ]
    
    ball = {'pos': [640, 360], 'vel': [3, 1], 'color': (255, 255, 255)}
    
    for _ in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (0, 100, 0) # Green field
        
        # Draw Half Line
        cv2.line(frame, (640, 0), (640, 720), (255, 255, 255), 2)
        
        # Update and Draw Players
        for p in players:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            
            # Bounce
            if p['pos'][0] < 0 or p['pos'][0] > width: p['vel'][0] *= -1
            if p['pos'][1] < 0 or p['pos'][1] > height: p['vel'][1] *= -1
            
            cv2.circle(frame, (int(p['pos'][0]), int(p['pos'][1])), 20, p['color'], -1)
            
        # Update and Draw Ball
        ball['pos'][0] += ball['vel'][0]
        ball['pos'][1] += ball['vel'][1]
        if ball['pos'][0] < 0 or ball['pos'][0] > width: ball['vel'][0] *= -1
        if ball['pos'][1] < 0 or ball['pos'][1] > height: ball['vel'][1] *= -1
        
        cv2.circle(frame, (int(ball['pos'][0]), int(ball['pos'][1])), 10, ball['color'], -1)
        
        out.write(frame)
        
    out.release()
    print(f"Mock video saved to {filename}")

if __name__ == "__main__":
    create_mock_video()
