import cv2
import os

class FaceDetector:
    def __init__(self):
        # Load OpenCV's pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face_in_player(self, frame, player_box):
        """
        player_box = (x1, y1, x2, y2)
        Returns: face_bbox (fx, fy, fw, fh) or None
        """
        x1, y1, x2, y2 = map(int, player_box)
        
        # Ensure box is within frame
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        # Crop upper body (faces are usually in the top 60% of the player box)
        h = y2 - y1
        face_region_h = int(h * 0.6)
        face_region = frame[y1 : y1 + face_region_h, x1:x2]

        if face_region.size == 0:
            return None

        # Convert to grayscale for Haar Cascade
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(faces) > 0:
            # Return the first detection's bounding box relative to the original frame
            (fx, fy, fw, fh) = faces[0]
            return (fx + x1, fy + y1, fw, fh)

        return None
