import cv2
import numpy as np

class AppearanceModel:
    def extract(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Resize to normalize
        crop = cv2.resize(crop, (64, 128))

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Color histogram (very compact)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0,180,0,256])
        cv2.normalize(hist, hist)

        # Shape ratio
        h, w = crop.shape[:2]
        ratio = h / (w + 1e-6)

        return {
            "hist": hist.flatten(),
            "ratio": ratio
        }

    def similarity(self, a, b):
        if a is None or b is None:
            return 0.0

        color_sim = cv2.compareHist(
            a["hist"], b["hist"], cv2.HISTCMP_CORREL
        )

        ratio_sim = 1 - abs(a["ratio"] - b["ratio"])

        return 0.7 * color_sim + 0.3 * ratio_sim
