import mediapipe
print("mediapipe dir:", dir(mediapipe))
try:
    from mediapipe.python.solutions import face_detection
    print("Imported from mediapipe.python.solutions")
except ImportError:
    print("Failed to import from mediapipe.python.solutions")

try:
    import mediapipe.solutions.face_detection
    print("Imported mediapipe.solutions.face_detection")
except ImportError:
    print("Failed to import mediapipe.solutions.face_detection")
