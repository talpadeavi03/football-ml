try:
    import mediapipe as mp
    print("mediapipe imported")
    print("solutions:", hasattr(mp, 'solutions'))
    if hasattr(mp, 'solutions'):
        print("face_detection:", hasattr(mp.solutions, 'face_detection'))
except Exception as e:
    print("Error:", e)
