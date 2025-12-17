import cv2
import os
import numpy as np
from deepface import DeepFace

class FaceProcessor:
    def __init__(self, faces_dir):
        self.faces_dir = faces_dir
        self.known_face_embeddings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Loads face images from the data/faces directory and encodes them."""
        if not os.path.exists(self.faces_dir):
            print(f"Warning: Faces directory {self.faces_dir} does not exist.")
            return

        print("Loading known faces with DeepFace...")
        for filename in os.listdir(self.faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(self.faces_dir, filename)
                try:
                    # Get embedding using Facenet512 (fast and accurate)
                    # enforce_detection=False allows loading cropped faces
                    embedding_objs = DeepFace.represent(img_path=filepath, model_name="Facenet512", enforce_detection=False)
                    
                    if embedding_objs:
                        embedding = embedding_objs[0]["embedding"]
                        self.known_face_embeddings.append(embedding)
                        
                        # Use filename without extension as the name
                        raw_name = os.path.splitext(filename)[0]
                        # Cleanup: remove 'photo-X-' prefix and replace hyphens with spaces
                        # Example: photo-1-amol-sir -> amol sir
                        clean_name = raw_name
                        if clean_name.startswith("photo-"):
                            parts = clean_name.split('-')
                            # photo, 1, amol, sir ...
                            # Remove first two parts if they are 'photo' and a number
                            if len(parts) > 2 and parts[1].replace('.', '', 1).isdigit():
                                clean_name = " ".join(parts[2:])
                            else:
                                clean_name = " ".join(parts)
                        else:
                            clean_name = raw_name.replace('-', ' ')
                            
                        clean_name = clean_name.title() # Capitalize
                        
                        self.known_face_names.append(clean_name)
                        print(f"Loaded face: {clean_name} (from {filename})")
                    else:
                        print(f"Warning: No face found in {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        print(f"Total known faces loaded: {len(self.known_face_names)}")

    def recognize_faces(self, frame, rois=None):
        """
        Recognizes faces in the given frame.
        Returns list of (top, right, bottom, left), name
        """
        # DeepFace expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = []
        
        # If ROIs provided (from YOLO), verify them
        if rois:
            for (x1, y1, x2, y2) in rois:
                face_img = rgb_frame[y1:y2, x1:x2]
                if face_img.size == 0: continue
                
                try:
                    # Get embedding for the crop
                    target_objs = DeepFace.represent(img_path=face_img, model_name="Facenet512", enforce_detection=False)
                    if not target_objs: continue
                    
                    target_embedding = target_objs[0]["embedding"]
                    
                    # Compare with known faces
                    best_match_name = "Unknown"
                    min_dist = float('inf')
                    
                    for idx, known_emb in enumerate(self.known_face_embeddings):
                        # Cosine distance
                        dist = self.find_cosine_distance(known_emb, target_embedding)
                        if dist < min_dist:
                            min_dist = dist
                            # Relaxed threshold for better recall on low-res video (0.3 -> 0.5)
                            if dist < 0.50: 
                                best_match_name = self.known_face_names[idx]
                    
                    results.append(((y1, x2, y2, x1), best_match_name))
                    
                except Exception as e:
                    # print(f"Face rec error: {e}")
                    pass
        
        return results

    def find_cosine_distance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
