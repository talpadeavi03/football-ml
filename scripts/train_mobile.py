from ultralytics import YOLO
import torch

def train_mobile_model():
    # 1. Load the Nano model (smallest, fastest, best for Android)
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # 2. Train the model
    # We use a very small batch size to fit in 8GB RAM (CPU/GPU)
    # 'data' should point to a yaml file describing the dataset (created later)
    try:
        results = model.train(
            data='football.yaml', # Using football dataset
            epochs=2,          # Quick test with 2 epochs (~2-3 hours)
            imgsz=640,
            batch=2,           # Low batch size for 8GB RAM
            device='cpu',      # Force CPU if no GPU (or '0' for GPU)
            workers=12,        # Use all 12 CPU cores for data loading
            project='football_mobile_project',
            name='yolov8n_mobile',
            exist_ok=True
        )
        
        # 3. Export to TFLite (for Android)
        print("Exporting to TFLite for Android...")
        model.export(format='tflite', optimize=True) # Quantization for smaller size
        
    except Exception as e:
        print(f"Training Error: {e}")

if __name__ == "__main__":
    train_mobile_model()
