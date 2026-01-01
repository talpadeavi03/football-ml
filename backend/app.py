from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os

app = FastAPI(title="Football ML Inference API")

# Load model - assuming it will be in a 'models' directory in the container
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n_football.onnx")

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/health")
def health():
    return {
        "status": "ok" if model else "error",
        "model": "yolov8-football",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        img_bytes = await file.read()
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        results = model(img)
        # Extract boxes, classes, and confidences
        boxes = results[0].boxes.xyxy.tolist()
        classes = results[0].boxes.cls.tolist()
        confidences = results[0].boxes.conf.tolist()

        return {
            "detections": [
                {"box": b, "class": int(c), "confidence": float(conf)}
                for b, c, conf in zip(boxes, classes, confidences)
            ],
            "count": len(boxes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
