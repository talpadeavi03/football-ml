# MLOps Deployment Documentation: Football-ML

This document details the full-stack cloud deployment and MLOps strategy for the `football-ml` project.

## 1. Architecture Overview

The system follows a "Control Plane vs. Compute Plane" architecture:
- **Training**: Google Colab (GPU)
- **Model Registry**: GitHub (Versioned artifacts)
- **Inference (Compute)**: Hugging Face Spaces / Render (Dockerized FastAPI)
- **API Gateway (Control)**: Cloudflare Workers
- **Edge Delivery**: Android APK (TFLite)

## 2. MLOps Pipeline

### Phase 1: Training & Export
1. Train YOLOv8 on SoccerNet dataset in Google Colab.
2. Export to **ONNX** (for Cloud) and **TFLite** (for Mobile).
3. Version models in `models/vX.Y.Z/`.

### Phase 2: CI/CD
GitHub Actions automates:
- Testing backend logic.
- Building Docker images.
- Deploying to inference platforms.

### Phase 3: Inference & Security
- **FastAPI** serves the model via a REST API.
- **Cloudflare Workers** provide:
    - API Key Authentication.
    - Rate Limiting.
    - Canary Deployments (Traffic splitting).

## 3. Deployment Steps

### Backend (FastAPI)
```bash
cd backend
docker build -t football-ml-api .
# Deploy to HF Spaces or Render
```

### Edge (Android)
1. Load `model.tflite` into Android Studio assets.
2. Use CameraX for real-time frames.
3. Fallback to Cloud API for heavy analytics.

## 4. Advanced Features
- **A/B Testing**: Split traffic between model versions at the edge.
- **Drift Detection**: Monitor prediction confidence and input statistics.
- **Feature Flags**: Toggle cloud inference without app updates.
