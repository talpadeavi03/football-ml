# Master Documentation: Full-Production MLOps, Cloud & DevOps Pipeline

This document serves as the ultimate guide for the `football-ml` project, detailing how to run a professional-grade ML system entirely on the cloud for free.

---

## 🚀 1. Architecture: The "Free-Tier" Power Stack

We use a split-plane architecture to ensure scalability, security, and zero cost.

| Component | Service | Role | Why? |
| :--- | :--- | :--- | :--- |
| **Training** | Google Colab | GPU Compute | Free T4/L4 GPUs for YOLOv8 training. |
| **Source Control** | GitHub | Code & CI/CD | Versioning, Actions, and Model Registry. |
| **Inference API** | Hugging Face Spaces | Compute Plane | Free Docker hosting with 16GB RAM. |
| **API Gateway** | Cloudflare Workers | Control Plane | Edge security, Auth, and Routing. |
| **Mobile App** | Android (Kotlin) | Delivery Layer | On-device TFLite + Cloud fallback. |

---

## 🛠 2. MLOps Pipeline (The Lifecycle)

### Phase 1: Training & Model Registry
- **Training**: Done in Google Colab. Models are exported as `.onnx` (Cloud) and `.tflite` (Mobile).
- **Versioning**: Models are stored in `ml/models/vX.Y.Z/` with a `metadata.json` for auditability.
- **Rollback**: Managed via a `latest` symlink/pointer. Rollback is a simple pointer update.

### Phase 2: CI/CD (DevOps)
- **GitHub Actions**: Triggered on changes to `backend/` or `ml/models/`.
- **Pipeline**: Linting -> Testing -> Docker Build -> Deploy.
- **Docker**: Ensures "it works on my machine" works everywhere.

### Phase 3: Inference as a Service
- **FastAPI**: A high-performance Python API serving YOLOv8.
- **Endpoints**:
    - `GET /health`: System status.
    - `POST /predict`: Receives image, returns JSON detections.

---

## 🛡 3. Cloudflare: The Security & Control Layer

Cloudflare Workers act as a "Zero-Trust" gateway between the APK and the Backend.

- **Authentication**: `x-api-key` header validation.
- **Rate Limiting**: Prevents abuse of the free-tier backend.
- **Canary Deployments**: 10% of traffic can be routed to a new model version for testing.
- **A/B Testing**: Compare Model A vs. Model B performance in real-time.

---

## 📱 4. Mobile Delivery (Android)

The APK uses a **Hybrid Inference** model:
1. **Real-time**: On-device `yolov8n_football.tflite` for zero latency.
2. **Heavy Analytics**: Cloud API call via Cloudflare for complex match analysis.
3. **Security**: Device-specific tokens generated on first launch to prevent unauthorized API usage.

---

## 📈 5. Advanced MLOps: Observability & Drift

- **Metrics**: Latency, detection counts, and confidence scores are logged.
- **Drift Detection**: If the mean confidence of predictions drops below a threshold, a GitHub Issue is automatically created for retraining.
- **Feature Flags**: Toggle "Cloud Inference" or "Experimental Features" via `infra/config/feature_flags.json` without updating the APK.

---

## 🏁 6. Project Management (GitHub)

### Milestones
1. **Foundation**: Repo restructuring & Training pipeline.
2. **DevOps**: Dockerized backend & CI/CD automation.
3. **MLOps**: Versioning, Canary, and Drift detection.
4. **Delivery**: Android integration & Secure Edge routing.

### Example Issues
- `[MLOps] Implement Canary Routing in Cloudflare`
- `[DevOps] Add Docker Build step to GitHub Actions`
- `[ML] Export YOLOv8 to ONNX with dynamic shapes`

---

## 💰 7. Staying "Free Forever" Strategy
- **Cloudflare**: 100k requests/day (Free).
- **HF Spaces**: 24/7 CPU hosting (Free).
- **GitHub Actions**: 2,000 mins/month (Free).
- **Google Colab**: On-demand GPU (Free).

**Rule**: Always prefer on-device compute (TFLite) to save cloud resources.
