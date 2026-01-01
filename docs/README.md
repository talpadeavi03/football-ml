# Football-ML: Production-Grade MLOps & Cloud Deployment

This project implements a full-stack football analytics system with on-device TFLite inference and a secure cloud fallback via FastAPI and Cloudflare.

## 🔗 Connecting Services

To link all components (GitHub, Backend, Cloudflare, Mobile), follow the [Connection Guide](file:///home/sudo69/pro/infra/config/env_setup.sh) or refer to the [Master Documentation](file:///home/sudo69/pro/ml_mlops_cloud_devops_full_production.md).

### Quick Setup:
1. **Train & Export**: Use Google Colab to train YOLOv8 and export to ONNX/TFLite.
2. **Deploy Backend**: Push `backend/` to Hugging Face Spaces (Docker).
3. **Deploy Edge**: Deploy `infra/cloudflare/worker.js` to Cloudflare Workers.
4. **Configure Secrets**: Add `API_KEY` and `BACKEND_URL` to Cloudflare and GitHub.
5. **Build APK**: Update the Android app with your Cloudflare URL and build.

## 📁 Repository Structure
- `ml/`: Training scripts and model registry.
- `backend/`: FastAPI inference service.
- `infra/`: Cloudflare Workers and environment config.
- `mobile/`: Android application source.
- `docs/`: Detailed guides and documentation.

---
*Original Notes:*
- Use antigravity to setup the whole project using pull request.
- Game logic is wrong while sorting Team A and Team B as the old logic was at the start of match the teams seperated at the center line was the input of the faces.
- I didnt have enough data to feed to train the ML as of now for accuracy.
