# Step-by-Step Cloud Deployment Guide: Football-ML

Follow these steps to deploy your full-stack MLOps pipeline to the cloud for free.

---

## 🟢 Step 1: Model Preparation (Google Colab)
1. **Open Colab**: Use the training script in `ml/training/train_mobile.py`.
2. **Train**: Run the training on the SoccerNet dataset using a GPU.
3. **Export**:
   - Export to ONNX: `model.export(format="onnx")`
   - Export to TFLite: `model.export(format="tflite", optimize=True)`
4. **Commit**: Move the exported models to `ml/models/v1.0.0/` in your repo and push to GitHub.

---

## 🔵 Step 2: Backend Deployment (Hugging Face Spaces)
1. **Create Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2. **Select Docker**: Choose "Docker" as the SDK.
3. **Connect GitHub**: Link your `football-ml` repository.
4. **Set Context**: Set the "Docker context" to `backend/`.
5. **Add Secrets**: In Space Settings, add:
   - `MODEL_PATH`: `ml/models/v1.0.0/model.onnx`
6. **Deploy**: HF will automatically build and run your FastAPI app. Note the public URL (e.g., `https://user-space.hf.space`).

---

## 🟠 Step 3: Edge Gateway Deployment (Cloudflare)
1. **Create Worker**: Go to the Cloudflare Dashboard -> Workers & Pages -> Create Application.
2. **Deploy Code**: Copy the content of `infra/cloudflare/worker.js` into the Worker editor.
3. **Configure Variables**: In Worker Settings -> Variables, add:
   - `API_KEY`: A secure random string (e.g., `my-secret-key-123`).
   - `BACKEND_URL`: Your Hugging Face Space URL from Step 2.
4. **Deploy**: Save and Deploy. Note your Worker URL (e.g., `https://football-ml.user.workers.dev`).

---

## 🟡 Step 4: CI/CD Setup (GitHub Actions)
1. **GitHub Secrets**: Go to your Repo -> Settings -> Secrets and variables -> Actions.
2. **Add Secrets**:
   - `HF_TOKEN`: Your Hugging Face Write Token (from HF Settings -> Tokens).
   - `CLOUDFLARE_API_TOKEN`: Your Cloudflare API Token.
   - `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare Account ID.
3. **Verify**: Push a small change to `backend/app.py` to trigger the `mlops-pipeline.yml` workflow.

---

## 🔴 Step 5: Mobile Integration (Android)
1. **Update Constants**: In your Android project, find the API configuration file (e.g., `Constants.kt`).
2. **Set URLs**:
   - `CLOUDFLARE_URL`: Your Cloudflare Worker URL from Step 3.
   - `API_KEY`: The same `API_KEY` you set in Cloudflare.
3. **Add TFLite**: Copy `model.tflite` from Step 1 into `app/src/main/assets/`.
4. **Build APK**: Generate the signed APK and install it on your device.

---

## 🏁 Step 6: End-to-End Verification
1. **Health Check**: Visit `https://your-worker.workers.dev/health` in your browser. It should return `{"status": "ok"}`.
2. **Test Prediction**: Open the APK, point it at a football match, and verify that detections are appearing (either via local TFLite or Cloud fallback).
3. **Check Logs**: Monitor Cloudflare Worker logs and HF Space logs for any errors.

---

**Congratulations!** You are now running a professional MLOps pipeline on the cloud for $0/month.
