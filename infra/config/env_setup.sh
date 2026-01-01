#!/bin/bash

# Environment Setup Script for Football-ML
# This script helps you set up the necessary environment variables and secrets.

echo "🚀 Setting up Football-ML Cloud Connection..."

# 1. GitHub Secrets
echo "------------------------------------------------"
echo "🔐 Step 1: GitHub Secrets (for CI/CD)"
echo "Add these to your GitHub Repo -> Settings -> Secrets and variables -> Actions:"
echo "- HF_TOKEN: Your Hugging Face Write Token"
echo "- CLOUDFLARE_API_TOKEN: Your Cloudflare API Token"
echo "- CLOUDFLARE_ACCOUNT_ID: Your Cloudflare Account ID"
echo "------------------------------------------------"

# 2. Cloudflare Worker Variables
echo "🌐 Step 2: Cloudflare Worker Variables"
echo "Add these via 'wrangler.toml' or Cloudflare Dashboard:"
echo "- API_KEY: A secure random string (e.g., $(openssl rand -hex 16))"
echo "- BACKEND_URL: The URL of your deployed FastAPI (e.g., https://user-space.hf.space)"
echo "------------------------------------------------"

# 3. Backend Environment Variables
echo "🐍 Step 3: Backend Environment Variables"
echo "Add these to your HF Space or Render environment:"
echo "- MODEL_PATH: models/yolov8n_football.onnx"
echo "------------------------------------------------"

# 4. Android Configuration
echo "📱 Step 4: Android Configuration"
echo "Update 'mobile/android/app/src/main/java/com/example/football/Constants.kt':"
echo "val CLOUDFLARE_URL = \"https://your-worker.workers.dev\""
echo "val API_KEY = \"your-secure-api-key\""
echo "------------------------------------------------"

echo "✅ Connection guide generated. Follow the steps above to link your services."
