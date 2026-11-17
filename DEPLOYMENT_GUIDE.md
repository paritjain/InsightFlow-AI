# InsightFlow AI - Complete Production Deployment Guide

This guide provides the fastest and most optimal paths to deploy the complete InsightFlow AI platform (Next.js 16 frontend + FastAPI / Python backend).

---

## 🛠️ Pre-Flight Fixes Applied to Your Repository

Before deployment, the following critical issues were diagnosed and resolved in your repository:

1. **Created Missing `backend/requirements.txt`**: Added all required dependencies (`fastapi`, `uvicorn`, `duckdb`, `pymupdf`, `rapidfuzz`, `scikit-learn`, `pandas`, `openpyxl`, `langgraph`, etc.).
2. **Fixed `.gitignore` Rule**: Removed rule blocking `backend/*.txt` so `requirements.txt` is tracked and pushed to Git.
3. **Dynamic Port Binding**: Updated `backend/Dockerfile` to bind to `${PORT:-8000}` dynamically for cloud environments (Render, Railway, Fly.io).
4. **Environment-Aware API Rewrites**: Updated `frontend/next.config.ts` to dynamically proxy `/api/*` requests to `BACKEND_API_URL` or `NEXT_PUBLIC_API_URL`.
5. **Multi-Container Networking**: Configured `docker-compose.yml` and `docker-compose.prod.yml` with Caddy reverse-proxy for automatic Let's Encrypt SSL.
6. **1-Click Cloud Blueprint**: Added `render.yaml` for automatic multi-service deployment on Render.
7. **Git Repository Initialized**: Cleaned directory nesting and made an initial root commit ready to push.

---

## 🚀 Choose Your Deployment Option

| Option | Speed | Cost | Best For |
|---|---|---|---|
| **Option 1: Vercel (Frontend) + Render (Backend)** | ~5 mins | Free tier | Best cloud performance, zero server management, global CDN |
| **Option 2: 1-Click Render Blueprint (`render.yaml`)** | ~4 mins | Free tier | Simplest setup: deploys both frontend & backend together |
| **Option 3: Single VPS / Cloud VM (Docker Compose)** | ~3 mins | \$4–\$6/mo VPS | Full control, single server, unlimited usage, private data |
| **Option 4: Cloudflare Tunnel (From Current Machine)** | ~1 min | 100% Free | Instant public URL directly from your local computer |

---

## 🌐 Option 1: Vercel (Frontend) + Render (Backend) [Recommended]

This is the industry standard for Next.js + FastAPI architectures.

### Step 1: Push Repository to GitHub
Run these commands in your project root:
```bash
# 1. Create a new empty repository on github.com (e.g. "insightflow-ai")
# 2. Link and push:
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy Backend to Render (Free)
1. Go to [render.com](https://render.com) and click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the settings:
   - **Name**: `insightflow-backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables**, add:
   - `PYTHONUNBUFFERED`: `1`
   - `PROJECT_CACHE_MAX_SIZE`: `50`
   - *(Optional)* `OPENAI_API_KEY`: *(your OpenAI key if using LLM features)*
5. Click **Create Web Service**.
6. Once deployed, copy your backend URL (e.g., `https://insightflow-backend.onrender.com`).

### Step 3: Deploy Frontend to Vercel (Free)
1. Go to [vercel.com](https://vercel.com) and click **Add New...** → **Project**.
2. Select your GitHub repository.
3. In project settings:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click edit and select `frontend`
4. Expand **Environment Variables** and add:
   - `NEXT_PUBLIC_API_URL`: `https://insightflow-backend.onrender.com` (paste your Render backend URL, without trailing slash)
5. Click **Deploy**.
6. Your site is live on your Vercel URL (e.g., `https://insightflow.vercel.app`)!

---

## ⚡ Option 2: 1-Click Render Blueprint (`render.yaml`)

We included a `render.yaml` file in your repository.

1. Push your repository to GitHub (see Step 1 above).
2. Go to [dashboard.render.com/blueprints](https://dashboard.render.com/blueprints).
3. Click **New Blueprint Instance**.
4. Select your repository. Render will automatically read `render.yaml` and configure:
   - The FastAPI backend service
   - The Next.js frontend service
   - The automatic environment variable link connecting frontend to backend
5. Click **Apply**. Both services will build and deploy automatically.

---

## 🐳 Option 3: Single VPS / Cloud Server (Docker Compose)

Deploy to any Linux server (DigitalOcean Droplet, AWS EC2, Hetzner, Linode, Ubuntu VM).

### Quick Start (Local or Port Access)
1. SSH into your VPS:
   ```bash
   git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git
   cd <YOUR-REPO-NAME>
   ```
2. Start the stack:
   ```bash
   docker compose up -d --build
   ```
   - **Frontend**: `http://<YOUR_SERVER_IP>:3000`
   - **Backend**: `http://<YOUR_SERVER_IP>:8000`

### Production with Domain & Automatic HTTPS (Caddy)
We provided `docker-compose.prod.yml` and `Caddyfile`.
1. Point your domain (e.g., `insightflow.example.com`) to your server IP via an `A` record.
2. Run:
   ```bash
   DOMAIN=insightflow.example.com docker compose -f docker-compose.prod.yml up -d --build
   ```
   - Caddy automatically obtains a free Let's Encrypt TLS certificate.
   - Both frontend and backend API (`/api/*`) are served securely over `https://insightflow.example.com`.

---

## ⚡ Option 4: Instant Public Access via Cloudflare Tunnel (1 Minute)

If you want to immediately showcase this running application to users on the internet right now without cloud hosting:

1. Start the unified server locally:
   ```bash
   npm start
   ```
2. Download and run Cloudflare `cloudflared` (or install via `winget install Cloudflare.cloudflared`):
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
3. Cloudflare prints a secure public HTTPS URL (e.g., `https://random-words.trycloudflare.com`) accessible worldwide immediately.

---

## 📋 Environment Variables Reference

| Variable | Service | Default | Description |
|---|---|---|---|
| `PORT` | Backend | `8000` | Port for FastAPI backend |
| `FRONTEND_URL` | Backend | `http://127.0.0.1:3000` | Address of frontend for backend reverse proxy |
| `PROJECT_CACHE_MAX_SIZE` | Backend | `50` | Maximum LRU dataset cache capacity |
| `NEXT_PUBLIC_API_URL` | Frontend | `http://localhost:8000` | Public backend URL accessed by client browser |
| `BACKEND_API_URL` | Frontend | `http://127.0.0.1:8000` | Internal backend URL for Next.js server rewrites |
| `OPENAI_API_KEY` | Backend | *(Optional)* | Enables LLM natural language responses |
