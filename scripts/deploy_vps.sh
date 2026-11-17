#!/usr/bin/env bash
# InsightFlow AI - Automated Production VPS Deployment Script
set -e

echo "=================================================="
echo " InsightFlow AI Production Deployment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[!] Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Check domain configuration
DOMAIN="${1:-localhost}"
echo "[+] Deploying with DOMAIN=${DOMAIN}"

export DOMAIN="${DOMAIN}"
if [ "${DOMAIN}" = "localhost" ]; then
    echo "[+] Starting standard Docker Compose on ports 3000 and 8000..."
    docker compose up -d --build
    echo ""
    echo "[✓] Deployment complete!"
    echo "    Frontend: http://localhost:3000"
    echo "    Backend API Docs: http://localhost:8000/docs"
else
    echo "[+] Starting Production Docker Compose with Caddy HTTPS..."
    docker compose -f docker-compose.prod.yml up -d --build
    echo ""
    echo "[✓] Deployment complete with Automatic SSL!"
    echo "    Web App: https://${DOMAIN}"
    echo "    Backend API Docs: https://${DOMAIN}/docs"
fi
