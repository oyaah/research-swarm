#!/usr/bin/env bash
set -euo pipefail

BUILD=0
WITH_UI=0
for arg in "$@"; do
  if [[ "$arg" == "--build" ]]; then BUILD=1; fi
  if [[ "$arg" == "--with-ui" ]]; then WITH_UI=1; fi
done

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
  else
    echo "[bootstrap] .env.example missing; create .env manually."
  fi
fi

if [[ $BUILD -eq 1 ]]; then
  echo "[bootstrap] Building and starting stack"
  if [[ $WITH_UI -eq 1 ]]; then
    docker compose --profile ui up --build -d
  else
    docker compose up --build -d
  fi
else
  echo "[bootstrap] Starting stack with hot-reload (no rebuild)"
  if [[ $WITH_UI -eq 1 ]]; then
    docker compose --profile ui up -d
  else
    docker compose up -d
  fi
fi

echo "[bootstrap] Waiting for services to be ready..."
RETRIES=30
for i in $(seq 1 $RETRIES); do
  if curl -sf http://localhost:8000/healthz >/dev/null 2>&1; then
    echo "[bootstrap] API ready!"
    break
  fi
  if [[ $i -eq $RETRIES ]]; then
    echo "[bootstrap] WARNING: API not responding after ${RETRIES} attempts. Check: docker compose logs api"
  fi
  sleep 2
done

echo "[bootstrap] API: http://localhost:8000"
if [[ $WITH_UI -eq 1 ]]; then
  echo "[bootstrap] Frontend: http://localhost:8501"
else
  echo "[bootstrap] Frontend disabled by default. Add --with-ui to start it."
fi
echo "[bootstrap] Code changes will auto-reload in running containers."
