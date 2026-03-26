#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U ".[browser]"
python -m playwright install chromium

echo "[playwright] installed package + chromium browser"
