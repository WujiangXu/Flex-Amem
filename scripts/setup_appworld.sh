#!/usr/bin/env bash
# AppWorld in its own venv (it pins pydantic<2): installs appworld 0.1.3.post1 with the pinned bridge packages and
# downloads its data into artifacts/appworld/root. Each episode runs src/flexamem/bench/appworld/bridge.py in this venv.
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv artifacts/appworld/venv
artifacts/appworld/venv/bin/pip install -r requirements_appworld.lock
mkdir -p artifacts/appworld/root
cd artifacts/appworld/root
APPWORLD_ROOT=$PWD ../venv/bin/appworld install
APPWORLD_ROOT=$PWD ../venv/bin/appworld download data
