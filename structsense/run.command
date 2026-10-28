#!/bin/bash
# StructSense AI launcher (macOS) - double-click in Finder to start the app.
# Streamlit serves both the backend (simulation, AI) and the frontend (web UI).
cd "$(dirname "$0")" || exit 1

if [ ! -x ".venv/bin/python" ]; then
    echo "[setup] Creating virtual environment with Python 3.14..."
    PY=$(command -v python3.14 || command -v python3)
    if ! "$PY" -m venv .venv; then
        echo "[error] Could not create a virtual environment. Install Python 3.14 from python.org."
        read -r -p "Press Enter to close..."; exit 1
    fi
    echo "[setup] Installing dependencies - first run only, this can take a few minutes..."
    .venv/bin/python -m pip install --upgrade pip
    if ! .venv/bin/python -m pip install -r requirements.txt; then
        echo "[error] Dependency install failed. Delete the .venv folder and try again."
        read -r -p "Press Enter to close..."; exit 1
    fi
fi

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[setup] Created .env - add your GROQ_API_KEY to it for the AI Verdict."
fi

# Stop an already-running StructSense server so the port is free
lsof -ti tcp:8501 | xargs kill 2>/dev/null

echo
echo " StructSense AI is starting at http://localhost:8501"
echo " Close this window or press Ctrl+C to stop."
echo
.venv/bin/python -m streamlit run app.py --server.port 8501
