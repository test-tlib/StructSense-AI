@echo off
rem StructSense AI launcher (Windows) - double-click to start the app.
rem Streamlit serves both the backend (simulation, AI) and the frontend (web UI).
setlocal
cd /d "%~dp0"
title StructSense AI

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment with Python 3.14...
    py -3.14 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo [error] Could not create a virtual environment. Install Python 3.14 from python.org.
        pause
        exit /b 1
    )
    echo [setup] Installing dependencies - first run only, this can take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [error] Dependency install failed. Delete the .venv folder and try again.
        pause
        exit /b 1
    )
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [setup] Created .env - add your GROQ_API_KEY to it for the AI Verdict.
)

echo.
echo  StructSense AI is starting at http://localhost:8501
echo  Close this window or press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501

pause
