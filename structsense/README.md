# StructSense AI — Structural Failure Detection System

AI-powered structural safety analysis using PPO Reinforcement Learning, IS 456/1893 engineering checks, and Groq GPT-OSS 120B reasoning.

---

## Getting Started

### Prerequisites

- Python 3.14+ (older versions cause a NumPy compatibility error)
- Optional: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), only needed for scanned (image-only) PDFs
- Optional: a free [Groq API key](https://console.groq.com/keys) for the AI Verdict (without one, a rule-based IS-code engine is used)

### Install

From inside the `structsense/` directory:

**macOS / Linux**
```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then paste your GROQ_API_KEY into .env
```

**Windows**
```powershell
py -3.14 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Tesseract: `brew install tesseract` (macOS), `conda install -c conda-forge tesseract`, or the
[Windows installer](https://github.com/UB-Mannheim/tesseract/wiki). If it isn't on your PATH, set
`TESSERACT_CMD` in `.env` to the binary's full path.

### Run the App

**One-click:** double-click `run.bat` (Windows) or `run.command` (macOS). On first run it creates
the virtual environment, installs dependencies and creates `.env`, then starts the app and opens your browser.

**Manually:**
```bash
.venv/bin/python -m streamlit run app.py        # macOS / Linux
.venv\Scripts\python -m streamlit run app.py    # Windows
```

The app will open at **http://localhost:8501**

### Check the setup

```bash
.venv/bin/python _check.py
```

Validates the rule-based fallback offline and, if `GROQ_API_KEY` is set, makes one live Groq call.

---

## Usage

1. Choose **Upload Documents** to extract parameters from PDF structural reports, or switch to **Manual Input** to adjust sliders directly.
2. Enter your **Groq API Key** in the sidebar, or put it in `.env` (free at [console.groq.com](https://console.groq.com/keys)).
3. Click **RUN SIMULATION** to train the PPO agent and get an AI safety verdict.
4. Explore results across the **Results**, **Trajectory**, **Risk Zones**, and **AI Verdict** tabs.
5. Export a PDF report, CSV data, or research abstract from the **Export** tab.

---

## Tech Stack

- **Streamlit** — Dashboard UI
- **Stable-Baselines3 + PyTorch** — PPO Reinforcement Learning
- **Groq GPT-OSS 120B** — AI structural reasoning
- **IS 456 / IS 1893** — Indian Standard code checks
- **Plotly** — Interactive visualizations
