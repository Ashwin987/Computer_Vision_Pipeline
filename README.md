---
title: AI Tactical Coach
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Football Tactics — CV Pipeline + Dashboard

See [LICENSE](LICENSE) for usage terms — all rights reserved.

This repository combines two previously separate codebases into one:

- **`cv_pipeline/`** — the computer-vision pipeline (`run_cv_analysis.py`, pitch
  calibration, player/ball tracking, tactical-event detection, and the trained model
  weights it depends on). See `cv_pipeline/README.md` for details on that pipeline
  specifically, and `Real-Time_Soccer_Analytics_Pipeline_v3.pdf` at its root for the
  full technical writeup.
- **`dashboard/`** — the Streamlit app (`app.py`) that drives the CV pipeline as a
  subprocess, renders the resulting tactical analysis, generates an AI coach report
  and training plan, and includes a RAG chatbot (`chatbot.py`) grounded in a match's
  real data.

`dashboard/app.py` locates the CV pipeline relative to its own file location
(`Path(__file__).resolve().parent.parent / "cv_pipeline"`), so the two folders must
stay siblings under the same repo root for the dashboard to find and launch the
pipeline correctly.

## Running as a Hugging Face Space

Streamlit is no longer a native Spaces SDK, so this repo ships a `Dockerfile` +
`entrypoint.sh` that installs `requirements.txt` and runs `streamlit run
dashboard/app.py` directly. Set one Space **Secret** named `GEMINI_API_KEY` via the
Space's Settings UI — `entrypoint.sh` materializes it into
`dashboard/.streamlit/secrets.toml` at container start (Streamlit's own secrets
mechanism, which `app.py` already reads via `st.secrets`), so no code change was needed
for the deployed environment specifically.

Live video upload works on this deployment but is not the guaranteed demo path: this
pipeline's own measured runtimes mean a full analysis on free CPU hardware can take
several hours. The two curated **Instant Demo** matches (pre-computed, no live compute
needed) are the reliable, immediate walkthrough.

## Scope note

This repo intentionally excludes bulk raw/dev video footage that isn't needed to run
either half (raw match clips, development/pilot/holdout/review clip stashes, and
CV-pipeline test runs not tied to a curated match) — it includes the trained model
weights, all source code, and the specific CV output data the dashboard's curated demo
matches depend on, so it's a runnable pipeline + working dashboard rather than a full
video archive of everything ever produced during development.

## Requirements

The root `requirements.txt` covers both halves (the dashboard's Streamlit/AI stack and
the CV pipeline's tracking/detection stack, traced from `run_cv_analysis.py`'s real
import graph) — this is what the Dockerfile installs, and what a local install should
use too, rather than `dashboard/requirements.txt` alone. A Gemini API key is required
for the dashboard's AI features — set it locally via `dashboard/.env`
(`GEMINI_API_KEY=...`) or `dashboard/.streamlit/secrets.toml` (`master_key = "..."`),
neither of which is committed to this repo.
