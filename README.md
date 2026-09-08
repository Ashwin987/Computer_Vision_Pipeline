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

## Running on Streamlit Community Cloud (active deployment target)

Deployed directly from this GitHub repo via [share.streamlit.io](https://share.streamlit.io)
— no Dockerfile involved for this path. **`dashboard/requirements.txt` (next to the
entrypoint, not the repo root) is the one Community Cloud actually installs from** —
confirmed empirically, not from docs: with a requirements.txt at *both* the root and
next to the entrypoint, Community Cloud's build silently preferred the one next to the
entrypoint (its own build log: `"More than one requirements file detected... Used: uv
with .../dashboard/requirements.txt"`), which briefly broke the deploy when a root-only
merged file existed and the stale `dashboard/`-local one (missing `pypdf` and the whole
CV-pipeline stack) got used instead. Fixed by keeping exactly one requirements.txt,
at `dashboard/requirements.txt`, covering both halves (since `run_cv_analysis.py` runs
as a subprocess in the same environment as the dashboard). `.streamlit/config.toml`
*does* need to stay at the repo root when the entrypoint is in a subdirectory (per
Streamlit's docs — confirmed correct, unlike the requirements.txt assumption above) —
kept as a duplicate of `dashboard/.streamlit/config.toml`, which stays too for local
`streamlit run` from inside `dashboard/`. When deploying, set **Main file path** to
`dashboard/app.py`, and paste the Gemini key into the app's **Advanced settings →
Secrets** as `master_key = "..."` — Community Cloud's native `st.secrets` support means
`app.py`'s existing `master_key = st.secrets["master_key"]` needs no adapter here.

**No `packages.txt`, deliberately**: an earlier version of this repo had one
(`libgl1`/`libglib2.0-0`/`ffmpeg`), but it turned out unnecessary and was actively
harmful — its mere presence triggers Community Cloud's apt-get step, which hit a
platform-wide bug (Debian's `bullseye-security` release file expired on Streamlit's own
base image, breaking `apt-get update` entirely, confirmed as a currently-active,
widely-reported issue, not anything specific to this repo). Checked what those packages
were actually for: `opencv-python-headless` is specifically built to not need
`libgl1`/GTK/X11 (that's the point of the "headless" variant), and `moviepy` depends on
`imageio-ffmpeg`, which bundles its own portable ffmpeg binary via pip — no system
`ffmpeg` needed. Removing `packages.txt` entirely sidesteps the broken apt source with
no functional loss.

**Live video upload is enabled but not the reliable demo path on this platform's free
tier specifically**: Community Cloud guarantees only ~1GB RAM per app (bursting to
~3GB), which the CV pipeline's PyTorch/YOLO/OpenCV stack can exceed on its own before
even accounting for the multi-hour CPU runtime this project has already measured on
demanding footage. The in-app notice on the upload flow says so plainly. The two
curated **Instant Demo** matches (pre-computed, no live compute needed) are the
reliable, immediate walkthrough.

### Alternate path: Hugging Face Spaces (Docker SDK) — present but not active

`Dockerfile` + `entrypoint.sh` still ship in this repo for a Docker-based Space, kept
rather than deleted since they're harmless and may be useful if a paid HF plan is used
later (Docker Spaces require one; that's why this repo moved to Community Cloud
instead). `entrypoint.sh` adapts HF's env-var secret delivery into
`dashboard/.streamlit/secrets.toml` at container start — that adapter is specific to
this path and isn't used by the Community Cloud deployment above.

## Scope note

This repo intentionally excludes bulk raw/dev video footage that isn't needed to run
either half (raw match clips, development/pilot/holdout/review clip stashes, and
CV-pipeline test runs not tied to a curated match) — it includes the trained model
weights, all source code, and the specific CV output data the dashboard's curated demo
matches depend on, so it's a runnable pipeline + working dashboard rather than a full
video archive of everything ever produced during development.

## Requirements

`dashboard/requirements.txt` covers both halves (the dashboard's Streamlit/AI stack and
the CV pipeline's tracking/detection stack, traced from `run_cv_analysis.py`'s real
import graph) — this is what both deployment paths install, and what a local install
should use. A Gemini API key is required
for the dashboard's AI features — set it locally via `dashboard/.env`
(`GEMINI_API_KEY=...`) or `dashboard/.streamlit/secrets.toml` (`master_key = "..."`),
neither of which is committed to this repo.
