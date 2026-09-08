#!/bin/sh
# Adapts HF Spaces' secret-delivery mechanism (a plain environment variable
# on a Docker SDK space) to app.py's existing, already-verified
# st.secrets["master_key"] read path (Streamlit's own secrets.toml
# mechanism) - without touching that already-working code. Set the Space
# Secret named GEMINI_API_KEY via the Space's Settings UI; never committed.
set -e
mkdir -p dashboard/.streamlit
printf 'master_key = "%s"\n' "$GEMINI_API_KEY" > dashboard/.streamlit/secrets.toml
exec streamlit run dashboard/app.py --server.port=7860 --server.address=0.0.0.0
