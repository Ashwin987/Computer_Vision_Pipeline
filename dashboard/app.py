import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import time
import tempfile
import cv2
import math
from google import genai
from google.genai import types
from streamlit_autorefresh import st_autorefresh
import training_plan as tp
import chatbot as cb
import corner_kicks as ck
import numpy as np
import random
import io
from moviepy import VideoFileClip
import concurrent.futures
import uuid
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak

import streamlit as st

# Securely pull the key from the hidden local vault (or Cloud vault later)
master_key = st.secrets["master_key"]

API_KEYS = [master_key] if master_key else []

st.set_page_config(page_title="Elite Tactical Scout AI", layout="wide")

# ==========================================
# THEME — one token set shared by every screen (mirrors dashboard_mockup_4.html's
# :root custom properties). config.toml's [theme] block sets Streamlit's own
# native dark palette (buttons/tabs/progress bar/sliders/checkboxes all pick it
# up for free); this stylesheet layers on the mockup-specific pieces Streamlit
# has no native equivalent for - panel cards, the pill tab bar, the dashed
# upload dropzone, and the bespoke landing/loading page markup below.
# ==========================================
THEME_CSS = """
<style>
:root {
  --bg: #0b0e14;
  --panel: #12161f;
  --panel2: #171c27;
  --border: #232a38;
  --accent: #4f8cff;
  --accent2: #22c55e;
  --danger: #ef4444;
  --warn: #f59e0b;
  --text: #e6e9f0;
  --muted: #8a93a6;
}

/* ---- App shell ---- */
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }
hr { border-color: var(--border) !important; }

/* ---- Metrics as panel cards ---- */
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; text-transform: uppercase; font-size: 11px; letter-spacing: .5px; }
/* The actual truncation happens on the inner <p> Streamlit renders inside the
   label's markdown container, which ships with white-space:nowrap +
   text-overflow:ellipsis baked in (confirmed via computed-style inspection -
   the outer label/stMetricLabel rule above does NOT reach this element). */
[data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  overflow-wrap: break-word;
}
/* Long metric values (e.g. "Middle Third") used to truncate with an ellipsis at
   narrower viewports - wrap instead of clipping, and shrink slightly on long
   values via clamp() rather than a fixed size, so short values (e.g. "-17.0")
   stay large while long ones still fit on one or two lines without cutting off. */
[data-testid="stMetricValue"] {
  font-weight: 700;
  white-space: normal !important;
  overflow-wrap: break-word;
  line-height: 1.15;
  font-size: clamp(14px, 2vw, 28px) !important;
}

/* ---- Bordered containers (st.container(border=True)) as panel cards ---- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) {
  background: var(--panel);
  border: 1px solid var(--border) !important;
  border-radius: 12px;
}

/* ---- Buttons ---- */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
  border-radius: 8px !important;
  border: 1px solid var(--border) !important;
}
.stButton>button[kind="primary"], .stButton>button[kind="primaryFormSubmit"] {
  background: linear-gradient(90deg, #059669, #22c55e) !important;
  color: #06281a !important;
  border: none !important;
  font-weight: 700 !important;
}

/* ---- Tabs as pill bar (mirrors mockup .tabs / .tab) ----
   !important throughout: Streamlit's baseweb/emotion tab styles are injected
   as auto-generated, highly-specific classes that otherwise beat plain
   attribute selectors - without it this only wins on the outermost st.tabs()
   and nested st.tabs() (e.g. the Tactical Visualizations sub-tabs) fall back
   to unstyled text. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 6px !important;
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 5px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  border-radius: 7px !important;
  font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
}

/* ---- File uploader dropzone ---- */
[data-testid="stFileUploaderDropzone"] {
  background: var(--panel2) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 14px !important;
}

/* ---- Progress bar (mirrors mockup's .loading-progress-fill green gradient) ---- */
div[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, #059669, #22c55e) !important;
}
div[data-testid="stProgress"] > div > div {
  background: var(--panel2) !important;
}

/* ---- Alerts (st.info/success/warning/error) ---- */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* ============ Bespoke landing / loading page markup ============ */
.tac-landing-nav { display: flex; align-items: center; justify-content: space-between; padding: 4px 4px 22px 4px; }
.tac-landing-nav .brand { font-size: 15px; font-weight: 800; letter-spacing: 0.3px; color: var(--text); }
.tac-landing-nav .brand span { color: var(--accent2); }
.tac-landing-nav .nav-links { display: flex; gap: 28px; font-size: 12.5px; color: var(--muted); }

.tac-hero { text-align: center; padding: 12px 12px 30px 12px; }
.tac-eyebrow {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 10.5px; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase;
  color: var(--accent2);
  background: rgba(34,197,94,0.08);
  border: 1px solid rgba(34,197,94,0.25);
  padding: 6px 14px; border-radius: 20px;
  margin-bottom: 22px;
}
.tac-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent2); box-shadow: 0 0 8px var(--accent2); display:inline-block; }
.tac-hero h1 {
  font-size: 44px; line-height: 1.12; font-weight: 800; letter-spacing: -1px;
  max-width: 780px; margin: 0 auto 18px auto; color: var(--text);
}
.tac-hero h1 .hl { color: var(--accent2); }
.tac-hero p.lede {
  font-size: 16px; color: var(--muted); max-width: 560px; line-height: 1.6;
  margin: 0 auto 8px auto;
}

.tac-proof-row { display: flex; justify-content: center; gap: 56px; padding: 30px 24px 10px 24px; flex-wrap: wrap; }
.tac-proof-item { text-align: center; }
.tac-proof-item .n { font-size: 24px; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; }
.tac-proof-item .l { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 2px; }

/* "Built under constraint" rotating stat cards - pure CSS animation (no JS:
   st.markdown's unsafe_allow_html strips <script> tags), one shared
   @keyframes cycle with staggered animation-delay per card so exactly one
   of the 5 cards is visible at a time, tiling a 20s loop with no gaps. */
.tac-constraint-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; text-align: center; margin: 34px 0 10px 0; }
.tac-constraint-wrap { max-width: 640px; margin: 0 auto 8px auto; position: relative; height: 84px; }
.tac-constraint-card {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 24px; text-align: center; opacity: 0;
  animation: tac-rotate 20s infinite;
}
.tac-constraint-card .n { font-size: 18px; font-weight: 800; color: var(--accent2); }
.tac-constraint-card .l { font-size: 12.5px; color: var(--muted); margin-top: 5px; line-height: 1.4; max-width: 520px; }
.tac-constraint-card:nth-child(1) { animation-delay: 0s; }
.tac-constraint-card:nth-child(2) { animation-delay: 4s; }
.tac-constraint-card:nth-child(3) { animation-delay: 8s; }
.tac-constraint-card:nth-child(4) { animation-delay: 12s; }
.tac-constraint-card:nth-child(5) { animation-delay: 16s; }
@keyframes tac-rotate {
  0% { opacity: 0; }
  2% { opacity: 1; }
  16% { opacity: 1; }
  20% { opacity: 0; }
  100% { opacity: 0; }
}

.tac-upload-divider { display: flex; align-items: center; gap: 12px; margin: 18px 0; color: var(--muted); font-size: 11px; }
.tac-upload-divider::before, .tac-upload-divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* Loading page */
.tac-loading-header { text-align: center; margin-bottom: 10px; }
.tac-loading-header h2 { font-size: 22px; margin-bottom: 6px; }
.tac-loading-header .status-line { font-size: 13px; color: var(--accent2); font-weight: 600; }

.tac-eta-row { display: flex; justify-content: center; gap: 12px; margin: 18px 0 18px 0; flex-wrap: wrap; }
.tac-eta-box { background: var(--panel2); border: 1px solid var(--border); border-radius: 12px; padding: 14px 26px; min-width: 120px; text-align:center; }
.tac-eta-val { font-size: 22px; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; }
.tac-eta-val.pct { color: var(--accent2); }
.tac-eta-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-top: 3px; }

.tac-stages { display: flex; justify-content: space-between; margin: 20px 0 6px 0; font-size: 10px; color: var(--muted); }
.tac-stages .stg { display: flex; flex-direction: column; align-items: center; gap: 6px; flex: 1; }
.tac-stages .stg .pip { width: 8px; height: 8px; border-radius: 50%; background: var(--border); }
.tac-stages .stg.done .pip { background: var(--accent2); }
.tac-stages .stg.active .pip { background: var(--accent2); box-shadow: 0 0 0 4px rgba(34,197,94,0.18); }
.tac-stages .stg.done, .tac-stages .stg.active { color: #d7dce8; }
.tac-stages .stg-sub { font-size: 9.5px; color: var(--muted); font-weight: 600; }
.tac-stages .stg.done .stg-sub { color: var(--accent2); }
.tac-stages .stg.active .stg-sub { color: #d7dce8; }

.tac-log-panel { background: #080a0e; border: 1px solid var(--border); border-radius: 12px; margin: 18px 0; overflow: hidden; }
.tac-log-panel-head { display: flex; align-items: center; gap: 7px; font-size: 10.5px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--muted); padding: 10px 16px; border-bottom: 1px solid var(--border); }
.tac-log-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent2); box-shadow: 0 0 6px var(--accent2); }
.tac-log-lines { font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace; font-size: 11.5px; padding: 12px 16px; display: flex; flex-direction: column; gap: 7px; max-height: 150px; overflow: hidden; }
.tac-log-line { color: #5b6b7a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tac-log-line.recent { color: #9aa5b8; }
.tac-log-line.current { color: var(--accent2); }

.tac-quote { font-size: 13px; font-style: italic; color: var(--muted); text-align: center; padding: 14px 18px; border-left: 2px solid var(--accent2); background: var(--panel); border-radius: 0 10px 10px 0; margin: 8px 0 18px 0; }
.tac-quote span { color: #d7dce8; font-style: normal; font-weight: 600; }

.tac-footnote { font-size: 11.5px; color: var(--muted); text-align: center; line-height: 1.6; margin-top: 4px; }

.tac-cv-tag { font-size: 10px; background: linear-gradient(90deg, #059669, #22c55e); padding: 2px 7px; border-radius: 10px; color: white; font-weight: 600; }
.tac-ai-tag { font-size: 10px; background: linear-gradient(90deg, #7c3aed, #4f8cff); padding: 2px 7px; border-radius: 10px; color: white; font-weight: 600; }

/* ---- Sidebar nav restyled as a pill bar (mirrors the main tab-list look) ---- */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
  gap: 4px !important;
  background: var(--panel2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  flex-direction: column !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  border-radius: 7px !important;
  padding: 7px 10px !important;
  margin: 0 !important;
  font-weight: 600 !important;
  font-size: 13px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: var(--accent) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
  color: #fff !important;
}
.tac-sidebar-actions {
  background: var(--panel2); border: 1px solid var(--border); border-radius: 10px;
  padding: 10px; margin-bottom: 12px;
}
.tac-nav-heading {
  font-size: 10.5px; font-weight: 700; letter-spacing: .8px; text-transform: uppercase;
  color: var(--muted); margin: 4px 0 6px 2px;
}

/* ---- Reusable "?" hover tooltip for non-st.metric custom cards (see
   html_tooltip() in app.py) - matches the native st.metric help-icon look. ---- */
.tac-tt {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; border-radius: 50%;
  border: 1px solid var(--muted); color: var(--muted);
  font-size: 10px; font-weight: 700; cursor: help; margin-left: 5px;
  vertical-align: middle;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

def _fmt_mmss(total_seconds):
    """mm:ss, matching the mockup's fmtMMSS()."""
    total_seconds = max(0, int(total_seconds))
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}"

MANAGER_QUOTES = [
    ("I am not a perfectionist, but I like to feel that things are done well.", "Cristiano Ronaldo"),
    ("Playing football is very simple, but playing simple football is the hardest thing there is.", "Johan Cruyff"),
    ("The objective is to move the opponent, not the ball.", "Pep Guardiola"),
    ("Without the ball, you can't win. With the ball, you have a chance.", "Xavi Hernandez"),
    ("If you control the midfield, you control the game.", "Sir Alex Ferguson"),
]

def _chunk_log_line(minute_idx, data):
    """One live-log line built only from fields the chunk's real Gemini
    response actually contains - possession + zone, plus any counter-attack/
    gegenpress flags if that minute triggered one."""
    poss = data.get("team_in_possession", "Unknown")
    zone = str(data.get("ball_zone", "unknown")).replace("_", " ").title()
    flags = []
    for side, label in (("a", str(data.get("team_a_color", "Team A"))), ("b", str(data.get("team_b_color", "Team B")))):
        trans = str(data.get(f"team_{side}_transition_threat", "")).lower().strip()
        if trans in ("counter_attack", "fast_vertical_transition"):
            flags.append(f"{trans.replace('_', ' ')} ({label})")
        if str(data.get(f"team_{side}_pressing_trigger", "")).lower().strip() == "gegenpress":
            flags.append(f"gegenpress ({label})")
    suffix = f" — {', '.join(flags)}" if flags else ""
    return f"Minute {minute_idx} chunk analyzed — {poss} in possession, {zone}{suffix}"

def render_loading_screen(placeholder, *, pct, elapsed_s, eta_s, status_line, stages, log_lines, quote):
    """Renders the mockup's #page-loading design into one st.empty() placeholder.
    stages: list of (label, sub_text, state) where state in {'pending','active','done'}.
    log_lines: list of (text, state) where state in {'old','recent','current'}, oldest first."""
    stages_html = "".join(
        f'<div class="stg {state}"><div class="pip"></div>{label}<span class="stg-sub">{sub}</span></div>'
        for label, sub, state in stages
    )
    log_html = "".join(
        f'<div class="tac-log-line {state}">{text}</div>' for text, state in log_lines
    ) or '<div class="tac-log-line current">Starting...</div>'
    quote_text, quote_author = quote

    html = f"""
    <div class="tac-loading-header">
      <h2>⚽ Analyzing your match</h2>
      <div class="status-line">🔄 {status_line}</div>
    </div>
    <div class="tac-eta-row">
      <div class="tac-eta-box"><div class="tac-eta-val pct">{pct}%</div><div class="tac-eta-lbl">complete</div></div>
      <div class="tac-eta-box"><div class="tac-eta-val">{_fmt_mmss(elapsed_s)}</div><div class="tac-eta-lbl">elapsed</div></div>
      <div class="tac-eta-box"><div class="tac-eta-val">~{_fmt_mmss(eta_s)}</div><div class="tac-eta-lbl">estimated remaining</div></div>
    </div>
    """
    placeholder_container = placeholder.container()
    with placeholder_container:
        st.markdown(html, unsafe_allow_html=True)
        st.progress(min(1.0, max(0.0, pct / 100.0)))
        st.markdown(f'<div class="tac-stages">{stages_html}</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="tac-log-panel">
                <div class="tac-log-panel-head"><span class="tac-log-dot"></span> Live progress</div>
                <div class="tac-log-lines">{log_html}</div>
            </div>
            <div class="tac-quote">"{quote_text}" — <span>{quote_author}</span></div>
            <div class="tac-footnote">
                This runs at full accuracy — no data is skipped or approximated to save time.
                Once this finishes you'll land on your full tactical dashboard.
            </div>""",
            unsafe_allow_html=True,
        )

def metric_card(container, label, value, help_text):
    """Thin, named wrapper over st.metric's native help= tooltip (the small
    "?" icon already used by a few boxes in this app, e.g. zone_help/press_help)
    - exists so every metric box in the app goes through one call instead of
    ad hoc st.metric(...) with help sometimes forgotten."""
    container.metric(label=label, value=value, help=help_text)

# Shared verbatim everywhere momentum score appears (home screen, dashboard,
# CV tab) - Part 2.3 requires identical wording in every location, not
# slightly different explanations per view. Uses the same {TEAM_A}/{TEAM_B}
# token + cb.substitute_team_tokens pattern used everywhere else (coach
# report, chatbot answers) - resolve at render time, don't hardcode names
# here. The sign convention named below (positive=team_a, negative=team_b)
# is the same hardcoded mapping chatbot.py's MOMENTUM_FIELDS/
# _momentum_team_sentence uses for "which team had the advantage" chat
# answers - both trace back to compute_dashboard_df's real calculation
# (net_momentum = team_a_raw_threat - team_b_raw_threat), not derived twice.
MOMENTUM_SCORE_HELP = (
    "Momentum score measures which minute of the match had the most attacking intensity "
    "and pressure, combining possession, territory, and pressing events. This window (the "
    "single highest-scoring minute) is what gets the full computer-vision breakdown below "
    "— not the whole match. A positive score favors {TEAM_A}; a negative score favors "
    "{TEAM_B}."
)

# Consistent category colors across BOTH teams' pies (Part 1.2) - keyed by the
# raw lowercase attacking_bias values Gemini returns, not the title-cased
# display labels, so it's a single source of truth for both pie charts.
ATTACKING_BIAS_COLORS = {
    'left_flank': '#66b3ff',
    'right_flank': '#99ff99',
    'central_channel': '#ff9999',
}

# Shared, theme-consistent heatmap colorscales (Part 1 of the polish pass) -
# Plotly's built-in 'Blues'/'Purples' fade to near-white at the low end,
# which looks out of place against this app's dark theme. Both scales share
# the same dark, near-panel low anchor (never white) so every heatmap in the
# app reads as one consistent system rather than each getting a one-off fix.
THEME_COLORSCALE_BLUE = [[0.0, '#1a2030'], [0.5, '#3a5f9e'], [1.0, '#4f8cff']]
THEME_COLORSCALE_PURPLE = [[0.0, '#1a2030'], [0.5, '#4f8cff'], [1.0, '#9b5fe0']]

def themed_plotly_layout(fig, height=380):
    """One shared dark-theme layout so every chart in the app matches --panel
    background / --text foreground instead of Plotly's white default, and so
    every figure gets the same height/margins regardless of its data shape
    (Part 1.2's "identical dimensions regardless of slice-count" requirement)."""
    fig.update_layout(
        height=height,
        paper_bgcolor="#12161f", plot_bgcolor="#12161f",
        font=dict(color="#e6e9f0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=30, r=20, t=30, b=30),
    )
    return fig

# ==========================================
# Shared dashboard-data + chart-building helpers (Part 5.2 / PDF export).
#
# Factored out of Step 3's dashboard rendering so the PDF generator computes
# and draws charts through the EXACT same code the live dashboard already
# uses - not a second, parallel implementation that could silently drift
# from what's on screen. Step 3 itself is updated to call these too.
# ==========================================

def compute_dashboard_df(raw_df, team_a, team_b, color_a, color_b):
    """Takes the raw per-minute dataframe (already validated to have
    'team_in_possession') and returns it enriched with every derived column
    the dashboard tab and its charts need - same formulas as always, just
    callable from more than one place now."""
    df = raw_df.copy()
    df.replace(["", " ", "Unknown", "N/A", None], np.nan, inplace=True)
    df = df.ffill().bfill()

    df['team_a_has_ball'] = (df['team_in_possession'].str.lower() == color_a.lower()).astype(int)
    df['team_b_has_ball'] = (df['team_in_possession'].str.lower() == color_b.lower()).astype(int)

    df['team_a_trans_threat'] = df.get('team_a_transition_threat', pd.Series(['none'] * len(df))).str.lower().str.strip()
    df['team_b_trans_threat'] = df.get('team_b_transition_threat', pd.Series(['none'] * len(df))).str.lower().str.strip()

    zone_map = {'attacking_third': 3, 'middle_third': 1.5, 'defensive_third': 0.5}
    df['zone_numeric'] = df.get('ball_zone', pd.Series(['middle_third'] * len(df))).map(zone_map).fillna(1.5)

    tempo_map = {'fast_direct': 2, 'sustained_high_pressure': 2, 'patient_possession': 1, 'none': 0, 'static': 0, 'dead_ball_stoppage': 0}
    df['team_a_tempo_num'] = df.get('team_a_attacking_tempo', pd.Series(['none'] * len(df))).map(tempo_map).fillna(1)
    df['team_b_tempo_num'] = df.get('team_b_attacking_tempo', pd.Series(['none'] * len(df))).map(tempo_map).fillna(1)

    team_a_hs = pd.to_numeric(df.get('team_a_half_space_occupancy', 0), errors='coerce').fillna(0)
    team_b_hs = pd.to_numeric(df.get('team_b_half_space_occupancy', 0), errors='coerce').fillna(0)

    ta_counter_boost = np.where(df['team_a_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)
    tb_counter_boost = np.where(df['team_b_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)

    df['team_a_raw_threat'] = df['team_a_has_ball'] * ((df['zone_numeric'] * 2.0) + (df['team_a_tempo_num'] * 1.5) + (team_a_hs * 1.0) + ta_counter_boost)
    df['team_b_raw_threat'] = df['team_b_has_ball'] * ((df['zone_numeric'] * 2.0) + (df['team_b_tempo_num'] * 1.5) + (team_b_hs * 1.0) + tb_counter_boost)

    df['net_momentum'] = df['team_a_raw_threat'] - df['team_b_raw_threat']
    df['smoothed_net_momentum'] = df['net_momentum'].rolling(window=2, min_periods=1).mean().round(2)
    return df

def compute_possession_heatmap_data(df, team_a, team_b, color_a, color_b):
    zones = ['defensive_third', 'middle_third', 'attacking_third']
    hm_a = [df[(df['team_in_possession'].str.lower() == color_a.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]
    hm_b = [df[(df['team_in_possession'].str.lower() == color_b.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]
    return pd.DataFrame([hm_a, hm_b], columns=zones, index=[team_a, team_b])

def compute_block_heatmap_data(df, team_a, team_b):
    block_map = {'low': 'defensive_third', 'mid': 'middle_third', 'high': 'attacking_third'}
    a_blocks = df['team_a_block_height'].str.lower().map(block_map).value_counts()
    b_blocks = df['team_b_block_height'].str.lower().map(block_map).value_counts()
    position_df = pd.DataFrame({team_a: a_blocks, team_b: b_blocks}).fillna(0).T
    for col in ['defensive_third', 'middle_third', 'attacking_third']:
        if col not in position_df.columns:
            position_df[col] = 0
    return position_df[['defensive_third', 'middle_third', 'attacking_third']]

def _minute_num_from_timestamp(ts):
    """'05:00-06:00' -> 5 (plain minute number, matching how a coach would
    refer to it). Falls back to the raw string if it isn't the expected
    per-minute timestamp shape."""
    try:
        return int(str(ts).split(':')[0])
    except (ValueError, IndexError):
        return ts

def compute_block_heatmap_minutes(df, team_a, team_b):
    """Same (team, zone) grid shape as compute_block_heatmap_data, but each
    cell holds the real minute numbers that produced that count - not
    estimated or evenly redistributed, since the per-minute timestamp is
    still on `df` at this point (confirmed before building this - Part 2)."""
    block_map = {'low': 'defensive_third', 'mid': 'middle_third', 'high': 'attacking_third'}
    zones = ['defensive_third', 'middle_third', 'attacking_third']
    grid = []
    for col in ('team_a_block_height', 'team_b_block_height'):
        mapped_zone = df[col].str.lower().map(block_map)
        row = []
        for z in zones:
            mins = sorted(_minute_num_from_timestamp(t) for t in df.loc[mapped_zone == z, 'timestamp'])
            row.append(", ".join(str(m) for m in mins) if mins else "none")
        grid.append(row)
    return grid

def build_bias_pie_mpl(bias_series_raw, valid_channels=('left_flank', 'right_flank', 'central_channel')):
    """Returns a matplotlib Figure for one team's attacking-bias pie, or None
    if there's no data - same drawing code the dashboard's download button
    already used (Plotly's PNG export needs kaleido, which hangs on this
    machine - see Part 1.2's comment - so matplotlib remains the one real
    static-image path, reused here rather than duplicated for the PDF)."""
    bias = bias_series_raw.dropna()
    bias = bias[bias.str.lower().str.strip().isin(valid_channels)].value_counts()
    if bias.empty:
        return None
    clean_idx = [str(idx).lower().strip().replace(' ', '_') for idx in bias.index]
    display_labels = [c.replace('_', ' ').title() for c in clean_idx]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(bias, labels=display_labels,
           colors=[ATTACKING_BIAS_COLORS.get(c, '#cccccc') for c in clean_idx],
           autopct=lambda p: f'{p:.1f}%\n({int(round(p * sum(bias) / 100.0)):d} times)')
    return fig

def build_heatmap_mpl(data_df, cmap):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(data_df, annot=True, cmap=cmap, fmt='g', linewidths=.5, ax=ax)
    ax.set_ylabel('')
    ax.set_xticklabels([label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()])
    ax.set_xlabel('Pitch Zone')
    return fig

def build_momentum_mpl(df, team_a, team_b):
    y_vals = df['smoothed_net_momentum']
    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = np.arange(len(df))
    ax.fill_between(x_vals, y_vals, 0, where=(y_vals >= 0), color='#d9383a', alpha=0.8, label=f'{team_a}', interpolate=True)
    ax.fill_between(x_vals, y_vals, 0, where=(y_vals <= 0), color='#333333', alpha=0.8, label=f'{team_b}', interpolate=True)
    max_momentum = max(abs(y_vals.max()), abs(y_vals.min()))
    y_lim = max(max_momentum, 1.0)
    ax.set_ylim(-y_lim - 1, y_lim + 1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ticks = np.arange(0, len(df), 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(df['timestamp'].iloc[ticks], rotation=45)
    ax.set_ylabel('Absolute Attacking Threat')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# ==========================================
# Part 5.2: full PDF export.
#
# Every section below reads through the exact same helpers the live UI
# already uses (get_cv_job_status_safe, generate_cv_observations,
# tp.load_training_plan, compute_dashboard_df + the build_*_mpl chart
# functions) - no second, parallel way of pulling the same data. Works
# identically for a curated match or a live upload: both end up as the same
# (source, key) + raw_data + ai_report_text inputs by the time they reach
# this function.
#
# Chart images use the same matplotlib path the on-screen "Download Chart"
# buttons already use, not Plotly's to_image()/kaleido - that hangs on this
# machine (see Part 1.2), was never actually wired into this app despite an
# earlier plan assuming otherwise, and reusing the one real static-image
# path here is exactly "don't build a second export mechanism."
# ==========================================

def _esc_rl(s):
    """Escapes for reportlab's Paragraph mini-XML (it interprets <b>, <i>,
    <br/> etc. as real tags, so raw &/</> in report text must be escaped
    first or it can break/mis-render)."""
    s = str(s if s is not None else "")
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def _markdown_to_flowables(text, styles):
    """Lightweight ###/**bold** markdown -> reportlab Paragraphs. The coach
    report is generated with exactly this small subset of markdown (see the
    writing_prompt in Step 3), so a full markdown parser isn't needed."""
    flowables = []
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        escaped = _esc_rl(para)
        bolded = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
        if bolded.startswith('### '):
            flowables.append(Paragraph(bolded[4:], styles['Heading2']))
        elif bolded.startswith('## '):
            flowables.append(Paragraph(bolded[3:], styles['Heading1']))
        else:
            flowables.append(Paragraph(bolded.replace('\n', '<br/>'), styles['BodyText']))
        flowables.append(Spacer(1, 8))
    return flowables

def _cv_summary_flowables(cv_output_dir, team_a, team_b, cv_team_mapping, styles):
    flowables = []
    if not cv_output_dir:
        flowables.append(Paragraph("No CV Deep Analysis was launched for this match.", styles['BodyText']))
        return flowables
    status = get_cv_job_status_safe(cv_output_dir)
    if status.get('status') != 'complete':
        flowables.append(Paragraph(
            f"CV Deep Analysis is not yet complete for this match (status: {_esc_rl(status.get('status', 'unknown'))}).",
            styles['BodyText']))
        return flowables
    stats = None
    stats_file = status.get('stats_file')
    if stats_file:
        resolved = _resolve_cv_path(stats_file)
        if resolved.exists():
            try:
                with open(resolved, 'r') as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, OSError):
                stats = None
    if not stats:
        flowables.append(Paragraph("CV stats file was not available when this PDF was generated.", styles['BodyText']))
        return flowables

    flowables.append(Paragraph("Rendered Output Explanations", styles['Heading3']))
    for fname, label in CV_OUTPUT_VIDEO_LABELS.items():
        desc = CV_OUTPUT_DESCRIPTIONS.get(fname, "")
        flowables.append(Paragraph(f"<b>{_esc_rl(label)}:</b> {_esc_rl(desc)}", styles['BodyText']))
        flowables.append(Spacer(1, 4))

    flowables.append(Spacer(1, 8))
    flowables.append(Paragraph("Observations — specific to this match", styles['Heading3']))
    observations = generate_cv_observations(stats, team_a, team_b, cv_team_mapping)
    if observations:
        for obs in observations:
            flowables.append(Paragraph(f"• {_esc_rl(obs)}", styles['BodyText']))
    else:
        flowables.append(Paragraph("No tactical-event observations were available for this window.", styles['BodyText']))
    return flowables

def _training_plan_flowables(plan, styles):
    flowables = []
    if not plan:
        flowables.append(Paragraph("No training plan has been generated for this match yet.", styles['BodyText']))
        return flowables

    team_plan = plan.get('team_plan')
    flowables.append(Paragraph("Team Plan", styles['Heading2']))
    if team_plan and team_plan.get('days'):
        flowables.append(Paragraph(_esc_rl(team_plan.get('source_note', '')), styles['Italic']))
        flowables.append(Spacer(1, 6))
        for d in team_plan['days']:
            edited = set(d.get('edited_fields') or [])
            chat_confirmed = set(d.get('chat_confirmed_fields') or [])
            label_note = " (manually edited)" if ('focus_label' in edited or 'focus_category' in edited) else ""
            flowables.append(Paragraph(f"<b>{_esc_rl(d.get('day'))} — {_esc_rl(d.get('focus_label'))}</b>{label_note}", styles['BodyText']))
            drills_note = " (manually edited)" if 'drills' in edited else ""
            for dr in (d.get('drills') or []):
                flowables.append(Paragraph(
                    f"&nbsp;&nbsp;• {_esc_rl(dr.get('title'))} ({_esc_rl(dr.get('duration_min'))} min) — "
                    f"{_esc_rl(dr.get('note'))}{drills_note}", styles['BodyText']))
            # Same 3-way precedence as the live UI's badge (_src_pill in
            # training_plan.py): manually edited > confirmed via chat >
            # grounded - a later hand-edit after a chat-confirm wins visually
            # here too, matching the calendar exactly.
            if 'why_stat' in edited:
                why_note = " (manually edited — not verified against match data)"
            elif 'why_stat' in chat_confirmed:
                why_note = " (confirmed via chat — an AI-proposed edit the user approved)"
            else:
                why_note = " (grounded in match data)"
            flowables.append(Paragraph(f"<i>Why: {_esc_rl(d.get('why_stat'))}</i>{why_note}", styles['BodyText']))
            flowables.append(Spacer(1, 8))
    else:
        flowables.append(Paragraph("No team plan has been generated for this match yet.", styles['BodyText']))

    flowables.append(Spacer(1, 10))
    player_plan = plan.get('player_plan')
    flowables.append(Paragraph("Player Plans", styles['Heading2']))
    if player_plan and player_plan.get('players'):
        flowables.append(Paragraph(_esc_rl(player_plan.get('scope_note', tp.PLAYER_PLAN_SCOPE_NOTE)), styles['Italic']))
        flowables.append(Spacer(1, 6))
        for p in player_plan['players']:
            flowables.append(Paragraph(
                f"<b>P{_esc_rl(p.get('player_id'))} ({_esc_rl(p.get('team_label'))})</b> — "
                f"top speed {p.get('top_speed_kmh') or 0:.1f} km/h, avg {p.get('avg_speed_kmh') or 0:.1f} km/h, "
                f"distance {p.get('total_distance_m') or 0:.0f} m", styles['BodyText']))
            for s in (p.get('sessions') or []):
                edited = set(s.get('edited_fields') or [])
                chat_confirmed = set(s.get('chat_confirmed_fields') or [])
                if 'note' in edited:
                    note_flag = " (manually edited — not verified against match data)"
                elif 'note' in chat_confirmed:
                    note_flag = " (confirmed via chat — an AI-proposed edit the user approved)"
                else:
                    note_flag = " (grounded in match data)"
                flowables.append(Paragraph(
                    f"&nbsp;&nbsp;• {_esc_rl(s.get('day'))}: {_esc_rl(s.get('title'))} — {_esc_rl(s.get('note'))}{note_flag}",
                    styles['BodyText']))
            flowables.append(Spacer(1, 8))
    else:
        flowables.append(Paragraph("No player plan has been generated for this match yet "
                                    "(this needs the CV Deep Analysis job to be complete).", styles['BodyText']))
    return flowables

def _chart_image_flowables(df, team_a, team_b, color_a, color_b, styles):
    flowables = [Paragraph("Dashboard Charts", styles['Heading1'])]
    bias_a_df = df[(df['team_a_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
    bias_b_df = df[(df['team_b_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
    chart_specs = [
        (f"{team_a} Attacking Bias", lambda: build_bias_pie_mpl(bias_a_df.get('team_a_attacking_bias', pd.Series(dtype=str))), 3 * inch, 3 * inch),
        (f"{team_b} Attacking Bias", lambda: build_bias_pie_mpl(bias_b_df.get('team_b_attacking_bias', pd.Series(dtype=str))), 3 * inch, 3 * inch),
        ("Active Possession Distribution", lambda: build_heatmap_mpl(compute_possession_heatmap_data(df, team_a, team_b, color_a, color_b), 'Blues'), 6 * inch, 3 * inch),
        ("Team Positioning (Block Height)", lambda: build_heatmap_mpl(compute_block_heatmap_data(df, team_a, team_b), 'Purples'), 6 * inch, 3 * inch),
        ("Segment Momentum", lambda: build_momentum_mpl(df, team_a, team_b), 6 * inch, 3 * inch),
    ]
    for label, build_fn, w, h in chart_specs:
        try:
            fig = build_fn()
        except Exception:
            fig = None
        flowables.append(Paragraph(label, styles['Heading3']))
        if fig is None:
            flowables.append(Paragraph("No data available for this chart in this segment.", styles['BodyText']))
        else:
            flowables.append(Image(io.BytesIO(fig_to_png_bytes(fig)), width=w, height=h))
        flowables.append(Spacer(1, 10))
    return flowables

def generate_full_pdf_report(team_a, team_b, color_a, color_b, raw_data, ai_report_text,
                              cv_output_dir, cv_team_mapping, source, key):
    """Assembles the full PDF: coach report -> CV summary -> training plan
    (with Part 5.1's edited_fields provenance annotations) -> dashboard
    charts. Works the same for a curated match or a live upload - both are
    just (source, key) + raw_data + ai_report_text by the time they get here."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    flowables = [
        Paragraph(f"{_esc_rl(team_a)} vs {_esc_rl(team_b)} — Full Match Report", styles['Title']),
        Spacer(1, 14),
        Paragraph("Coach Report", styles['Heading1']),
    ]
    if ai_report_text:
        flowables.extend(_markdown_to_flowables(ai_report_text, styles))
    else:
        flowables.append(Paragraph("No coach report has been generated for this match yet.", styles['BodyText']))
    flowables.append(PageBreak())

    flowables.append(Paragraph("CV Deep Analysis Summary", styles['Heading1']))
    flowables.extend(_cv_summary_flowables(cv_output_dir, team_a, team_b, cv_team_mapping, styles))
    flowables.append(PageBreak())

    flowables.append(Paragraph("Training Plan", styles['Heading1']))
    plan = tp.load_training_plan(source, key, CURATED_MATCHES_DIR, CACHE_DIR) if source else None
    flowables.extend(_training_plan_flowables(plan, styles))
    flowables.append(PageBreak())

    try:
        df = compute_dashboard_df(pd.DataFrame(raw_data), team_a, team_b, color_a, color_b)
        flowables.extend(_chart_image_flowables(df, team_a, team_b, color_a, color_b, styles))
    except Exception as e:
        flowables.append(Paragraph("Dashboard Charts", styles['Heading1']))
        flowables.append(Paragraph(f"Charts could not be generated for this match: {_esc_rl(e)}", styles['BodyText']))

    doc.build(flowables)
    buf.seek(0)
    return buf.getvalue()

def html_tooltip(text):
    """A "?" tooltip matching metric_card's look, for custom HTML cards (e.g.
    the training-plan calendar) that aren't a real st.metric and so can't use
    its native help= param. text is plain text (goes into a title= attribute,
    so keep it short - browsers don't wrap long title tooltips well)."""
    safe = text.replace('"', "&quot;")
    return f'<span class="tac-tt" title="{safe}">?</span>'

# --- CV pipeline (separate codebase, sibling folder in this repo) integration
# constants. Resolved relative to this file, not a hardcoded absolute path -
# a hardcoded string here was found (and fixed) twice before (once during a
# GPU-rental test, once more during this repo consolidation, where a second,
# independent hardcoded copy of this same path turned up in get_pdf_download_
# button below), so this is the one place either should ever be computed from. ---
CV_PIPELINE_DIR = Path(__file__).resolve().parent.parent / "cv_pipeline"
CV_PIPELINE_SCRIPT = CV_PIPELINE_DIR / "run_cv_analysis.py"

# --- Instant Demo (curated matches) + processing cache (repeat uploads) ---
CURATED_MATCHES_DIR = Path(__file__).parent / "curated_matches"
# CACHE_DIR used to live under the app's own source directory
# (Path(__file__).parent / ".cache") - that works for local dev, but
# Streamlit Community Cloud mounts the cloned repo read-only, so every write
# through here (the processing cache manifest below, active_jobs.json,
# cache-sourced training plans) crashed with "attempt to write a readonly
# database"/OSError in production, the same failure mode diagnosed for
# ChromaDB (see chatbot.py's CHROMA_DIR). Redirected to the system temp dir,
# which is writable in effectively any hosting environment. This makes all
# of this data ephemeral (cleared on a container restart) rather than
# persisting indefinitely - acceptable here since every consumer already
# either rebuilds on demand (chroma collections, the CV job registry) or is
# genuinely disposable (the repeat-upload cache); it's the same tradeoff
# already accepted for ChromaDB, just applied consistently.
CACHE_DIR = Path(tempfile.gettempdir()) / "tactical_scout_dashboard_cache"
CACHE_MANIFEST_PATH = CACHE_DIR / "processing_cache.json"
# Lightweight in-flight-job registry (video_hash -> {pid, output_dir, ...}).
# Stopgap per the orphaned-subprocess investigation: no real job database
# exists yet, so this file is the only thing standing between a repeat
# upload and a redundant, colliding second run_cv_analysis.py launch.
ACTIVE_JOBS_PATH = CACHE_DIR / "active_jobs.json"

valid_keys = [k.strip() for k in API_KEYS if k.strip() and "YOUR_" not in k]

# Initialize Session States
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.raw_data = []
    st.session_state.color_a = "Team A"
    st.session_state.color_b = "Team B"
    st.session_state.team_a = "Team A"
    st.session_state.team_b = "Team B"
    st.session_state.view_mode = 'dashboard'
    st.session_state.ai_report = None
    st.session_state.history = []
    st.session_state.video_hash = None
    st.session_state.gemini_cache_hit = False

if st.session_state.step != 1:
    # Step 1 (landing page) carries its own hero branding - this top title is
    # only for the post-upload steps (loading / team mapping / dashboard).
    st.title("⚽ AI Tactical Coaching Dashboard")

# ==========================================
# METHODOLOGY TEXT 
# ==========================================
methodology_text = """
### 📊 Tactical Glossary & Metrics

This dashboard utilizes a custom logic engine to transform raw video data into actionable tactical insights. Below is the methodology for each core metric:

**Match Dominance (Net Attacking Threat Score)**
A zero-sum mathematical momentum calculation. It computes a team's real-time threat by assigning distinct weighted values to their pitch position (Attacking Third = Highest), attacking tempo (Fast Direct = Highest), and spatial half-space occupancy. A heavy +5 point multiplier is applied strictly to high-value counter-attacks and fast vertical transitions. Only the team controlling the primary possession registers a threat score for that minute.

**Average Threat Score**
Calculates the mean value of the Threat Score exclusively during the minutes a team held possession. 

**Primary Zone**
Identifies the specific third of the pitch (Defensive, Middle, or Attacking) where a team logged the highest volume of sustained possession over the course of the match segment.

**Time in Attack**
Evaluates the literal fractional minutes a team spent actively establishing possession and attacking inside the opponent's defensive third.

**Active Possession Distribution (Heatmap)**
Measures the territorial footprints of both teams across all three zones. *Note: This does not necessarily sum to 90 minutes.* It explicitly filters out purely neutral moments, rapid transitions, or major stoppages to provide a pure reflection of established, active possession.

**Defensive Block Height (Positioning)**
Measures the *starting position* of a team's defensive wall out of possession:
* **Low Block:** Retreating deep into their own penalty area to defend.
* **Mid Block:** Holding their defensive line near the midfield circle.
* **High Block:** Pushing defenders aggressively high up the pitch into the opponent's half.

**Pressing Intensity (1-10 Scale)**
An AI-calculated metric measuring the aggression of the team's tackling and closing down. (1-3: Passive, 4-6: Engaging near halfway line, 7-10: Aggressive).

**Attacking Bias (Geometry)**
Calculates the exact sequence frequency a team successfully funneled an attack down a specific geometric channel. Sterile defensive possession is filtered out. The 'Central Channel' is strictly geometrically defined as the exact physical width of the 18-yard penalty box.

---
### 🧠 The Backend Architecture & Prompt-Based Fine-Tuning
This dashboard is powered by a custom multimodal data pipeline that converts raw match footage into structured, mathematical tactical arrays using Google's **Gemini 2.5 Flash** vision-language model.

**1. Local Micro-Batching Protocol**
The Python backend physically slices the match into strict 1-minute intervals locally, uploading and processing them sequentially to guarantee the AI maintains "tunnel vision" on specific tactical actions.

**2. Architectural Constraint Fine-Tuning**
The model is strictly fine-tuned through systemic prompt architecture and logic locks to avoid common vision-model hallucinations (e.g., center-frame bias).

---
### 💰 Scalability & Cost (rough planning estimate — not a commitment)
The figures below are early, non-binding estimates for future planning discussions, not a fixed budget or a promise of what any deployment will actually cost. They're included for transparency about how this system could scale, nothing more.

**Chatbot / function-calling (once built)**
At current or early scale, ongoing cost is expected to be near-negligible: the vector store (ChromaDB) runs free and local, and every Gemini call involved (embedding, routing, generation, function-calling) uses Flash-tier pricing — realistically fractions of a cent to a few cents per typical user session.

**Hugging Face Spaces deployment**
The plan is to commit `bundle.json` files and rendered `output_videos/` directly into the Space's own git repository — no separate database or object-storage service needed at this scale. Hugging Face Spaces automatically handles Git LFS for the larger binary files. The Gemini API key is stored as a Space secret, never committed as a file.

**GPU rental, if/when full-match processing is revisited**
Renting (not buying) a consumer-grade GPU on a reliable cloud host runs roughly $0.30–$0.50/hour. Once environment setup is already done, a full batch processing session would likely cost single-digit dollars total. This is a realistic estimate, not a guess made in the abstract — GPU acceleration of the fallback ball-detection stage specifically is already confirmed and measured (Section 10.7 of the linked project report), so the cost-per-hour figure above reflects real, tested hardware behavior, not just a rate card.

**Broader scaling estimate (illustrative: 10–20 paying clients)**
A rough planning estimate lands around $150–$800/month at that scale. GPU compute and storage/database are the cost categories that would actually grow with volume; Gemini API usage and general hosting tend to stay comparatively small unless usage becomes very heavy. Again — a rough estimate for internal planning, not a number to hold anyone to.

---
### 👨‍💻 About the Creator
I am a Master’s student in **Applied Statistics and Data Science at UCLA** and a UC Riverside Alumni in Computer Science. My technical expertise sits at the intersection of AI, Machine Learning, and Full-Stack Data Science. 
🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/ashwin-ramaseshan-a63188201/)
"""

def get_pdf_download_button():
    pdf_path = str(CV_PIPELINE_DIR / "Real-Time_Soccer_Analytics_Pipeline_v3.pdf")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        return st.download_button(
            label="📄 Download Full Project Report (PDF)",
            data=pdf_bytes,
            file_name="Real-Time_Soccer_Analytics_Pipeline_v3.pdf",
            mime="application/pdf",
        )
    else:
        return st.warning(f"Project Report PDF not found at {pdf_path}. Please check the file path.")

def get_cv_job_status(cv_output_dir):
    """Reads the status.json a launched run_cv_analysis.py subprocess writes
    into cv_output_dir. Returns the parsed dict as-is (status one of
    'running'/'complete'/'error', per run_cv_analysis.py's StatusWriter) or
    {'status': 'not_started'} if the job hasn't written anything yet, or
    {'status': 'unknown'} if the file exists but can't be parsed (e.g. read
    racing a partial write - StatusWriter writes atomically via os.replace,
    so this should be rare).

    CAUTION: do not call this while the job might still be actively running
    and rewriting status.json at high frequency (calibration/tracking write
    very often) - on Windows, another process merely having this file open
    for reading can make the pipeline's own os.replace() fail with
    PermissionError, which actually crashed a real multi-hour run earlier in
    this project. Use get_cv_job_stdout_tail()/render_cv_deep_analysis_tab's
    dispatch (which only calls this once a terminal marker is seen in the
    safe, append-only stdout log, or when that log doesn't exist at all -
    i.e. an older bundle that finished before this log file existed)."""
    status_path = Path(cv_output_dir) / "status.json"
    if not status_path.exists():
        return {"status": "not_started"}
    try:
        with open(status_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"status": "unknown"}

_STAGE_LINE_RE = re.compile(r"^\[(\d+)/(\d+)\]\s*(.*)$", re.MULTILINE)

def get_cv_job_stdout_tail(cv_output_dir, n_lines=500):
    """Safe to call anytime, including while the job is actively running -
    pipeline_stdout.log is only ever appended to by the child process, never
    replaced/renamed, so concurrent reads never race a rename the way
    status.json does. Returns '' if the log doesn't exist (a bundle built
    before this file existed, or the subprocess hasn't written anything yet)."""
    log_path = Path(cv_output_dir) / "pipeline_stdout.log"
    if not log_path.exists():
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n_lines:])
    except OSError:
        return ""

def _parse_stage_from_log(log_text):
    """Last '[N/19] <label>' stage header in the log, if any - the safe,
    log-based equivalent of status.json's stage_num/total_stages/label."""
    matches = _STAGE_LINE_RE.findall(log_text)
    if not matches:
        return None
    stage_num, total_stages, label = matches[-1]
    return {"stage_num": int(stage_num), "total_stages": int(total_stages), "label": label.strip()}

def _log_terminal_state(log_text):
    """None while still running, else 'complete'/'error' - both mean the
    subprocess has exited, so status.json is no longer being actively
    rewritten and becomes safe to read."""
    if "Traceback (most recent call last):" in log_text:
        return "error"
    if "[run_cv_analysis] done in" in log_text:
        return "complete"
    return None

def get_cv_job_status_safe(cv_output_dir):
    """The dispatch every UI call site should use instead of calling
    get_cv_job_status()/reading status.json directly. Reads the safe,
    append-only stdout log first; only falls through to a real status.json
    read when the log proves the subprocess has already exited (a terminal
    marker present) or the log file doesn't exist at all (an older bundle
    that predates this log, which can only mean it already finished - no
    live run is redirecting output to a log path that then vanished)."""
    log_path = Path(cv_output_dir) / "pipeline_stdout.log"
    log_text = get_cv_job_stdout_tail(cv_output_dir)
    terminal = _log_terminal_state(log_text)
    if terminal is not None or not log_text:
        return get_cv_job_status(cv_output_dir)
    stage_info = _parse_stage_from_log(log_text) or {}
    # started_at from the log file's own creation time, not status.json's
    # field of the same name - keeps the ETA math in render_cv_processing_state
    # working without ever reading status.json during an active run. No
    # substage/frame-level progress in this path (that's a bigger, fragile
    # regex job against run_cv_analysis.py's own print formats) - falls back
    # to whole-stage granularity, same as any stage that doesn't report it.
    try:
        started_at = datetime.fromtimestamp(log_path.stat().st_ctime, tz=timezone.utc).isoformat()
    except OSError:
        started_at = None
    return {"status": "running", "started_at": started_at, **stage_info}

def _cv_bundle_is_valid(cv_output_dir):
    """Re-checks a CV bundle's status live off disk - never trust a
    cached/curated registry's word for it, since the folder could have been
    moved, deleted, or the job could have errored out since it was recorded.
    Uses the safe dispatch (get_cv_job_status_safe), not a direct status.json
    read: this function is called on every page load (curated-match listing,
    the home screen's live "Clips validated" count, cache-manifest checks),
    so a direct read here would repeatedly open status.json for any bundle
    whose CV job might still be actively running elsewhere - exactly the
    access pattern that crashed a real multi-hour run on Windows earlier in
    this project."""
    if not cv_output_dir:
        return False
    # Resolved here, once, rather than requiring every caller to remember to
    # do it - found missing at exactly this call site during the repo
    # consolidation's fresh-clone check (curated matches' now-relative
    # cv_output_dir was being used as-is, resolving against the dashboard's
    # own cwd instead of CV_PIPELINE_DIR, so no curated match ever validated
    # on a fresh clone). Safe for cache-manifest entries too: their
    # cv_output_dir is already absolute, and _resolve_cv_path passes
    # absolute paths through unchanged.
    return get_cv_job_status_safe(str(_resolve_cv_path(cv_output_dir))).get('status') == 'complete'

def _load_curated_matches():
    """Scans curated_matches/<id>/bundle.json for developer-curated instant-
    demo matches. Only returns entries that verifiably have BOTH pieces
    (Gemini raw_data+ai_report inline, and a CV bundle that still completes
    validly on disk right now) - a stale or half-built bundle is silently
    excluded rather than shown broken."""
    matches = []
    if not CURATED_MATCHES_DIR.exists():
        return matches
    for bundle_path in sorted(CURATED_MATCHES_DIR.glob("*/bundle.json")):
        try:
            with open(bundle_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not bundle.get("raw_data") or not bundle.get("ai_report"):
            continue
        if not _cv_bundle_is_valid(bundle.get("cv_output_dir")):
            continue
        _apply_curated_override(bundle)
        matches.append(bundle)
    return matches

# Field edits to a curated match (rename, Part 3's CV team-mapping
# confirmation) used to write straight back into curated_matches/<id>/
# bundle.json via _update_match_fields - that path lives inside the
# git-cloned source tree, which Streamlit Community Cloud mounts read-only
# (the same failure mode diagnosed for ChromaDB/CACHE_DIR), so any edit to
# a curated match would fail there. Edits now write to a small per-match
# override file under the writable CACHE_DIR instead, merged on top of the
# shipped bundle.json at read time - the shipped file itself is never
# touched. Like the rest of CACHE_DIR, this is ephemeral: an edit made on
# a deployed container doesn't survive a restart, same tradeoff already
# accepted for ChromaDB and the processing cache.
CURATED_OVERRIDES_DIR = CACHE_DIR / "curated_overrides"

def _curated_override_path(key):
    return CURATED_OVERRIDES_DIR / f"{key}.json"

def _apply_curated_override(bundle):
    """Merges a curated match's writable override file (if any) on top of
    its shipped bundle dict, in place."""
    override_path = _curated_override_path(bundle.get("id"))
    if not override_path.exists():
        return
    try:
        with open(override_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    bundle.update(overrides)

def _load_cache_manifest():
    if not CACHE_MANIFEST_PATH.exists():
        return {}
    try:
        with open(CACHE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def _save_cache_manifest(manifest):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(CACHE_MANIFEST_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, CACHE_MANIFEST_PATH)

def _cache_get(video_hash):
    return _load_cache_manifest().get(video_hash)

def _cache_upsert(video_hash, **fields):
    """Merges fields into the manifest entry for video_hash (creating it if
    new) and writes through atomically. Used both right after a fresh Gemini
    extraction finishes and, separately, whenever the CV tab notices a
    background job has completed - each writes only the piece it just
    produced, so a partially-cached video fills in over time with no manual
    step."""
    manifest = _load_cache_manifest()
    entry = manifest.get(video_hash, {})
    entry.update(fields)
    manifest[video_hash] = entry
    _save_cache_manifest(manifest)
    return entry

def _activate_match_bundle(bundle, source=None, key=None):
    """Loads a fully-qualified bundle (curated match or a full cache hit)
    straight into session state and jumps to the finished dashboard - no
    Gemini call, no CV subprocess launch. Clears video_hash so a later CV-tab
    view of THIS bundle never write-throughs into the cache manifest under
    a stale hash left over from an earlier real upload in the same session.

    source/key identify where this bundle lives on disk ("curated"/<id> or
    "cache"/<video_hash>) so the CV tab's one-time team-mapping confirmation
    (Part 3) can write its answer back to the right place via
    _update_match_fields - video_hash alone can't be used for this since it's
    cleared below."""
    st.session_state.raw_data = bundle["raw_data"]
    st.session_state.team_a = bundle["team_a"]
    st.session_state.team_b = bundle["team_b"]
    st.session_state.color_a = bundle["color_a"]
    st.session_state.color_b = bundle["color_b"]
    st.session_state.ai_report = bundle["ai_report"]
    # Found during the repo-consolidation fresh-clone check: curated bundle.json
    # files shipped in this repo store cv_output_dir as a path relative to
    # CV_PIPELINE_DIR (e.g. "output_videos/liverpool_psg_verified"), not an
    # absolute one - a stale absolute path here would silently only work on
    # the machine it was curated on. _resolve_cv_path already exists for
    # exactly this (absolute passes through unchanged, relative resolves
    # against CV_PIPELINE_DIR), so route through it here too.
    _bundle_cv_dir = bundle.get("cv_output_dir")
    st.session_state.cv_job_output_dir = str(_resolve_cv_path(_bundle_cv_dir)) if _bundle_cv_dir else None
    st.session_state.cv_segment_timestamp = bundle.get("cv_segment_timestamp")
    st.session_state.cv_segment_momentum_score = bundle.get("cv_segment_momentum_score")
    st.session_state.cv_team_mapping = bundle.get("cv_team_mapping")
    st.session_state.active_match_source = source
    st.session_state.active_match_key = key
    st.session_state.video_hash = None
    st.session_state.step = 3
    st.session_state.view_mode = "dashboard"

def _display_name_for_bundle(bundle):
    """User-provided name wins. Falls back to a curated bundle's original
    'label' (so the pre-existing "White vs Maroon" bundle keeps working with
    zero migration), then to a plain team-colors guess as a last resort for
    any record that somehow has neither."""
    return (
        bundle.get("display_name")
        or bundle.get("label")
        or f"{bundle.get('team_a', 'Team A')} vs {bundle.get('team_b', 'Team B')}"
    )

def _load_instant_demo_matches():
    """Everything eligible for the Instant Demo picker: developer-curated
    matches (curated_matches/*/bundle.json) plus any processing-cache entry
    that's fully complete (Gemini + CV both done, CV re-verified live off
    disk). Each item is tagged with 'source'/'key' so activation and
    rename actions can route back to the right storage location."""
    items = []
    for bundle in _load_curated_matches():
        items.append({
            "source": "curated",
            "key": bundle.get("id"),
            "display_name": _display_name_for_bundle(bundle),
            "bundle": bundle,
        })
    for video_hash, entry in _load_cache_manifest().items():
        if entry.get("raw_data") and entry.get("ai_report") and _cv_bundle_is_valid(entry.get("cv_output_dir")):
            items.append({
                "source": "cache",
                "key": video_hash,
                "display_name": _display_name_for_bundle(entry),
                "bundle": entry,
            })
    return items

def _update_match_fields(item, **fields):
    """Persists arbitrary field updates (display_name, team_a, team_b, the
    Part 3 team1/team2 CV mapping, etc.) to an Instant Demo item, routing to
    curated_matches/<id>/bundle.json or the cache manifest depending on
    where the record actually lives. Generalized from the original rename-
    only helper so Part 3's team-name editing and CV team-mapping
    confirmation reuse the same single write-through path."""
    fields = {k: (v.strip() if isinstance(v, str) else v) for k, v in fields.items() if v not in (None, "")}
    if not fields:
        return
    if item["source"] == "curated":
        bundle_path = CURATED_MATCHES_DIR / item["key"] / "bundle.json"
        if not bundle_path.exists():
            return
        override_path = _curated_override_path(item["key"])
        override_path.parent.mkdir(parents=True, exist_ok=True)
        overrides = {}
        if override_path.exists():
            try:
                with open(override_path, "r", encoding="utf-8") as f:
                    overrides = json.load(f)
            except (json.JSONDecodeError, OSError):
                overrides = {}
        overrides.update(fields)
        tmp = str(override_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2)
        os.replace(tmp, override_path)
    else:
        _cache_upsert(item["key"], **fields)

def _rename_match(item, new_name):
    _update_match_fields(item, display_name=new_name)

# ==========================================
# CV subprocess process-list helpers + in-flight job registry
# Stopgap per the orphaned-subprocess investigation: no real job database
# exists, so these do the two things that were missing - (1) let an admin
# actually find/kill jobs this app launched, and (2) let a repeat upload
# detect a job already in flight instead of launching a duplicate.
# ==========================================

def _list_app_launched_cv_pids():
    """PIDs (+cmdline) of run_cv_analysis.py processes launched BY THIS APP
    specifically - found the same way this project's own manual debugging
    has all along: querying the OS process list via PowerShell and matching
    on command line, not just process name (every python.exe looks
    identical by name alone). Scoped to cmdlines that also reference this
    app's own tactical_scout_cv_sessions temp convention, so a
    run_cv_analysis.py invoked some other way (e.g. a manual test run
    against output_videos/) is never touched."""
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return []
    matches = []
    for row in data:
        cmdline = row.get("CommandLine") or ""
        if "run_cv_analysis.py" in cmdline and "tactical_scout_cv_sessions" in cmdline:
            pid = row.get("ProcessId")
            if pid:
                matches.append((int(pid), cmdline))
    return matches

def _kill_pid(pid):
    try:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"],
            capture_output=True, text=True, timeout=15,
        )
        return True
    except (subprocess.SubprocessError, OSError):
        return False

def _pid_is_alive(pid):
    """True only if pid is still a live, THIS-APP-launched run_cv_analysis.py
    process - re-verifies the command line rather than trusting bare PID
    existence, since PIDs get reused by the OS."""
    if not pid:
        return False
    return any(p == pid for p, _ in _list_app_launched_cv_pids())

def _load_active_jobs():
    if not ACTIVE_JOBS_PATH.exists():
        return {}
    try:
        with open(ACTIVE_JOBS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def _save_active_jobs(jobs):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = str(ACTIVE_JOBS_PATH) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)
    os.replace(tmp, ACTIVE_JOBS_PATH)

def _get_active_job(video_hash):
    """Returns the in-flight-job record for video_hash if its process is
    still genuinely alive, else None. Self-healing: a stale entry (process
    no longer running) is pruned from the registry right here, so dead
    entries never accumulate without a separate cleanup pass."""
    if not video_hash:
        return None
    jobs = _load_active_jobs()
    entry = jobs.get(video_hash)
    if not entry:
        return None
    if _pid_is_alive(entry.get("pid")):
        return entry
    del jobs[video_hash]
    _save_active_jobs(jobs)
    return None

def _register_active_job(video_hash, pid, output_dir, match_name):
    jobs = _load_active_jobs()
    jobs[video_hash] = {
        "pid": pid,
        "output_dir": output_dir,
        "match_name": match_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_active_jobs(jobs)

def _unregister_active_job(video_hash):
    if not video_hash:
        return
    jobs = _load_active_jobs()
    if video_hash in jobs:
        del jobs[video_hash]
        _save_active_jobs(jobs)

def _render_admin_panel():
    """Maintenance-only panel - visible only with ?admin=1 in the URL, not
    a real user-facing feature. Explicitly a TEMPORARY STOPGAP per the
    orphaned-subprocess investigation: until a real job registry/database
    exists, this is the manual way to stop everything this app has
    launched and reset the current session. It never touches saved data
    (processing cache manifest, curated matches, completed CV bundles)."""
    with st.sidebar.expander("🧹 Admin: Clean Everything / Start Over"):
        st.caption(
            "Stops any run_cv_analysis.py jobs **this app** launched (checked live "
            "against the OS process list) and resets your current session back to "
            "the landing page. Does **not** delete the processing cache, curated "
            "matches, or any completed Gemini/CV results - those are untouched. "
            "Temporary stopgap until a real job registry exists."
        )
        if st.button("🧹 Clean Everything / Start Over", key="admin_clean_everything"):
            targets = _list_app_launched_cv_pids()
            killed = [pid for pid, _ in targets if _kill_pid(pid)]
            # The registry only ever tracks processes this app itself
            # launched, so wiping it here is safe: anything that failed to
            # die would just be re-detected as dead (or re-discovered by
            # PID) on the next liveness check regardless.
            _save_active_jobs({})
            st.session_state.clear()
            st.toast(f"🧹 Killed {len(killed)} CV job(s): {killed}. Session reset.")
            st.rerun()

# run_cv_analysis.py's save_video() writes XVID-in-.avi, which browsers do not
# decode natively - st.video() would silently show a blank/broken player.
# Transcode once to H.264/.mp4 before displaying.
def _ensure_browser_playable_video(avi_path):
    avi_path = Path(avi_path)
    # Curated matches ship a pre-transcoded sibling next to the .avi (e.g.
    # output1_web.mp4 next to output1.avi) - use it directly, no transcode.
    shipped_mp4 = avi_path.with_name(avi_path.stem + "_web.mp4")
    if shipped_mp4.exists():
        return shipped_mp4

    # Fallback for outputs with no pre-shipped web.mp4 (e.g. barca_madrid_pt1
    # is only missing outputs 2/4/5/6). This used to cache the transcode next
    # to the source .avi via avi_path.with_name(...) - that path lives under
    # cv_pipeline/output_videos/ inside the git-cloned source tree, which
    # Streamlit Community Cloud mounts read-only (the same failure mode
    # diagnosed for ChromaDB/CACHE_DIR: a write into the source tree raises
    # OSError there). It silently degraded to the "download raw file"
    # fallback instead of crashing, which is why this went unnoticed until
    # now. Cached under the writable CACHE_DIR instead, namespaced by the
    # source directory's name since every match's output dir reuses the same
    # output1.avi..output6.avi filenames.
    cache_dir = CACHE_DIR / "web_video_cache" / avi_path.parent.name
    mp4_path = cache_dir / shipped_mp4.name
    if mp4_path.exists():
        return mp4_path
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with VideoFileClip(str(avi_path)) as clip:
            clip.write_videofile(str(mp4_path), codec="libx264", audio=False, logger=None)
        return mp4_path
    except Exception:
        return None

def _resolve_cv_path(path_str):
    """status.json's 'outputs'/'stats_file' entries are written by
    run_cv_analysis.py using whatever --output-dir it was given. Jobs this
    app launches always pass an absolute --output-dir, so those resolve
    fine as-is; but a job run manually/historically inside the CV repo
    (e.g. its own default relative 'output_videos') writes paths relative
    to the CV repo's cwd, not this Streamlit app's cwd. Absolute paths pass
    through unchanged; relative ones are resolved against CV_PIPELINE_DIR,
    the only base directory that makes sense for this cross-repo path.

    The curated matches' status.json files were originally written on Windows,
    where run_cv_analysis.py's output-dir join produced backslash separators
    (e.g. "output_videos\\liverpool_psg_verified\\output1.avi") - valid as a
    JSON string, but on Linux (Community Cloud) a backslash is just a regular
    filename character, not a separator, so Path() treated the whole string
    as one unresolvable component. Normalizing to forward slashes first fixes
    both these committed files and any future job run on Windows."""
    p = Path(path_str.replace("\\", "/"))
    return p if p.is_absolute() else (CV_PIPELINE_DIR / p)

# Duplicated from cv_pipeline/pitch_calibrator.py's PITCH_KP_WORLD rather than
# imported - that module does `from ultralytics import YOLO` at import time,
# and ultralytics isn't installed in the dashboard's own environment (dropped
# from dashboard/requirements.txt; see that file's header comment). Re-verify
# this stays in sync with pitch_calibrator.py if that table is ever edited.
# Landmark names below are filled in ONLY for the subset (12 of 48) whose
# coordinate exactly matches a standard FIFA pitch landmark computed from the
# geometry constants documented in pitch_calibrator.py's own header comment
# (corners, penalty/six-yard box corners, centre-circle/halfway intersection).
# The other 36 indices are additional correspondence points the report
# (Section 2.4.3-2.4.4) describes as assigned "by eye" during training data
# labeling, with no published index-to-landmark name table anywhere in this
# codebase or the technical report - inventing specific names for them would
# not be verifiable, so they're shown with their real coordinate and no
# invented name instead. Indices 8 and 32 share the identical coordinate
# (16.5, 0.0) - the report (Section 2.4.6) describes exactly this symptom
# (two indices assigned one coordinate) as a sign of an uncorrected mapping
# error; this specific pair does not appear to have been fixed.
PITCH_REFERENCE_POINTS = {
     0: ( 26.18,  0.56, None),
     1: (  0.00,  0.00, "Corner — left goal line, far touchline"),
     2: ( 35.42,  6.87, None),
     3: (  0.00, 68.00, "Corner — left goal line, near touchline"),
     4: (  5.50, 68.00, None),
     5: ( 12.32, 50.75, None),
     6: ( 26.74, 15.70, None),
     7: ( 30.94, 27.34, None),
     8: ( 16.50,  0.00, None),
     9: ( 16.50, 13.84, "Left penalty box corner — top edge, inside the pitch"),
    10: ( 16.50, 54.16, "Left penalty box corner — bottom edge, inside the pitch"),
    11: ( 16.50, 68.00, None),
    12: (  0.00, 24.84, "Left six-yard box corner — top edge, on the goal line"),
    13: ( 53.90,  0.42, None),
    14: ( 54.74, 23.98, None),
    15: ( 52.50, 43.15, "Centre circle ∩ halfway line — near-touchline side"),
    16: ( 54.88, 39.82, None),
    17: ( 37.94, 28.32, None),
    18: ( 28.42, 66.88, None),
    19: ( 32.48, 41.64, None),
    20: ( 48.16, 30.99, None),
    21: ( 25.76, 22.85, None),
    22: ( 33.04, 10.09, None),
    23: ( 22.68,  7.29, None),
    24: ( 31.92,  0.56, None),
    25: ( 88.50, 54.16, "Right penalty box corner — bottom edge, inside the pitch"),
    26: ( 88.50, 68.00, None),
    27: ( 46.00,  0.00, None),
    28: ( 42.28, 23.13, None),
    29: ( 47.00, 43.15, None),
    30: ( 35.84, 36.87, None),
    31: ( 88.50, 34.00, None),
    32: ( 16.50,  0.00, None),
    33: ( 58.00,  0.00, None),
    34: ( 40.60,  1.26, None),
    35: ( 88.50, 13.84, "Right penalty box corner — top edge, inside the pitch"),
    36: ( 30.38, 53.28, None),
    37: ( 33.60, 64.07, None),
    38: ( 36.40, 61.97, None),
    39: ( 31.36, 53.84, None),
    40: ( 64.00,  0.00, None),
    41: ( 88.50,  0.00, None),
    42: ( 38.08, 57.91, None),
    43: ( 35.84, 59.73, None),
    44: (  0.00, 13.84, "Left penalty box corner — top edge, on the goal line"),
    45: (  0.00, 54.16, "Left penalty box corner — bottom edge, on the goal line"),
    46: (105.00, 13.84, "Right penalty box corner — top edge, on the goal line"),
    47: (105.00, 54.16, "Right penalty box corner — bottom edge, on the goal line"),
}

CV_OUTPUT_VIDEO_LABELS = {
    'output1.avi': 'Tracking + Speed & Distance',
    'output2.avi': 'Tactical Events Carousel',
    'output3.avi': 'Player Stamina Panel',
    'output4.avi': 'Per-Player Tactical Map',
    'output5.avi': 'Team Pitch-Control Heatmap',
    'output6.avi': 'Movement Trails',
}

# Real descriptions, confirmed from run_cv_analysis.py / render_output*.py's own
# rendering code and docstrings (Part 2.5) - not guessed.
CV_OUTPUT_DESCRIPTIONS = {
    'output1.avi': (
        "The base tracking video: every player gets a team-colored ellipse marker (yellow for "
        "the referee), the ball gets a green triangle, and whoever the model thinks has the ball "
        "gets a red triangle overhead. Each player's live speed (km/h) and distance covered (m) "
        "are drawn beneath them, in a different color when tracking confidence is low."
    ),
    'output2.avi': (
        "The same tracking video, with a ranked list of this window's tactical events "
        "(BREAK, PRESS, SPACE, etc.) in a side panel, refreshed every 20 seconds so you can see "
        "what the model flagged as significant while it happened."
    ),
    'output3.avi': (
        "A stamina dashboard alongside the tracking video: each player's energy bar drains based "
        "on their tracked movement intensity (fastest at sprint speed, slowest when walking), "
        "updated every 10 seconds — an estimate from this window's movement, not a medical measurement."
    ),
    'output4.avi': (
        "A grid overlay where every cell is colored by whichever individual player is closest to "
        "it — a per-player space-control map, not a team-level one, blended semi-transparently over "
        "the source footage."
    ),
    'output5.avi': (
        "A team-level version of the same idea: an 8×8 grid colored by which team controls more of "
        "that area of the pitch, giving a heatmap of territorial control rather than individual "
        "positioning."
    ),
    'output6.avi': (
        "Fading movement trails (about 3 seconds long) showing where players ran, corrected for "
        "camera pan/zoom so the trails reflect real pitch movement rather than the camera's own motion."
    ),
}

# All 10 real tactical-event types this pipeline can detect (Part 2.4) - not
# just the 3 examples that happen to show up most often. Confirmed against
# tactical_events_detector.py / event_ranking.py's own type list and metric-
# string formats.
TACTICAL_EVENT_GLOSSARY = {
    'BREAK': "A player carrying the ball at high speed away from defensive pressure, into space.",
    'SPRINT': "A high-speed sprint off the ball — one of the fastest movements tracked in this window.",
    'BURST': "A sudden jump in speed — an explosive change of pace rather than a sustained sprint.",
    'PRESS': "Multiple defenders closing down the ball carrier together.",
    'RECOVERY': "A fast sprint back into defensive position after losing the ball or being caught upfield.",
    'OVERLAP': "An overlapping run past a teammate to create width or an extra passing option.",
    'SPACE': "A player receiving the ball with significantly more open space around them than average.",
    'LATERAL_RUN': "A significant sideways run across the pitch, often to stretch the opponent's shape.",
    'DROP': "A player dropping deep (backward) to receive the ball and help build play.",
    'ISOLATED': "A player caught far from any teammate, with little immediate support.",
}

def _cv_team_label(team_num, team_a, team_b, team_mapping=None):
    """Best-available label for the CV pipeline's numeric team1/team2 - falls
    back to a plain "Team {n}" until a real mapping has been confirmed (Part
    3's one-time manual confirmation, not yet wired as of Phase B)."""
    if team_mapping:
        key = team_mapping.get(str(team_num)) or team_mapping.get(team_num)
        if key == 'team_a':
            return team_a
        if key == 'team_b':
            return team_b
    return f"Team {team_num}"

def generate_cv_observations(stats, team_a, team_b, team_mapping=None, top_n=3):
    """Plain-English callouts computed directly from this bundle's real
    tactical_events highlights (Part 2.5) - never hardcoded, so it's correct
    for every match. Picks the top-N highlights by score and describes each.

    NOTE: highlights entries carry a 'window' (a coarse ~20s bucket index),
    not a 'frame' or exact-second timestamp (confirmed directly against a
    real stats.json - event_ranking.py's internal per-event dict has 'frame',
    but that field does not survive into the highlights list this app reads).
    An earlier version of this function assumed a 'frame' key and silently
    defaulted to 0 for every event, fabricating a "0:00" timestamp that
    wasn't real data - fixed by not claiming a specific moment at all."""
    if not stats:
        return []
    player_team = {p.get('player_id'): p.get('team') for p in stats.get('players', [])}
    highlights = stats.get('tactical_events', {}).get('highlights', [])
    if not highlights:
        return []
    ranked = sorted(highlights, key=lambda h: h.get('score', 0), reverse=True)[:top_n]

    observations = []
    for i, h in enumerate(ranked):
        team_num = player_team.get(h.get('player_id'))
        team_label = _cv_team_label(team_num, team_a, team_b, team_mapping) if team_num else "A player"
        event_type = str(h.get('type', '')).upper()
        metric = h.get('metric', '')
        superlative = "the standout moment" if i == 0 else "another key moment"
        observations.append(
            f"{team_label}'s {event_type.replace('_', ' ').title()} — {metric}, "
            f"{superlative} in this window (Player {h.get('player_id')})."
        )
    return observations

def render_cv_processing_state(status, cv_output_dir):
    # Part 4.2: re-run this script every few seconds while a job is running,
    # so the progress bar/ETA update live without the user having to leave
    # and come back to this tab. Reads only the safe stdout log (via
    # get_cv_job_status_safe upstream) - never status.json - so a much
    # higher rerun frequency doesn't reintroduce the earlier crash risk.
    st_autorefresh(interval=4000, key=f"cv_autorefresh_{cv_output_dir}")

    st.markdown("#### ⏳ Deep analysis running in the background")
    st.write(
        "The 30-second peak-momentum window was extracted during upload and is now being processed by the "
        "CV pipeline — player tracking, team classification, speed & stamina, tactical events, and movement "
        "analysis. The AI dashboard above is already complete; this section fills in automatically once "
        "processing finishes."
    )
    stage_num = status.get('stage_num')
    total_stages = status.get('total_stages')
    label = status.get('label')
    started_at = status.get('started_at')
    substage = status.get('substage')

    # 3-stat row (elapsed / stage / percent), all derived from status.json's real
    # started_at timestamp and stage_num - not hardcoded. Remaining time is a
    # linear extrapolation from "seconds per stage so far", which is only an
    # estimate (render stages run far longer than e.g. 'init'), same caveat the
    # mockup's own illustrative number carries.
    #
    # When the current stage reports frame-level 'substage' progress (only
    # calibration and tracking do - most of the 19 stages don't), blend it in
    # as a fractional stage so the bar/percent creep forward within a long
    # stage instead of sitting frozen until the whole stage completes. Falls
    # back to whole-stage granularity for every stage that doesn't report it.
    if stage_num and total_stages and started_at:
        try:
            started_dt = datetime.fromisoformat(started_at)
            elapsed_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
        except ValueError:
            elapsed_s = None

        substage_frac = None
        if substage and substage.get('frames_total'):
            substage_frac = max(0.0, min(1.0, (substage.get('pct') or 0) / 100.0))

        fine_fraction = ((stage_num - 1) + substage_frac) / total_stages if substage_frac is not None else stage_num / total_stages
        pct = round(fine_fraction * 100)
        eta_s = (elapsed_s / fine_fraction) * (1 - fine_fraction) if elapsed_s is not None and fine_fraction > 0 else None

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="tac-eta-box"><div class="tac-eta-val">{"~" + _fmt_mmss(eta_s) if eta_s is not None else "—"}</div>'
            f'<div class="tac-eta-lbl">estimated time remaining</div></div>', unsafe_allow_html=True)
        c2.markdown(
            f'<div class="tac-eta-box"><div class="tac-eta-val">{stage_num}<span style="font-size:14px;color:var(--muted)">/{total_stages}</span></div>'
            f'<div class="tac-eta-lbl">pipeline stage</div></div>', unsafe_allow_html=True)
        c3.markdown(
            f'<div class="tac-eta-box"><div class="tac-eta-val pct">{pct}%</div>'
            f'<div class="tac-eta-lbl">complete</div></div>', unsafe_allow_html=True)
        st.markdown("")

        progress_text = f"Stage {stage_num} of {total_stages}"
        if label:
            progress_text += f" — {label}"
        if substage_frac is not None:
            progress_text += f" — frame {substage['frames_done']}/{substage['frames_total']} ({substage.get('pct'):.0f}%)"
        st.progress(fine_fraction, text=progress_text)
    else:
        st.progress(0, text="Waiting for the CV job to start...")
    st.success("✓ You can navigate away from this tab — results are saved to disk and will be here when you return.")

    # Part 4.3: a second, more detailed view of the exact same safe log this
    # tab's progress bar already reads (get_cv_job_status_safe) - stage-by-
    # stage detection counts, ball-fallback trigger/recovery stats, etc.
    with st.expander("🔍 View Live Logs"):
        log_tail = get_cv_job_stdout_tail(cv_output_dir, n_lines=50)
        if log_tail:
            st.code(log_tail, language=None)
        else:
            st.caption("No pipeline output yet — this fills in once the subprocess starts writing.")

def _bgr_to_hex(bgr):
    """team_colors_bgr entries are [B, G, R] floats (OpenCV convention) -
    convert to a CSS hex color for the swatch UI."""
    if not bgr or len(bgr) < 3:
        return "#888888"
    b, g, r = bgr[0], bgr[1], bgr[2]
    return f"#{int(max(0, min(255, r))):02x}{int(max(0, min(255, g))):02x}{int(max(0, min(255, b))):02x}"

def _persist_cv_team_mapping(mapping):
    """Writes the Part 3 team1/team2 <-> team_a/team_b confirmation back to
    wherever this match's record actually lives, using the source/key
    _activate_match_bundle stashed - falls back to the live-upload cache
    path (keyed by video_hash) for a match that never went through
    _activate_match_bundle at all (i.e. a fresh, non-cached upload)."""
    st.session_state.cv_team_mapping = mapping
    key = st.session_state.get("active_match_key")
    source = st.session_state.get("active_match_source")
    if key and source:
        _update_match_fields({"source": source, "key": key}, cv_team_mapping=mapping)
    elif st.session_state.get("video_hash"):
        _cache_upsert(st.session_state.video_hash, cv_team_mapping=mapping)

def _render_team_mapping_confirmation(stats, team_a, team_b):
    """One-time manual confirmation (Part 3's approved design, in place of an
    automatic color-distance guess that could mislabel a player's team in a
    coaching tool) linking the CV pipeline's own team1/team2 jersey-color
    clustering to the real team_a/team_b names. Shown once per match; the
    answer is persisted so it never needs to be asked again."""
    if st.session_state.get('cv_team_mapping'):
        return
    team_colors = (stats or {}).get('team_resolution', {}).get('team_colors_bgr')
    if not team_colors:
        return
    swatch1, swatch2 = _bgr_to_hex(team_colors.get('team1')), _bgr_to_hex(team_colors.get('team2'))
    with st.container(border=True):
        st.markdown("**🎨 Confirm team colors** — one-time, so real team names show up correctly below.")
        st.caption(
            "This CV pipeline classifies players by jersey color independently of the tactical analysis "
            "above, so we need you to confirm which is which, once, for this match."
        )
        # Swatch style is inlined rather than a shared CSS class: a real match's
        # measured jersey colors can be dark and low-contrast against both each
        # other and this app's dark background (confirmed - not hypothetical -
        # during Phase C verification), so every swatch gets an explicit light
        # ring + shadow regardless of how dark the underlying color is.
        swatch_style = ("width:16px;height:16px;border-radius:50%;display:inline-block;"
                         "margin-right:6px;vertical-align:middle;border:2px solid rgba(255,255,255,0.55);"
                         "box-shadow:0 0 0 1px rgba(0,0,0,0.4);")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<span style="{swatch_style}background:{swatch1};"></span> **Team 1** is:',
                        unsafe_allow_html=True)
            team1_choice = st.radio("Team 1 is:", [team_a, team_b], key="cv_team1_choice", label_visibility="collapsed")
        with c2:
            st.markdown(f'<span style="{swatch_style}background:{swatch2};"></span> **Team 2** is:',
                        unsafe_allow_html=True)
            st.caption(team_b if team1_choice == team_a else team_a)
        if st.button("Confirm", key="cv_team_mapping_confirm"):
            team2_choice = team_b if team1_choice == team_a else team_a
            mapping = {
                "1": "team_a" if team1_choice == team_a else "team_b",
                "2": "team_a" if team2_choice == team_a else "team_b",
            }
            _persist_cv_team_mapping(mapping)
            st.rerun()

def render_cv_completed_state(status, cv_output_dir):
    outputs = status.get('outputs', {})
    stats_file = status.get('stats_file')

    stats = None
    if stats_file:
        resolved_stats_path = _resolve_cv_path(stats_file)
        if resolved_stats_path.exists():
            try:
                with open(resolved_stats_path, 'r') as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                st.error(f"Could not read stats file: {e}")

    _render_team_mapping_confirmation(stats, st.session_state.get('team_a', 'Team A'), st.session_state.get('team_b', 'Team B'))
    team_mapping = st.session_state.get('cv_team_mapping')

    video_col, side_col = st.columns([1.4, 1])

    with video_col:
        st.markdown("**📹 Rendered Analysis**")
        available = [(fname, label) for fname, label in CV_OUTPUT_VIDEO_LABELS.items() if fname in outputs]
        if not available:
            st.warning("CV job completed, but no rendered video outputs were recorded in status.json.")
        else:
            labels = [label for _, label in available]
            selected_label = st.radio("Choose a rendered output:", labels, horizontal=True, key="cv_video_switcher")
            selected_fname = next(fname for fname, lbl in available if lbl == selected_label)
            st.caption(CV_OUTPUT_DESCRIPTIONS.get(selected_fname, ""))
            video_path = _resolve_cv_path(outputs[selected_fname])
            if video_path.exists():
                with st.spinner("Preparing video for playback (one-time transcode)..."):
                    playable_path = _ensure_browser_playable_video(video_path)
                if playable_path and playable_path.exists():
                    st.video(str(playable_path))
                else:
                    st.warning("Couldn't prepare this render for in-browser playback. You can still download the raw file.")
                    with open(video_path, 'rb') as f:
                        st.download_button(f"Download raw {selected_label} (.avi)", f.read(),
                                            file_name=video_path.name, key=f"dl_cv_{selected_fname}")
            else:
                st.error(f"Expected video file missing on disk: {video_path}")

    with side_col:
        if stats:
            st.markdown("**Top Speeds — this window**")
            for p in stats.get('players', [])[:8]:
                conf = p.get('top_speed_confidence', 'high')
                speed_str = f"{p.get('top_speed_kmh', 0):.1f} km/h"
                team_display = _cv_team_label(p.get('team'), st.session_state.get('team_a', 'Team A'),
                                               st.session_state.get('team_b', 'Team B'), team_mapping)
                player_label = f"P{p.get('player_id')} ({team_display})"
                if conf == 'high':
                    st.write(f"**{player_label}** — **{speed_str}** ✅ high confidence")
                else:
                    st.caption(f"{player_label} — {speed_str} ⚠️ {conf} confidence")

            st.markdown("**Window Stats**")
            team_res = stats.get('team_resolution', {})
            calib = stats.get('calibration', {})
            ball = stats.get('ball', {})
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                metric_card(st, "Team resolution", f"{team_res.get('resolution_rate_pct', 'N/A')}%",
                            "The percentage of detected player tracks this window's team-classification "
                            "step could confidently assign to Team 1 or Team 2 by jersey color, versus "
                            "left unresolved (e.g. a referee, a brief partial-frame detection, or a kit "
                            "color too close to call).")
                metric_card(st, "Ball detection rate", f"{ball.get('final_detection_rate_pct', 'N/A')}%",
                            "The percentage of this window's frames where the ball was located — either "
                            "detected directly or filled in by the fallback detector/interpolation when "
                            "the primary detector missed it.")
            with wcol2:
                metric_card(st, "Calibration confidence", calib.get('mean_confidence', 'N/A'),
                            "How confidently the pitch-calibration model located the 48 pitch reference "
                            "points (corners, box lines, center circle, etc.) it uses to map broadcast "
                            "pixels onto real pitch coordinates for this window's frames.")
                metric_card(st, "Unique Tracking IDs", team_res.get('total_tracked_players', 'N/A'),
                            "This counts every distinct tracking ID our model assigned during the window, "
                            "including the same player being re-identified multiple times after being "
                            "briefly occluded or leaving/re-entering frame. It is not a headcount — a "
                            "real match has ~22 players on the pitch at once.")
                metric_card(st, "Ball tracking", "Dual-model",
                            "A fast primary detector, backed by a slower, higher-accuracy fallback for "
                            "the frames it misses.")
            with wcol1:
                metric_card(st, "Pitch reference points", str(len(PITCH_REFERENCE_POINTS)),
                            "Every frame is calibrated against 48 fixed landmarks on the pitch — "
                            "corners, penalty spots, the centre circle, and other line intersections — "
                            "which is what makes it possible to convert a player's position in the "
                            "video into a real position on the pitch.")
                metric_card(st, "Automation", "Fully automated",
                            "No manual annotation at run time — every match is processed the same way, "
                            "start to finish.")

            with st.expander(f"View all {len(PITCH_REFERENCE_POINTS)} pitch reference points"):
                st.caption(
                    "Real-world pitch coordinates (metres) the calibration model is trained to "
                    "locate, from a 105×68m pitch with the origin at one corner. 12 of these "
                    "correspond to a standard, independently verifiable landmark (a corner, a box "
                    "corner, the centre circle/halfway intersection); the other 36 are additional "
                    "correspondence points the technical report describes as labeled by visual "
                    "inspection during model training, with no published name for each one beyond "
                    "its coordinate — shown here honestly rather than with an invented name."
                )
                ref_rows = [
                    {
                        "Index": i,
                        "World coordinate (x, y) — metres": f"({x:.2f}, {y:.2f})",
                        "Landmark": name if name else "—",
                    }
                    for i, (x, y, name) in sorted(PITCH_REFERENCE_POINTS.items())
                ]
                st.dataframe(ref_rows, hide_index=True, use_container_width=True)
                st.caption(
                    "Note: indices 8 and 32 currently share the identical coordinate (16.50, 0.00). "
                    "The technical report (Section 2.4.6) documents this exact symptom — two indices "
                    "assigned one coordinate — as a sign of an uncorrected mapping error elsewhere in "
                    "this table; this specific pair does not appear to have been fixed yet."
                )
        else:
            st.info("Stats file not yet available on disk.")

    st.markdown("---")
    st.markdown("**Tactical Events — in this window**")
    st.write(
        "These are individual high-value moments our tracking detected in this window — a player "
        "breaking away with speed, finding open space, or a coordinated press closing down the ball. "
        "`score` is this pipeline's own relative-significance ranking within the window (higher = more "
        "tactically significant) — it isn't a standardized industry metric — and `metric` is the real "
        "measurement behind that score (a speed, a distance, a count of pressers)."
    )
    with st.expander("Event type glossary (all 10)"):
        for event_type, definition in TACTICAL_EVENT_GLOSSARY.items():
            st.markdown(f"**{event_type.replace('_', ' ').title()}** — {definition}")

    if stats:
        highlights = stats.get('tactical_events', {}).get('highlights', [])
        if highlights:
            for h in highlights[:10]:
                st.write(f"- **{str(h.get('type', '')).upper()}** — Player {h.get('player_id')} · "
                         f"score {h.get('score')} · {h.get('metric', '')}")

            team_a_name = st.session_state.get('team_a', 'Team A')
            team_b_name = st.session_state.get('team_b', 'Team B')
            observations = generate_cv_observations(stats, team_a_name, team_b_name, st.session_state.get('cv_team_mapping'))
            if observations:
                st.markdown("**Observations — specific to this match**")
                for obs in observations:
                    st.write(f"- {obs}")
        else:
            st.caption("No tactical events detected in this window.")

    _render_corner_kicks_section(stats)

def _render_corner_kicks_section(reference_stats):
    """Manual corner-kick marking + team-shape/metrics, reusing the CURRENT
    match's own already-confirmed team_resolution.team_colors_bgr as the
    reference for resolving each corner's independently-clustered team1/
    team2 labels back to team_a/team_b (see
    corner_kicks.resolve_corner_team_mapping's docstring for why this
    can't be assumed constant across separate CV runs - confirmed to
    actually flip on one of the three real corners this feature was built
    and verified against).

    A mark here only has a real team-shape/metrics render once its window
    has been through BOTH run_cv_analysis.py (with --no-ball-fallback,
    since none of this needs ball position - see that flag's own
    docstring) AND reconstruct_positions.py (see that script's docstring
    for why a second, offline step is needed at all). Marking a corner
    here does NOT trigger either of those - this only records/reads
    metadata and renders from data that already exists on disk. A mark
    with no player_positions.json yet shows an honest "not processed yet"
    message, never a broken render."""
    st.markdown("---")
    st.markdown("**⚽ Corner Kicks**")
    st.caption(
        "Manually mark a corner-kick window within this match and see each team's real tracked "
        "shape (never a role/marking assignment - just measured positions and honest geometry)."
    )

    source, key = _get_active_match_identity()
    if not source:
        st.info("This match doesn't have a saved identity yet, so corner marks can't be saved across reloads.")
        return

    marks = ck.load_marks(CACHE_DIR, key, CURATED_MATCHES_DIR)
    team_a = st.session_state.get('team_a', 'Team A')
    team_b = st.session_state.get('team_b', 'Team B')
    team_name = {"team_a": team_a, "team_b": team_b}

    if marks:
        labels = [
            f"{m['timestamp_label']} — {team_name.get(m['attacking_team'], m['attacking_team'])} attacking"
            for m in marks
        ]
        chosen_label = st.selectbox("Marked corners:", labels, key="corner_kick_select")
        mark = marks[labels.index(chosen_label)]

        positions = ck.load_player_positions(CV_PIPELINE_DIR, mark.get("cv_output_dir"))
        if positions is None:
            st.warning(
                f"'{mark['timestamp_label']}' is marked but not processed yet — its CV output "
                "folder hasn't been attached, or hasn't been through reconstruct_positions.py. "
                "No render to show until that's done."
            )
        else:
            reference_colors = (reference_stats or {}).get('team_resolution', {}).get('team_colors_bgr')
            if not reference_colors:
                st.warning("This match's own team colors aren't available, so this corner's team1/team2 can't be reliably matched to the real teams.")
            else:
                corner_mapping = ck.resolve_corner_team_mapping(positions, reference_colors)
                attacking_label = team_name.get(mark["attacking_team"], mark["attacking_team"])
                defending_label = team_name.get(mark["defending_team"], mark["defending_team"])

                shape_dir = CACHE_DIR / "corner_shapes"
                shape_dir.mkdir(parents=True, exist_ok=True)
                # .avi (XVID), not .mp4 - same reason every other CV-rendered
                # video in this app needs _ensure_browser_playable_video's
                # transcode: confirmed live that cv2.VideoWriter's raw output
                # isn't reliably playable by st.video() directly.
                shape_path = shape_dir / f"{ck.safe_filename_part(key)}_{ck.safe_filename_part(mark['id'])}.avi"
                if not shape_path.exists():
                    with st.spinner("Rendering team-shape video (one-time, cached after this)..."):
                        ck.render_team_shape_video(
                            positions, corner_mapping, mark["attacking_team"], mark["defending_team"],
                            attacking_label, defending_label, shape_path,
                        )
                playable_shape_path = _ensure_browser_playable_video(shape_path)
                if playable_shape_path and playable_shape_path.exists():
                    st.video(str(playable_shape_path))
                else:
                    st.warning("Couldn't prepare the team-shape video for in-browser playback.")
                st.caption(
                    f"Top-down view, real tracked positions only. {attacking_label} (orange, attacking) "
                    f"and {defending_label} (red, defending) are each connected by a convex hull — the "
                    "simplest non-crossing outline around that team's outfield players, not a tactical "
                    "role assignment of any kind."
                )

                metrics = ck.compute_corner_metrics(positions, corner_mapping, mark["attacking_team"], mark["defending_team"])
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    metric_card(st, f"{defending_label}'s last defender", f"{metrics['last_defender_distance_m']} m",
                                "Average, across every tracked frame of this window, of the distance from "
                                "the deepest defender to their own goal line — how high or deep the "
                                "defensive line was set. This is a window average, not a single freeze-"
                                "frame at the instant of delivery: this window's ball isn't tracked "
                                "closely enough (ball detection was intentionally skipped here — none of "
                                "this feature uses it) to isolate that exact moment, so it also reflects "
                                "the moments just before and after delivery.")
                with mcol2:
                    metric_card(st, "Compactness (both teams)",
                                f"{attacking_label} {metrics['attacking_compactness_m']} m / {defending_label} {metrics['defending_compactness_m']} m",
                                "Average pairwise distance between a team's own outfield players — how "
                                "spread out (higher) or tight (lower) their shape was, averaged the same "
                                "way as the last-defender distance above.")
                with mcol3:
                    if metrics['marking_distances']:
                        closest = metrics['marking_distances'][0]
                        metric_card(st, "Closest marking distance", f"P{closest['player_id']}: {closest['distance_m']} m",
                                    "For every defender, the average distance (across this window) to "
                                    "their nearest attacker — a real nearest-neighbor calculation, not a "
                                    "claim about who is tactically assigned to mark whom. Showing the "
                                    "closest pairing here; see the full per-defender list below.")
                with st.expander(f"All {len(metrics['marking_distances'])} defenders' marking distances"):
                    st.dataframe(
                        [{"Defender": f"P{d['player_id']}", "Distance to nearest attacker (m, avg)": d['distance_m']}
                         for d in metrics['marking_distances']],
                        hide_index=True, use_container_width=True,
                    )

                st.markdown("**This corner's own rendered outputs**")
                st.caption(
                    "Movement trails and per-player space control, already computed for this exact "
                    "window by the same CV pipeline (reused as-is, not recomputed here)."
                )
                other_col1, other_col2 = st.columns(2)
                for col, fname, label in ((other_col1, 'output6.avi', 'Movement Trails'), (other_col2, 'output3.avi', 'Per-Player Tactical Map (Voronoi)')):
                    with col:
                        st.caption(label)
                        vid_path = CV_PIPELINE_DIR / "output_videos" / mark["cv_output_dir"] / fname
                        if vid_path.exists():
                            playable = _ensure_browser_playable_video(vid_path)
                            if playable and playable.exists():
                                st.video(str(playable))
                            else:
                                st.caption("Couldn't prepare this render for in-browser playback.")
                        else:
                            st.caption("Not available for this corner.")

        if st.button("🗑️ Delete this mark", key="corner_kick_delete"):
            ck.delete_mark(CACHE_DIR, key, mark["id"])
            st.rerun()

    with st.expander("+ Mark a new corner"):
        st.caption(
            "Records the mark's metadata only — it does not launch CV processing. A newly-marked "
            "corner needs its window run through the CV pipeline (with --no-ball-fallback) and then "
            "reconstruct_positions.py before a team-shape render or metrics can appear here; until "
            "then it will honestly show as not yet processed."
        )
        new_label = st.text_input("Timestamp label (e.g. '3:06-3:15'):", key="corner_new_label")
        new_attacking = st.selectbox("Attacking team:", [team_a, team_b], key="corner_new_attacking")
        new_output_dir = st.text_input(
            "CV output folder name (optional — leave blank until processed):",
            key="corner_new_output_dir",
            help="The folder name under cv_pipeline/output_videos/ for this corner's already-completed CV + reconstruction run, if it exists yet.",
        )
        if st.button("Save mark", key="corner_new_save"):
            if not new_label.strip():
                st.error("Enter a timestamp label first.")
            else:
                attacking_token = "team_a" if new_attacking == team_a else "team_b"
                defending_token = "team_b" if attacking_token == "team_a" else "team_a"
                mark_id = ck.safe_filename_part(new_label.strip())
                ck.upsert_mark(CACHE_DIR, key, mark_id, new_label.strip(), attacking_token, defending_token,
                                new_output_dir.strip() or None)
                st.success(f"Saved mark '{new_label.strip()}'.")
                st.rerun()

def render_cv_deep_analysis_tab():
    st.subheader("🎬 CV Deep Analysis")
    st.caption(
        "Full-pipeline computer vision analysis of the match's highest-momentum window — auto-selected from "
        "the per-minute momentum data driving the Segment Momentum chart, no manual scrubbing needed."
    )

    seg_ts = st.session_state.get('cv_segment_timestamp')
    seg_score = st.session_state.get('cv_segment_momentum_score')

    if seg_ts is None:
        st.info(
            "No CV analysis window was identified for this match (this happens if the momentum-window "
            "extraction step didn't run - e.g. a Demo Mode CSV was loaded instead of a live video)."
        )
        return

    banner_col1, banner_col2 = st.columns([3, 1])
    with banner_col1:
        # Streamlit's stHorizontalBlock stretches every column to the row's
        # tallest sibling by default (the Momentum Score metric card, ~92px),
        # but this single line of text is only ~20px tall - left top-aligned,
        # that mismatch left a real ~70px dead zone below the text before the
        # divider (measured directly, not guessed). Vertically centering the
        # text within the stretched column height fixes the actual cause
        # instead of just shrinking the gap after it with negative margin.
        st.markdown(
            '<div style="display:flex;align-items:center;height:100%;min-height:76px;">'
            f'<span><b>Auto-selected window:</b> <code>{seg_ts}</code> — the highest-momentum minute in this match segment.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with banner_col2:
        momentum_help = cb.substitute_team_tokens(
            MOMENTUM_SCORE_HELP, st.session_state.get('team_a', 'Team A'), st.session_state.get('team_b', 'Team B')
        )
        metric_card(st, "Momentum Score", seg_score if seg_score is not None else "N/A", momentum_help)

    st.caption(
        "This window was chosen because it's the match's highest-momentum minute — the CV pipeline "
        "currently analyzes one window per match rather than the full 90 minutes, to keep processing "
        "time practical on the hardware available. The main cost driver is the ball-tracking fallback "
        "detector: GPU acceleration measured a 65–70× speedup on that stage specifically (Section 10.7 "
        "of the technical report), so with a larger GPU budget the same pipeline could be extended "
        "toward the full match — that hasn't been measured end-to-end at this point, just the per-stage "
        "speedup that would make it practical."
    )

    st.markdown("---")

    cv_output_dir = st.session_state.get('cv_job_output_dir')
    if not cv_output_dir:
        st.warning(
            "CV pipeline job was not launched for this match (the launch step may have failed non-fatally, "
            "or this match was loaded from history/demo mode)."
        )
        return

    status = get_cv_job_status_safe(cv_output_dir)
    job_status = status.get('status', 'not_started')

    if job_status in ('not_started', 'running'):
        render_cv_processing_state(status, cv_output_dir)
    elif job_status == 'complete':
        # PART 3: opportunistic write-through - the CV job runs detached, so
        # there's no callback for "it just finished"; the first time this tab
        # is viewed after completion is when we find out, and that's when we
        # record it. video_hash is only set for a real (non-curated, non-demo)
        # upload, so this can never write into an unrelated cache entry.
        if st.session_state.get('video_hash'):
            _cache_upsert(
                st.session_state.video_hash,
                cv_output_dir=cv_output_dir,
                cv_match_name=st.session_state.get('cv_job_match_name'),
                cv_completed_at=status.get('updated_at'),
            )
            # Job finished - it's no longer "in flight", so drop it from the
            # active-job registry (harmless if it was never registered, e.g.
            # a resumed/reused job).
            _unregister_active_job(st.session_state.video_hash)
        render_cv_completed_state(status, cv_output_dir)
    elif job_status == 'error':
        if st.session_state.get('video_hash'):
            _unregister_active_job(st.session_state.video_hash)
        st.error(f"CV analysis failed: {status.get('message', 'Unknown error')}")
    else:
        st.warning(f"Unrecognized CV job status: {job_status}")

def _get_active_match_identity():
    """(source, key) for the currently-active match, for training-plan
    persistence - reuses the same identity Phase C's team-mapping
    confirmation already writes back through (_activate_match_bundle's
    active_match_source/active_match_key), falling back to a video_hash-
    keyed cache entry for a fresh (not-yet-cached) live upload. Returns
    (None, None) when this match has no stable identity at all (e.g. Demo
    Mode CSV) - callers must handle that by keeping edits session-only."""
    source = st.session_state.get('active_match_source')
    key = st.session_state.get('active_match_key')
    if source and key:
        return source, key
    if st.session_state.get('video_hash'):
        return "cache", st.session_state.video_hash
    return None, None

def render_training_plan_tab():
    st.subheader("🏋️ Training Plan")
    st.caption(
        "A 7-day team training schedule generated from this match's real tactical patterns, "
        "plus individual player plans from the CV pipeline's tracked physical data."
    )

    source, key = _get_active_match_identity()
    if not source:
        st.info(
            "This match doesn't have a saved identity yet (it isn't a cached upload or an Instant "
            "Demo match), so a generated plan can't be persisted across reloads — you can still "
            "generate and edit one for this session."
        )

    identity = (source, key)
    if st.session_state.get('training_plan_draft_key') != identity:
        st.session_state.training_plan_draft = tp.load_training_plan(source, key, CURATED_MATCHES_DIR, CACHE_DIR) if source else None
        st.session_state.training_plan_draft_key = identity

    plan = st.session_state.training_plan_draft

    if not plan:
        if st.button("⚡ Generate Training Plan"):
            if not valid_keys:
                st.error("No Gemini API key configured.")
            else:
                with st.spinner("Generating team + player training plans from real match data..."):
                    # ai_report stores literal {TEAM_A}/{TEAM_B} tokens (see Step
                    # 3's writing_prompt) - substitute real names before handing it
                    # to another Gemini prompt as free-text context, or that
                    # prompt sees confusing placeholder tokens instead of names.
                    team_plan = tp.generate_team_plan(
                        st.session_state.raw_data,
                        cb.substitute_team_tokens(st.session_state.ai_report, st.session_state.team_a, st.session_state.team_b),
                        st.session_state.team_a, st.session_state.team_b, valid_keys[0],
                    )
                    player_plan = None
                    cv_insights = None
                    cv_output_dir = st.session_state.get('cv_job_output_dir')
                    if cv_output_dir:
                        cv_status = get_cv_job_status_safe(cv_output_dir)
                        if cv_status.get('status') == 'complete' and cv_status.get('stats_file'):
                            resolved = _resolve_cv_path(cv_status['stats_file'])
                            if resolved.exists():
                                with open(resolved, 'r') as f:
                                    stats_json = json.load(f)
                                player_plan = tp.generate_player_plan(
                                    stats_json, st.session_state.team_a, st.session_state.team_b,
                                    st.session_state.get('cv_team_mapping'), _cv_team_label, valid_keys[0],
                                )
                                # Second, additive layer (grounded in tactical-event
                                # highlights + tracking-coverage - see
                                # generate_cv_insights' docstring for why only these
                                # two of five originally-considered CV signals are
                                # used) - never replaces player_plan above.
                                if player_plan:
                                    cv_insights = tp.generate_cv_insights(
                                        stats_json, st.session_state.team_a, st.session_state.team_b,
                                        st.session_state.get('cv_team_mapping'), _cv_team_label,
                                        player_plan['players'], valid_keys[0],
                                    )
                if team_plan is None:
                    st.error("Failed to generate the team plan after multiple attempts. Please try again.")
                else:
                    new_plan = {"team_plan": team_plan, "player_plan": player_plan, "cv_insights": cv_insights}
                    st.session_state.training_plan_draft = new_plan
                    if source:
                        tp.save_training_plan(source, key, new_plan, CURATED_MATCHES_DIR, CACHE_DIR)
                    st.rerun()
        return

    sub_team, sub_player = st.tabs(["Team Plan", "Player Plans"])
    with sub_team:
        _render_team_plan_subtab(source, key)
    with sub_player:
        _render_player_plan_subtab(source, key)

def _render_team_plan_subtab(source, key):
    team_plan = st.session_state.training_plan_draft.get("team_plan")
    if not team_plan:
        st.info("No team plan available.")
        return

    cv_insights = st.session_state.training_plan_draft.get("cv_insights") or {}
    team_insights = cv_insights.get("team_insights") or []
    st.components.v1.html(
        tp.render_team_calendar_html(team_plan, team_insights), height=560 + (170 * len(team_insights)), scrolling=True
    )

    # Same fix as _render_player_plan_subtab's match_prefix: these widget
    # keys used to be keyed only by day index (i)/drill index (j), with no
    # match identity at all - switching matches (or a Reset+regenerate)
    # could leave a text_input showing session-state from a PREVIOUS match's
    # Monday instead of the freshly-generated one, since Streamlit only
    # honors a widget's value= the first time that exact key ever appears.
    match_prefix = f"{source}_{key}"

    st.markdown("---")
    st.markdown("**✏️ Edit this week's plan**")
    days = team_plan.get("days", [])
    day_names = [d.get("day", f"Day {i}") for i, d in enumerate(days)]

    swap_col1, swap_col2, swap_col3 = st.columns([2, 2, 1])
    with swap_col1:
        swap_a = st.selectbox("Swap day:", day_names, key=f"tp_swap_a_{match_prefix}")
    with swap_col2:
        swap_b = st.selectbox("with day:", day_names, key=f"tp_swap_b_{match_prefix}", index=min(1, len(day_names) - 1))
    with swap_col3:
        st.write("")
        if st.button("🔁 Swap", key=f"tp_swap_btn_{match_prefix}") and swap_a != swap_b:
            ia, ib = day_names.index(swap_a), day_names.index(swap_b)
            # Swap CONTENT, not list position, so the week always renders
            # Monday-first regardless of which two days were swapped.
            days[ia]["focus_label"], days[ib]["focus_label"] = days[ib]["focus_label"], days[ia]["focus_label"]
            days[ia]["focus_category"], days[ib]["focus_category"] = days[ib]["focus_category"], days[ia]["focus_category"]
            days[ia]["drills"], days[ib]["drills"] = days[ib]["drills"], days[ia]["drills"]
            days[ia]["why_stat"], days[ib]["why_stat"] = days[ib]["why_stat"], days[ia]["why_stat"]
            st.rerun()

    for i, d in enumerate(days):
        day_prefix = f"{match_prefix}_{d.get('day', i)}"
        with st.expander(f"Edit {d.get('day', f'Day {i}')}"):
            d['focus_label'] = st.text_input("Focus label:", value=d.get('focus_label', ''), key=f"tp_focus_{day_prefix}")
            cats = tp.FOCUS_CATEGORIES
            d['focus_category'] = st.selectbox(
                "Category:", cats, index=cats.index(d.get('focus_category')) if d.get('focus_category') in cats else 0,
                key=f"tp_cat_{day_prefix}",
            )
            d['why_stat'] = st.text_area("Why (real stat):", value=d.get('why_stat', ''), key=f"tp_why_{day_prefix}", height=68)

            drills = d.get('drills', [])
            remove_idx = None
            for j, dr in enumerate(drills):
                dc1, dc2, dc3, dc4 = st.columns([3, 1, 4, 1])
                dr['title'] = dc1.text_input("Drill title", value=dr.get('title', ''), key=f"tp_dt_{day_prefix}_{j}", label_visibility="collapsed")
                dr['duration_min'] = dc2.number_input("min", value=int(dr.get('duration_min', 15) or 15), key=f"tp_dd_{day_prefix}_{j}", label_visibility="collapsed", min_value=0, step=5)
                dr['note'] = dc3.text_input("Note", value=dr.get('note', ''), key=f"tp_dn_{day_prefix}_{j}", label_visibility="collapsed")
                if dc4.button("🗑️", key=f"tp_drm_{day_prefix}_{j}"):
                    remove_idx = j
            if remove_idx is not None:
                drills.pop(remove_idx)
                st.rerun()
            if st.button("+ Add drill", key=f"tp_dadd_{day_prefix}"):
                drills.append({"title": "New drill", "duration_min": 15, "note": ""})
                st.rerun()
            d['drills'] = drills

    team_plan['days'] = days
    # Per-field edit tracking (not per-day) - compares each day's tracked
    # fields against the pristine snapshot captured at generation time, so
    # e.g. editing only a drill's duration doesn't falsely mark that day's
    # why_stat as unverified. Recomputed every rerun so it's already correct
    # in the draft by the time "Save changes" persists it - no separate
    # tracking mechanism.
    tp.recompute_edited_fields(team_plan['days'], team_plan.get('_original_days', []), tp.TEAM_DAY_TRACKED_FIELDS)
    st.session_state.training_plan_draft['team_plan'] = team_plan
    _render_training_plan_save_reset("tp_team", source, key)

def _render_player_plan_subtab(source, key):
    player_plan = st.session_state.training_plan_draft.get("player_plan")
    if not player_plan:
        st.info(
            "No player plan available — this needs the CV Deep Analysis job to be complete for "
            "this match (player physical stats come from that pipeline, not the tactical data)."
        )
        return

    players = player_plan.get("players", [])
    if not players:
        st.info("No players with sufficient tracking confidence in this window.")
        return

    # match_prefix scopes every widget key below to this specific match, not
    # just this render - without it, switching matches (or a player landing
    # at the same list position as a previously-viewed player in another
    # match) could show a text_input/text_area's STALE session-state value
    # from a completely different player instead of this one's real,
    # freshly-generated title/note/tag. Streamlit only uses a widget's
    # value= argument the very first time that key appears; once a key has
    # session-state, value= is ignored on every later rerun.
    match_prefix = f"{source}_{key}"

    labels = [f"P{p['player_id']} ({p.get('team_label', '?')})" for p in players]
    chosen_label = st.selectbox("Choose a player:", labels, key=f"tp_player_select_{match_prefix}")
    chosen_idx = labels.index(chosen_label)
    player = players[chosen_idx]
    # Player identity for widget keys - NOT chosen_idx (list position), since
    # that's the same staleness risk one level up: the player at a given
    # position can differ between generations/matches even though the index
    # doesn't change.
    player_prefix = f"{match_prefix}_{player['player_id']}"

    cv_insights = st.session_state.training_plan_draft.get("cv_insights") or {}
    player_insights = (cv_insights.get("player_insights") or {}).get(str(player['player_id'])) or []
    st.components.v1.html(
        tp.render_player_card_html(player, player_insights), height=440 + (170 * len(player_insights)), scrolling=True
    )

    st.markdown("---")
    st.markdown(f"**✏️ Edit {chosen_label}'s plan**")
    sessions = player.get('sessions', [])
    remove_idx = None
    for j, s in enumerate(sessions):
        # Keyed by day, not just j: removing an earlier session shifts every
        # later session's list position (j) down by one, which would
        # otherwise make it inherit the widget state left behind by whatever
        # used to occupy that position. j is kept only as a tiebreaker for
        # the rare case of two sessions sharing a day (e.g. "+ Add session"'s
        # hardcoded "Mon" default alongside an existing Monday session).
        session_key = f"{player_prefix}_{s.get('day', 'session')}_{j}"
        with st.expander(f"Edit {s.get('day', f'Session {j}')}"):
            s['title'] = st.text_input("Title:", value=s.get('title', ''), key=f"tp_ps_title_{session_key}")
            s['note'] = st.text_area("Note:", value=s.get('note', ''), key=f"tp_ps_note_{session_key}", height=68)
            s['tag'] = st.text_input("Tag:", value=s.get('tag', ''), key=f"tp_ps_tag_{session_key}")
            if st.button("🗑️ Remove this session", key=f"tp_ps_rm_{session_key}"):
                remove_idx = j
    if remove_idx is not None:
        sessions.pop(remove_idx)
        st.rerun()
    if st.button("+ Add session", key=f"tp_ps_add_{player_prefix}"):
        sessions.append({"day": "Mon", "title": "New session", "note": "", "tag": ""})
        st.rerun()

    original_players = player_plan.get('_original_players', [])
    original_sessions = original_players[chosen_idx].get('sessions', []) if chosen_idx < len(original_players) else []
    tp.recompute_edited_fields(sessions, original_sessions, tp.PLAYER_SESSION_TRACKED_FIELDS)
    player['sessions'] = sessions
    players[chosen_idx] = player
    player_plan['players'] = players
    st.session_state.training_plan_draft['player_plan'] = player_plan
    _render_training_plan_save_reset("tp_player", source, key)

def _render_training_plan_save_reset(key_prefix, source, key):
    st.caption("Resetting regenerates BOTH the team and player plans from scratch (they're stored together).")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("💾 Save changes", key=f"{key_prefix}_save"):
            if source:
                tp.save_training_plan(source, key, st.session_state.training_plan_draft, CURATED_MATCHES_DIR, CACHE_DIR)
                st.success("Saved.")
            else:
                st.warning("This match has no saved identity — changes stay for this session only.")
    with bcol2:
        if st.button("↺ Reset to AI-generated plan", key=f"{key_prefix}_reset"):
            if source:
                tp.delete_training_plan(source, key, CURATED_MATCHES_DIR, CACHE_DIR)
            st.session_state.training_plan_draft = None
            st.rerun()

def extract_video_segment(source_path, start_sec, end_sec, output_path):
    """Cuts [start_sec, end_sec) out of source_path and writes it to output_path.

    Uses an ffmpeg subprocess with stream-copy (no re-encode) instead of
    moviepy's write_videofile. moviepy relays every decoded frame through
    Python/numpy to re-encode it, which holds the GIL for the whole cut and
    fully re-decodes the source - concurrent ThreadPoolExecutor workers doing
    this in parallel serialize on the GIL and thrash the disk reading the
    same master file, making concurrency slower than running one at a time
    (measured: ~30s/chunk sequential vs 300-750s/chunk at 6 concurrent
    workers). ffmpeg -ss before -i does a fast keyframe seek and -c copy
    remuxes without decoding, so it runs as a genuine OS-level subprocess
    with no GIL involvement, and only reads the bytes it needs.

    Stream-copy cuts land on the nearest keyframe at/before start_sec (can't
    cut a compressed stream mid-GOP without decoding), so the slice boundary
    can drift by up to one keyframe interval. That's an acceptable trade-off
    for minute-granularity tactical analysis; if frame-accurate cuts are ever
    needed, fall through to the re-encode path below.
    """
    duration = end_sec - start_sec
    copy_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(source_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(output_path),
    ]
    result = subprocess.run(copy_cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        # Stream-copy can fail to produce a valid cut right at the very start
        # of the file (no preceding keyframe) - re-encode as a fallback.
        encode_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(source_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-an",
            str(output_path),
        ]
        result = subprocess.run(encode_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg slicing failed: {result.stderr[-1000:]}")

def process_single_minute(start_min, duration_sec, temp_video_path, api_key, master_team_a_color, master_team_b_color, chunk_dir):
    client = genai.Client(api_key=api_key)
    end_min = min(start_min + 1, duration_sec / 60.0)

    # Was the bare relative filename f"temp_chunk_{start_min}.mp4" - written
    # to the process's current working directory. On Streamlit Community
    # Cloud that's the git-cloned repo root, which is mounted read-only (the
    # same failure mode diagnosed for ChromaDB/CACHE_DIR), so this write
    # would fail on every single live upload attempt, from the very first
    # minute - independent of anything else broken in that flow. chunk_dir
    # is a writable, per-video temp directory the caller creates once.
    chunk_path = str(chunk_dir / f"temp_chunk_{start_min}.mp4")
    try:
        extract_video_segment(temp_video_path, start_min * 60, end_min * 60, chunk_path)
    except Exception as e:
        return {"error": f"Slicing failed: {e}", "minute": start_min}

    try:
        video_file = client.files.upload(file=chunk_path)
        timeout_counter = 0
        max_checks = 30 
        while video_file.state.name == "PROCESSING":
            if timeout_counter >= max_checks:
                try: client.files.delete(name=video_file.name)
                except: pass
                return {"error": "Google Cloud processing timed out.", "minute": start_min}
            time.sleep(3)
            video_file = client.files.get(name=video_file.name)
            timeout_counter += 1
            
        if video_file.state.name == "FAILED":
            return {"error": "Cloud processing failed.", "minute": start_min}
    except Exception as e:
         return {"error": f"Upload failed: {e}", "minute": start_min}

    if master_team_a_color is None:
        color_instruction = "1. Identify the two teams by their primary kit colors. Label them Team A and Team B."
    else:
        color_instruction = f"1. CRITICAL: Team A kit color is exactly '{master_team_a_color}' and Team B kit color is exactly '{master_team_b_color}'."

    data_prompt = f"""
    Watch this 1-minute soccer match clip.
    {color_instruction}
    2. Output EXACTLY 1 JSON object describing the tactics for this specific 60-second window. Do not output an array.
    
    POSSESSION RULE (MAJORITY RULES): Assign 'team_in_possession' to the team that held the ball for the absolute majority of this 60-second interval. Do NOT award dual possession. 
    
    HIGH-VALUE STEAL EXCEPTION: If the defending team steals the ball and registers a clear goal-scoring threat or shot inside the opponent's penalty box during this minute, they "steal" the minute. You MUST assign 'team_in_possession' to the counter-attacking team and assign them a 'fast_direct' tempo.
    
    ZONAL MAJORITY RULE: 'ball_zone' MUST strictly reflect where the ball spent the majority of the 60 seconds relative to the team in possession. If a team defends deep for 55 seconds and counters for 5 seconds, the zone is 'defensive_third'.
    
    FRACTIONAL ATTACKING TIME: You must output two integers (0-60) for 'team_a_attack_sec' and 'team_b_attack_sec' representing exactly how many literal seconds each team spent in the attacking third during this minute.
    
    CRITICAL VOCABULARY UPGRADE (GEGENPRESSING): If a team immediately swarms the ball high up the pitch after losing it, you MUST tag their 'pressing_trigger' as 'gegenpress' and their 'block_height' as 'high'. Do this EVEN IF the opponent bypasses the press and forces them to defend deep later in the minute. Do not fall for the "Recovery Illusion."
    
    CRITICAL VOCABULARY UPGRADE (TRANSITIONS): A 'counter_attack' is when a team sits deep, wins the ball, and breaks. A 'fast_vertical_transition' is when a team uses rapid, direct passing to bypass an opponent's high press. Use these tags correctly in 'transition_threat'.
    
    CRITICAL VOCABULARY UPGRADE (POSSESSION): If a team's primary goal is to pin the opponent in their own half (Zonal Possession), you MUST tag their 'attacking_tempo' as 'sustained_high_pressure', not just generic 'patient_possession'.
    
    PHASE 2 FIX - GEOMETRIC DEFINITION OF CENTRAL CHANNEL: The 'central_channel' is STRICTLY defined as the physical width of the 18-yard penalty box. If the ball is operating outside the width of the 18-yard box, you MUST classify the 'attacking_bias' as 'left_flank' or 'right_flank'. Do not fall for the broadcast camera illusion.
    
    EFFECTIVE PLAYING TIME INSTRUCTION: If the majority of the minute is spent dealing with an injury, a player walking to set up a corner, or extreme time-wasting, set BOTH teams' attacking tempo to "dead_ball_stoppage". 

    Global Variables:
    - "team_a_color": string
    - "team_b_color": string
    - "team_in_possession": color of the ONE team with primary ball control (Apply exception rule if needed)
    - "ball_zone": "defensive_third", "middle_third", or "attacking_third" (Where ball spent the majority of the minute)
    - "team_a_attack_sec": Integer (0-60)
    - "team_b_attack_sec": Integer (0-60)
    
    Team A Variables:
    - "team_a_pressing_intensity": Integer from 1 to 10
    - "team_a_block_height": "low", "mid", or "high"
    - "team_a_half_space_occupancy": Integer (0 to 4 players)
    - "team_a_vertical_compactness": "tight", "standard", or "stretched"
    - "team_a_build_up_shape": "3-2", "2-3", "4-2", or "3-box-3"
    - "team_a_attacking_tempo": "fast_direct", "patient_possession", "sustained_high_pressure", "none", or "dead_ball_stoppage"
    - "team_a_transition_threat": "counter_attack", "fast_vertical_transition", "sustained_build", or "none"
    - "team_a_striker_profile": "false_9", "target_man", or "channel_runner"
    - "team_a_fullback_role": "overlapping", "inverted", or "defensive"
    - "team_a_pressing_trigger": "gegenpress", "loss_of_possession", "backward_pass", "poor_touch", or "none"
    - "team_a_rest_defense_shape": "3-2", "2-3", or "unstructured"
    - "team_a_attacking_bias": "left_flank", "right_flank", or "central_channel"
    - "team_a_defensive_line_action": "drop_deep" or "step_up"
    
    Team B Variables:
    - "team_b_pressing_intensity": Integer from 1 to 10
    - "team_b_block_height": "low", "mid", or "high"
    - "team_b_half_space_occupancy": Integer (0 to 4 players)
    - "team_b_vertical_compactness": "tight", "standard", or "stretched"
    - "team_b_build_up_shape": "3-2", "2-3", "4-2", or "3-box-3"
    - "team_b_attacking_tempo": "fast_direct", "patient_possession", "sustained_high_pressure", "none", or "dead_ball_stoppage"
    - "team_b_transition_threat": "counter_attack", "fast_vertical_transition", "sustained_build", or "none"
    - "team_b_striker_profile": "false_9", "target_man", or "channel_runner"
    - "team_b_fullback_role": "overlapping", "inverted", or "defensive"
    - "team_b_pressing_trigger": "gegenpress", "loss_of_possession", "backward_pass", "poor_touch", or "none"
    - "team_b_rest_defense_shape": "3-2", "2-3", or "unstructured"
    - "team_b_attacking_bias": "left_flank", "right_flank", or "central_channel"
    - "team_b_defensive_line_action": "drop_deep" or "step_up"
    """
    
    max_retries = 3 
    minute_data = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[video_file, data_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            minute_data = json.loads(response.text)
            if isinstance(minute_data, list) and len(minute_data) > 0:
                minute_data = minute_data[0]
                
            minute_data["timestamp"] = f"{start_min:02d}:00-{(start_min+1):02d}:00"
            minute_data["_minute_index"] = start_min 
            break 
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3) 
                continue
            else:
                minute_data = {"error": "Parsing failed", "minute": start_min}
    
    try: client.files.delete(name=video_file.name)
    except: pass
    if os.path.exists(chunk_path): os.remove(chunk_path)
        
    return minute_data

# Maintenance-only, not a real user-facing feature - see _render_admin_panel's
# docstring. Gated behind a query param specifically so it never shows up in
# the normal landing-page flow a real user sees.
if st.query_params.get("admin") == "1":
    _render_admin_panel()

# ==========================================
# STEP 1: UI RENDER
# ==========================================
if st.session_state.step == 1:

    if 'show_demo_panel' not in st.session_state:
        st.session_state.show_demo_panel = False

    # Past Reports stays in the sidebar - not part of the mockup's landing page,
    # but existing functionality that must keep working.
    if len(st.session_state.history) > 0:
        with st.sidebar:
            st.subheader("📁 Past Reports")
            history_options = {"(Select a past report...)": -1}
            for i, h in enumerate(st.session_state.history):
                history_options[f"Match {i+1}: {h['team_a']} vs {h['team_b']}"] = i

            selected_history = st.selectbox("Load History", list(history_options.keys()))
            if history_options[selected_history] != -1:
                h_idx = history_options[selected_history]
                st.session_state.team_a = st.session_state.history[h_idx]['team_a']
                st.session_state.team_b = st.session_state.history[h_idx]['team_b']
                st.session_state.color_a = st.session_state.history[h_idx]['color_a']
                st.session_state.color_b = st.session_state.history[h_idx]['color_b']
                st.session_state.raw_data = st.session_state.history[h_idx]['raw_data']
                st.session_state.ai_report = st.session_state.history[h_idx]['ai_report']
                st.session_state.step = 3
                st.session_state.view_mode = 'dashboard'
                st.rerun()

    # --- PART 1: LANDING PAGE (mirrors dashboard_mockup_4.html #page-landing) ---
    landing_placeholder = st.empty()
    with landing_placeholder.container():
        st.markdown("""
        <div class="tac-landing-nav">
          <div class="brand">⚽ AI Tactical<span> Coach</span></div>
          <div class="nav-links"><span>How it works</span><span>Recruiter Demo</span><span>About</span></div>
        </div>
        <div class="tac-hero">
          <div class="tac-eyebrow"><span class="dot"></span> Computer vision + AI, grounded in measured data</div>
          <h1>Simple to use. <span class="hl">Built to win.</span></h1>
          <p class="lede">Upload a match video. Get a tactical dashboard, real player tracking
          pipelines, a threat model, a training plan, and a coach's report — all from one
          system, backed by computer vision.</p>
        </div>
        <div class="tac-constraint-label">Built under constraint</div>
        <div class="tac-constraint-wrap">
          <div class="tac-constraint-card">
            <div class="n">48 pitch reference points</div>
            <div class="l">Detected automatically in every frame to calibrate image-to-pitch coordinates.</div>
          </div>
          <div class="tac-constraint-card">
            <div class="n">5 distinct analytical outputs</div>
            <div class="l">Tracking, tactical events, stamina, per-player control, and team pitch-control — from one pipeline.</div>
          </div>
          <div class="tac-constraint-card">
            <div class="n">Dual-model ball tracking</div>
            <div class="l">A fast primary detector, backed by a slower, higher-accuracy fallback for the frames it misses.</div>
          </div>
          <div class="tac-constraint-card">
            <div class="n">Fully automated</div>
            <div class="l">No manual annotation at run time — every match is processed the same way, start to finish.</div>
          </div>
          <div class="tac-constraint-card">
            <div class="n">GPU-bound, not redesign-bound</div>
            <div class="l">The fallback ball detector measured a 65–70× speedup on GPU — more compute, not a rebuild, is what full-match analysis needs.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _, upload_col, _ = st.columns([1, 2, 1])
        with upload_col:
            with st.container(border=True):
                st.markdown("##### 📤 Drop a match video, or click to browse")
                st.warning(
                    "⏱️ **A full analysis run on this deployment's free tier is likely to fail outright, "
                    "not just run slowly.** Streamlit Community Cloud's free tier guarantees only ~1GB RAM "
                    "per app (bursting to ~3GB depending on load) — this pipeline's tracking/detection "
                    "stack (PyTorch, YOLO, OpenCV) commonly needs more than that on its own for a real "
                    "video, on top of a multi-hour CPU runtime for demanding footage (see the Methodology "
                    "page for measured numbers). Expect either a resource-limit error or a very long wait "
                    "with no guarantee the session survives it, keeping this tab open included — there's no "
                    "way to resume an interrupted run. For an immediate, complete, reliable walkthrough, use "
                    "**Instant Demo (Curated Matches)** below instead."
                )
                uploaded_video = st.file_uploader(
                    "Choose a video file (Max 2GB)", type=["mp4", "mov", "avi"], label_visibility="collapsed"
                )
                st.caption("MP4 · MOV · up to 45 min · requires an active Gemini API key")
                extract_btn = st.button("🚀 Analyze This Match", type="primary", use_container_width=True)

                st.markdown('<div class="tac-upload-divider">or</div>', unsafe_allow_html=True)

                if st.button("▶ See a sample report (Recruiter Demo)", use_container_width=True):
                    st.session_state.show_demo_panel = not st.session_state.show_demo_panel

                uploaded_csv = None
                demo_btn = False
                if st.session_state.show_demo_panel:
                    st.caption(
                        "Upload a pre-processed tactical CSV to instantly load a sample report "
                        "(bypasses the AI video processing phase)."
                    )
                    uploaded_csv = st.file_uploader("Upload Tactical CSV", type=["csv"], key="demo_csv_uploader")
                    demo_btn = st.button("Load Instant Demo Match", use_container_width=True)

                st.markdown('<div class="tac-upload-divider">or</div>', unsafe_allow_html=True)

                if st.button("⚡ Instant Demo (Curated Matches)", use_container_width=True):
                    st.session_state.show_instant_demo_panel = not st.session_state.get("show_instant_demo_panel", False)

                instant_demo_bundle = None
                if st.session_state.get("show_instant_demo_panel"):
                    instant_demo_items = _load_instant_demo_matches()
                    if not instant_demo_items:
                        st.info(
                            "No saved match is fully processed yet — a match needs both a completed "
                            "Gemini tactical dataset and a completed CV deep-analysis bundle on disk "
                            "before it can appear here."
                        )
                    else:
                        st.caption(
                            "Pre-processed real matches — jumps straight to the finished dashboard, "
                            "no live AI calls."
                        )
                        item_names = [it["display_name"] for it in instant_demo_items]
                        chosen_name = st.selectbox("Choose a saved match:", item_names, key="instant_demo_picker")
                        chosen_item = next(it for it in instant_demo_items if it["display_name"] == chosen_name)
                        if st.button("View Instant Demo", use_container_width=True, key="instant_demo_go"):
                            instant_demo_bundle = chosen_item["bundle"]

                        with st.expander("✏️ Rename this match / set real team names"):
                            renamed = st.text_input(
                                "Display name:", value=chosen_item["display_name"], key="rename_input",
                            )
                            st.caption(
                                "Display name is just the title shown in the match picker above — it's "
                                "cosmetic only. To make team names show up correctly *inside* the dashboard "
                                "and CV tab (e.g. player labels, chart legends), also set the two real team "
                                "names below — changing one does not change the other."
                            )
                            # Part 3: curated/Instant Demo bundles skip the live-upload
                            # flow's Step 2 team-mapping screen entirely, so this is
                            # where they get the same team_a/team_b naming. Defaults to
                            # whatever the bundle currently has (usually a color title
                            # like "White") - deliberately not guessed at automatically,
                            # since which real team wore which color isn't verifiable
                            # from the data alone.
                            rn_col1, rn_col2 = st.columns(2)
                            with rn_col1:
                                renamed_team_a = st.text_input(
                                    "Team A real name:", value=chosen_item["bundle"].get("team_a", ""), key="rename_team_a",
                                )
                            with rn_col2:
                                renamed_team_b = st.text_input(
                                    "Team B real name:", value=chosen_item["bundle"].get("team_b", ""), key="rename_team_b",
                                )
                            if st.button("Save Name", key="rename_save"):
                                _update_match_fields(chosen_item, display_name=renamed, team_a=renamed_team_a, team_b=renamed_team_b)
                                st.rerun()

        # Part 4.1: a real, live count - not hardcoded - reusing the exact same
        # "fully processed" criteria _load_instant_demo_matches() already
        # enforces (Gemini + CV both done, CV re-verified live off disk).
        live_clips_validated = len(_load_instant_demo_matches())
        st.markdown(f"""
        <div class="tac-proof-row">
          <div class="tac-proof-item"><div class="n">{live_clips_validated}</div><div class="l">Clips validated</div></div>
          <div class="tac-proof-item"><div class="n">7</div><div class="l">Leagues / broadcasts</div></div>
          <div class="tac-proof-item"><div class="n">~25 min</div><div class="l">Deep CV analysis</div></div>
          <div class="tac-proof-item"><div class="n">0</div><div class="l">Guessed stats</div></div>
        </div>
        """, unsafe_allow_html=True)

    # --- INSTANT DEMO EXECUTION (Part 2: curated matches, no live AI/CV calls) ---
    if instant_demo_bundle is not None:
        landing_placeholder.empty()
        _activate_match_bundle(instant_demo_bundle, source=chosen_item["source"], key=chosen_item["key"])
        st.rerun()

    # --- DEMO MODE EXECUTION ---
    if demo_btn:
        if uploaded_csv is None:
            st.error("⚠️ Please upload a CSV file first to use Demo Mode.")
        else:
            try:
                demo_df = pd.read_csv(uploaded_csv)
                st.session_state.raw_data = demo_df.to_dict('records')

                # Try to grab colors if they exist in the CSV
                if 'team_a_color' in demo_df.columns:
                    st.session_state.color_a = str(demo_df['team_a_color'].iloc[0]).title()
                    st.session_state.color_b = str(demo_df['team_b_color'].iloc[0]).title()
                else:
                    st.session_state.color_a = "Team A"
                    st.session_state.color_b = "Team B"

                st.session_state.video_hash = None
                st.session_state.gemini_cache_hit = False
                st.session_state.step = 2
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Error reading CSV file: {e}")

    # --- LIVE EXTRACTION EXECUTION ---
    if extract_btn:
        if not valid_keys:
            st.error("Please add your Gemini API Key in the configuration section.")
        elif uploaded_video is None:
            st.error("Please upload a video file.")
        else:
            # --- PART 3: PROCESSING CACHE - hash the upload BEFORE doing any
            # work, so an exact repeat of a file already processed (in full
            # or in part) never re-runs work that's already sitting on disk.
            # SHA-256 over the raw file bytes: cryptographically collision-
            # resistant (no realistic chance of two different videos hashing
            # equal), which is what "never serve wrong results for a
            # different video" needs here - this isn't defending against an
            # adversary, just against accidental misidentification.
            video_bytes = uploaded_video.read()
            video_hash = hashlib.sha256(video_bytes).hexdigest()
            st.session_state.video_hash = video_hash
            cached = _cache_get(video_hash)
            gemini_cached = bool(cached and cached.get("raw_data") and cached.get("ai_report"))
            cv_cached_valid = bool(cached and _cv_bundle_is_valid(cached.get("cv_output_dir")))
            st.session_state.gemini_cache_hit = gemini_cached

            if gemini_cached and cv_cached_valid:
                # Full match: identical experience to the Part 2 curated picker.
                landing_placeholder.empty()
                st.toast("This exact video was already fully processed — loading instantly from cache.")
                _activate_match_bundle(cached, source="cache", key=video_hash)
                st.rerun()

            landing_placeholder.empty()

            # --- PART 2: LOADING PAGE (mirrors dashboard_mockup_4.html #page-loading) ---
            loading_placeholder = st.empty()
            extraction_start_time = time.time()
            loading_quote = random.choice(MANAGER_QUOTES)
            log_entries = []  # plain-text lines, oldest first - real chunk outcomes only

            def _tick(pct, status_line, stages, completed_for_eta=None):
                elapsed_s = time.time() - extraction_start_time
                if completed_for_eta:
                    eta_s = (elapsed_s / completed_for_eta[0]) * max(0, completed_for_eta[1] - completed_for_eta[0])
                else:
                    eta_s = 0
                tail = log_entries[-5:]
                log_lines = [
                    (text, "current" if i == len(tail) - 1 else ("recent" if i == len(tail) - 2 else "old"))
                    for i, text in enumerate(tail)
                ]
                render_loading_screen(
                    loading_placeholder, pct=pct, elapsed_s=elapsed_s, eta_s=eta_s,
                    status_line=status_line, stages=stages, log_lines=log_lines, quote=loading_quote,
                )

            _tick(2, "Saving your video and scanning duration...", [
                ("Upload", "running", "active"), ("Team colors", "—", "pending"),
                ("Per-minute analysis", "—", "pending"), ("Threat model", "—", "pending"),
                ("Coach report", "—", "pending"),
            ])

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_video_path = tmp_file.name

            cap = cv2.VideoCapture(temp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            if fps > 0:
                duration_sec = frames / fps
                TOTAL_MINUTES = math.ceil(duration_sec / 60)
            else:
                TOTAL_MINUTES = 1
            cap.release()

            st.toast(f"Detected video length: {TOTAL_MINUTES} minutes.")

            log_entries.append(f"Video saved — {TOTAL_MINUTES} minute(s) detected")
            _tick(4, "Establishing team colors from minute 0...", [
                ("Upload", "done", "done"), ("Team colors", "running", "active"),
                ("Per-minute analysis", "—", "pending"), ("Threat model", "—", "pending"),
                ("Coach report", "—", "pending"),
            ])

            try:
                if gemini_cached:
                    # PART 3: partial cache hit (Gemini done, CV not) - reuse the
                    # cached dataset+report verbatim, skip every live Gemini call.
                    all_tactical_data = list(cached["raw_data"])
                    master_team_a_color = cached["color_a"]
                    master_team_b_color = cached["color_b"]
                    st.session_state.color_a = cached["color_a"]
                    st.session_state.color_b = cached["color_b"]
                    st.session_state.ai_report = cached["ai_report"]
                    log_entries.append(
                        f"Cached tactical dataset found for this exact video — "
                        f"{master_team_a_color} vs {master_team_b_color} — skipping AI analysis"
                    )
                    _tick(90, "Loaded cached tactical dataset — preparing CV analysis...", [
                        ("Upload", "done", "done"), ("Team colors", "done", "done"),
                        ("Per-minute analysis", f"{TOTAL_MINUTES} / {TOTAL_MINUTES}", "done"),
                        ("Threat model", "—", "pending"), ("Coach report", "done", "done"),
                    ])
                else:
                    all_tactical_data = []
                    master_team_a_color = None
                    master_team_b_color = None

                    chunk_dir = Path(tempfile.gettempdir()) / "tactical_scout_chunks" / st.session_state.video_hash
                    chunk_dir.mkdir(parents=True, exist_ok=True)

                    first_min_data = process_single_minute(0, duration_sec, temp_video_path, valid_keys[0], None, None, chunk_dir)

                    if "error" not in first_min_data:
                        master_team_a_color = first_min_data.get("team_a_color", "Team A")
                        master_team_b_color = first_min_data.get("team_b_color", "Team B")
                        st.session_state.color_a = str(master_team_a_color).title()
                        st.session_state.color_b = str(master_team_b_color).title()
                        all_tactical_data.append(first_min_data)
                        log_entries.append(f"Team colors identified — {master_team_a_color} vs {master_team_b_color}")
                    else:
                        log_entries.append("Team color detection failed on minute 0 — falling back to per-chunk identification")

                    completed_count = 1
                    _tick(round(completed_count / TOTAL_MINUTES * 100),
                          f"Reading per-minute tactical patterns — chunk {completed_count} of {TOTAL_MINUTES}", [
                        ("Upload", "done", "done"), ("Team colors", "done", "done"),
                        ("Per-minute analysis", f"{completed_count} / {TOTAL_MINUTES}", "active"),
                        ("Threat model", "—", "pending"), ("Coach report", "—", "pending"),
                    ], completed_for_eta=(completed_count, TOTAL_MINUTES))

                    if TOTAL_MINUTES > 1:
                        max_workers = 3

                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [executor.submit(process_single_minute, start_min, duration_sec, temp_video_path, valid_keys[0], master_team_a_color, master_team_b_color, chunk_dir) for start_min in range(1, TOTAL_MINUTES)]

                            for future in concurrent.futures.as_completed(futures):
                                result = future.result()
                                minute_idx = result.get('_minute_index', '?')
                                if "error" not in result:
                                    all_tactical_data.append(result)
                                    log_entries.append(_chunk_log_line(minute_idx, result))
                                else:
                                    log_entries.append(f"Minute {minute_idx} chunk failed — {result.get('error', 'unknown error')}")

                                completed_count += 1
                                _tick(round(completed_count / TOTAL_MINUTES * 100),
                                      f"Reading per-minute tactical patterns — chunk {completed_count} of {TOTAL_MINUTES}", [
                                    ("Upload", "done", "done"), ("Team colors", "done", "done"),
                                    ("Per-minute analysis", f"{completed_count} / {TOTAL_MINUTES}", "active"),
                                    ("Threat model", "—", "pending"), ("Coach report", "—", "pending"),
                                ], completed_for_eta=(completed_count, TOTAL_MINUTES))

                    all_tactical_data.sort(key=lambda x: x.get('_minute_index', 0))

                _tick(97, "Computing threat model and saving peak-momentum window...", [
                    ("Upload", "done", "done"), ("Team colors", "done", "done"),
                    ("Per-minute analysis", f"{TOTAL_MINUTES} / {TOTAL_MINUTES}", "done"),
                    ("Threat model", "running", "active"), ("Coach report", "—", "pending"),
                ])

                # --- CV HOOK: extract the highest-momentum 30s window while the
                # source video still exists (the finally block below deletes it).
                # This does NOT run the CV pipeline itself - it only identifies
                # the peak window and saves a durable segment for a later step.
                try:
                    if len(all_tactical_data) > 0:
                        color_a_local = str(master_team_a_color or "Team A").lower()
                        color_b_local = str(master_team_b_color or "Team B").lower()

                        peak_df = pd.DataFrame(all_tactical_data)
                        zone_map = {'attacking_third': 3, 'middle_third': 1.5, 'defensive_third': 0.5}
                        tempo_map = {'fast_direct': 2, 'sustained_high_pressure': 2, 'patient_possession': 1, 'none': 0, 'static': 0, 'dead_ball_stoppage': 0}

                        team_a_has_ball = (peak_df.get('team_in_possession', pd.Series(dtype=str)).astype(str).str.lower() == color_a_local).astype(int)
                        team_b_has_ball = (peak_df.get('team_in_possession', pd.Series(dtype=str)).astype(str).str.lower() == color_b_local).astype(int)

                        zone_numeric = peak_df.get('ball_zone', pd.Series(['middle_third'] * len(peak_df))).map(zone_map).fillna(1.5)
                        ta_tempo_num = peak_df.get('team_a_attacking_tempo', pd.Series(['none'] * len(peak_df))).map(tempo_map).fillna(1)
                        tb_tempo_num = peak_df.get('team_b_attacking_tempo', pd.Series(['none'] * len(peak_df))).map(tempo_map).fillna(1)
                        ta_hs = pd.to_numeric(peak_df.get('team_a_half_space_occupancy', 0), errors='coerce').fillna(0)
                        tb_hs = pd.to_numeric(peak_df.get('team_b_half_space_occupancy', 0), errors='coerce').fillna(0)

                        ta_trans = peak_df.get('team_a_transition_threat', pd.Series(['none'] * len(peak_df))).astype(str).str.lower().str.strip()
                        tb_trans = peak_df.get('team_b_transition_threat', pd.Series(['none'] * len(peak_df))).astype(str).str.lower().str.strip()
                        ta_boost = np.where(ta_trans.isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)
                        tb_boost = np.where(tb_trans.isin(['counter_attack', 'fast_vertical_transition']), 5.0, 0.0)

                        ta_threat = team_a_has_ball * ((zone_numeric * 2.0) + (ta_tempo_num * 1.5) + (ta_hs * 1.0) + ta_boost)
                        tb_threat = team_b_has_ball * ((zone_numeric * 2.0) + (tb_tempo_num * 1.5) + (tb_hs * 1.0) + tb_boost)
                        net_momentum = ta_threat - tb_threat

                        top_idx = int(net_momentum.abs().idxmax())
                        peak_minute = int(all_tactical_data[top_idx].get('_minute_index', top_idx))
                        peak_score = round(float(net_momentum.iloc[top_idx]), 2)
                        peak_timestamp = all_tactical_data[top_idx].get('timestamp', f"{peak_minute:02d}:00-{peak_minute+1:02d}:00")

                        peak_start_sec = peak_minute * 60
                        peak_end_sec = peak_start_sec + 30

                        if 'cv_session_id' not in st.session_state:
                            st.session_state.cv_session_id = uuid.uuid4().hex

                        session_dir = Path(tempfile.gettempdir()) / "tactical_scout_cv_sessions" / st.session_state.cv_session_id
                        session_dir.mkdir(parents=True, exist_ok=True)
                        peak_segment_path = session_dir / "peak_momentum_segment.mp4"

                        extract_video_segment(temp_video_path, peak_start_sec, peak_end_sec, peak_segment_path)

                        if peak_segment_path.exists():
                            st.session_state.cv_segment_path = str(peak_segment_path)
                            st.session_state.cv_segment_timestamp = peak_timestamp
                            st.session_state.cv_segment_momentum_score = peak_score
                            st.session_state.cv_segment_start_sec = peak_start_sec
                            log_entries.append(f"Peak momentum window identified — {peak_timestamp} (score {peak_score})")
                            print(f"[CV HOOK] Peak segment saved: {peak_segment_path} "
                                  f"(exists={peak_segment_path.exists()}, size={peak_segment_path.stat().st_size} bytes) | "
                                  f"window={peak_timestamp} | momentum={peak_score}")

                            # PART 3: write-through the fresh Gemini result into the
                            # processing cache so the *next* upload of this exact
                            # file (same hash) skips straight to cached data. Only
                            # fires for a genuinely fresh extraction - a cache-hit
                            # replay has nothing new to write.
                            if not gemini_cached:
                                _cache_upsert(
                                    video_hash,
                                    video_name=uploaded_video.name,
                                    team_a=str(master_team_a_color).title(),
                                    team_b=str(master_team_b_color).title(),
                                    color_a=str(master_team_a_color).title(),
                                    color_b=str(master_team_b_color).title(),
                                    raw_data=all_tactical_data,
                                    ai_report=None,
                                    gemini_completed_at=datetime.now(timezone.utc).isoformat(),
                                    cv_segment_timestamp=peak_timestamp,
                                    cv_segment_momentum_score=peak_score,
                                )
                        else:
                            print(f"[CV HOOK] WARNING: expected segment file not found at {peak_segment_path}")
                except Exception as e:
                    # Non-fatal: the CV segment is a bonus step, it must never break
                    # the existing Gemini extraction flow if something goes wrong here.
                    print(f"[CV HOOK] Peak segment extraction failed (non-fatal): {e}")

                # --- CV PIPELINE LAUNCH: fire the actual CV analysis subprocess
                # against the segment saved above. Launched detached so it keeps
                # running after this script execution (and any later reruns)
                # finish - we never hold a process handle in session_state,
                # since a rerun re-executes this script from scratch and would
                # lose it anyway. Only the output directory (where status.json
                # will appear) is stashed. Non-fatal: must never break the
                # existing Gemini flow, same principle as the hook above.
                try:
                    active_job = _get_active_job(video_hash)
                    if cv_cached_valid:
                        # PART 3: CV side already cached (only reachable if Gemini
                        # needed a fresh run but a valid CV bundle already existed
                        # for this hash) - reuse it instead of launching a redundant job.
                        st.session_state.cv_job_output_dir = cached["cv_output_dir"]
                        st.session_state.cv_job_match_name = cached.get("cv_match_name")
                        log_entries.append("Reusing already-completed CV deep-analysis bundle for this exact video")
                    elif active_job:
                        # Orphaned-subprocess-investigation fix: a job for this
                        # exact video hash is already running (verified live
                        # against the OS process list, not just a stale record) -
                        # track it instead of launching a redundant, colliding
                        # second run_cv_analysis.py.
                        st.session_state.cv_job_output_dir = active_job["output_dir"]
                        st.session_state.cv_job_match_name = active_job.get("match_name")
                        log_entries.append(
                            "A CV deep-analysis job for this exact video is already running "
                            "(PID " + str(active_job.get("pid")) + ") — tracking it instead of starting a duplicate"
                        )
                    elif st.session_state.get('cv_segment_path'):
                        if CV_PIPELINE_SCRIPT.exists():
                            def _cv_slug(value):
                                s = re.sub(r'[^a-z0-9]+', '_', str(value).lower())
                                return re.sub(r'_+', '_', s).strip('_') or "match"

                            cv_match_name = (
                                f"{_cv_slug(master_team_a_color)}_vs_{_cv_slug(master_team_b_color)}"
                                f"_{st.session_state.cv_session_id[:8]}"
                            )
                            cv_output_base = (
                                Path(tempfile.gettempdir()) / "tactical_scout_cv_sessions"
                                / st.session_state.cv_session_id / "cv_pipeline_output"
                            )
                            cv_final_dir = cv_output_base / cv_match_name

                            creationflags = 0
                            if os.name == 'nt':
                                creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

                            # Part 4.2/4.3 needs live progress + a raw-log view, which means
                            # something has to actually capture this subprocess's stdout -
                            # previously discarded via DEVNULL, so there was nothing to show.
                            # Redirect to a real file instead of touching status.json while the
                            # job is running (that caused a real multi-hour-job crash on Windows
                            # in an earlier build - see get_cv_job_stdout_log's docstring).
                            cv_final_dir.mkdir(parents=True, exist_ok=True)
                            cv_stdout_log_path = cv_final_dir / "pipeline_stdout.log"
                            cv_stdout_log_file = open(cv_stdout_log_path, "w", encoding="utf-8", errors="replace")
                            try:
                                proc = subprocess.Popen(
                                    [sys.executable, str(CV_PIPELINE_SCRIPT),
                                     "--video", st.session_state.cv_segment_path,
                                     "--output-dir", str(cv_output_base),
                                     "--match-name", cv_match_name],
                                    cwd=str(CV_PIPELINE_DIR),
                                    stdin=subprocess.DEVNULL,
                                    stdout=cv_stdout_log_file,
                                    stderr=subprocess.STDOUT,
                                    creationflags=creationflags,
                                    close_fds=True,
                                )
                            finally:
                                # Popen duplicates the handle into the child on Windows: safe
                                # to close this side immediately, the detached child keeps
                                # writing to its own copy after this script's run ends.
                                cv_stdout_log_file.close()

                            st.session_state.cv_job_output_dir = str(cv_final_dir)
                            st.session_state.cv_job_match_name = cv_match_name
                            if video_hash:
                                _register_active_job(video_hash, proc.pid, str(cv_final_dir), cv_match_name)
                            print(f"[CV LAUNCH] Started detached run_cv_analysis.py for "
                                  f"'{cv_match_name}' -> {cv_final_dir} (PID {proc.pid})")
                        else:
                            print(f"[CV LAUNCH] WARNING: CV pipeline script not found at "
                                  f"{CV_PIPELINE_SCRIPT}, skipping launch.")
                except Exception as e:
                    print(f"[CV LAUNCH] Failed to launch CV pipeline (non-fatal): {e}")

                for d in all_tactical_data:
                    if '_minute_index' in d:
                        del d['_minute_index']

                _tick(100, "Building your dashboard...", [
                    ("Upload", "done", "done"), ("Team colors", "done", "done"),
                    ("Per-minute analysis", f"{TOTAL_MINUTES} / {TOTAL_MINUTES}", "done"),
                    ("Threat model", "done", "done"),
                    ("Coach report", "done", "done") if gemini_cached else ("Coach report", "—", "pending"),
                ])

                st.session_state.raw_data = all_tactical_data

                loading_placeholder.empty()
                st.success("🚀 Parallel Data Extraction Complete!")

                st.session_state.step = 2
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                if os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass

# ==========================================
# STEP 2: TEAM MAPPING (User Input)
# ==========================================
elif st.session_state.step == 2:
    st.sidebar.header("2. Team Mapping")

    st.info("🎯 **TEAM MAPPING**")
    st.write(f"**AI detected Team A as {st.session_state.color_a}.**")
    user_a = st.text_input(f"Enter real team name (or press Enter to keep '{st.session_state.color_a}'):", placeholder=st.session_state.color_a)

    st.write(f"**AI detected Team B as {st.session_state.color_b}.**")
    user_b = st.text_input(f"Enter real team name (or press Enter to keep '{st.session_state.color_b}'):", placeholder=st.session_state.color_b)

    # Name this saved report - only the first time a real upload's Gemini
    # result lands in the processing cache (video_hash set, no display_name
    # recorded yet). Cache hits, demo CSV mode, and a video already named on
    # an earlier run all skip this - nothing new to name.
    show_naming_prompt = False
    if st.session_state.get('video_hash'):
        _existing_entry = _cache_get(st.session_state.video_hash)
        show_naming_prompt = not (_existing_entry and _existing_entry.get('display_name'))

    match_name_input = None
    if show_naming_prompt:
        st.markdown("---")
        st.write("**📝 Name this match report**")
        st.caption("Shown in the Instant Demo picker later. Accept the suggestion or type your own.")
        match_name_input = st.text_input(
            "Report name:", value=f"{st.session_state.color_a} vs {st.session_state.color_b}",
            label_visibility="collapsed",
        )

    if st.button("Generate Dashboard"):
        st.session_state.team_a = user_a.strip() if user_a.strip() else st.session_state.color_a
        st.session_state.team_b = user_b.strip() if user_b.strip() else st.session_state.color_b
        # PART 3: a Gemini cache hit already carries its cached AI report -
        # don't clobber it. Any non-cached run (including demo CSV mode)
        # still resets it here so a stale report from a previous match in
        # this browser session never leaks into a new one.
        if not st.session_state.get('gemini_cache_hit'):
            st.session_state.ai_report = None
        if st.session_state.get('video_hash'):
            update_fields = dict(team_a=st.session_state.team_a, team_b=st.session_state.team_b)
            if show_naming_prompt:
                # Blank/whitespace input (skipped or cleared the prompt) falls
                # back to a timestamped default rather than an empty name.
                chosen_name = (match_name_input or "").strip()
                if not chosen_name:
                    chosen_name = f"Match — {datetime.now().strftime('%b %d, %Y %H:%M')}"
                update_fields["display_name"] = chosen_name
            _cache_upsert(st.session_state.video_hash, **update_fields)
        st.session_state.step = 3
        st.session_state.view_mode = 'dashboard'
        st.rerun()

# ==========================================
# STEP 3: MATH, AI DIAGNOSIS & DASHBOARD
# ==========================================
elif st.session_state.step == 3:
    
    df = pd.DataFrame(st.session_state.raw_data)
    
    # Catch bad/missing API responses before they crash Pandas
    if 'team_in_possession' not in df.columns:
        st.error("⚠️ AI Extraction Failed: The cloud API timed out or returned malformed data. If you are a recruiter reviewing this portfolio, please use the 'Quick Demo Mode' on the home screen.")
        if st.button("⬅️ Start Over"):
            st.session_state.step = 1
            st.session_state.raw_data = []
            st.rerun()
        st.stop()

    team_a = st.session_state.team_a
    team_b = st.session_state.team_b
    color_a = st.session_state.color_a
    color_b = st.session_state.color_b
    ai_report_text = st.session_state.ai_report

    # Part 3 (polish pass): proactively surface the rename prompt at the top
    # of the page instead of leaving it tucked inside an expander the user
    # has to know exists. Detection is generic - checks whether team_a/b are
    # still literally equal to their own raw color-cluster fallback, not a
    # hardcoded "Red"/"Blue" string check - so it fires for any jersey color,
    # and for a cached/curated match exactly the same as a fresh upload.
    if team_a.strip().lower() == color_a.strip().lower() and team_b.strip().lower() == color_b.strip().lower():
        with st.container(border=True):
            st.warning(
                "🏷️ **This match is still showing raw jersey-color labels.** Set real team names "
                "so they show up correctly everywhere below — dashboard, coach report, CV tab, and charts."
            )
            pn_col1, pn_col2, pn_col3 = st.columns([2, 2, 1])
            with pn_col1:
                prompted_team_a = st.text_input(f"Team currently shown as '{color_a}':", key="prompt_team_a_name")
            with pn_col2:
                prompted_team_b = st.text_input(f"Team currently shown as '{color_b}':", key="prompt_team_b_name")
            with pn_col3:
                st.write("")
                if st.button("Save names", key="prompt_team_save"):
                    new_a, new_b = prompted_team_a.strip(), prompted_team_b.strip()
                    if new_a:
                        st.session_state.team_a = new_a
                    if new_b:
                        st.session_state.team_b = new_b
                    if new_a or new_b:
                        psource, pkey = _get_active_match_identity()
                        if psource:
                            _update_match_fields({"source": psource, "key": pkey},
                                                  team_a=st.session_state.team_a, team_b=st.session_state.team_b)
                    st.rerun()

    if len(st.session_state.history) > 0 and 'load_idx' in st.session_state:
         pass

    df = compute_dashboard_df(df, team_a, team_b, color_a, color_b)

    ta_counters = int(df['team_a_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']).sum())
    tb_counters = int(df['team_b_trans_threat'].isin(['counter_attack', 'fast_vertical_transition']).sum())

    team_a_avg_dom = round(df[df['smoothed_net_momentum'] > 0]['smoothed_net_momentum'].mean(), 2) if not df[df['smoothed_net_momentum'] > 0].empty else 0
    team_b_avg_dom = round(abs(df[df['smoothed_net_momentum'] < 0]['smoothed_net_momentum'].mean()), 2) if not df[df['smoothed_net_momentum'] < 0].empty else 0
    
    team_a_peak_idx = df['smoothed_net_momentum'].idxmax()
    team_a_peak_time = df.loc[team_a_peak_idx, 'timestamp'] if pd.notna(team_a_peak_idx) else "Unknown"
    
    team_b_peak_idx = df['smoothed_net_momentum'].idxmin()
    team_b_peak_time = df.loc[team_b_peak_idx, 'timestamp'] if pd.notna(team_b_peak_idx) else "Unknown"

    ta_att_sec = pd.to_numeric(df.get('team_a_attack_sec', 0), errors='coerce').fillna(0).sum()
    tb_att_sec = pd.to_numeric(df.get('team_b_attack_sec', 0), errors='coerce').fillna(0).sum()
    
    ta_att_count = round(ta_att_sec / 60, 1)
    tb_att_count = round(tb_att_sec / 60, 1)

    def get_mode(column): return df[column].mode()[0] if column in df.columns and not df[column].dropna().empty else "Unknown"
    
    ta_block_height = get_mode('team_a_block_height')
    ta_press_intensity = get_mode('team_a_pressing_intensity')
    ta_build_up = get_mode('team_a_build_up_shape')
    ta_tempo = get_mode('team_a_attacking_tempo')
    ta_trigger = get_mode('team_a_pressing_trigger')
    ta_striker = get_mode('team_a_striker_profile')
    ta_fullback = get_mode('team_a_fullback_role')
    ta_bias = get_mode('team_a_attacking_bias')
    ta_def_action = get_mode('team_a_defensive_line_action')
    ta_half_space = get_mode('team_a_half_space_occupancy')
    
    tb_block_height = get_mode('team_b_block_height')
    tb_press_intensity = get_mode('team_b_pressing_intensity')
    tb_build_up = get_mode('team_b_build_up_shape')
    tb_tempo = get_mode('team_b_attacking_tempo')
    tb_trigger = get_mode('team_b_pressing_trigger')
    tb_striker = get_mode('team_b_striker_profile')
    tb_fullback = get_mode('team_b_fullback_role')
    tb_bias = get_mode('team_b_attacking_bias')
    tb_def_action = get_mode('team_b_defensive_line_action')
    tb_half_space = get_mode('team_b_half_space_occupancy')

    sidebar_placeholder = st.sidebar.empty()

    if st.session_state.ai_report is None and st.session_state.view_mode != 'ticker':
        dashboard_spinner = st.empty()
        with dashboard_spinner.container():
            with st.spinner("Running universal tactical engine and generating deep AI report..."):
                client = genai.Client(api_key=valid_keys[0]) 
                
                writing_prompt = f"""
                Act as an elite soccer tactical data scientist. You have been handed a statistical readout from a computer vision model that analyzed a segment of a match between {team_a} and {team_b}. 
                
                YOUR MISSION: 
                1. DIAGNOSE THE TACTICS: Look at the raw metrics below and use your vast knowledge of soccer strategy to figure out what tactical systems both teams were employing.
                2. WRITE THE REPORT: Explain to the coaching staff exactly HOW they are executing these systems mechanically.
                
                THE HARD DATA:
                - {team_a} Avg Threat Score: {team_a_avg_dom}/10 (Peak at: {team_a_peak_time})
                - {team_b} Avg Threat Score: {team_b_avg_dom}/10 (Peak at: {team_b_peak_time})
                - Total Fractional Time in Attack (Minutes): {team_a} ({ta_att_count}), {team_b} ({tb_att_count})
                - Total High-Value Transitions (Counters/Fast Vertical): {team_a} ({ta_counters}), {team_b} ({tb_counters})

                PREDOMINANT TACTICAL MODES:
                - {team_a}: Def: Block: {ta_block_height} | Press: {ta_press_intensity}/10 | Line: {ta_def_action} | Trigger: {ta_trigger}. Off: Build: {ta_build_up} | Tempo: {ta_tempo} | Bias: {ta_bias} | Striker: {ta_striker}. Space: Half-Space: {ta_half_space} | Fullbacks: {ta_fullback}
                - {team_b}: Def: Block: {tb_block_height} | Press: {tb_press_intensity}/10 | Line: {tb_def_action} | Trigger: {tb_trigger}. Off: Build: {tb_build_up} | Tempo: {tb_tempo} | Bias: {tb_bias} | Striker: {tb_striker}. Space: Half-Space: {tb_half_space} | Fullbacks: {tb_fullback}

                CRITICAL INSTRUCTION FOR WRITING:
                - For EVERY single section and sub-section below, you MUST write a rich, highly detailed analytical paragraph (at least 4-5 sentences).
                - DO NOT use brief bullet points. Expand deeply on the tactical theory and what it means for the game.
                - DO NOT use raw variable names or key-value pairs (like 'line: step_up') anywhere in your text. Translate all data into natural, free-flowing, professional scouting language.
                - YOU MUST PLACE A DOUBLE LINE BREAK BETWEEN EVERY SINGLE NUMBERED POINT so the Markdown formats cleanly.
                - TRANSITIONAL THREAT: Analyze the team's transitional threat based on the Total Transitions Logged data. Do NOT explicitly list the raw count of transitions. Instead, write a narrative analysis explaining how they successfully absorbed pressure and used fast vertical transitions or counters to bypass the opponent's structure.
                - CRITICAL - TEAM NAME TOKENS: throughout your ENTIRE response, never write {team_a}'s real name or color - instead write the exact literal placeholder text {{TEAM_A}} (including in headers and the middle of sentences). Never write {team_b}'s real name or color either - write the exact literal placeholder text {{TEAM_B}} instead. This applies with NO exceptions anywhere in the report, so that team names can be substituted in later without regenerating this text. Do not explain or acknowledge the tokens - just use them exactly as shown in the structure below.

                Format your analysis EXACTLY like this structure (note the literal {{TEAM_A}}/{{TEAM_B}} tokens - use them verbatim, do not replace them with real names):

                ### Tactical Diagnosis & Match Flow
                [Write a massive, comprehensive paragraph identifying the overarching tactical battle and flow of the game. Address if it was one-sided or highly open-ended.]

                ### {{TEAM_A}} Tactical Profile

                **1. SYSTEM IDENTIFICATION:** [Write a detailed 4-5 sentence paragraph identifying philosophy based on data.]

                **2. POSSESSION & TERRITORY:** [Write a detailed 4-5 sentence paragraph analyzing their Fractional Time in Attack and Build-up.]

                **3. DEFENSIVE MECHANICS:** [Write a detailed 4-5 sentence paragraph analyzing Line Action, Pressing Intensity, Trigger.]

                **4. TRANSITION & ATTACK:** [Write a detailed 4-5 sentence paragraph analyzing Bias, Striker Profile, Tempo, and seamlessly integrate their transitional/counter-attacking strategy without stating the raw number.]

                **5. SPATIAL MANIPULATION:** [Write a detailed 4-5 sentence paragraph analyzing Half-Space and Fullback Role.]

                **6. VULNERABILITIES:** [Write a detailed 4-5 sentence paragraph explaining the inherent flaws of this system.]

                ### {{TEAM_B}} Tactical Profile

                **1. SYSTEM IDENTIFICATION:** [Write a detailed 4-5 sentence paragraph identifying philosophy based on data.]

                **2. POSSESSION & TERRITORY:** [Write a detailed 4-5 sentence paragraph analyzing their Fractional Time in Attack and Build-up.]

                **3. DEFENSIVE MECHANICS:** [Write a detailed 4-5 sentence paragraph analyzing Line Action, Pressing Intensity, Trigger.]

                **4. TRANSITION & ATTACK:** [Write a detailed 4-5 sentence paragraph analyzing Bias, Striker Profile, Tempo, and seamlessly integrate their transitional/counter-attacking strategy without stating the raw number.]

                **5. SPATIAL MANIPULATION:** [Write a detailed 4-5 sentence paragraph analyzing Half-Space and Fullback Role.]

                **6. VULNERABILITIES:** [Write a detailed 4-5 sentence paragraph explaining the inherent flaws of this system.]

                ### DATA-DRIVEN COACHING ADJUSTMENTS

                **For {{TEAM_B}}:** 1. [First specific tactical change paragraph.]

                2. [Second specific tactical change paragraph.]

                **For {{TEAM_A}}:** 1. [First specific tactical change paragraph.]

                2. [Second specific tactical change paragraph.]
                """
                
                max_report_retries = 5
                for attempt in range(max_report_retries):
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash', 
                            contents=writing_prompt,
                            config=types.GenerateContentConfig(temperature=0.1) 
                        )
                        st.session_state.ai_report = response.text
                        ai_report_text = st.session_state.ai_report
                        # PART 3: write-through the freshly generated report into
                        # the cache entry the Gemini extraction already created,
                        # so the next upload of this exact video gets a full hit.
                        if st.session_state.get('video_hash'):
                            _cache_upsert(st.session_state.video_hash, ai_report=ai_report_text)
                        break
                    except Exception as e:
                        if attempt < max_report_retries - 1:
                            time.sleep(5)
                            continue
                        else:
                            st.error("❌ Failed to generate report after max retries due to strict API limits.")
                            st.stop()
        dashboard_spinner.empty()
    
    with sidebar_placeholder.container():
        st.success("Analysis Complete!")
        with st.container(border=True):
            if st.button("💾 Save Report to History", use_container_width=True):
                if st.session_state.ai_report is not None:
                    st.session_state.history.append({
                        'team_a': st.session_state.team_a,
                        'team_b': st.session_state.team_b,
                        'raw_data': st.session_state.raw_data,
                        'ai_report': st.session_state.ai_report,
                        'color_a': st.session_state.color_a,
                        'color_b': st.session_state.color_b
                    })
                    st.success("Report Saved!")

            # Part 4.4: distinct from "Save Report to History" (session-only,
            # lost on refresh) - writes into the same durable processing
            # cache Instant Demo/repeat-upload reuse already draws from.
            cache_key = st.session_state.get('video_hash')
            cache_name_input = None
            if not cache_key:
                cache_name_input = st.text_input(
                    "Name for this saved match:",
                    placeholder=f"{st.session_state.team_a} vs {st.session_state.team_b}",
                    key="save_to_cache_name",
                )
            if st.button("📦 Save to Cache", use_container_width=True):
                if st.session_state.ai_report is None:
                    st.warning("Generate the full report before saving to cache.")
                else:
                    effective_key = cache_key
                    if not effective_key:
                        name = (cache_name_input or f"{st.session_state.team_a} vs {st.session_state.team_b}").strip()
                        effective_key = "manual_" + hashlib.sha256(name.encode()).hexdigest()[:16]
                    existing_entry = _cache_get(effective_key)
                    confirm_flag = f"_confirm_overwrite_cache_{effective_key}"
                    if existing_entry and not st.session_state.get(confirm_flag):
                        st.warning(
                            f"A cached entry ('{existing_entry.get('display_name', effective_key)}') already "
                            f"exists for this video/name. Click **Save to Cache** again to overwrite it."
                        )
                        st.session_state[confirm_flag] = True
                    else:
                        _cache_upsert(
                            effective_key,
                            team_a=st.session_state.team_a, team_b=st.session_state.team_b,
                            color_a=st.session_state.color_a, color_b=st.session_state.color_b,
                            raw_data=st.session_state.raw_data, ai_report=st.session_state.ai_report,
                            display_name=(cache_name_input or f"{st.session_state.team_a} vs {st.session_state.team_b}").strip(),
                            cv_output_dir=st.session_state.get('cv_job_output_dir'),
                            cv_segment_timestamp=st.session_state.get('cv_segment_timestamp'),
                            cv_segment_momentum_score=st.session_state.get('cv_segment_momentum_score'),
                        )
                        st.session_state[confirm_flag] = False
                        st.success("Saved to cache — it'll appear in Instant Demo once its CV analysis is complete.")

            if st.button("🗑️ Start Over (Clear Screen)", use_container_width=True):
                st.session_state.step = 1
                st.session_state.view_mode = 'dashboard'
                st.session_state.ai_report = None
                st.session_state.raw_data = []
                st.rerun()

        st.markdown('<div class="tac-nav-heading">Navigate</div>', unsafe_allow_html=True)
        nav_selection = st.radio("Go to:", ["Match Dashboard", "Methodology & Project Report"], label_visibility="collapsed")
        
        if len(st.session_state.history) > 0:
            st.markdown("---")
            st.subheader("📁 Past Reports")
            history_options = {"(Active Session)": -1}
            for i, h in enumerate(st.session_state.history):
                history_options[f"Match {i+1}: {h['team_a']} vs {h['team_b']}"] = i
                
            selected_history = st.selectbox("Load History", list(history_options.keys()))
            if history_options[selected_history] != -1:
                h_idx = history_options[selected_history]
                st.session_state.raw_data = st.session_state.history[h_idx]['raw_data']
                st.session_state.team_a = st.session_state.history[h_idx]['team_a']
                st.session_state.team_b = st.session_state.history[h_idx]['team_b']
                st.session_state.color_a = st.session_state.history[h_idx]['color_a']
                st.session_state.color_b = st.session_state.history[h_idx]['color_b']
                st.session_state.ai_report = st.session_state.history[h_idx]['ai_report']
                st.session_state.load_idx = h_idx
                st.rerun()

    if nav_selection == "Methodology & Project Report":
        get_pdf_download_button()
        st.markdown(methodology_text)

    elif nav_selection == "Match Dashboard":
        
        zone_help = "Shows the pitch zone where the team held the ball most frequently. Attacking Third indicates high pressure, Middle Third reflects controlled build-up, and Defensive Third indicates being pinned back."
        press_help = "Calculated from 1-10 based on defensive line height and intensity. \n\n• 1-3: Low Block (Passive/Defending the box)\n• 4-6: Mid Block (Engaging near halfway line)\n• 7-10: High Press (Aggressive/Hunting the ball)"
        att_time_help = "The true fractional number of minutes this team held sustained possession inside the opponent's defensive third."
        bias_help = "Shows the percentage of successful attacking sequences funneled down the left flank, right flank, or center. Minutes where the team was purely defending have been filtered out to provide a 100% pure offensive analysis."
        poss_heat_help = "Displays the total minutes each team held active, established possession. Purely neutral moments, transitions, or major stoppages are excluded to provide a true reflection of territorial control."
        block_heat_help = "Understanding the 'Team Positioning' Chart:\n\nWhile the first heatmap shows where the team had the ball, this second heatmap shows where the team built their defensive wall when they didn't have the ball.\n\n• Defensive Third (Low Block): The team 'parked the bus.' They retreated deep into their own penalty area to defend.\n• Middle Third (Mid Block): The team held their defensive line near the midfield circle, staying compact.\n• Attacking Third (High Press): The team pushed their defenders aggressively high up the pitch into the opponent's half to trap them."
        mom_help = "A zero-sum measure of Absolute Threat. Only the team controlling the minute scores points. Spikes represent counter-attacks or deep penetration."
        avg_mom_help = "The average Threat Score generated by this team while in control of the match. It evaluates pitch position, attacking tempo, and transitions."

        if st.session_state.view_mode == 'ticker':
            if st.button("⬅️ Back to Match Charts"):
                st.session_state.view_mode = 'dashboard'
                st.rerun()
                
            st.header("⏱️ Minute-by-Minute Live Match Ticker")
            st.markdown("---")
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                
                a_poss = int(row.get('team_a_has_ball', 0)) == 1
                b_poss = int(row.get('team_b_has_ball', 0)) == 1
                zone = str(row['ball_zone']).replace('_', ' ').title()

                if a_poss:
                    narrative = f"**{team_a}** held possession in the **{zone}**, operating at a **{str(row['team_a_attacking_tempo']).replace('_', ' ').title()}** tempo. **{team_b}** defended with a **{str(row['team_b_block_height']).title()} Block** (Pressing Intensity: **{str(row['team_b_pressing_intensity'])}/10**)."
                elif b_poss:
                    narrative = f"**{team_b}** held possession in the **{zone}**, operating at a **{str(row['team_b_attacking_tempo']).replace('_', ' ').title()}** tempo. **{team_a}** defended with a **{str(row['team_a_block_height']).title()} Block** (Pressing Intensity: **{str(row['team_a_pressing_intensity'])}/10**)."
                else:
                    narrative = f"Neutral phase in the **{zone}**."
                    
                st.markdown(f"#### Minute {timestamp}")
                st.write(narrative)
                st.markdown("---")

        else:
            if st.button("⏱️ View Live Match Ticker (Minute-by-Minute Timeline)"):
                st.session_state.view_mode = 'ticker'
                st.rerun()

            st.header("Match Segment Overview")

            tab_dashboard, tab_coach, tab_cv, tab_training, tab_chat = st.tabs(
                ["📊 Data Dashboard", "🎯 Coach Report", "🎬 CV Deep Analysis", "🏋️ Training Plan", "💬 Ask the Assistant"])

            with tab_dashboard:
                st.subheader("Global Control")
                col1, col2, col3, col4 = st.columns(4)
            
                with col1:
                    ta_poss_df = df[df['team_a_has_ball'] == 1]['ball_zone']
                    if not ta_poss_df.empty and ta_poss_df.value_counts().iloc[0] >= 2:
                        ta_zone_display = ta_poss_df.value_counts().index[0].replace('_', ' ').title()
                    else:
                        ta_zone_display = "None"
                    metric_card(st, f"{team_a} Primary Zone", ta_zone_display, zone_help)
                
                with col2:
                    tb_poss_df = df[df['team_b_has_ball'] == 1]['ball_zone']
                    if not tb_poss_df.empty and tb_poss_df.value_counts().iloc[0] >= 2:
                        tb_zone_display = tb_poss_df.value_counts().index[0].replace('_', ' ').title()
                    else:
                        tb_zone_display = "None"
                    metric_card(st, f"{team_b} Primary Zone", tb_zone_display, zone_help)
                
                with col3:
                    metric_card(st, f"{team_a} Time in Attack (min)", f"{ta_att_count}", att_time_help)
                with col4:
                    metric_card(st, f"{team_b} Time in Attack (min)", f"{tb_att_count}", att_time_help)
                
                st.markdown("---")
            
                st.subheader("Advanced Tactical Metrics")
                col5, col6, col7 = st.columns(3)
            
                ta_press_avg = round(pd.to_numeric(df['team_a_pressing_intensity'], errors='coerce').mean(), 1)
                tb_press_avg = round(pd.to_numeric(df['team_b_pressing_intensity'], errors='coerce').mean(), 1)
            
                block_help = "The primary (most frequent) starting position of the team's defensive wall out of possession."
            
                with col5:
                    metric_card(st, f"{team_a} Avg Threat", team_a_avg_dom, avg_mom_help)
                    metric_card(st, f"{team_b} Avg Threat", team_b_avg_dom, avg_mom_help)
                
                with col6:
                    metric_card(st, f"{team_a} Avg Pressing Intensity", f"{ta_press_avg} / 10", press_help)
                    metric_card(st, f"{team_b} Avg Pressing Intensity", f"{tb_press_avg} / 10", press_help)
                
                with col7:
                    metric_card(st, f"{team_a} Primary Block Height", ta_block_height.replace('_', ' ').title() + " Block", block_help)
                    metric_card(st, f"{team_b} Primary Block Height", tb_block_height.replace('_', ' ').title() + " Block", block_help)
            
                st.markdown("---")
                st.header("Tactical Visualizations")
            
                tab1, tab2, tab3, tab4 = st.tabs(["Attacking Bias (Geometry)", "Active Possession Distribution", "Team Positioning (Block Height)", "Segment Momentum"])
                with tab1:
                    st.subheader("Attacking Channels", help=bias_help)
                    valid_channels = ['left_flank', 'right_flank', 'central_channel']

                    def _render_bias_pie(col, team_label, filtered_df, bias_col):
                        """filtered_df: this team's active-possession rows (with 'timestamp'
                        still attached - Part 2 confirmed this is available at this point,
                        not yet collapsed into a bare count). Renders an interactive Plotly
                        pie fixed at the same size/legend/colors as the other team's, with a
                        per-slice hover drilldown listing the real minutes behind that count -
                        plus keeps the existing matplotlib PNG-download path unchanged."""
                        with col:
                            st.markdown(f"**{team_label}**")
                            bias_series_raw = filtered_df[bias_col]
                            sub = filtered_df[[bias_col, 'timestamp']].copy()
                            sub[bias_col] = sub[bias_col].astype(str).str.lower().str.strip()
                            sub = sub[sub[bias_col].isin(valid_channels)]
                            if sub.empty:
                                st.info("No active build-up or attacking sequences logged for this segment.")
                                return
                            # Real per-minute detail, not estimated/evenly split - Part 2's
                            # drilldown requirement, backed by an explicit check that this
                            # data is still here at chart-build time (it is: `df` retains
                            # every per-minute row and its timestamp throughout).
                            grouped = sub.groupby(bias_col)['timestamp'].apply(
                                lambda ts: ", ".join(str(_minute_num_from_timestamp(t)) for t in sorted(ts, key=_minute_num_from_timestamp))
                            )
                            counts = sub.groupby(bias_col).size()
                            clean_idx = list(grouped.index)
                            display_labels = [c.replace('_', ' ').title() for c in clean_idx]
                            minute_lists = list(grouped.values)

                            pie_fig = go.Figure(data=[go.Pie(
                                labels=display_labels, values=[counts[c] for c in clean_idx], hole=0.35,
                                marker=dict(colors=[ATTACKING_BIAS_COLORS.get(c, '#cccccc') for c in clean_idx]),
                                textinfo="percent+value", customdata=minute_lists,
                                hovertemplate="%{label}: %{value} times (%{percent})<br>Minutes: %{customdata}<extra></extra>",
                            )])
                            themed_plotly_layout(pie_fig, height=340)
                            st.plotly_chart(pie_fig, use_container_width=True, key=f"pie_{team_label}")

                            # Unchanged matplotlib render, kept only to back the PNG download
                            # button (Plotly's PNG export needs a Chromium-based renderer
                            # (kaleido) that hangs in this environment - matplotlib already
                            # works, so it's kept purely as the export path, not displayed).
                            # Also the same path Part 5.2's PDF export uses to embed this
                            # chart - see build_bias_pie_mpl.
                            fig = build_bias_pie_mpl(bias_series_raw)
                            if fig is not None:
                                st.download_button("💾 Download Chart", fig_to_png_bytes(fig), f"{team_label}_attacking_bias.png",
                                                    "image/png", key=f"dl_pie_{team_label}")

                    col_pie1, col_pie2 = st.columns(2)
                    bias_a_df = df[(df['team_a_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
                    bias_b_df = df[(df['team_b_has_ball'] == 1) & (df['ball_zone'].isin(['middle_third', 'attacking_third']))]
                    _render_bias_pie(col_pie1, team_a, bias_a_df, 'team_a_attacking_bias')
                    _render_bias_pie(col_pie2, team_b, bias_b_df, 'team_b_attacking_bias')
                    
                with tab2:
                    st.subheader("Active Possession Distribution (Total Minutes)", help=poss_heat_help)

                    zones = ['defensive_third', 'middle_third', 'attacking_third']
                    zone_labels = ['Defensive Third', 'Middle Third', 'Attacking Third']
                    hm_a = [df[(df['team_in_possession'].str.lower() == color_a.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]
                    hm_b = [df[(df['team_in_possession'].str.lower() == color_b.lower()) & (df['ball_zone'] == z)].shape[0] for z in zones]

                    heat_fig = go.Figure(data=go.Heatmap(
                        z=[hm_a, hm_b], x=zone_labels, y=[team_a, team_b],
                        colorscale=THEME_COLORSCALE_BLUE, texttemplate="%{z}",
                        hovertemplate="%{y} · %{x}: %{z} min<extra></extra>",
                    ))
                    themed_plotly_layout(heat_fig, height=280)
                    heat_fig.update_layout(xaxis_title="Pitch Zone")
                    st.plotly_chart(heat_fig, use_container_width=True, key="heat_possession")

                    heatmap_data = compute_possession_heatmap_data(df, team_a, team_b, color_a, color_b)
                    fig3 = build_heatmap_mpl(heatmap_data, 'Blues')
                    st.download_button("💾 Download Heatmap", fig_to_png_bytes(fig3), "possession_heatmap.png", "image/png", key="dl_heat1")

                with tab3:
                    st.subheader("Team Positioning (Defensive Block Height)", help=block_heat_help)
                    position_df = compute_block_heatmap_data(df, team_a, team_b)
                    zone_labels = ['Defensive Third', 'Middle Third', 'Attacking Third']

                    # Part 2: hover-based drilldown, not click - this Streamlit version
                    # (1.32.2, confirmed) predates plotly_chart's on_select/selection_mode
                    # (added 1.35+), so native click events aren't available here at all,
                    # not just unreliable. Hover + customdata needs no new dependency and
                    # no event-handling risk, matching the brief's preferred fallback.
                    minutes_grid = compute_block_heatmap_minutes(df, team_a, team_b)
                    block_fig = go.Figure(data=go.Heatmap(
                        z=position_df.values, x=zone_labels, y=list(position_df.index),
                        colorscale=THEME_COLORSCALE_PURPLE, texttemplate="%{z}",
                        customdata=minutes_grid,
                        hovertemplate="%{y} · %{x}: %{z} min<br>Minutes: %{customdata}<extra></extra>",
                    ))
                    themed_plotly_layout(block_fig, height=280)
                    block_fig.update_layout(xaxis_title="Pitch Zone")
                    st.plotly_chart(block_fig, use_container_width=True, key="heat_block")

                    fig5 = build_heatmap_mpl(position_df, 'Purples')
                    st.download_button("💾 Download Block Heatmap", fig_to_png_bytes(fig5), "defensive_block_heatmap.png", "image/png", key="dl_heat2")

                with tab4:
                    st.subheader("Match Dominance (Absolute Threat)", help=mom_help)
                    y_vals = df['smoothed_net_momentum']
                    y_pos = y_vals.clip(lower=0)
                    y_neg = y_vals.clip(upper=0)

                    mom_fig = go.Figure()
                    mom_fig.add_trace(go.Scatter(x=df['timestamp'], y=y_pos, name=team_a, mode='lines',
                                                  line=dict(width=0), fill='tozeroy', fillcolor='rgba(217,56,58,0.8)'))
                    mom_fig.add_trace(go.Scatter(x=df['timestamp'], y=y_neg, name=team_b, mode='lines',
                                                  line=dict(width=0), fill='tozeroy', fillcolor='rgba(51,51,51,0.8)'))
                    mom_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    themed_plotly_layout(mom_fig, height=420)
                    mom_fig.update_layout(yaxis_title="Absolute Attacking Threat", xaxis_tickangle=-45,
                                           hovermode="x unified")
                    st.plotly_chart(mom_fig, use_container_width=True, key="mom_chart")

                    fig4 = build_momentum_mpl(df, team_a, team_b)
                    st.download_button("💾 Download Momentum Graph", fig_to_png_bytes(fig4), "momentum_chart.png", "image/png", key="dl_mom")

                st.markdown("---")
                st.subheader("💾 Export Raw Data")
                st.write("Download the fully structured, minute-by-minute tactical dataset to run your own models.")
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Tactical Data (CSV)",
                    data=csv_data,
                    file_name=f"{team_a}_vs_{team_b}_tactical_data.csv",
                    mime='text/csv'
                )
            with tab_coach:
                st.header("🤖 In-Depth AI Diagnostic Report")
                # The stored report has literal {TEAM_A}/{TEAM_B} tokens baked in
                # instead of real names (see the writing_prompt in Step 3) so a
                # rename never requires regenerating it - substitute at display
                # time, every render site, via the one shared helper.
                display_report = cb.substitute_team_tokens(st.session_state.ai_report, team_a, team_b)
                st.markdown(display_report)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="💾 Download AI Report (Text)",
                        data=display_report.encode('utf-8'),
                        file_name=f"{team_a}_vs_{team_b}_AI_Report.md",
                        mime='text/markdown'
                    )
                with dl_col2:
                    # Part 5.2: works identically for a curated match or a
                    # live upload - both resolve to the same (source, key)
                    # via _get_active_match_identity, and every section reads
                    # through the same helpers the live UI already uses.
                    pdf_source, pdf_key = _get_active_match_identity()
                    try:
                        pdf_bytes = generate_full_pdf_report(
                            team_a, team_b, color_a, color_b, st.session_state.raw_data,
                            display_report, st.session_state.get('cv_job_output_dir'),
                            st.session_state.get('cv_team_mapping'), pdf_source, pdf_key,
                        )
                        st.download_button(
                            label="📄 Download Full Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"{team_a}_vs_{team_b}_Full_Report.pdf",
                            mime='application/pdf',
                        )
                    except Exception as e:
                        st.error(f"Could not generate the full PDF report: {e}")


            with tab_cv:
                render_cv_deep_analysis_tab()

            with tab_training:
                render_training_plan_tab()

            with tab_chat:
                stats_json_for_chat = None
                cv_output_dir_chat = st.session_state.get('cv_job_output_dir')
                if cv_output_dir_chat:
                    cv_status_chat = get_cv_job_status_safe(cv_output_dir_chat)
                    if cv_status_chat.get('status') == 'complete' and cv_status_chat.get('stats_file'):
                        resolved_chat = _resolve_cv_path(cv_status_chat['stats_file'])
                        if resolved_chat.exists():
                            with open(resolved_chat, 'r') as f:
                                stats_json_for_chat = json.load(f)
                chat_source, chat_key = _get_active_match_identity()
                cb.render_chatbot_tab(
                    df, stats_json_for_chat, team_a, team_b, ai_report_text,
                    valid_keys[0] if valid_keys else None, chat_source, chat_key,
                    lambda s, k, plan: tp.save_training_plan(s, k, plan, CURATED_MATCHES_DIR, CACHE_DIR),
                )

