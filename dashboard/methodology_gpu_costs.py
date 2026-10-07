"""GPU compute and cost block for the Methodology page.

Static planning figures only: nothing here is computed from match data, read
from disk or fetched over the network. Kept in its own module so the
Methodology branch in app.py only needs a single call.
"""
import pandas as pd
import streamlit as st

from methodology_cost_calculator import render_cost_calculator

# Marker in app.py's methodology_text that this block is rendered just above,
# so it sits directly under the existing "Scalability & Cost" section.
ABOUT_SECTION_MARKER = "---\n### 👨‍💻 About the Creator"

COMPUTE_TIME_ROWS = [
    ("10 min", "2.2 to 3.3 h"),
    ("45 min (one half)", "9.8 to 15 h"),
    ("90 min (full match)", "19.5 to 30 h"),
]

SINGLE_GPU_COST_ROWS = [
    ("Budget marketplace (T4 or RTX 4090)", "about $0.40/h", "$0.90 to $1.30", "$4 to $6", "$8 to $12"),
    ("Mid (L4 or A10 class)", "about $0.80/h", "$1.80 to $2.70", "$8 to $12", "$16 to $24"),
    ("Datacenter (A100)", "about $1.50/h", "$3.30 to $5.00", "$15 to $23", "$29 to $45"),
    ("Serverless A100 (Modal or RunPod serverless, premium)", "about $2.50/h", "$5.50 to $8.30", "$24 to $38", "$49 to $75"),
]

PARALLEL_COST_ROWS = [
    ("Budget", "about $0.40/h", "$4 to $6", "$5 to $8", "$8 to $12", "$10 to $15"),
    ("Mid", "about $0.80/h", "$8 to $12", "$10 to $15", "$16 to $24", "$20 to $30"),
    ("Datacenter A100", "about $1.50/h", "$15 to $23", "$18 to $28", "$29 to $45", "$37 to $56"),
    ("Serverless A100", "about $2.50/h", "$24 to $38", "$30 to $47", "$49 to $75", "$61 to $94"),
]

GPU_COUNT_ROWS = [
    ("1", "19.5 to 30 h"),
    ("10", "2 to 3 h"),
    ("30", "39 to 60 min"),
    ("60", "20 to 30 min"),
    ("90", "13 to 20 min"),
]

SUMMARY_MD = r"""
**How it was calculated**
* Compute time = video seconds x 13 to 20. A 90-minute match is 5,400 seconds of video, which works out to 19.5 to 30 hours of compute.
* Cost = compute hours x the hourly rate.
* A parallel run splits the same work across N GPUs, so N = compute hours / target hours. The total cost stays about the same, plus about 25% overhead.

**Why a GPU**

The pipeline runs a detector, a tracker and a pitch-keypoint model on every sampled frame. On a CPU this takes far longer (earlier training estimates on CPU were 15 to 25 hours). A rented GPU is billed only while it runs.

**Benefits**
* Pay per video: roughly \$10 to \$15 per full match on the budget tier.
* No hardware to own.
* Can scale out to more GPUs when speed matters.
* The cost per match is small next to analyst time.

**Caveats**
* All figures are planning estimates, not measurements.
* Hourly rates are approximate and must be checked live.
* The 13 to 20 seconds per second figure is unmeasured on the target GPU, so a 10-minute test should come first.
* Parallel runs need the video split into chunks, with tracker IDs and camera calibration carried across chunk boundaries. The pipeline does not do this today.
* This is planned for the full-match work after the current demo. It is not live functionality.
"""


def _table(rows, columns):
    # First column becomes the index so st.table shows it as the row label
    # instead of a 0..n counter.
    # st.table renders cells as markdown, where a pair of "$" would be read as
    # LaTeX, so dollar signs are escaped.
    df = pd.DataFrame([[cell.replace("$", "\\$") for cell in row] for row in rows], columns=columns)
    st.table(df.set_index(columns[0]))


def render_gpu_compute_and_cost():
    with st.container(key="methodology_gpu_costs"):
        st.markdown("---")
        st.markdown("### 🖥️ GPU Compute & Cost (planning estimate)")
        st.markdown(
            "What it would take to process longer videos on rented GPUs. This is planned work for "
            "full-match processing after the current demo. None of it is live in this dashboard."
        )

        st.markdown("**Table 1: Compute needed on one GPU**")
        st.caption(
            "Assumption: 13 to 20 seconds of compute per second of video "
            "(planning estimate; the GPU it was measured on is not recorded)."
        )
        _table(COMPUTE_TIME_ROWS, ["Video", "Compute time"])

        st.markdown("**Table 2: Cost to process one video on one GPU (GPU rental only)**")
        _table(SINGLE_GPU_COST_ROWS, ["GPU option", "Rate", "10 min", "45 min", "90 min"])

        st.markdown("**Table 3: Finishing fast by running chunks in parallel**")
        st.caption(
            "Target: 45-min half in 10 min, 90-min match in 20 min. Needs about 60 to 90 GPUs at once. "
            "Costs include about 25% overhead for model load and spin-up."
        )
        _table(PARALLEL_COST_ROWS, [
            "GPU option", "Rate",
            "45-min half in 10 min: GPU only", "45-min half in 10 min: with overhead",
            "90-min match in 20 min: GPU only", "90-min match in 20 min: with overhead",
        ])

        st.markdown("**Table 4: How many GPUs for a 90-min match**")
        _table(GPU_COUNT_ROWS, ["GPUs at once", "Finish time"])

        st.markdown(SUMMARY_MD)

        render_cost_calculator()
