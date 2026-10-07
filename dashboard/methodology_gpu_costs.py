"""GPU compute and cost block for the Methodology page.

Static planning figures only: nothing here is computed from match data, read
from disk or fetched over the network. Kept in its own module so the
Methodology branch in app.py only needs a single call.
"""
import pandas as pd
import streamlit as st

from methodology_cost_calculator import (
    GPU_TIERS,
    STARTUP_MINUTES_PER_GPU,
    estimate,
    format_usd_range,
    render_cost_calculator,
)

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

TABLE3_TIER_NAMES = {
    "budget": "Budget",
    "mid": "Mid",
    "a100": "Datacenter A100",
    "serverless_a100": "Serverless A100",
}

GPU_TIER_ROWS = [
    ("Budget marketplace", "T4 (2018); RTX 4090",
     "16 GB (T4); 24 GB (RTX 4090)",
     "T4: an older data-centre card, common on free notebook services. RTX 4090: a fast consumer gaming card, "
     "often rented from individuals on peer-to-peer marketplaces.",
     "Cheapest per hour. The RTX 4090 is often a strong speed per dollar for small vision models "
     "(approximate, from public spec sheets; verify before relying).",
     "Machines vary in reliability and network speed; they can be interrupted; not built for sustained "
     "data-centre use; many at once may not be available.",
     "about $0.40/h (approximate; not checked against live prices)"),
    ("Mid", "L4; A10 (A10G on AWS)",
     "24 GB",
     "Data-centre cards for inference: efficient, low-power, available from the big cloud providers.",
     "Reliable and efficient for video and vision inference; a balance of price and speed.",
     "Costs more than marketplace cards. For raw speed a 4090 is faster, so an L4 can be slower than a 4090 "
     "for the same job.",
     "about $0.80/h (approximate; not checked against live prices)"),
    ("Datacenter (A100)", "A100 40 GB; A100 80 GB",
     "40 GB or 80 GB",
     "A large data-centre card built for training and big batches. Memory bandwidth 1.56 TB/s (40 GB) "
     "or 2.04 TB/s (80 GB).",
     "Very reliable and widely available from cloud providers.",
     "Our detector, tracker and keypoint models are small and do not need that memory, so much of the extra "
     "cost buys capacity we do not use.",
     "about $1.50/h (approximate; not checked against live prices)"),
    ("Serverless A100 (premium)", "A100 80 GB on Modal (40 GB option too)",
     "80 GB (Modal also offers 40 GB)",
     "The same class of GPU, billed per second only while it runs, scaling to dozens of GPUs at once with no "
     "machine to manage.",
     "Fast turnaround at scale, with no idle charges while nothing runs.",
     "Highest hourly rate. Modal's GPU rates are preemptible: work can be interrupted and rescheduled. "
     "Cold-start delay. Code must be packaged for the platform. The realistic option for finishing a full match "
     "in about 20 minutes with 60 to 90 GPUs at once.",
     "about $2.50/h (Modal A100 80 GB, $0.000694/s; its 40 GB is $2.10/h)"),
]

GPU_TIER_CHOICE_MD = r"""
**Which one for which job**
* **Marketplace:** cheap batch runs when time does not matter.
* **Mid:** steady reliability for regular work.
* **A100:** only if we later train large models.
* **Serverless:** fast turnaround at scale.

**Caveat:** all four tiers currently use the same assumed speed (the speed factor in the calculator), so the
calculator shows the higher-priced tiers as more expensive. A faster card finishes sooner, so the price comparison
is not yet fair. Measure speeds with a 10-minute GPU timing test before comparing tiers.

None of this is a quote. Prices vary by provider and time of day, and the spec figures are approximate: they come
from public spec sheets and must be verified before relying on them.
"""


def _table3_rows():
    rows = []
    for key, info in GPU_TIERS.items():
        half = estimate(45, 10, key, ["cv_models"])
        match = estimate(90, 20, key, ["cv_models"])
        hg, mg = half["gpu"], match["gpu"]
        rows.append((
            TABLE3_TIER_NAMES[key], f"about ${info['usd_per_hour']:.2f}/h",
            format_usd_range(hg["compute_cost_low"], hg["compute_cost_high"]),
            format_usd_range(half["total_low"], half["total_high"]),
            format_usd_range(mg["compute_cost_low"], mg["compute_cost_high"]),
            format_usd_range(match["total_low"], match["total_high"]),
        ))
    return rows


def _table3_gpu_range():
    half = estimate(45, 10, "budget", ["cv_models"])["gpu"]
    match = estimate(90, 20, "budget", ["cv_models"])["gpu"]
    return (min(half["gpus_low"], match["gpus_low"]), max(half["gpus_high"], match["gpus_high"]))


GPU_COUNT_ROWS = [
    ("1", "19.5 to 30 h"),
    ("10", "2 to 3 h"),
    ("30", "39 to 60 min"),
    ("60", "20 to 30 min"),
    ("90", "13 to 20 min"),
]

SUMMARY_MD_TEMPLATE = r"""
**How it was calculated**
* Compute time = video seconds x 13 to 20. A 90-minute match is 5,400 seconds of video, which works out to 19.5 to 30 hours of compute.
* Cost = compute hours x the hourly rate.
* A parallel run splits the same work across N GPUs, so N = compute hours / target hours.
* Each GPU used also pays for model load and container start, STARTUP_MIN minutes per GPU (an unmeasured planning assumption), so total cost rises with the number of GPUs.

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
        lo, hi = _table3_gpu_range()
        st.caption(
            f"Target: 45-min half in 10 min, 90-min match in 20 min. Needs about {lo} to {hi} GPUs at once. "
            f"Costs with model load include {STARTUP_MINUTES_PER_GPU:g} minutes of model load and container start "
            "per GPU (unmeasured planning assumption). Generated from the calculator's own cost function."
        )
        _table(_table3_rows(), [
            "GPU option", "Rate",
            "45-min half in 10 min: GPU only", "45-min half in 10 min: with model load",
            "90-min match in 20 min: GPU only", "90-min match in 20 min: with model load",
        ])

        st.markdown("**Table 4: How many GPUs for a 90-min match**")
        _table(GPU_COUNT_ROWS, ["GPUs at once", "Finish time"])

        st.markdown(SUMMARY_MD_TEMPLATE.replace("STARTUP_MIN", f"{STARTUP_MINUTES_PER_GPU:g}"))

        st.markdown("**What the GPU options are**")
        st.markdown(
            "Each option is a kind of rented graphics card. The table explains what each one is, its strengths "
            "and weaknesses, and the hourly rate this page uses."
        )
        _table(GPU_TIER_ROWS, [
            "Option", "Example GPUs", "Memory", "What it is", "Strengths", "Weaknesses and risks", "Rate used here",
        ])
        st.markdown(GPU_TIER_CHOICE_MD)

        render_cost_calculator()
