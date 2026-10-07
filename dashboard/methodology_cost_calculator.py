"""Interactive cost calculator for the Methodology page (under the GPU tables).

Everything here is arithmetic on the constants below: no match data is read,
nothing is fetched and no API is called. estimate() is a pure function so it
can be unit-tested without Streamlit (see test_methodology_cost_calculator.py).
"""
import math

import pandas as pd
import streamlit as st

# ======================================================================
# ASSUMPTIONS - every number the calculator uses lives in this one block.
# ======================================================================

# GPU rental rates per tier, USD per hour. Same approximate figures as the
# GPU tables above the calculator; hourly rates must be checked live.
GPU_TIERS = {
    "budget": {"label": "Budget marketplace (T4 or RTX 4090)", "usd_per_hour": 0.40},
    "mid": {"label": "Mid (L4 or A10 class)", "usd_per_hour": 0.80},
    "a100": {"label": "Datacenter (A100)", "usd_per_hour": 1.50},
    "serverless_a100": {"label": "Serverless A100 (premium)", "usd_per_hour": 2.50},
}

# Compute seconds per second of video for the CV models (low, high).
# Planning assumption, NOT derived from the code and unmeasured on the target
# GPU. The one timing recorded in this repo is a CPU-only run.
CV_COMPUTE_SEC_PER_VIDEO_SEC = (13.0, 20.0)

# Extra cost when the work is split across several GPUs (model load, spin-up).
PARALLEL_OVERHEAD_FRACTION = 0.25

# The tables above stop at 90 GPUs at once; anything needing more is flagged.
TABLE_MAX_GPUS = 90

# Gemini pricing, USD per 1M tokens, standard paid tier.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_USD_PER_M_INPUT_TOKENS = 0.30        # text / image / video input
GEMINI_USD_PER_M_AUDIO_INPUT_TOKENS = 1.00  # audio track of an uploaded clip
GEMINI_USD_PER_M_OUTPUT_TOKENS = 2.50       # output, including thinking tokens
GEMINI_RATE_SOURCE = "ai.google.dev/gemini-api/docs/pricing (read 2026-10-06)"
GEMINI_RATE_NOTE = "approximate, verify before relying on it"

# How Gemini counts an uploaded video clip (ai.google.dev/gemini-api/docs/
# video-understanding): 1 frame per second at 66 tokens (low resolution) to
# 258 tokens (default resolution), plus 32 audio tokens per second. The code
# does not set a resolution, so both ends are kept as the (low, high) range.
GEMINI_VIDEO_TOKENS_PER_SEC = (66, 258)
GEMINI_AUDIO_TOKENS_PER_SEC = 32

# Thinking tokens per call (low, high). The code leaves thinking at the model
# default and never logs token usage, so the high end is an ASSUMED allowance
# that needs measurement. Billed at the output rate.
GEMINI_THINKING_TOKENS_PER_CALL = (0, 2000)

# Per-product call counts and token estimates, read from the code. Token
# counts are prompt/response characters divided by 4 (rule of thumb), with
# response sizes taken from the two saved demo matches. All approximate.
# Retries (up to 3 per call, 5 for the report) are not counted.
MATCH_REPORT = {
    # app.py process_single_minute: one call per 1-minute clip.
    "minute_call_prompt_tokens": 1800,      # ~7,200-character instruction prompt
    "minute_call_output_tokens": 300,       # one ~1,200-character JSON object
    # app.py coach report: one call per match.
    "report_calls": 1,
    "report_prompt_tokens": 2300,           # ~9,200-character template
    "report_prompt_tokens_per_video_min": 100,  # 22 per-minute timelines
    "report_output_tokens": 2300,           # ~9,200-character report
}
TRAINING_PLAN = {
    # training_plan.py: generate_team_plan once per team.
    "team_plan_calls": 2,
    "team_plan_prompt_tokens": 1740,        # template + first 4,000 chars of the report
    "team_plan_prompt_tokens_per_video_min": 25,  # 5 per-minute timelines
    "team_plan_output_tokens": 2500,
    # generate_player_plan: one call, up to 12 player cards.
    "player_plan_calls": 1,
    "player_plan_prompt_tokens": 800,
    "player_plan_output_tokens": 8200,
    # generate_cv_insights: one call.
    "cv_insights_calls": 1,
    "cv_insights_prompt_tokens": 2000,
    "cv_insights_output_tokens": 1700,
}

UNKNOWN_LABEL = "unknown: needs measurement"
ESTIMATE_LABEL = "planning estimate, not a quote"
CHUNKING_WARNING = (
    "This turnaround needs parallel chunking: needs the video split into chunks with tracker IDs "
    "and camera calibration carried across chunk boundaries; not built yet; planned after the current demo."
)
FASTER_THAN_TABLES_WARNING = (
    f"This turnaround is faster than any option in the tables above, which stop at {TABLE_MAX_GPUS} GPUs "
    "at once. Treat it as outside the planned range."
)

# Product checklist, in display order.
PRODUCTS = {
    "match_report": "Match report",
    "cv_models": "Computer vision models (incl. Tactical Map and Game Board)",
    "training_plan": "Training plan",
    "corner_kicks": "Corner kicks",
}

# ======================================================================
# Calculation (pure, no Streamlit)
# ======================================================================


def _gemini_usd(input_tokens, audio_tokens, output_tokens):
    return (
        input_tokens * GEMINI_USD_PER_M_INPUT_TOKENS
        + audio_tokens * GEMINI_USD_PER_M_AUDIO_INPUT_TOKENS
        + output_tokens * GEMINI_USD_PER_M_OUTPUT_TOKENS
    ) / 1_000_000


def _match_report(video_minutes):
    m = MATCH_REPORT
    minute_calls = math.ceil(video_minutes)
    video_seconds = video_minutes * 60
    costs = []
    for end in (0, 1):
        thinking = GEMINI_THINKING_TOKENS_PER_CALL[end]
        minute_cost = _gemini_usd(
            minute_calls * m["minute_call_prompt_tokens"] + video_seconds * GEMINI_VIDEO_TOKENS_PER_SEC[end],
            video_seconds * GEMINI_AUDIO_TOKENS_PER_SEC,
            minute_calls * (m["minute_call_output_tokens"] + thinking),
        )
        report_cost = m["report_calls"] * _gemini_usd(
            m["report_prompt_tokens"] + m["report_prompt_tokens_per_video_min"] * video_minutes,
            0,
            m["report_output_tokens"] + thinking,
        )
        costs.append(minute_cost + report_cost)
    calls = minute_calls + m["report_calls"]
    how = (
        f"Gemini API ({GEMINI_MODEL}), no GPU. {calls} calls: one per minute of video "
        f"({minute_calls}) plus {m['report_calls']} to write the report. Tokens per call x price per token."
    )
    return costs[0], costs[1], how


def _training_plan(video_minutes):
    t = TRAINING_PLAN
    calls = t["team_plan_calls"] + t["player_plan_calls"] + t["cv_insights_calls"]
    input_tokens = (
        t["team_plan_calls"] * (t["team_plan_prompt_tokens"] + t["team_plan_prompt_tokens_per_video_min"] * video_minutes)
        + t["player_plan_calls"] * t["player_plan_prompt_tokens"]
        + t["cv_insights_calls"] * t["cv_insights_prompt_tokens"]
    )
    output_tokens = (
        t["team_plan_calls"] * t["team_plan_output_tokens"]
        + t["player_plan_calls"] * t["player_plan_output_tokens"]
        + t["cv_insights_calls"] * t["cv_insights_output_tokens"]
    )
    costs = [
        _gemini_usd(input_tokens, 0, output_tokens + calls * GEMINI_THINKING_TOKENS_PER_CALL[end])
        for end in (0, 1)
    ]
    how = (
        f"Gemini API ({GEMINI_MODEL}), no GPU. {calls} calls per match: one plan per team, one player plan, "
        "one tracking-insights call. Tokens per call x price per token. Reads the match report and the "
        "CV output, which are costed separately."
    )
    return costs[0], costs[1], how


def _cv_models(video_minutes, target_minutes, tier):
    rate = GPU_TIERS[tier]["usd_per_hour"]
    video_seconds = video_minutes * 60
    hours = [video_seconds * s / 3600 for s in CV_COMPUTE_SEC_PER_VIDEO_SEC]
    if target_minutes is None:
        gpus = [1, 1]
    else:
        gpus = [max(1, math.ceil(h / (target_minutes / 60) - 1e-9)) for h in hours]
    # Overhead only applies at an end of the range that really is split across GPUs.
    costs = [h * rate * (1 + PARALLEL_OVERHEAD_FRACTION if n > 1 else 1) for h, n in zip(hours, gpus)]
    finish_minutes = [
        h * 60 if (target_minutes is None or n == 1) else target_minutes
        for h, n in zip(hours, gpus)
    ]
    low_s, high_s = (f"{s:g}" for s in CV_COMPUTE_SEC_PER_VIDEO_SEC)
    how = (
        f"Rented GPU. Compute hours = video seconds x {low_s} to {high_s} / 3600, "
        f"x about ${rate:.2f}/h"
        + (f", plus about {PARALLEL_OVERHEAD_FRACTION:.0%} overhead for running in parallel." if gpus[1] > 1 else ".")
        + " Runs the player and ball detector, tracker, pitch-keypoint model and team classification; "
        "the Tactical Map and Game Board reuse that output and add no model runs."
    )
    gpu = {
        "compute_hours_low": hours[0],
        "compute_hours_high": hours[1],
        "gpus_low": gpus[0],
        "gpus_high": gpus[1],
        "finish_minutes_low": finish_minutes[0],
        "finish_minutes_high": finish_minutes[1],
        "needs_chunking": gpus[1] > 1,
        "faster_than_tables": gpus[1] > TABLE_MAX_GPUS,
    }
    return costs[0], costs[1], how, gpu


CORNER_KICKS_UNKNOWN_REASON = (
    "No model or API of its own: it measures team shape from CV output for each manually marked corner. "
    "The cost is the CV run on those corner windows, and the code fixes neither how many corners are "
    "marked per match nor a timing for that run (it skips the ball pass and adds a position-rebuild step)."
)


def estimate(video_minutes, target_minutes, tier, products):
    """Low to high cost for the selected products.

    video_minutes: length of the video, > 0.
    target_minutes: wanted turnaround in minutes (> 0), or None for "no rush
        (single GPU)". Only affects the CV models.
    tier: a key of GPU_TIERS.
    products: iterable of PRODUCTS keys.

    Raises ValueError on any input it cannot cost. Products whose cost cannot
    be derived come back with status "unknown" and are left out of the total.
    """
    if isinstance(video_minutes, bool) or not isinstance(video_minutes, (int, float)) \
            or not math.isfinite(video_minutes) or video_minutes <= 0:
        raise ValueError("Video length must be a number of minutes greater than zero.")
    if target_minutes is not None and (
            isinstance(target_minutes, bool) or not isinstance(target_minutes, (int, float))
            or not math.isfinite(target_minutes) or target_minutes < 1 or target_minutes != int(target_minutes)):
        raise ValueError("Target turnaround must be a whole number of minutes, 1 or more.")
    if tier not in GPU_TIERS:
        raise ValueError(f"Unknown GPU tier: {tier!r}.")
    selected = list(dict.fromkeys(products))
    bad = [p for p in selected if p not in PRODUCTS]
    if bad:
        raise ValueError(f"Unknown product: {bad[0]!r}.")

    items, gpu = [], None
    for key in PRODUCTS:
        if key not in selected:
            continue
        item = {"key": key, "label": PRODUCTS[key], "status": "ok", "low": None, "high": None}
        if key == "match_report":
            item["low"], item["high"], item["how"] = _match_report(video_minutes)
        elif key == "training_plan":
            item["low"], item["high"], item["how"] = _training_plan(video_minutes)
        elif key == "cv_models":
            item["low"], item["high"], item["how"], gpu = _cv_models(video_minutes, target_minutes, tier)
        else:
            item["status"], item["how"] = "unknown", CORNER_KICKS_UNKNOWN_REASON
        items.append(item)

    known = [i for i in items if i["status"] == "ok"]
    has_unknown = len(known) != len(items)
    warnings = []
    if gpu and gpu["needs_chunking"]:
        warnings.append(CHUNKING_WARNING)
    if gpu and gpu["faster_than_tables"]:
        warnings.append(FASTER_THAN_TABLES_WARNING)
    return {
        "items": items,
        "total_low": sum(i["low"] for i in known),
        "total_high": sum(i["high"] for i in known),
        "has_unknown": has_unknown,
        "total_label": "Total, excluding unknown items" if has_unknown else "Total",
        "gpu": gpu,
        "warnings": warnings,
    }


def format_usd(amount):
    if amount < 0.01:
        return "under $0.01"
    return f"${amount:,.2f}" if amount < 10 else f"${amount:,.0f}"


def format_usd_range(low, high):
    low_s, high_s = format_usd(low), format_usd(high)
    return low_s if low_s == high_s else f"{low_s} to {high_s}"


def format_duration(minutes):
    return f"{minutes:.0f} min" if minutes < 90 else f"{round(minutes / 60, 1):g} h"


def format_duration_range(low, high):
    low_s, high_s = format_duration(low), format_duration(high)
    if low_s == high_s:
        return low_s
    if low_s.endswith(" h") and high_s.endswith(" h"):
        return f"{low_s[:-2]} to {high_s}"
    if low_s.endswith(" min") and high_s.endswith(" min"):
        return f"{low_s[:-4]} to {high_s}"
    return f"{low_s} to {high_s}"


# ======================================================================
# Streamlit block
# ======================================================================

_MINUTES_KEY = "mcc_video_minutes"
_NO_RUSH = "No rush (single GPU)"
_TARGET = "Target turnaround"


def _md(text):
    # Streamlit markdown reads a pair of "$" as LaTeX, so dollar signs are escaped.
    return text.replace("$", "\\$")


def _set_minutes(value):
    st.session_state[_MINUTES_KEY] = float(value)


def _assumptions_table():
    rows = [(f"GPU rate: {t['label']}", f"about ${t['usd_per_hour']:.2f}/h", "Approximate; check live")
            for t in GPU_TIERS.values()]
    low_s, high_s = (f"{s:g}" for s in CV_COMPUTE_SEC_PER_VIDEO_SEC)
    rows += [
        ("CV compute per second of video", f"{low_s} to {high_s} s", "Planning assumption; unmeasured on the target GPU"),
        ("Parallel overhead", f"{PARALLEL_OVERHEAD_FRACTION:.0%}", "Planning assumption"),
        (f"Gemini input ({GEMINI_MODEL})", f"${GEMINI_USD_PER_M_INPUT_TOKENS:.2f} per 1M tokens",
         f"{GEMINI_RATE_SOURCE}; {GEMINI_RATE_NOTE}"),
        ("Gemini audio input", f"${GEMINI_USD_PER_M_AUDIO_INPUT_TOKENS:.2f} per 1M tokens",
         f"{GEMINI_RATE_SOURCE}; {GEMINI_RATE_NOTE}"),
        ("Gemini output, incl. thinking", f"${GEMINI_USD_PER_M_OUTPUT_TOKENS:.2f} per 1M tokens",
         f"{GEMINI_RATE_SOURCE}; {GEMINI_RATE_NOTE}"),
        ("Video tokens per second of clip",
         f"{GEMINI_VIDEO_TOKENS_PER_SEC[0]} to {GEMINI_VIDEO_TOKENS_PER_SEC[1]}, plus {GEMINI_AUDIO_TOKENS_PER_SEC} audio",
         "Gemini video docs; resolution is not set in the code"),
        ("Thinking tokens per Gemini call",
         f"{GEMINI_THINKING_TOKENS_PER_CALL[0]} to {GEMINI_THINKING_TOKENS_PER_CALL[1]:,}",
         "Assumed allowance; needs measurement"),
        ("Match report: per-minute call",
         f"about {MATCH_REPORT['minute_call_prompt_tokens']:,} prompt + video tokens in, "
         f"{MATCH_REPORT['minute_call_output_tokens']} out",
         "1 call per minute of video; from prompt length in the code"),
        ("Match report: report call",
         f"about {MATCH_REPORT['report_prompt_tokens']:,} + {MATCH_REPORT['report_prompt_tokens_per_video_min']} per video minute in, "
         f"{MATCH_REPORT['report_output_tokens']:,} out",
         "1 call per match"),
        ("Training plan: team plan",
         f"about {TRAINING_PLAN['team_plan_prompt_tokens']:,} + {TRAINING_PLAN['team_plan_prompt_tokens_per_video_min']} per video minute in, "
         f"{TRAINING_PLAN['team_plan_output_tokens']:,} out",
         f"{TRAINING_PLAN['team_plan_calls']} calls per match (one per team)"),
        ("Training plan: player plan",
         f"about {TRAINING_PLAN['player_plan_prompt_tokens']:,} in, {TRAINING_PLAN['player_plan_output_tokens']:,} out",
         "1 call per match"),
        ("Training plan: tracking insights",
         f"about {TRAINING_PLAN['cv_insights_prompt_tokens']:,} in, {TRAINING_PLAN['cv_insights_output_tokens']:,} out",
         "1 call per match"),
        ("Corner kicks", UNKNOWN_LABEL, "Corner count and run time are not fixed by the code"),
    ]
    df = pd.DataFrame([[_md(c) for c in r] for r in rows], columns=["Assumption", "Value", "Basis"])
    st.table(df.set_index("Assumption"))


def render_cost_calculator():
    with st.container(key="methodology_cost_calculator"):
        st.markdown("#### 🧮 Cost calculator (planning estimate, not a quote)")
        st.markdown(
            "Pick a video length, how fast it should finish, a GPU tier and the products to include. "
            "The figures come from the assumptions listed at the bottom; nothing is measured or run here."
        )

        if _MINUTES_KEY not in st.session_state:
            st.session_state[_MINUTES_KEY] = 90.0
        st.number_input("Video length (minutes)", min_value=0.0, step=1.0, format="%g", key=_MINUTES_KEY)
        preset_cols = st.columns(3)
        for col, preset in zip(preset_cols, (10, 45, 90)):
            col.button(f"{preset} min", key=f"mcc_preset_{preset}", on_click=_set_minutes, args=(preset,),
                       width='stretch')

        speed = st.radio("How fast do you want it?", [_NO_RUSH, _TARGET], key="mcc_speed", horizontal=True)
        target_minutes = None
        if speed == _TARGET:
            target_minutes = st.number_input("Target turnaround (minutes)", value=20, step=1, format="%d",
                                             key="mcc_target_minutes")

        tier = st.selectbox(
            "GPU tier", list(GPU_TIERS), key="mcc_tier",
            format_func=lambda k: f"{GPU_TIERS[k]['label']}, about ${GPU_TIERS[k]['usd_per_hour']:.2f}/h",
        )

        st.markdown("**Products to include**")
        selected = [key for key, label in PRODUCTS.items() if st.checkbox(label, value=True, key=f"mcc_product_{key}")]

        st.markdown("---")
        try:
            result = estimate(st.session_state[_MINUTES_KEY], target_minutes, tier, selected)
        except ValueError as exc:
            st.error(f"{exc} No estimate shown.")
            return
        if not result["items"]:
            st.info("Select at least one product to see an estimate.")
            return

        st.markdown(f"**Cost per product ({ESTIMATE_LABEL})**")
        rows = [
            (i["label"],
             format_usd_range(i["low"], i["high"]) if i["status"] == "ok" else UNKNOWN_LABEL,
             i["how"])
            for i in result["items"]
        ]
        df = pd.DataFrame([[_md(c) for c in r] for r in rows], columns=["Product", "Cost range", "How it was calculated"])
        st.table(df.set_index("Product"))

        if any(i["status"] == "ok" for i in result["items"]):
            total = format_usd_range(result["total_low"], result["total_high"])
        else:
            total = "nothing costed yet"
        st.markdown(_md(f"**{result['total_label']}: {total}** ({ESTIMATE_LABEL})"))
        if result["has_unknown"]:
            st.caption("Corner kicks is not in the total because its cost cannot be derived from the code yet.")

        gpu = result["gpu"]
        if gpu:
            gpus = str(gpu["gpus_low"]) if gpu["gpus_low"] == gpu["gpus_high"] else f"{gpu['gpus_low']} to {gpu['gpus_high']}"
            finish = format_duration_range(gpu["finish_minutes_low"], gpu["finish_minutes_high"])
            col_gpus, col_finish = st.columns(2)
            col_gpus.metric("GPUs needed at once", gpus)
            col_finish.metric("Expected finish time (CV models)", finish)
            if gpu["gpus_low"] == gpu["gpus_high"] == 1:
                st.caption("Assumes 1 GPU.")
            elif gpu["gpus_low"] == gpu["gpus_high"]:
                st.caption(f"Assumes about {gpu['gpus_low']} GPUs running in parallel.")
            else:
                st.caption(f"Assumes about {gpu['gpus_low']} to {gpu['gpus_high']} GPUs running in parallel.")
            st.caption(
                f"Both figures are a {ESTIMATE_LABEL}. The turnaround applies to the CV models only; "
                "how long the Gemini calls take is not modelled."
            )
        else:
            st.caption(f"No GPU is needed for the selected products ({ESTIMATE_LABEL}).")

        for warning in result["warnings"]:
            st.warning(warning)

        with st.expander("Assumptions behind these figures"):
            _assumptions_table()
            st.caption(
                "Token counts are prompt and response lengths from the code divided by 4, which is a rule of thumb. "
                "Retries are not counted. Every figure is a planning estimate, not a quote."
            )
