"""
Training plan generation + rendering for the AI Tactical Coaching Dashboard
(Part 5.1). Kept out of app.py (already 2000+ lines) - imported from there.

Two plans, from genuinely different data sources:
- TEAM plan: aggregated over the FULL raw_data array (every analyzed minute)
  plus the existing coach report text - legitimate to summarize across the
  whole analyzed segment, since that's real per-minute Gemini tactical data.
- PLAYER plan: from the CV pipeline's stats.json players[] list ONLY - a
  single ~30-second peak-momentum window, never a full match. Explicitly
  scoped and captioned as such everywhere it's shown (PLAYER_PLAN_SCOPE_NOTE)
  - no full-match fatigue/stamina-curve claims. The mockup this UI's visual
  style is based on made exactly that mistake (a fabricated "pace drop-off,
  final 15 min" stat); this module deliberately does not reproduce it.

Persisted separately from bundle.json (curated_matches/<id>/training_plan.json,
or .cache/training_plans/<video_hash>.json for a live upload) so a bad manual
edit can never touch the underlying, verified match analysis data.
"""
import copy
import html as _html
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
from google import genai
from google.genai import types

PLAYER_PLAN_SCOPE_NOTE = (
    "Based on the match's peak-momentum window only — not a full-match average. "
    "Full-match fatigue tracking is on our roadmap."
)

FOCUS_CATEGORY_COLORS = {
    'press': '#8fb4ff',
    'trans': '#ffb27a',
    'build': '#a7e3b0',
    'rest': '#9aa3b5',
}
FOCUS_CATEGORIES = list(FOCUS_CATEGORY_COLORS.keys())

# ==========================================
# EDIT TRACKING
#
# Per-field, not per-day/session, so editing e.g. just a drill's duration
# doesn't falsely mark that day's why_stat as unverified. Detected by
# comparing the CURRENT (possibly-edited) value against a pristine snapshot
# captured once at generation time (_original_days / _original_players,
# never mutated afterward) - not against "whatever was loaded last," so an
# edit that's later typed back to its original value correctly un-flags.
#
# --accent (blue) for a manual edit here; --accent2 (green) is reserved for
# "grounded in real match data" (reusing the mockup's existing chat-message
# .src citation-pill pattern - a colored dot + short source text) so it stays
# free for a future chat-confirmed-edit state without colliding with a
# meaning it already carries. NOTE: the training_plans_mockup.html on disk
# has no "Updated via chat" badge to reuse (only that chat .src pill) - if a
# newer mockup with that specific pattern exists, it wasn't the one read for
# this fix; flagged rather than guessed at.
# ==========================================
EDITED_COLOR = '#4f8cff'   # --accent
GROUNDED_COLOR = '#22c55e'  # --accent2
CHAT_CONFIRMED_COLOR = '#f5a623'  # --warn - "confirmed via chat" (chatbot.py)

TEAM_DAY_TRACKED_FIELDS = ['focus_label', 'focus_category', 'why_stat', 'drills']
PLAYER_SESSION_TRACKED_FIELDS = ['title', 'note', 'tag']


def recompute_edited_fields(items, original_items, tracked_fields):
    """Compares each item's tracked fields against its pristine original
    snapshot and updates item['edited_fields'] in place. Call this every
    rerun right after native edit widgets write values back into `items` -
    by the time "Save changes" persists the draft, it already carries
    correct flags, so save/edit/reset logic itself needs no changes."""
    for i, item in enumerate(items):
        original = original_items[i] if original_items and i < len(original_items) else {}
        item['edited_fields'] = [f for f in tracked_fields if item.get(f) != original.get(f)]


def _esc(s):
    return _html.escape(str(s if s is not None else ""))


# ==========================================
# GENERATION
# ==========================================

def compute_team_source_stats(raw_data):
    """Real, computable aggregates over every analyzed minute - grounds both
    the Gemini prompt and the on-screen 'why' stats, so nothing in the plan
    is invented. Same underlying columns app.py's Step 3 dashboard already
    computes modes/counts from, just re-derived here for this module's own
    prompt (kept independent rather than threading dashboard locals through)."""
    df = pd.DataFrame(raw_data)
    n_minutes = len(df)

    def freq(col, values):
        if col not in df.columns:
            return 0
        return int(df[col].astype(str).str.lower().isin(values).sum())

    return {
        "n_minutes": n_minutes,
        "team_a_gegenpress_minutes": freq('team_a_pressing_trigger', ['gegenpress']),
        "team_b_gegenpress_minutes": freq('team_b_pressing_trigger', ['gegenpress']),
        "team_a_counter_minutes": freq('team_a_transition_threat', ['counter_attack', 'fast_vertical_transition']),
        "team_b_counter_minutes": freq('team_b_transition_threat', ['counter_attack', 'fast_vertical_transition']),
        "team_a_defensive_fullback_minutes": freq('team_a_fullback_role', ['defensive']),
        "team_b_defensive_fullback_minutes": freq('team_b_fullback_role', ['defensive']),
        "team_a_overlapping_fullback_minutes": freq('team_a_fullback_role', ['overlapping']),
        "team_b_overlapping_fullback_minutes": freq('team_b_fullback_role', ['overlapping']),
        "team_a_dropdeep_minutes": freq('team_a_defensive_line_action', ['drop_deep']),
        "team_b_dropdeep_minutes": freq('team_b_defensive_line_action', ['drop_deep']),
    }


def generate_team_plan(raw_data, ai_report_text, team_a, team_b, api_key):
    """One Gemini call -> a 7-day team training schedule. Returns None on
    total failure (caller shows an error, never a placeholder plan)."""
    stats = compute_team_source_stats(raw_data)
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an elite soccer fitness/tactics coach. Using the coach report and the real
per-minute statistics below (from {stats['n_minutes']} analyzed minutes of a match
between {team_a} and {team_b}), build a 7-day team training schedule for the week
following this match, for {team_a} specifically (the team whose staff commissioned
this report).

COACH REPORT (for tactical context):
{ai_report_text[:4000]}

REAL PER-MINUTE STATISTICS (use these exact numbers in "why_stat" — never invent
different numbers):
- {team_a} used gegenpress in {stats['team_a_gegenpress_minutes']} of {stats['n_minutes']} minutes; {team_b} in {stats['team_b_gegenpress_minutes']}.
- {team_a} was involved in a counter-attack or fast vertical transition in {stats['team_a_counter_minutes']} minutes; {team_b} in {stats['team_b_counter_minutes']}.
- {team_a}'s fullbacks played defensive (not attacking) in {stats['team_a_defensive_fullback_minutes']} minutes and overlapping in {stats['team_a_overlapping_fullback_minutes']}; {team_b}'s were defensive in {stats['team_b_defensive_fullback_minutes']}, overlapping in {stats['team_b_overlapping_fullback_minutes']}.
- {team_a} dropped its defensive line deep in {stats['team_a_dropdeep_minutes']} minutes; {team_b} in {stats['team_b_dropdeep_minutes']}.

Return EXACTLY 7 days (Monday through Sunday), each addressing a real weakness or
pattern from the numbers above. Include at least one full recovery/rest day and a
light matchday-minus-1 day. Each day object must have:
- "day": one of Monday/Tuesday/Wednesday/Thursday/Friday/Saturday/Sunday
- "focus_label": short label, 2-4 words (e.g. "Pressing Trigger")
- "focus_category": one of "press", "trans", "build", "rest" (use "rest" only for
  genuine recovery/rest days)
- "drills": a list of 1-2 objects, each with "title" (string), "duration_min"
  (integer), "note" (one sentence)
- "why_stat": one sentence citing a REAL number from above justifying this day's
  focus (a rest day may instead explain its recovery purpose)

Respond with ONLY a JSON array of exactly 7 such day objects.
"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            days = json.loads(response.text)
            if isinstance(days, dict) and 'days' in days:
                days = days['days']
            if isinstance(days, list) and len(days) > 0:
                for d in days:
                    d['edited_fields'] = []
                    d['chat_confirmed_fields'] = []
                return {
                    "days": days,
                    # Pristine snapshot, deep-copied so later edits to `days`
                    # can never alias into this - the only thing edit-tracking
                    # ever compares against. Never touched again except by a
                    # fresh Generate (Reset wipes the whole plan, this included).
                    "_original_days": copy.deepcopy(days),
                    "source_note": (
                        f"Generated from real per-minute tactical patterns across the "
                        f"{stats['n_minutes']} analyzed minutes of this match."
                    ),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            time.sleep(3)
    return None


def _player_stat_cards(stats_json, team_a, team_b, team_mapping, cv_team_label_fn, limit=12):
    cards = []
    for p in stats_json.get('players', []):
        if p.get('top_speed_confidence') not in ('high', 'medium'):
            continue
        team_label = cv_team_label_fn(p.get('team'), team_a, team_b, team_mapping)
        cards.append({
            "player_id": p.get('player_id'),
            "team_label": team_label,
            "top_speed_kmh": p.get('top_speed_kmh'),
            "avg_speed_kmh": p.get('avg_speed_kmh'),
            "total_distance_m": p.get('total_distance_m'),
            "confidence": p.get('top_speed_confidence'),
        })
    cards.sort(key=lambda x: x.get('total_distance_m') or 0, reverse=True)
    return cards[:limit]


def generate_player_plan(stats_json, team_a, team_b, team_mapping, cv_team_label_fn, api_key):
    """Player stat numbers are real data straight from stats.json - no Gemini
    call needed for those. One Gemini call turns them into session text,
    explicitly barred from inventing any full-match/fatigue-curve framing."""
    player_cards = _player_stat_cards(stats_json, team_a, team_b, team_mapping, cv_team_label_fn)
    if not player_cards:
        return None

    client = genai.Client(api_key=api_key)
    players_payload = json.dumps([
        {k: v for k, v in c.items() if k != 'team_label'} for c in player_cards
    ], indent=2)

    prompt = f"""
You are an elite soccer fitness coach. Below is REAL tracked physical data for
{len(player_cards)} players from a computer-vision analysis of a single ~30-second
peak-momentum window of a match (NOT the full match, NOT a full-match average).

{players_payload}

For each player (by player_id), propose a 5-day training week (Monday-Friday), one
session per day, grounded ONLY in the numbers given above.

CRITICAL RULES:
- Never mention "full match", "90 minutes", "final minutes", or "fatigue curve" — you
  only have this one ~30-second window's totals, not a time series.
- Do not invent a stamina/fatigue trend.
- Every session's "note" must reference a real number from the data above.

Return ONLY a JSON object mapping each player_id (as a string) to an object with
"sessions": a list of exactly 5 objects (Mon-Fri), each with "day", "title", "note"
(one sentence), and "tag" (a short category like "Top speed" or "Endurance").
"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash', contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
            )
            sessions_by_player = json.loads(response.text)
            if isinstance(sessions_by_player, dict):
                for card in player_cards:
                    pid = str(card['player_id'])
                    sessions = (sessions_by_player.get(pid) or {}).get('sessions', [])
                    for s in sessions:
                        s['edited_fields'] = []
                        s['chat_confirmed_fields'] = []
                    card['sessions'] = sessions
                return {
                    "players": player_cards,
                    "_original_players": copy.deepcopy(player_cards),
                    "scope_note": PLAYER_PLAN_SCOPE_NOTE,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            time.sleep(3)
    return None


# ==========================================
# RENDERING (HTML fragments for st.components.v1.html - this reuses the
# training_plans_mockup.html CSS almost verbatim, since it already matches
# this app's theme tokens exactly)
# ==========================================

_BASE_CSS = """
<style>
  :root{
    --bg:#0b0e14; --panel:#12161f; --panel-raised:#171c28; --line:#232a38;
    --accent:#4f8cff; --accent2:#22c55e; --amber:#f5a623;
    --text:#e6e9f0; --muted:#8a93a6; --muted2:#5b6478;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Roboto,sans-serif;line-height:1.5;}
  .source-strip{display:flex;gap:10px;align-items:center;background:var(--panel);
    border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:20px;font-size:13px;color:var(--muted);}
  .source-strip .dot{width:6px;height:6px;border-radius:50%;background:var(--accent);flex:none;}
  .source-strip b{color:var(--text);font-weight:600;}
  .scope-note{font-size:12px;color:var(--amber);background:rgba(245,166,35,.08);
    border:1px solid rgba(245,166,35,.3);border-radius:10px;padding:10px 14px;margin-bottom:18px;}
  .week-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:24px;}
  .day{background:var(--panel);border:1px solid var(--line);border-radius:11px;
    padding:14px 12px;min-height:190px;display:flex;flex-direction:column;gap:8px;position:relative;}
  .day.rest{background:linear-gradient(180deg, rgba(90,100,120,.06), transparent);}
  .day.edited{border-color:rgba(79,140,255,.55);}
  .day.chat{border-color:rgba(245,166,35,.55);}
  .edited-badge{position:absolute;top:8px;right:8px;font-size:9px;font-weight:700;letter-spacing:.4px;
    text-transform:uppercase;color:#fff;background:#4f8cff;padding:2px 6px;border-radius:5px;}
  .day-name{font-size:11.5px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.6px;}
  .day-focus{font-size:14.5px;font-weight:650;margin-bottom:2px;}
  .drill{font-size:12.5px;color:var(--muted);border-left:2px solid var(--line);padding-left:8px;}
  .drill b{color:var(--text);font-weight:600;display:block;font-size:12.5px;}
  .drill span.dur{color:var(--muted2);}
  .why-row{display:flex;gap:14px;margin-top:2px;padding-top:12px;border-top:1px solid var(--line);flex-wrap:wrap;}
  .why{flex:1;min-width:180px;background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px 16px;}
  .why .lbl{font-size:11px;color:var(--muted2);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;}
  .why .stat{font-size:13.5px;color:var(--text);}
  .why .stat b{color:#8fb4ff;}
  /* Grounded/edited source-citation pill - extends the mockup's chat-message
     .src pattern (colored dot + short text) to why_stat / player-note claims,
     since that's the mockup's existing visual language for "this claim traces
     to a real data source," not an invented pattern. */
  .src-pill{display:inline-flex;align-items:center;gap:5px;font-size:10.5px;color:var(--muted2);
    border:1px solid var(--line);border-radius:6px;padding:2px 7px;margin-top:8px;}
  .src-pill .d{width:5px;height:5px;border-radius:50%;flex:none;}
  .src-pill.grounded .d{background:var(--accent2);}
  .src-pill.edited{color:#8fb4ff;border-color:rgba(79,140,255,.4);}
  .src-pill.edited .d{background:var(--accent);}
  .src-pill.chat{color:#f5c37a;border-color:rgba(245,166,35,.4);}
  .src-pill.chat .d{background:var(--amber);}
  .player-head{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap;}
  .pcard{flex:1;min-width:160px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;cursor:help;}
  .pcard .lbl{font-size:11px;color:var(--muted2);font-weight:600;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;}
  .pcard .big{font-size:24px;font-weight:700;}
  .pcard .trend{font-size:12.5px;color:var(--muted);margin-top:4px;}
  .plan-list{display:flex;flex-direction:column;gap:10px;}
  .plan-item{display:flex;gap:14px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);
    border-radius:11px;padding:14px 16px;}
  .plan-item.edited{border-color:rgba(79,140,255,.55);}
  .plan-item.chat{border-color:rgba(245,166,35,.55);}
  .plan-item .day-tag{flex:none;width:64px;font-size:11px;font-weight:700;color:var(--muted);
    text-transform:uppercase;letter-spacing:.5px;padding-top:1px;}
  .plan-item .content b{font-size:14px;display:block;margin-bottom:3px;}
  .plan-item .content p{margin:0;font-size:13px;color:var(--muted);}
  .plan-item .content .tag{display:inline-block;margin-top:7px;font-size:11px;padding:3px 8px;border-radius:6px;
    background:rgba(245,166,35,.12);color:#f5b95a;font-weight:600;}
</style>
"""


def _src_pill(is_edited, is_chat_confirmed=False):
    """Small citation pill (colored dot + text) - three provenance states,
    per final_three_state_mockup.html: green 'grounded' (untouched), amber
    'confirmed via chat' (an AI-authored proposal the user explicitly
    confirmed - chatbot.py), blue 'manually edited' (typed directly).
    Precedence when a field carries both flags: edited > chat-confirmed >
    grounded - a later hand-edit is the more recent human action and wins
    visually, while both flags stay in the JSON (see chatbot.py's
    _rebase_and_mark)."""
    if is_edited:
        return '<div class="src-pill edited"><div class="d"></div> Manually edited — not verified against match data</div>'
    if is_chat_confirmed:
        return '<div class="src-pill chat"><div class="d"></div> Confirmed via chat</div>'
    return '<div class="src-pill grounded"><div class="d"></div> Grounded in match data</div>'


def render_team_calendar_html(team_plan):
    days = team_plan.get("days", [])
    day_divs, why_divs = [], []
    for d in days:
        cat = d.get("focus_category", "build")
        rest_cls = " rest" if cat == "rest" else ""
        edited_fields = d.get("edited_fields") or []
        chat_fields = d.get("chat_confirmed_fields") or []
        if edited_fields:
            day_edited_cls, day_badge = " edited", '<div class="edited-badge">Edited</div>'
        elif chat_fields:
            day_edited_cls = " chat"
            day_badge = '<div class="edited-badge" style="background:var(--amber);">Via Chat</div>'
        else:
            day_edited_cls, day_badge = "", ""
        drills_html = "".join(
            f'<div class="drill"><b>{_esc(dr.get("title"))}</b>'
            f'<span class="dur">{_esc(dr.get("duration_min"))} min</span> — {_esc(dr.get("note"))}</div>'
            for dr in d.get("drills", [])
        )
        color = FOCUS_CATEGORY_COLORS.get(cat, '#e6e9f0')
        day_divs.append(
            f'<div class="day{rest_cls}{day_edited_cls}">{day_badge}'
            f'<div class="day-name">{_esc(d.get("day"))}</div>'
            f'<div class="day-focus" style="color:{color}">{_esc(d.get("focus_label"))}</div>'
            f'{drills_html}</div>'
        )
        if cat != "rest" and d.get("why_stat"):
            why_divs.append(
                f'<div class="why"><div class="lbl">Why {_esc(d.get("focus_label"))}</div>'
                f'<div class="stat">{_esc(d.get("why_stat"))}</div>'
                f'{_src_pill("why_stat" in edited_fields, "why_stat" in chat_fields)}</div>'
            )
    source_note = team_plan.get("source_note", "")
    return (
        _BASE_CSS
        + f'<div class="source-strip"><div class="dot"></div>{_esc(source_note)}</div>'
        + f'<div class="week-grid">{"".join(day_divs)}</div>'
        + f'<div class="why-row">{"".join(why_divs[:3])}</div>'
    )


def _render_session_item(s):
    edited_fields = s.get("edited_fields") or []
    chat_fields = s.get("chat_confirmed_fields") or []
    if edited_fields:
        item_edited_cls, item_badge = " edited", '<div class="edited-badge">Edited</div>'
    elif chat_fields:
        item_edited_cls = " chat"
        item_badge = '<div class="edited-badge" style="background:var(--amber);">Via Chat</div>'
    else:
        item_edited_cls, item_badge = "", ""
    return (
        f'<div class="plan-item{item_edited_cls}" style="position:relative;">{item_badge}'
        f'<div class="day-tag">{_esc(s.get("day"))}</div>'
        f'<div class="content"><b>{_esc(s.get("title"))}</b><p>{_esc(s.get("note"))}</p>'
        + (f'<span class="tag">{_esc(s.get("tag"))}</span>' if s.get("tag") else '')
        + _src_pill("note" in edited_fields, "note" in chat_fields)
        + '</div></div>'
    )


def render_player_card_html(player):
    sessions_html = "".join(_render_session_item(s) for s in player.get("sessions", []))
    conf = player.get("confidence", "high")
    conf_badge = "✅ high confidence" if conf == "high" else f"⚠️ {_esc(conf)} confidence"
    top_speed = player.get('top_speed_kmh') or 0
    avg_speed = player.get('avg_speed_kmh') or 0
    dist = player.get('total_distance_m') or 0
    return (
        _BASE_CSS
        + f'<div class="scope-note">{_esc(PLAYER_PLAN_SCOPE_NOTE)}</div>'
        + '<div class="player-head">'
        + (f'<div class="pcard" title="Fastest single moment tracked for this player during the ~30s window. '
           f'Confidence reflects how reliably the tracker held this player\'s identity at that moment.">'
           f'<div class="lbl">Top speed (this window)</div><div class="big">{top_speed:.1f} km/h</div>'
           f'<div class="trend">{conf_badge}</div></div>')
        + (f'<div class="pcard" title="Mean speed across every tracked frame for this player in the window - '
           f'includes standing/walking moments, so it is well below top speed.">'
           f'<div class="lbl">Average speed (this window)</div><div class="big">{avg_speed:.1f} km/h</div></div>')
        + (f'<div class="pcard" title="Total ground covered across the tracked frames of this ~30s window - '
           f'not a full-match total.">'
           f'<div class="lbl">Distance covered (this window)</div><div class="big">{dist:.0f} m</div></div>')
        + '</div>'
        + f'<div class="plan-list">{sessions_html}</div>'
    )


# ==========================================
# PERSISTENCE (separate from bundle.json - see module docstring)
# ==========================================

def get_training_plan_path(source, key, curated_matches_dir, cache_dir):
    if source == "curated":
        return curated_matches_dir / key / "training_plan.json"
    plans_dir = cache_dir / "training_plans"
    return plans_dir / f"{key}.json"


def load_training_plan(source, key, curated_matches_dir, cache_dir):
    if not source or not key:
        return None
    path = get_training_plan_path(source, key, curated_matches_dir, cache_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_training_plan(source, key, plan, curated_matches_dir, cache_dir):
    if not source or not key:
        return False
    path = get_training_plan_path(source, key, curated_matches_dir, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, path)
    return True


def delete_training_plan(source, key, curated_matches_dir, cache_dir):
    if not source or not key:
        return
    path = get_training_plan_path(source, key, curated_matches_dir, cache_dir)
    try:
        path.unlink()
    except OSError:
        pass
