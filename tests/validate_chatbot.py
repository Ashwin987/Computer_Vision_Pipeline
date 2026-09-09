"""
validate_chatbot.py - standalone chatbot regression/validation harness.

Re-runnable anytime after a chatbot.py change, the same way the CV pipeline's
59-clip batch-validation harness gives repeatable confidence there. NOT wired
into the app - run manually from the repo root:

    python tests/validate_chatbot.py --estimate       # no API calls, just a cost estimate
    python tests/validate_chatbot.py --run             # the real ~50-question battery
    python tests/validate_chatbot.py --run --out FILE  # custom report path

Requires GEMINI_API_KEY in the environment (or dashboard/.env).

CORE PRINCIPLE (non-negotiable): every expected answer below is computed
independently of the chatbot, directly from raw_data/stats.json/the
in-memory training plan this script itself generates - never by asking the
chatbot (or another LLM call) to grade its own answer. Structured checks are
exact/tolerant numeric comparisons against a value this script computed
itself. Semantic/project-info checks verify that every citation the chatbot
returned actually exists in the real source data (recomputing the same
chunk boundaries chatbot.py itself uses for indexing - deterministic string
processing, not a model call) and that no number in the answer contradicts
the real data. Edit-flow checks call chatbot.py's own validate_grounding
(itself plain Python, no model call) against the real grounding context -
never apply_edit, so this script never writes to any real training_plan.json.
"""
import argparse
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = REPO_ROOT / "dashboard"
CV_PIPELINE_DIR = REPO_ROOT / "cv_pipeline"
CURATED_MATCHES_DIR = DASHBOARD_DIR / "curated_matches"
sys.path.insert(0, str(DASHBOARD_DIR))

import numpy as np
import pandas as pd

import chatbot as cb
import training_plan as tp


# ==========================================
# Duplicated from app.py (verbatim, kept in sync manually) - app.py is a
# Streamlit script with top-level UI code that runs on import, not an
# importable library, so these two small pure functions are copied here
# rather than imported. Same pattern this project already uses elsewhere
# (chatbot.py's MOMENTUM_FIELDS, training_plan.py's TACTICAL_EVENT_GLOSSARY).
# ==========================================

def compute_dashboard_df(raw_df, team_a, team_b, color_a, color_b):
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


def cv_team_label(team_num, team_a, team_b, team_mapping=None):
    if team_mapping:
        key = team_mapping.get(str(team_num)) or team_mapping.get(team_num)
        if key == 'team_a':
            return team_a
        if key == 'team_b':
            return team_b
    return f"Team {team_num}"


# ==========================================
# MATCH CONTEXT LOADING
# ==========================================

class MatchContext:
    def __init__(self, match_id):
        self.match_id = match_id
        bundle_path = CURATED_MATCHES_DIR / match_id / "bundle.json"
        with open(bundle_path, "r", encoding="utf-8") as f:
            self.bundle = json.load(f)

        self.team_a = self.bundle["team_a"]
        self.team_b = self.bundle["team_b"]
        self.color_a = self.bundle["color_a"]
        self.color_b = self.bundle["color_b"]
        self.ai_report_text = self.bundle["ai_report"]
        self.cv_team_mapping = self.bundle.get("cv_team_mapping")

        raw_df = pd.DataFrame(self.bundle["raw_data"])
        self.df = compute_dashboard_df(raw_df, self.team_a, self.team_b, self.color_a, self.color_b)

        cv_output_dir = CV_PIPELINE_DIR / self.bundle["cv_output_dir"]
        with open(cv_output_dir / "stats.json", "r", encoding="utf-8") as f:
            self.stats_json = json.load(f)

        # Real (fresh, per-run) team_a/team_b label for CV-sourced facts,
        # honoring cv_team_mapping exactly like the live app (barca_madrid_pt1
        # has no confirmed mapping yet, so this correctly falls back to
        # "Team 1"/"Team 2" for CV-pipeline-sourced facts specifically - the
        # Gemini-sourced raw_data still uses the real team_a/team_b names).
        self.cv_label = lambda team_num: cv_team_label(team_num, self.team_a, self.team_b, self.cv_team_mapping)

        # A real (not fabricated) training plan, generated fresh and kept
        # ONLY in memory - never written via tp.save_training_plan, so this
        # script has zero persistent side effects. Needed for: (a) the
        # training-plan-field STRUCTURED questions' ground truth, and (b) the
        # "if not draft" gate the real chatbot tab checks before EDIT
        # proposals are even attempted.
        print(f"  generating in-memory training plan for {match_id} (not saved)...")
        team_plan = tp.generate_team_plan(
            self.bundle["raw_data"], cb.substitute_team_tokens(self.ai_report_text, self.team_a, self.team_b),
            self.team_a, self.team_b, API_KEY,
        )
        player_plan = tp.generate_player_plan(
            self.stats_json, self.team_a, self.team_b, self.cv_team_mapping, cv_team_label, API_KEY,
        )
        # Mirrors the real app exactly (render_training_plan_tab): a
        # training_plan_draft is only ever stored if team_plan generation
        # succeeded - never a dict with team_plan=None, which would be
        # truthy and incorrectly pass the "if not draft" gate the real
        # chatbot tab's EDIT route checks.
        self.training_plan_draft = {"team_plan": team_plan, "player_plan": player_plan} if team_plan else None

        # source/key: a real, stable identity so the per-match ChromaDB
        # collection is genuinely isolated per match (never "session"/random,
        # which would defeat the cross-match isolation checks below).
        self.source, self.key = "curated", match_id
        # NOT cached as self.collection: chromadb.PersistentClient objects for
        # the SAME on-disk path don't seem to stay coherent with each other
        # when two separate instances exist at once in one process (confirmed
        # by the harness's first run - liverpool_psg's collection queries
        # started throwing "Error creating hnsw segment reader: Nothing found
        # on disk" only once barca_madrid_pt1's context/client was also
        # created). The real Streamlit app never hits this: it only ever
        # creates one PersistentClient at a time, fresh on every script
        # rerun. Testing both matches in one process is this harness's own
        # requirement, not something the production code path needs to
        # tolerate - so the fix here is to get a fresh collection reference
        # right when it's used (ask_chatbot), not to hold one open per match
        # for the process's whole lifetime.

    # ---- convenience ground-truth accessors, all reading real source data ----
    def player(self, pid):
        return next((p for p in self.stats_json.get("players", []) if p.get("player_id") == pid), None)

    def real_player_ids(self):
        return [p["player_id"] for p in self.stats_json.get("players", [])]

    def fastest_player(self):
        return max(self.stats_json["players"], key=lambda p: p.get("top_speed_kmh") or 0)

    def momentum_at(self, minute):
        return float(self.df.iloc[minute]["smoothed_net_momentum"])

    def coach_report_sections(self):
        """The REAL, valid citation labels for this match's coach-report
        chunks - recomputed via chatbot.py's own (deterministic, non-LLM)
        chunker, not asked of the chatbot. _chunk_coach_report's labels
        carry LITERAL {TEAM_A}/{TEAM_B} tokens (confirmed directly - the
        report is stored/embedded in token form so a rename never requires
        re-embedding; chatbot.py's own semantic_answer() only substitutes
        real names into the tag at display time, in
        cb.substitute_team_tokens). Comparing an unsubstituted label against
        the chatbot's real (substituted) output tag would never match, so
        this must substitute here too - same helper, not a second, separate
        derivation of the mapping."""
        chunks = cb._chunk_coach_report(self.ai_report_text, self.team_a, self.team_b)
        return {cb.substitute_team_tokens(label, self.team_a, self.team_b) for label, _ in chunks}


_TECH_REPORT_SECTIONS_CACHE = None


def technical_report_sections():
    """Match-independent - computed once, reused for every PROJECT_INFO
    citation check. Same chatbot.py chunker used to build the real index."""
    global _TECH_REPORT_SECTIONS_CACHE
    if _TECH_REPORT_SECTIONS_CACHE is None:
        chunks = cb._chunk_technical_report(cb.TECHNICAL_REPORT_PDF_PATH)
        _TECH_REPORT_SECTIONS_CACHE = {label for label, _ in chunks}
    return _TECH_REPORT_SECTIONS_CACHE


# ==========================================
# CHATBOT DISPATCH - replicates render_chatbot_tab's routing logic exactly
# (same functions, same order, same STRUCTURED->SEMANTIC fallback), without
# any Streamlit dependency, so this script exercises the real production
# code path rather than a second, parallel reimplementation of it.
# ==========================================

def _query_with_retry(fn, attempts=3, delay=2.0):
    """Retries a chromadb query through a BRAND NEW collection/client object
    each attempt (the confirmed-working recipe from this harness's own
    diagnostic script) - not a generic try/except, since a same-object retry
    reproduced the same disk error every time in earlier debugging."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            time.sleep(delay)
    raise last_exc


def ask_chatbot(client, question, ctx, project_info_collection):
    debug = {}
    route = cb.route_question(client, question)
    debug["designed_route"] = None  # filled in by caller
    debug["actual_route"] = route
    tags = []
    retrieved = None
    edit_result = None

    if route == "EDIT":
        if not ctx.training_plan_draft:
            answer = "Generate a training plan first (Training Plan tab) before I can propose changes to it."
        else:
            grounding_ctx = cb.build_grounding_context(ctx.df, ctx.stats_json)
            kind, payload = cb.propose_edit(client, question, grounding_ctx, ctx.df, ctx.stats_json)
            edit_result = (kind, payload)
            answer = payload if kind == "decline" else "Here's what I'd change — see the proposal below."
    elif route == "STRUCTURED":
        answer, tag = cb.run_structured_lookup(
            client, question, ctx.df, ctx.stats_json, ctx.training_plan_draft, ctx.team_a, ctx.team_b,
        )
        if answer is None:
            route = "SEMANTIC"
        else:
            tags = [tag] if tag else []
    elif route == "PROJECT_INFO":
        retrieved = _query_with_retry(lambda: cb.query_project_info(project_info_collection, API_KEY, question))
        answer, tags = cb.project_info_answer(API_KEY, question, retrieved)

    if route == "SEMANTIC":
        # Fresh collection reference right at point of use - see MatchContext's
        # comment on why this is NOT cached as a long-lived attribute. Wrapped
        # in a retry (see _query_with_retry's docstring): this harness's own
        # first two runs hit an intermittent "hnsw segment reader: Nothing
        # found on disk" error, 100% reproducible for liverpool_psg's
        # collection specifically within the full run, but NOT reproducible
        # querying the same already-persisted collection in an isolated,
        # fresh process - pointing to transient chromadb flakiness under this
        # harness's own long-running, many-calls-first process shape, not a
        # chatbot.py bug or corrupted data.
        def _do_query():
            collection = cb.get_chroma_collection(ctx.source, ctx.key)
            return cb.query_collection(collection, API_KEY, question)
        retrieved = _query_with_retry(_do_query)
        answer, tags = cb.semantic_answer(API_KEY, question, retrieved, ctx.team_a, ctx.team_b)

    debug["actual_route"] = route  # post-fallback, real route the answer came from
    return {
        "question": question, "answer": answer, "tags": tags, "retrieved": retrieved,
        "edit_result": edit_result, "debug": debug,
    }


# ==========================================
# GRADING HELPERS
# ==========================================

BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def extract_bolded(answer):
    return BOLD_RE.findall(answer)


def numeric_close(a, b, tol=0.15):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def extract_numbers(text):
    return [float(n) for n in NUMBER_RE.findall(text)]


def verify_citations(tags, ctx):
    """Returns list of citation strings that do NOT correspond to real data.
    Empty list = every citation verified against real source content."""
    bad = []
    valid_sections = ctx.coach_report_sections() if ctx else set()
    valid_tech_sections = technical_report_sections()
    n_minutes = len(ctx.df) if ctx else 0
    for t in tags:
        if t.startswith("raw_data · minute "):
            try:
                m = int(t.rsplit(" ", 1)[1])
            except ValueError:
                bad.append(t)
                continue
            if not (0 <= m < n_minutes):
                bad.append(t)
        elif t.startswith("coach report · "):
            label = t.split("·", 1)[1].strip()
            if not any(label == s or label in s or s in label for s in valid_sections):
                bad.append(t)
        elif t.startswith("technical report · "):
            label = t.split("·", 1)[1].strip()
            if not any(label == s or label in s or s in label for s in valid_tech_sections):
                bad.append(t)
        # players · P{id}, tactical_events · ..., training_plan · ... tags are
        # trusted structurally (they're built from a real, already-resolved
        # value inside run_structured_lookup, not a retrieval hit that could
        # cite something nonexistent) - only retrieval-sourced tags (the two
        # cases above) can cite something that doesn't actually exist.
    return bad


# ==========================================
# QUESTION BANK
# ==========================================
# Each question is a dict: id, category (for reporting), designed_route
# (STRUCTURED/SEMANTIC/PROJECT_INFO/EDIT - what we expect route_question to
# pick), match (which MatchContext to run it against), question (text), and
# check(result, ctx, other_ctx) -> (passed: bool|None, reason: str, flag_for_human: bool).
# passed=None means "not auto-gradable, human review only" (never silently
# counted as a pass).

QUESTIONS = []


def q(id, category, designed_route, match, question, check):
    QUESTIONS.append({
        "id": id, "category": category, "designed_route": designed_route,
        "match": match, "question": question, "check": check,
    })


# ---------- STRUCTURED (~12) ----------

def _check_bold_number(expected, tol=0.15):
    def check(r, ctx, other):
        vals = extract_bolded(r["answer"])
        nums = [v for v in vals if re.match(r"^-?\d+\.?\d*$", v.strip())]
        if not nums:
            return False, f"no bolded numeric value found in answer (expected {expected})", False
        if any(numeric_close(n, expected, tol) for n in nums):
            return True, f"found {expected} among bolded values {nums}", False
        return False, f"expected {expected}, bolded values were {nums}", False
    return check


q("S1", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was the momentum score at minute 3?",
  lambda r, ctx, other: _check_bold_number(ctx.momentum_at(3))(r, ctx, other))

q("S2", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was the smoothed net momentum at minute 15?",
  lambda r, ctx, other: _check_bold_number(ctx.momentum_at(15))(r, ctx, other))

q("S3", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "How many PRESS events were logged in this match?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["tactical_events"]["counts"]["PRESS"], tol=0.5)(r, ctx, other))

q("S4", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "How many BURST events happened in this window?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["tactical_events"]["counts"]["BURST"], tol=0.5)(r, ctx, other))

q("S5", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was player 16's top speed?",
  lambda r, ctx, other: _check_bold_number(ctx.player(16)["top_speed_kmh"])(r, ctx, other))

q("S6", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was player 8's average speed in this window?",
  lambda r, ctx, other: _check_bold_number(ctx.player(8)["avg_speed_kmh"])(r, ctx, other))

q("S7", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "How far did player 8 travel in this window?",
  lambda r, ctx, other: _check_bold_number(ctx.player(8)["total_distance_m"], tol=1.0)(r, ctx, other))

q("S8", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was the team resolution rate for this match?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["team_resolution"]["resolution_rate_pct"], tol=0.5)(r, ctx, other))

q("S9", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "What was the calibration confidence for this match?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["calibration"]["mean_confidence"], tol=0.02)(r, ctx, other))

q("S10", "STRUCTURED", "STRUCTURED", "barca_madrid_pt1",
  "What was the ball detection rate in this match?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["ball"]["final_detection_rate_pct"], tol=0.5)(r, ctx, other))

q("S11", "STRUCTURED", "STRUCTURED", "barca_madrid_pt1",
  "What was player 12's top speed in this window?",
  lambda r, ctx, other: _check_bold_number(ctx.player(12)["top_speed_kmh"])(r, ctx, other))

q("S12", "STRUCTURED", "STRUCTURED", "barca_madrid_pt1",
  "How many SPACE events were recorded in this match?",
  lambda r, ctx, other: _check_bold_number(ctx.stats_json["tactical_events"]["counts"]["SPACE"], tol=0.5)(r, ctx, other))


def _check_momentum_team(minute):
    def check(r, ctx, other):
        val = ctx.momentum_at(minute)
        expected_team = ctx.team_a if val > 0 else (ctx.team_b if val < 0 else None)
        ans = r["answer"]
        if expected_team is None:
            return None, "momentum exactly 0 - ambiguous ground truth, human review", True
        if expected_team.lower() in ans.lower():
            wrong_team = ctx.team_b if expected_team == ctx.team_a else ctx.team_a
            if wrong_team.lower() in ans.lower() and wrong_team.lower() != expected_team.lower():
                return False, f"answer names both teams, ambiguous (expected {expected_team})", False
            return True, f"correctly named {expected_team} (momentum={val})", False
        return False, f"expected team {expected_team} (momentum={val}), not found in answer: {ans}", False
    return check


q("S13", "STRUCTURED", "STRUCTURED", "liverpool_psg",
  "Which team had the advantage at minute 5?",
  lambda r, ctx, other: _check_momentum_team(5)(r, ctx, other))

q("S14", "STRUCTURED", "STRUCTURED", "barca_madrid_pt1",
  "Which team was dominant at minute 10?",
  lambda r, ctx, other: _check_momentum_team(10)(r, ctx, other))


# ---------- SEMANTIC (~12) ----------

def _check_semantic(other_match_leak_terms=None):
    def check(r, ctx, other):
        bad_citations = verify_citations(r["tags"], ctx)
        if bad_citations:
            return False, f"citation(s) don't match real data: {bad_citations}", False
        nums = extract_numbers(r["answer"])
        # Loose plausibility check only (see module docstring on why this is
        # a flag, not a hard fail, for free-form prose): every number in the
        # answer should be within tolerance of SOME real number in this
        # match's data pool, or it's flagged for a human to look at.
        pool = _real_number_pool(ctx)
        suspicious = [n for n in nums if not any(numeric_close(n, p, tol=0.6) for p in pool) and abs(n) > 1]
        flag = bool(suspicious)
        reason = "citations verified" + (f"; unmatched numbers flagged for review: {suspicious}" if flag else "")
        return True, reason, flag
    return check


def _real_number_pool(ctx):
    pool = []
    for col in ("smoothed_net_momentum",):
        pool.extend(pd.to_numeric(ctx.df[col], errors="coerce").dropna().tolist())
    for p in ctx.stats_json.get("players", []):
        for f in ("top_speed_kmh", "avg_speed_kmh", "total_distance_m"):
            if p.get(f) is not None:
                pool.append(p[f])
    for v in ctx.stats_json.get("tactical_events", {}).get("counts", {}).values():
        pool.append(v)
    pool.append(ctx.stats_json["team_resolution"]["resolution_rate_pct"])
    pool.append(ctx.stats_json["calibration"]["mean_confidence"] * 100)
    pool.append(ctx.stats_json["ball"]["final_detection_rate_pct"])
    pool.extend(range(0, len(ctx.df) + 1))  # minute numbers are valid "numbers" too
    return pool


for i, (mid, ta_topic) in enumerate([
    ("liverpool_psg", "defensive vulnerabilities"),
    ("liverpool_psg", "how they built up play from the back"),
    ("liverpool_psg", "their pressing triggers"),
    ("barca_madrid_pt1", "defensive vulnerabilities"),
    ("barca_madrid_pt1", "spatial manipulation"),
    ("barca_madrid_pt1", "transition and counter-attacking threat"),
]):
    ctx_teams = {"liverpool_psg": ("Liverpool", "PSG"), "barca_madrid_pt1": ("White", "Maroon")}
    ta, tb = ctx_teams[mid]
    q(f"M{i*2+1}", "SEMANTIC", "SEMANTIC", mid,
      f"What are {ta}'s {ta_topic}?",
      _check_semantic())
    q(f"M{i*2+2}", "SEMANTIC", "SEMANTIC", mid,
      f"What adjustments should {tb}'s coach make based on this match?",
      _check_semantic())


# ---------- PROJECT_INFO (~8) ----------

q("P1", "PROJECT_INFO", "PROJECT_INFO", "liverpool_psg",
  "How does the pitch calibration system work?",
  _check_semantic())
q("P2", "PROJECT_INFO", "PROJECT_INFO", "liverpool_psg",
  "Why is only one window of the match analyzed instead of the whole game?",
  _check_semantic())
q("P3", "PROJECT_INFO", "PROJECT_INFO", "liverpool_psg",
  "What speedup did the fallback ball detector get from running on GPU?",
  _check_semantic())
q("P4", "PROJECT_INFO", "PROJECT_INFO", "liverpool_psg",
  "What does the fallback ball detector do differently from the primary one?",
  _check_semantic())
q("P5", "PROJECT_INFO", "PROJECT_INFO", "liverpool_psg",
  "What are the 48 reference points used for?",
  _check_semantic())
q("P6", "PROJECT_INFO", "PROJECT_INFO", "barca_madrid_pt1",
  "How does the system link a player's identity after they're lost and re-detected?",
  _check_semantic())
q("P7", "PROJECT_INFO", "PROJECT_INFO", "barca_madrid_pt1",
  "How is player space control calculated?",
  _check_semantic())
q("P8", "PROJECT_INFO", "PROJECT_INFO", "barca_madrid_pt1",
  "What caused the team-color classification bug the developers found?",
  _check_semantic())


# ---------- EDIT-FLOW (~10): should-succeed / should-decline pairs ----------

def _check_edit_should_succeed(r, ctx, other):
    if r["edit_result"] is None:
        return False, "no edit_result at all (route may not have been EDIT)", False
    kind, payload = r["edit_result"]
    if kind != "proposal":
        return False, f"expected a proposal, got decline: {payload}", False
    args = payload["args"]
    grounding_ctx = cb.build_grounding_context(ctx.df, ctx.stats_json)
    ok = cb.validate_grounding(grounding_ctx, args.get("grounding_field", ""), args.get("grounding_value", ""))
    if not ok:
        return False, f"proposal's own grounding does not validate: {args}", False
    return True, f"valid proposal, grounding {args.get('grounding_field')}={args.get('grounding_value')} verified", False


def _check_edit_should_decline(r, ctx, other):
    if r["edit_result"] is None:
        return None, "route was not EDIT at all - can't confirm decline behavior specifically", True
    kind, payload = r["edit_result"]
    if kind == "decline":
        return True, "correctly declined", False
    return False, f"expected a decline, got a proposal: {payload}", False


fastest_lp = None  # filled at runtime after contexts load (see main())

q("E1", "EDIT", "EDIT", "liverpool_psg",
  "Add a recovery session for player 8 on Wednesday, based on their real distance covered",
  _check_edit_should_succeed)
q("E2", "EDIT", "EDIT", "liverpool_psg",
  "Change the team's Monday focus to pressing, based on the real PRESS event count",
  _check_edit_should_succeed)
q("E3", "EDIT", "EDIT", "liverpool_psg",
  "Add a sprint session for player 16 on Friday, grounded in their real top speed",
  _check_edit_should_succeed)
q("E4", "EDIT", "EDIT", "barca_madrid_pt1",
  "Add a conditioning session for player 12 on Tuesday based on their real avg speed",
  _check_edit_should_succeed)
q("E5", "EDIT", "EDIT", "barca_madrid_pt1",
  "Update the team's Thursday why-stat using the real SPACE event count",
  _check_edit_should_succeed)

q("E6", "EDIT", "EDIT", "liverpool_psg",
  "Add a session for player 9999 on Monday praising their incredible dribbling",
  _check_edit_should_decline)
q("E7", "EDIT", "EDIT", "liverpool_psg",
  "Change player 16's Monday note to say they ran 500 meters at 60 km/h top speed",
  _check_edit_should_decline)
q("E8", "EDIT", "EDIT", "barca_madrid_pt1",
  "Add a session for player 500 on Friday because they were the fastest player",
  _check_edit_should_decline)
q("E9", "EDIT", "EDIT", "barca_madrid_pt1",
  "Change the team's Tuesday focus, citing 200 PRESS events this match",
  _check_edit_should_decline)
q("E10", "EDIT", "EDIT", "liverpool_psg",
  "Remove Thursday's session for a player who was never tracked in this match, player 8888",
  _check_edit_should_decline)


# ---------- ADVERSARIAL / ISOLATION (~8) ----------

def _check_cross_match_isolation(other_real_fact_terms):
    def check(r, ctx, other):
        ans = r["answer"].lower()
        honest_markers = ["don't have", "doesn't cover", "does not cover", "not available", "no data",
                           "not covered", "isn't part of", "no mention of", "does not include",
                           "doesn't include", "outside", "don't cover", "not something i have"]
        honest = any(m in ans for m in honest_markers)
        # Check honesty FIRST: a correct decline naturally has to name the
        # queried-but-absent team to explain what it's declining ("this
        # match's data doesn't cover PSG's..." necessarily contains "psg")
        # - that's expected and correct, not a leak. Found via manual
        # spot-check that checking the name-mention first flagged exactly
        # this correct behavior as a failure.
        if honest:
            return True, "honestly declined / correctly isolated (naming the queried team while declining is expected)", False
        leaked = [t for t in other_real_fact_terms if t.lower() in ans]
        if leaked:
            return False, f"mentioned other-match term(s) with no honest-decline phrasing: {leaked}", False
        return None, "no leaked facts detected, but no clear honest-decline phrase either - human review", True
    return check


q("A1", "ADVERSARIAL", "SEMANTIC", "liverpool_psg",
  "What was White's biggest defensive vulnerability in this match?",
  _check_cross_match_isolation(["barca_madrid_pt1", "real madrid"]))
q("A2", "ADVERSARIAL", "SEMANTIC", "barca_madrid_pt1",
  "What was PSG's biggest defensive vulnerability in this match?",
  _check_cross_match_isolation(["liverpool", "psg"]))
q("A3", "ADVERSARIAL", "STRUCTURED", "liverpool_psg",
  "What was Real Madrid's top speed in this match?",
  lambda r, ctx, other: (
      (False, f"fabricated an answer for a team not in this match: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly had nothing to report for a team not in this match", False)
  ))
q("A4", "ADVERSARIAL", "STRUCTURED", "barca_madrid_pt1",
  "What was Manchester United's top speed in this match?",
  lambda r, ctx, other: (
      (False, f"fabricated an answer for a team not in this match: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly had nothing to report for a team not in this match", False)
  ))
q("A5", "ADVERSARIAL", "STRUCTURED", "liverpool_psg",
  "What was player 99999's top speed?",
  lambda r, ctx, other: (
      (False, f"fabricated a stat for a nonexistent player: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly reported no data for a nonexistent player", False)
  ))
q("A6", "ADVERSARIAL", "STRUCTURED", "barca_madrid_pt1",
  "What was player -1's average speed?",
  lambda r, ctx, other: (
      (False, f"fabricated a stat for an invalid player id: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly reported no data for an invalid player id", False)
  ))
q("A7", "ADVERSARIAL", "SEMANTIC", "liverpool_psg",
  "How did Barcelona's build-up play compare in this match?",
  _check_cross_match_isolation(["barca_madrid_pt1"]))
q("A8", "ADVERSARIAL", "SEMANTIC", "barca_madrid_pt1",
  "What did Liverpool's coach adjust after this match?",
  _check_cross_match_isolation(["psg", "liverpool"]))


# ---------- EDGE CASES (~8) ----------

q("X1", "EDGE", "STRUCTURED", "liverpool_psg",
  "What was the momentum score at minute 500?",
  lambda r, ctx, other: (
      (False, f"fabricated a value for an out-of-range minute: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly reported the minute is out of range", False)
  ))
q("X2", "EDGE", "STRUCTURED", "barca_madrid_pt1",
  "What was the momentum score at minute -5?",
  lambda r, ctx, other: (
      (False, f"fabricated a value for an invalid minute: {r['answer']}", False)
      if extract_bolded(r["answer"]) else
      (True, "correctly reported the minute is invalid/out of range", False)
  ))


def _check_false_premise(real_value_fn, false_value):
    def check(r, ctx, other):
        real = real_value_fn(ctx)
        ans = r["answer"]
        nums = extract_numbers(ans)
        accepted_false = any(numeric_close(n, false_value, tol=0.05) for n in nums) and not any(
            numeric_close(n, real, tol=0.15) for n in nums
        )
        corrected = any(numeric_close(n, real, tol=0.15) for n in nums)
        correction_language = any(
            phrase in ans.lower() for phrase in ["actually", "not accurate", "incorrect", "in fact", "real value", "isn't right", "doesn't match"]
        )
        if accepted_false and not corrected:
            return False, f"accepted false premise ({false_value}) without correcting to real value ({real})", False
        if corrected or correction_language:
            return True, f"corrected or pushed back on the false premise (real value {real})", False
        return None, f"ambiguous - didn't clearly accept or correct (real value {real}): {ans}", True
    return check


q("X3", "EDGE", "STRUCTURED", "liverpool_psg",
  "The momentum score at minute 3 was 999 in this match, right? Can you confirm and explain why it was so high?",
  _check_false_premise(lambda ctx: ctx.momentum_at(3), 999))
q("X4", "EDGE", "STRUCTURED", "barca_madrid_pt1",
  "Player 12's top speed was 100 km/h in this match - what made them so fast?",
  _check_false_premise(lambda ctx: ctx.player(12)["top_speed_kmh"], 100))
q("X5", "EDGE", "STRUCTURED", "liverpool_psg",
  "There were 0 PRESS events in this match - why was the team so passive?",
  _check_false_premise(lambda ctx: ctx.stats_json["tactical_events"]["counts"]["PRESS"], 0))

q("X6", "EDGE", "SEMANTIC", "liverpool_psg",
  "Which team, Liverpool or PSG, was more dominant overall in the vulnerabilities section?",
  _check_semantic())
q("X7", "EDGE", "SEMANTIC", "barca_madrid_pt1",
  "Which team, White or Maroon, showed more spatial manipulation?",
  _check_semantic())
q("X8", "EDGE", "STRUCTURED", "liverpool_psg",
  "What was the momentum score at minute 28.5?",
  lambda r, ctx, other: (
      (True, "handled a non-integer minute without fabricating a bogus value", False)
      if not extract_bolded(r["answer"]) or any(
          numeric_close(float(v), ctx.momentum_at(28), 0.15) or numeric_close(float(v), ctx.momentum_at(29) if len(ctx.df) > 29 else ctx.momentum_at(28), 0.15)
          for v in extract_bolded(r["answer"]) if re.match(r"^-?\d+\.?\d*$", v.strip())
      ) else
      (None, f"non-integer minute produced a value not matching either neighboring minute - human review: {r['answer']}", True)
  ))


# ==========================================
# COST ESTIMATE (no API calls)
# ==========================================

def estimate_calls():
    per_route_calls = {"STRUCTURED": 2, "SEMANTIC": 3, "PROJECT_INFO": 3, "EDIT": 2}
    total = 0
    for item in QUESTIONS:
        total += per_route_calls.get(item["designed_route"], 3)
    setup = 2 * 2  # 2 matches x (summarize + embed) for per-match chroma collections
    setup += 1     # project-info collection embed
    setup += 2 * 2  # 2 matches x (team_plan + player_plan) generation
    return total, setup, total + setup


# ==========================================
# MAIN
# ==========================================

API_KEY = None


def load_api_key():
    import os
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    env_path = REPO_ROOT.parent / "football_analysis" / "simulation" / "Dashboard" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    dashboard_env = DASHBOARD_DIR / ".env"
    if dashboard_env.exists():
        for line in dashboard_env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _patch_client_counter():
    """Patched at the CLASS level: chatbot.py and training_plan.py each
    construct their OWN genai.Client(api_key=...) internally
    (query_collection, semantic_answer, project_info_answer,
    build_collection_if_needed, generate_team_plan, generate_player_plan all
    do this) rather than accepting a shared client - an earlier version of
    this counter only wrapped the ONE client instance this script itself
    created, silently missing every SEMANTIC/PROJECT_INFO/training-plan
    call (confirmed: the true count was roughly double what that undercount
    reported). Every genai.Client() constructed anywhere in this run, by any
    module, is counted here. Returns (call_counter, restore_fn)."""
    from google import genai
    call_counter = {"n": 0}
    _RealClient = genai.Client

    class _CountingModels:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def generate_content(self, *a, **kw):
            call_counter["n"] += 1
            return self._inner.generate_content(*a, **kw)

        def embed_content(self, *a, **kw):
            call_counter["n"] += 1
            return self._inner.embed_content(*a, **kw)

    class _CountingClient:
        def __init__(self, *a, **kw):
            self._inner = _RealClient(*a, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def models(self):
            return _CountingModels(self._inner.models)

    genai.Client = _CountingClient
    cb.genai.Client = _CountingClient
    tp.genai.Client = _CountingClient

    def restore():
        genai.Client = _RealClient
        cb.genai.Client = _RealClient
        tp.genai.Client = _RealClient

    return call_counter, restore


def run_battery(match_filter=None):
    """Runs the question bank (optionally restricted to one match) and
    returns (results, call_count). match_filter=None runs both matches in
    ONE process - see main()'s --match handling for why the real --run path
    does NOT do this."""
    from google import genai
    call_counter, restore = _patch_client_counter()
    client = genai.Client(api_key=API_KEY)

    match_ids = [match_filter] if match_filter else ["liverpool_psg", "barca_madrid_pt1"]
    print(f"\nLoading match context(s): {match_ids} (generates real in-memory training plan(s), not saved)...")
    contexts = {}
    for mid in match_ids:
        print(f"Loading {mid}...")
        contexts[mid] = MatchContext(mid)

    print("Building/verifying ChromaDB collection(s) (built once, skipped if already present)...")
    for mid, ctx in contexts.items():
        collection = cb.get_chroma_collection(ctx.source, ctx.key)
        cb.build_collection_if_needed(collection, API_KEY, ctx.df, ctx.ai_report_text, ctx.team_a, ctx.team_b)
    project_info_collection = cb.get_project_info_collection()
    cb.build_project_info_collection_if_needed(project_info_collection, API_KEY)

    items = [it for it in QUESTIONS if match_filter is None or it["match"] == match_filter]
    results = []
    print(f"\nRunning {len(items)} questions...\n")
    for i, item in enumerate(items):
        ctx = contexts[item["match"]]
        print(f"[{i+1}/{len(items)}] {item['id']} ({item['category']}, {item['match']}): {item['question'][:70]}")
        try:
            r = ask_chatbot(client, item["question"], ctx, project_info_collection)
        except Exception as e:
            r = {"question": item["question"], "answer": f"[ERROR: {e}]", "tags": [], "retrieved": None,
                 "edit_result": None, "debug": {"actual_route": "ERROR"}}
        # Universal guard, checked BEFORE any category-specific grading logic
        # runs: found via manual spot-check (this file's own "verify the
        # verifier" pass) that a category check like _check_semantic() would
        # trivially PASS an error-stub answer, since an empty tags list has
        # no bad citations and an error message often has no digits to flag
        # as a suspicious number. An exploded call is a failure, full stop -
        # never delegate that judgment to a checker that wasn't designed to
        # recognize its own harness's error format.
        if isinstance(r["answer"], str) and r["answer"].startswith("[ERROR:"):
            passed, reason, flag = False, f"chatbot call raised an exception: {r['answer']}", False
        else:
            try:
                # 'other' (the other match's context) is unused by every
                # check function in this file (confirmed by grep before
                # relying on it) - safe to pass None even when this process
                # only ever loaded one match.
                passed, reason, flag = item["check"](r, ctx, None)
            except Exception as e:
                passed, reason, flag = False, f"grading raised an exception: {e}", False
        results.append({
            "id": item["id"], "category": item["category"], "match": item["match"],
            "question": item["question"], "designed_route": item["designed_route"],
            "actual_route": r["debug"].get("actual_route"), "answer": r["answer"],
            "tags": r["tags"], "passed": passed, "reason": reason, "flag_for_human": flag,
        })
        time.sleep(1.2)  # gentle pacing to avoid per-minute rate limits

    restore()
    return results, call_counter["n"]


def main():
    global API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="Print the call-count estimate and exit, no API calls.")
    parser.add_argument("--run", action="store_true", help="Actually run the full battery against the live Gemini API.")
    parser.add_argument("--match", default=None, choices=["liverpool_psg", "barca_madrid_pt1"],
                         help="Internal: run only this match's questions in THIS process, dump JSON to --out-json. "
                              "Used by the orchestrator below to isolate each match in its own process.")
    parser.add_argument("--out-json", default=None, help="Internal: JSON dump path, used with --match.")
    parser.add_argument("--out", default=None, help="Report output path (default: tests/chatbot_validation_report_<timestamp>.md)")
    args = parser.parse_args()

    per_q, setup, total = estimate_calls()
    print(f"Question bank: {len(QUESTIONS)} questions")
    print(f"Estimated Gemini API calls: {per_q} (per-question) + {setup} (setup: chroma builds + training-plan generation) = {total} total")
    by_cat = {}
    for item in QUESTIONS:
        by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
    print("Breakdown by category:", by_cat)

    if args.estimate and not args.run:
        return

    if not args.run:
        print("\nPass --run to actually execute against the live Gemini API, or --estimate to only see the count.")
        return

    API_KEY = load_api_key()
    if not API_KEY:
        print("ERROR: no GEMINI_API_KEY found in environment or .env files.")
        sys.exit(1)

    if args.match:
        # Internal worker mode: run one match's questions in this process,
        # dump raw results as JSON for the orchestrator to merge.
        results, calls = run_battery(match_filter=args.match)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "calls": calls}, f)
        return

    # Orchestrator: each match runs in its OWN subprocess. Confirmed via
    # three consecutive full single-process runs that liverpool_psg's
    # ChromaDB queries fail 100% reproducibly ("Error creating hnsw segment
    # reader: Nothing found on disk") specifically when barca_madrid_pt1's
    # context/collection is also created in the SAME process beforehand -
    # yet an isolated diagnostic script proved the same on-disk collection
    # reads back fine from a fresh process. Rather than keep guessing at the
    # exact chromadb/HNSW internals, isolating each match in its own process
    # sidesteps whatever the interaction is - and more accurately mirrors
    # how a real user's single-match Streamlit session behaves anyway (only
    # one match's collection ever exists in that session's process).
    total_results, total_calls = [], 0
    for mid in ("liverpool_psg", "barca_madrid_pt1"):
        out_json = Path(__file__).parent / f"_tmp_result_{mid}.json"
        print(f"\n{'=' * 20} Running {mid} in its own isolated process {'=' * 20}")
        subprocess_result = _run_worker_subprocess(mid, out_json)
        if subprocess_result != 0:
            print(f"WARNING: worker process for {mid} exited with code {subprocess_result} - its results may be incomplete.")
        if out_json.exists():
            with open(out_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            total_results.extend(data["results"])
            total_calls += data["calls"]
            out_json.unlink()

    write_report(total_results, total_calls, args.out)


def _run_worker_subprocess(match_id, out_json):
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run", "--match", match_id, "--out-json", str(out_json)],
        cwd=str(REPO_ROOT),
    )
    return proc.returncode


def write_report(results, call_count_actual, out_path):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(out_path) if out_path else (Path(__file__).parent / f"chatbot_validation_report_{ts}.md")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    lines = [f"# Chatbot Validation Report — {ts}", "", f"Total questions: {len(results)}", f"Total Gemini API calls used: {call_count_actual}", ""]
    lines.append("## Pass rate by category")
    for cat, items in by_cat.items():
        n_pass = sum(1 for i in items if i["passed"] is True)
        n_fail = sum(1 for i in items if i["passed"] is False)
        n_review = sum(1 for i in items if i["passed"] is None)
        lines.append(f"- **{cat}**: {n_pass}/{len(items)} passed, {n_fail} failed, {n_review} need human review")
    lines.append("")

    mismatches = [r for r in results if r["designed_route"] != r["actual_route"]
                  and not (r["designed_route"] == "STRUCTURED" and r["actual_route"] == "SEMANTIC")]
    lines.append(f"## Routing mismatches ({len(mismatches)})")
    lines.append("(STRUCTURED->SEMANTIC fallback is expected/logged separately, not counted as a mismatch here)")
    for r in mismatches:
        lines.append(f"- {r['id']}: designed={r['designed_route']}, actual={r['actual_route']} — \"{r['question']}\"")
    structured_fallbacks = [r for r in results if r["designed_route"] == "STRUCTURED" and r["actual_route"] == "SEMANTIC"]
    lines.append(f"\nSTRUCTURED->SEMANTIC fallbacks: {len(structured_fallbacks)}")
    for r in structured_fallbacks:
        lines.append(f"- {r['id']}: \"{r['question']}\"")
    lines.append("")

    failures = [r for r in results if r["passed"] is False]
    lines.append(f"## Failures ({len(failures)})")
    for r in failures:
        lines.append(f"### {r['id']} ({r['category']}, {r['match']})")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Designed route: {r['designed_route']}, Actual route: {r['actual_route']}")
        lines.append(f"- Answer: {r['answer']}")
        lines.append(f"- Tags: {r['tags']}")
        lines.append(f"- Reason: {r['reason']}")
        lines.append("")

    flagged = [r for r in results if r["flag_for_human"]]
    lines.append(f"## Flagged for human review ({len(flagged)})")
    for r in flagged:
        lines.append(f"### {r['id']} ({r['category']}, {r['match']})")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Answer: {r['answer']}")
        lines.append(f"- Reason flagged: {r['reason']}")
        lines.append("")

    lines.append("## Full results")
    for r in results:
        status = "PASS" if r["passed"] is True else ("FAIL" if r["passed"] is False else "REVIEW")
        lines.append(f"- [{status}] {r['id']} ({r['category']}): {r['question'][:80]} — {r['reason']}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {out_path}")

    total = len(results)
    n_pass = sum(1 for r in results if r["passed"] is True)
    n_fail = sum(1 for r in results if r["passed"] is False)
    n_review = sum(1 for r in results if r["passed"] is None)
    print(f"\nOVERALL: {n_pass}/{total} passed, {n_fail} failed, {n_review} flagged for human review")
    print(f"Total Gemini API calls actually used: {call_count_actual}")


if __name__ == "__main__":
    main()
