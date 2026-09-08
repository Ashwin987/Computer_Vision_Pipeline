"""
"Ask the Assistant" — two-layer RAG chatbot + training-plan edit flow.

Kept out of app.py (already 3000+ lines) - same rationale as training_plan.py.
Every answer must trace to this match's real data (raw per-minute df, CV
stats.json, or the coach report text) - never general knowledge. The one
place this writes anything is a narrow, closed set of training-plan edit
functions, gated behind an explicit user Confirm click and a code-level
grounding check (not just a prompt instruction) - see propose_edit /
validate_grounding / apply_edit below.

Two Gemini models are used throughout:
- EMBEDDING_MODEL for Layer 2's vector search (confirmed live: 'models/gemini-
  embedding-001' - 'models/text-embedding-004' does not exist for this API
  key/version, confirmed via client.models.list()).
- ROUTING_MODEL ('gemini-2.5-flash') for routing, structured-lookup function
  calls, the edit-proposal function call, and the semantic-answer generation
  call - the same model already used everywhere else in this app.
"""
import copy
import json
import re
import tempfile
import time
from pathlib import Path

import chromadb
import pandas as pd
import pypdf
import streamlit as st
from google import genai
from google.genai import types

EMBEDDING_MODEL = "models/gemini-embedding-001"
ROUTING_MODEL = "gemini-2.5-flash"

# Was Path(__file__).parent / ".cache" / "chroma" - inside the app's own
# source directory. Confirmed via a live traceback that Streamlit Community
# Cloud mounts the cloned repo read-only, so the moment ChromaDB tried to
# actually write a collection there, it failed with "attempt to write a
# readonly database" - blocking Training Plan and Ask the Assistant
# entirely. Redirected to the system temp dir, which is writable in
# essentially any hosting environment. Collections become ephemeral (rebuilt
# on first use after a container restart) rather than persisting
# indefinitely, which is fine: every collection is already built behind a
# `collection.count() > 0` guard specifically so on-demand rebuilding is a
# no-op once built, not a repeated cost.
CHROMA_DIR = Path(tempfile.gettempdir()) / "tactical_scout_chroma"

TEAM_DAY_EDITABLE_FIELDS = ["focus_label", "focus_category", "why_stat"]
PLAYER_SESSION_EDITABLE_FIELDS = ["title", "note", "tag"]

PROJECT_INFO_COLLECTION_NAME = "project_technical_report"
# chatbot.py doesn't import app.py (avoids a circular import - app.py already
# imports this module), so it computes this the same way app.py's
# CV_PIPELINE_DIR does rather than sharing the constant directly. This was
# a THIRD independent hardcoded copy of this path found during the repo
# consolidation (app.py had two, one already routed through CV_PIPELINE_DIR
# and one that wasn't) - all three now resolve the same relative way.
TECHNICAL_REPORT_PDF_PATH = (
    Path(__file__).resolve().parent.parent / "cv_pipeline" / "Real-Time_Soccer_Analytics_Pipeline_v3.pdf"
)


def substitute_team_tokens(text, team_a, team_b):
    """The coach report is generated once with literal {TEAM_A}/{TEAM_B}
    placeholder tokens instead of real names baked in (see the writing_prompt
    in app.py), so a rename never requires regenerating it - every render
    site (Coach Report tab, PDF export, this module's chatbot citations)
    substitutes real names in at display time instead."""
    if not text:
        return text
    return text.replace("{TEAM_A}", team_a or "Team A").replace("{TEAM_B}", team_b or "Team B")


def _minute_num(ts):
    try:
        return int(str(ts).split(":")[0])
    except (ValueError, IndexError):
        return None


def _safe_str(v):
    return "" if v is None else str(v)


def _norm_num(s):
    """Extracts the first float found in a string, or None. Lets grounding
    validation compare '31.0' against '31.0 km/h' against 31.0 (float)."""
    m = re.search(r"-?\d+\.?\d*", _safe_str(s))
    return float(m.group()) if m else None


# ==========================================
# LAYER 1 — STRUCTURED LOOKUPS (plain Python, no LLM in the lookup itself)
# ==========================================

def get_raw_data_field(df, minute, field):
    minute = int(minute)
    if not (0 <= minute < len(df)):
        return None, f"Minute {minute} is outside this match's analyzed range (0-{len(df) - 1})."
    if field not in df.columns:
        return None, f"'{field}' isn't a field this match's per-minute data tracks."
    value = df.iloc[minute][field]
    if pd.isna(value):
        return None, f"No value recorded for '{field}' at minute {minute}."
    return value, None


def count_tactical_events(stats_json, event_type):
    if not stats_json:
        return None, "No CV analysis is available for this match, so there's no tactical-event data to count."
    counts = stats_json.get("tactical_events", {}).get("counts", {})
    event_type = str(event_type).upper()
    if event_type not in counts:
        return None, (
            f"'{event_type}' isn't a tactical-event type recorded for this match "
            f"(recorded types: {', '.join(sorted(counts.keys())) or 'none'})."
        )
    # Only a whole-window total exists in stats.json - there is no reliable
    # per-team breakdown (highlights[] is a top-N sample, not the full list,
    # so summing it by team would undercount). Never fabricate a split.
    return counts[event_type], None


def get_player_stat(stats_json, player_id, field):
    if not stats_json:
        return None, "No CV analysis is available for this match, so there's no player tracking data."
    try:
        player_id = int(player_id)
    except (TypeError, ValueError):
        return None, f"'{player_id}' isn't a valid player id."
    players = {p.get("player_id"): p for p in stats_json.get("players", [])}
    if player_id not in players:
        return None, f"Player {player_id} wasn't tracked in this match's analyzed window."
    p = players[player_id]
    if field not in p:
        return None, f"'{field}' isn't a tracked field for player {player_id}."
    return p[field], None


def get_tactical_event_highlights(stats_json, event_type=None, top_n=3):
    if not stats_json:
        return [], "No CV analysis is available for this match."
    highlights = stats_json.get("tactical_events", {}).get("highlights", [])
    if event_type:
        highlights = [h for h in highlights if str(h.get("type", "")).upper() == str(event_type).upper()]
    if not highlights:
        return [], f"No {event_type or ''} highlights recorded for this match's analyzed window.".replace("  ", " ")
    highlights = sorted(highlights, key=lambda h: h.get("score", 0), reverse=True)[: int(top_n)]
    return highlights, None


def get_training_plan_field(training_plan_draft, target, day, field):
    if not training_plan_draft:
        return None, "No training plan has been generated for this match yet."
    if str(target).lower() == "team":
        days = (training_plan_draft.get("team_plan") or {}).get("days", [])
        idx = _find_day_index([d.get("day", "") for d in days], day)
        if idx is None:
            return None, f"'{day}' isn't a day in this team plan."
        return days[idx].get(field), None
    player_plan = training_plan_draft.get("player_plan") or {}
    players = player_plan.get("players", [])
    p_idx = next((i for i, p in enumerate(players) if str(p.get("player_id")) == str(target)), None)
    if p_idx is None:
        return None, f"Player {target} doesn't have a plan in this match's player plans."
    sessions = players[p_idx].get("sessions", [])
    s_idx = _find_day_index([s.get("day", "") for s in sessions], day)
    if s_idx is None:
        return None, f"'{day}' isn't a session day for player {target}."
    return sessions[s_idx].get(field), None


LOOKUP_TOOLS = [
    types.FunctionDeclaration(
        name="get_raw_data_field",
        description="Look up one field of this match's real per-minute tactical data (e.g. momentum score, pressing intensity, block height) at one minute.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "minute": {"type": "INTEGER", "description": "0-based minute index"},
                "field": {"type": "STRING", "description": "e.g. smoothed_net_momentum, team_a_pressing_intensity, ball_zone"},
            },
            "required": ["minute", "field"],
        },
    ),
    types.FunctionDeclaration(
        name="count_tactical_events",
        description="Count how many times one tactical event type (SPRINT, BURST, PRESS, RECOVERY, OVERLAP, SPACE, LATERAL_RUN, DROP, BREAK, ISOLATED) occurred in this match's analyzed CV window.",
        parameters={
            "type": "OBJECT",
            "properties": {"event_type": {"type": "STRING"}},
            "required": ["event_type"],
        },
    ),
    types.FunctionDeclaration(
        name="get_player_stat",
        description="Look up one real tracked physical stat for one player id (top_speed_kmh, avg_speed_kmh, total_distance_m, top_speed_confidence, frames_tracked, team).",
        parameters={
            "type": "OBJECT",
            "properties": {
                "player_id": {"type": "STRING"},
                "field": {"type": "STRING"},
            },
            "required": ["player_id", "field"],
        },
    ),
    types.FunctionDeclaration(
        name="get_tactical_event_highlights",
        description="Get the top-scoring tactical event highlights in this match's analyzed window, optionally filtered by event type.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "event_type": {"type": "STRING"},
                "top_n": {"type": "INTEGER"},
            },
            "required": [],
        },
    ),
    types.FunctionDeclaration(
        name="get_training_plan_field",
        description="Look up one field of this match's current training plan for the team or one player, on one day.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "target": {"type": "STRING", "description": "'team' or a player id"},
                "day": {"type": "STRING"},
                "field": {"type": "STRING"},
            },
            "required": ["target", "day", "field"],
        },
    ),
]


def run_structured_lookup(client, question, df, stats_json, training_plan_draft):
    """Stage B for the STRUCTURED route: one function-calling call maps the
    question to exactly one read-only lookup, which then runs as plain
    Python (no LLM). Returns (answer_text, source_tag) or (None, None) if
    Gemini didn't call a function (caller falls back to SEMANTIC)."""
    resp = client.models.generate_content(
        model=ROUTING_MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=LOOKUP_TOOLS)],
            system_instruction=(
                "You answer questions about one soccer match's real analyzed data by calling "
                "exactly one of the provided lookup functions with concrete arguments extracted "
                "from the question. Always call a function - never answer in plain text."
            ),
        ),
    )
    parts = resp.candidates[0].content.parts if resp.candidates else []
    fc = next((p.function_call for p in parts if getattr(p, "function_call", None)), None)
    if not fc:
        return None, None
    name, args = fc.name, dict(fc.args)

    if name == "get_raw_data_field":
        value, err = get_raw_data_field(df, args["minute"], args["field"])
        if err:
            return err, None
        return f"At minute {args['minute']}, {args['field'].replace('_', ' ')} was **{value}**.", f"raw_data · minute {args['minute']}"

    if name == "count_tactical_events":
        value, err = count_tactical_events(stats_json, args["event_type"])
        if err:
            return err, None
        return (
            f"**{value}** {args['event_type'].upper()} events were logged in this match's analyzed window.",
            f"tactical_events · {args['event_type'].upper()} count",
        )

    if name == "get_player_stat":
        value, err = get_player_stat(stats_json, args["player_id"], args["field"])
        if err:
            return err, None
        return (
            f"Player {args['player_id']}'s {args['field'].replace('_', ' ')} was **{value}**.",
            f"players · P{args['player_id']}",
        )

    if name == "get_tactical_event_highlights":
        highlights, err = get_tactical_event_highlights(
            stats_json, args.get("event_type"), args.get("top_n", 3)
        )
        if err:
            return err, None
        lines = [f"- **{h['type']}** (player {h.get('player_id')}) — {h.get('metric')}" for h in highlights]
        return "Top highlights in this match's analyzed window:\n" + "\n".join(lines), "tactical_events · highlights"

    if name == "get_training_plan_field":
        value, err = get_training_plan_field(training_plan_draft, args["target"], args["day"], args["field"])
        if err:
            return err, None
        return f"{args['target']} · {args['day']} · {args['field']}: **{value}**", f"training_plan · {args['target']} · {args['day']}"

    return None, None


# ==========================================
# LAYER 2 — SEMANTIC (ChromaDB, embedded; Gemini embeddings)
# ==========================================

_HEADER_RE = re.compile(r"^(### .+|\*\*\d+\.\s*[A-Z][A-Z &]+:?\*\*)", re.MULTILINE)


def _chunk_coach_report(ai_report_text, team_a, team_b):
    """Splits on the report's own real headers (### section, **N. TITLE:**
    sub-points) - not invented boundaries - so a citation like 'coach report
    · {team} Tactical Profile · Vulnerabilities' names a real section."""
    if not ai_report_text:
        return []
    matches = list(_HEADER_RE.finditer(ai_report_text))
    chunks = []
    section_label = "Introduction"
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(ai_report_text)
        header = m.group(1).strip("* :#").strip()
        body = ai_report_text[m.end():end].strip()
        if header.startswith(("Tactical Diagnosis", "DATA-DRIVEN")):
            section_label = header
            label = header
        elif re.match(r"^\d+\.\s", header):
            label = f"{section_label} · {header.split('.', 1)[1].strip().title()}"
        else:
            section_label = header
            label = header
        if body:
            chunks.append((label, body[:1500]))
    return chunks


def _collection_name(source, key):
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"match_{source}_{key}")[:60]
    return safe or "match_unknown"


def get_chroma_collection(source, key):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(_collection_name(source, key))


def build_collection_if_needed(collection, api_key, df, ai_report_text, team_a, team_b):
    if collection.count() > 0:
        return
    client = genai.Client(api_key=api_key)

    # One batched call summarizes every analyzed minute at once - same
    # "one call for the whole table" pattern generate_team_plan already uses
    # for 7 days, not N separate calls.
    minute_rows = df.to_dict(orient="records")
    summary_prompt = f"""
Below is one row of real per-minute tactical data for each analyzed minute of a match
between {team_a} and {team_b}. For EACH minute (by its 0-based index), write ONE short
natural-language sentence summarizing that minute's real data - no invented facts,
only what's in the row.

DATA (JSON array, index = minute):
{json.dumps(minute_rows, default=str)[:12000]}

Respond with ONLY a JSON array of exactly {len(minute_rows)} strings, one per minute, in order.
"""
    documents, metadatas, ids = [], [], []
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=ROUTING_MODEL, contents=summary_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
            )
            summaries = json.loads(resp.text)
            if isinstance(summaries, list) and len(summaries) > 0:
                for i, s in enumerate(summaries[:len(minute_rows)]):
                    documents.append(s)
                    metadatas.append({"type": "raw_data", "minute": i,
                                       "timestamp": _safe_str(minute_rows[i].get("timestamp"))})
                    ids.append(f"minute_{i}")
                break
        except Exception:
            time.sleep(3)

    for j, (label, body) in enumerate(_chunk_coach_report(ai_report_text, team_a, team_b)):
        documents.append(f"{label}: {body}")
        metadatas.append({"type": "coach_report", "section": label})
        ids.append(f"report_{j}")

    if not documents:
        return

    embed_resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=documents)
    embeddings = [e.values for e in embed_resp.embeddings]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_collection(collection, api_key, question, n_results=4):
    client = genai.Client(api_key=api_key)
    q_embed = client.models.embed_content(model=EMBEDDING_MODEL, contents=[question])
    q_vec = q_embed.embeddings[0].values
    n_results = min(n_results, collection.count()) or 1
    res = collection.query(query_embeddings=[q_vec], n_results=n_results)
    retrieved = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})
    return retrieved


def semantic_answer(api_key, question, retrieved, team_a=None, team_b=None):
    client = genai.Client(api_key=api_key)
    if not retrieved:
        return "I don't have enough information in this match's data to answer that.", []
    # Coach-report chunks are stored/embedded in token form ({TEAM_A}/{TEAM_B} -
    # see substitute_team_tokens) so a rename never requires re-embedding.
    # Substitute real names in here, right before the chunk text is used, so
    # Gemini writes its answer with real names instead of literal tokens.
    context = "\n\n".join(
        f"[{i}] {substitute_team_tokens(r['document'], team_a, team_b)}" for i, r in enumerate(retrieved)
    )
    prompt = f"""
Answer the question using ONLY the retrieved context below, from one real soccer match's
analyzed data and coach report. If the context doesn't actually answer the question,
say honestly that this match's data doesn't cover it - do not guess or use outside
knowledge.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Answer in 2-4 sentences.
"""
    resp = client.models.generate_content(model=ROUTING_MODEL, contents=prompt)
    tags = []
    for r in retrieved:
        m = r["metadata"]
        if m.get("type") == "raw_data":
            tags.append(f"raw_data · minute {m.get('minute')}")
        else:
            section = substitute_team_tokens(m.get("section", ""), team_a, team_b)
            tags.append(f"coach report · {section}")
    return resp.text.strip(), tags


# ==========================================
# PROJECT_INFO — separate, match-independent index over the technical report
# (Part 5). Never blended into the per-match Layer 2 collection/index above -
# a distinct ChromaDB collection, distinct query path, distinct tag style, so
# a claim about the pipeline itself is never confusable with a claim about
# one match's data.
# ==========================================

_REPORT_HEADER_RE = re.compile(
    r"(?<=[\.\?\!\s])(\d{1,2}(?:\.\d{1,2}){0,2})\s+([A-Z][A-Za-z0-9,:\-' ]{6,90}?)(?=\s+[A-Z][a-z]{2,}|\s+\d)"
)


def _chunk_technical_report(pdf_path):
    """Chunks the real technical-report PDF on its own numbered section
    headers (e.g. '10.7 GPU Acceleration...') - confirmed by direct
    extraction that this pattern matches real inline section headers in the
    body text, not just the table of contents. Short fragments (<200 chars,
    mostly TOC-listing matches with no real body before the next header) are
    dropped rather than embedded as noise."""
    reader = pypdf.PdfReader(str(pdf_path))
    full_text = "".join(page.extract_text() or "" for page in reader.pages)
    matches = list(_REPORT_HEADER_RE.finditer(full_text))
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[m.end():end].strip()
        if len(body) > 200:
            chunks.append((f"Section {m.group(1)}", body[:1500]))
    return chunks


def get_project_info_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(PROJECT_INFO_COLLECTION_NAME)


def build_project_info_collection_if_needed(collection, api_key):
    """Built once, ever - match-independent, so unlike the per-match
    collections this never needs rebuilding on a rename or a different match
    being viewed. Same count()==0 guard as the per-match path."""
    if collection.count() > 0:
        return
    if not TECHNICAL_REPORT_PDF_PATH.exists():
        return
    chunks = _chunk_technical_report(TECHNICAL_REPORT_PDF_PATH)
    if not chunks:
        return
    client = genai.Client(api_key=api_key)
    documents = [f"{label}: {body}" for label, body in chunks]
    metadatas = [{"section": label} for label, _ in chunks]
    ids = [f"report_{i}" for i in range(len(chunks))]
    embed_resp = client.models.embed_content(model=EMBEDDING_MODEL, contents=documents)
    embeddings = [e.values for e in embed_resp.embeddings]
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_project_info(collection, api_key, question, n_results=4):
    client = genai.Client(api_key=api_key)
    q_embed = client.models.embed_content(model=EMBEDDING_MODEL, contents=[question])
    q_vec = q_embed.embeddings[0].values
    n_results = min(n_results, collection.count()) or 1
    res = collection.query(query_embeddings=[q_vec], n_results=n_results)
    retrieved = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        retrieved.append({"document": doc, "metadata": meta, "distance": dist})
    return retrieved


def project_info_answer(api_key, question, retrieved):
    client = genai.Client(api_key=api_key)
    if not retrieved:
        return "I don't have that in the technical report.", []
    context = "\n\n".join(f"[{i}] {r['document']}" for i, r in enumerate(retrieved))
    prompt = f"""
Answer the question using ONLY the retrieved context below, from the project's real
technical report describing how this computer-vision pipeline works. This is a
question about the SYSTEM ITSELF (architecture, methodology, measured performance),
not about any specific match's data. If the context doesn't actually answer the
question, say honestly that the report doesn't cover it - do not guess.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Answer in 2-4 sentences.
"""
    resp = client.models.generate_content(model=ROUTING_MODEL, contents=prompt)
    tags = [f"technical report · {r['metadata'].get('section')}" for r in retrieved]
    return resp.text.strip(), tags


# ==========================================
# ROUTING
# ==========================================

def route_question(client, question):
    resp = client.models.generate_content(
        model=ROUTING_MODEL,
        contents=(
            'Classify this question about a soccer match analysis app into exactly one category:\n'
            '- STRUCTURED: asks for one specific exact fact about THIS MATCH (a stat at a specific '
            'minute, a player\'s measured value, a count of one event type, one training-plan field)\n'
            '- SEMANTIC: open-ended/interpretive question about THIS MATCH, not tied to one exact field '
            '(e.g. "when did they look most vulnerable", "what was their biggest weakness")\n'
            '- EDIT: asks to add, remove, change, or swap something in the TRAINING PLAN\n'
            '- PROJECT_INFO: about the pipeline/system ITSELF, not this match\'s data - how it works, '
            'what a metric means in general, why it has some limitation, its architecture or measured '
            'performance (e.g. "how does this system work", "why is only one window analyzed", "what '
            'does the momentum score mean")\n\n'
            f'Question: "{question}"\n\n'
            'Respond with ONLY JSON: {"route": "STRUCTURED"|"SEMANTIC"|"EDIT"|"PROJECT_INFO"}'
        ),
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
    )
    try:
        return json.loads(resp.text).get("route", "SEMANTIC")
    except (json.JSONDecodeError, AttributeError):
        return "SEMANTIC"


# ==========================================
# PART 4 — FUNCTION-CALLING EDIT FLOW
# ==========================================

EDIT_TOOLS = [
    types.FunctionDeclaration(
        name="add_session",
        description="Propose adding a new drill/session to the training plan for a specific day.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "target": {"type": "STRING", "description": "'team' or a player id"},
                "day": {"type": "STRING"},
                "title": {"type": "STRING"},
                "note": {"type": "STRING"},
                "duration_min": {"type": "INTEGER"},
                "grounding_field": {"type": "STRING", "description": "dotted path to a real stat justifying this, e.g. players.247.top_speed_kmh"},
                "grounding_value": {"type": "STRING", "description": "the real value at that path"},
            },
            "required": ["target", "day", "title", "note", "grounding_field", "grounding_value"],
        },
    ),
    types.FunctionDeclaration(
        name="remove_session",
        description="Propose removing the session/drill(s) on a specific day.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "target": {"type": "STRING"},
                "day": {"type": "STRING"},
                "grounding_field": {"type": "STRING"},
                "grounding_value": {"type": "STRING"},
            },
            "required": ["target", "day", "grounding_field", "grounding_value"],
        },
    ),
    types.FunctionDeclaration(
        name="modify_session",
        description="Propose changing one field of an existing day's plan.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "target": {"type": "STRING"},
                "day": {"type": "STRING"},
                "field": {"type": "STRING"},
                "new_value": {"type": "STRING"},
                "grounding_field": {"type": "STRING"},
                "grounding_value": {"type": "STRING"},
            },
            "required": ["target", "day", "field", "new_value", "grounding_field", "grounding_value"],
        },
    ),
    types.FunctionDeclaration(
        name="swap_days",
        description="Propose swapping the team plan's content between two days.",
        parameters={
            "type": "OBJECT",
            "properties": {
                "day_a": {"type": "STRING"},
                "day_b": {"type": "STRING"},
                "grounding_field": {"type": "STRING"},
                "grounding_value": {"type": "STRING"},
            },
            "required": ["day_a", "day_b", "grounding_field", "grounding_value"],
        },
    ),
]

EDIT_SYSTEM_INSTRUCTION = (
    "You propose edits to a soccer training plan, grounded ONLY in this match's real data. "
    "You will be given a block of real data for this match (player stats, tactical event "
    "counts, match-wide averages) before each request - use it to find a real, specific stat "
    "that justifies the requested edit. Every proposal MUST include grounding_field (the exact "
    "dotted path shown in the data block, e.g. players.247.top_speed_kmh or "
    "tactical_events.counts.PRESS) and grounding_value (the real value shown there) - this will "
    "be checked against the real data before anything is shown to the user. "
    "If the given data block genuinely has nothing that justifies the requested edit, DO NOT "
    "call a function - respond in plain text explaining that you don't have data to support it."
)


def _available_data_summary(df, stats_json):
    """Compact, real-numbers-only digest handed to the EDIT proposal call so
    Gemini has something concrete to cite - without this, the model has no
    visibility into this match's actual data and can only ask the user to
    supply a stat themselves (confirmed live: it correctly refuses to
    fabricate one, but that's not useful - it should look the real number up
    itself, the same way the mockup's assistant appears to).

    Includes EVERY tracked player, not just a top-N slice - confirmed live
    that a top-8-by-speed cutoff silently excluded a real, valid player (rank
    23 of 37) the user asked about by id, causing a false 'I don't have
    data for that player' decline. The full list is small (tens of players,
    a few fields each) so there's no real token-budget reason to truncate it
    and risk that failure mode again."""
    lines = []
    if stats_json and stats_json.get("players"):
        lines.append("Player stats (player_id: top_speed_kmh, avg_speed_kmh, total_distance_m):")
        for p in stats_json["players"]:
            lines.append(
                f"  players.{p.get('player_id')}: top_speed_kmh={p.get('top_speed_kmh')}, "
                f"avg_speed_kmh={p.get('avg_speed_kmh')}, total_distance_m={p.get('total_distance_m')}"
            )
    if stats_json and stats_json.get("tactical_events", {}).get("counts"):
        counts = stats_json["tactical_events"]["counts"]
        lines.append("Tactical event counts (this match's analyzed window):")
        for k, v in counts.items():
            lines.append(f"  tactical_events.counts.{k}: {v}")
    if df is not None and len(df) > 0:
        for col in ("team_a_pressing_intensity", "team_b_pressing_intensity", "smoothed_net_momentum"):
            if col in df.columns:
                lines.append(f"Average {col} across the match: {round(pd.to_numeric(df[col], errors='coerce').mean(), 2)}")
    return "\n".join(lines) if lines else "No real match data is available."


def build_grounding_context(df, stats_json):
    ctx = {}
    for i, row in df.reset_index(drop=True).iterrows():
        for col in df.columns:
            ctx[f"raw_data.{i}.{col}"] = row[col]
    if stats_json:
        for p in stats_json.get("players", []):
            pid = p.get("player_id")
            for k, v in p.items():
                ctx[f"players.{pid}.{k}"] = v
        for k, v in stats_json.get("tactical_events", {}).get("counts", {}).items():
            ctx[f"tactical_events.counts.{k}"] = v
    return ctx


def validate_grounding(ctx, grounding_field, grounding_value):
    if grounding_field not in ctx:
        return False
    actual = ctx[grounding_field]
    a_num, g_num = _norm_num(actual), _norm_num(grounding_value)
    if a_num is not None and g_num is not None:
        return abs(a_num - g_num) < 0.05
    return _safe_str(actual).strip().lower() == _safe_str(grounding_value).strip().lower()


def propose_edit(client, question, ctx, df=None, stats_json=None):
    """Returns ('proposal', {name, args}) | ('decline', text)."""
    data_summary = _available_data_summary(df, stats_json)
    prompt = f"AVAILABLE REAL DATA FOR THIS MATCH:\n{data_summary}\n\nUSER REQUEST: {question}"
    resp = client.models.generate_content(
        model=ROUTING_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=EDIT_TOOLS)],
            system_instruction=EDIT_SYSTEM_INSTRUCTION,
        ),
    )
    parts = resp.candidates[0].content.parts if resp.candidates else []
    fc = next((p.function_call for p in parts if getattr(p, "function_call", None)), None)
    if not fc:
        text = "".join(getattr(p, "text", "") or "" for p in parts).strip()
        return "decline", text or "I don't have data to support that edit."

    args = dict(fc.args)
    if not validate_grounding(ctx, args.get("grounding_field", ""), args.get("grounding_value", "")):
        return "decline", (
            f"I can't verify \"{args.get('grounding_value')}\" against this match's real data "
            f"at `{args.get('grounding_field')}`, so I won't propose this edit."
        )
    return "proposal", {"name": fc.name, "args": args}


def _find_day_index(day_names, query):
    query = str(query).strip().lower()
    for i, d in enumerate(day_names):
        if str(d).strip().lower() == query:
            return i
    for i, d in enumerate(day_names):
        if str(d).strip().lower().startswith(query[:3]):
            return i
    return None


def _rebase_and_mark(item, original_item, fields):
    """After a chat-confirmed write, rebase the pristine snapshot for the
    touched fields so the generic recompute_edited_fields pass (which runs
    unconditionally every render) doesn't ALSO flag them as manually edited -
    chat_confirmed_fields is the only durable record it was chat-driven."""
    confirmed = set(item.get("chat_confirmed_fields") or [])
    for f in fields:
        original_item[f] = copy.deepcopy(item.get(f))
        confirmed.add(f)
    item["chat_confirmed_fields"] = sorted(confirmed)


def describe_proposal(name, args):
    """Diff-style text for the proposal card, matching chatbot_edit_flow_mockup.html."""
    if name == "add_session":
        return (f"**{args['target']} · {args['day']}**\n\n"
                f"+ {args['title']}" + (f" — {args['duration_min']} min" if args.get("duration_min") else "")
                + f"\n{args['note']}\n\nGrounded in: `{args['grounding_field']}` = {args['grounding_value']}")
    if name == "remove_session":
        return (f"**{args['target']} · {args['day']}**\n\n- remove this day's session(s)\n\n"
                f"Grounded in: `{args['grounding_field']}` = {args['grounding_value']}")
    if name == "modify_session":
        return (f"**{args['target']} · {args['day']}**\n\n"
                f"{args['field']} → {args['new_value']}\n\n"
                f"Grounded in: `{args['grounding_field']}` = {args['grounding_value']}")
    if name == "swap_days":
        return (f"Swap **{args['day_a']}** ↔ **{args['day_b']}** (team plan)\n\n"
                f"Grounded in: `{args['grounding_field']}` = {args['grounding_value']}")
    return json.dumps(args)


def apply_edit(training_plan_draft, name, args):
    """Mutates training_plan_draft in place with 'confirmed via chat' provenance."""
    team_plan = training_plan_draft.get("team_plan") or {}
    player_plan = training_plan_draft.get("player_plan") or {}

    if name == "swap_days":
        days = team_plan.get("days", [])
        original_days = team_plan.get("_original_days", [])
        ia = _find_day_index([d.get("day", "") for d in days], args["day_a"])
        ib = _find_day_index([d.get("day", "") for d in days], args["day_b"])
        if ia is None or ib is None:
            return False
        fields = ["focus_label", "focus_category", "drills", "why_stat"]
        for f in fields:
            days[ia][f], days[ib][f] = days[ib][f], days[ia][f]
        if ia < len(original_days):
            _rebase_and_mark(days[ia], original_days[ia], fields)
        if ib < len(original_days):
            _rebase_and_mark(days[ib], original_days[ib], fields)
        return True

    target = args["target"]
    if str(target).lower() == "team":
        days = team_plan.get("days", [])
        original_days = team_plan.get("_original_days", [])
        idx = _find_day_index([d.get("day", "") for d in days], args["day"])
        if idx is None:
            return False
        original = original_days[idx] if idx < len(original_days) else {}

        if name == "add_session":
            days[idx].setdefault("drills", []).append({
                "title": args["title"], "duration_min": args.get("duration_min") or 15, "note": args["note"],
            })
            _rebase_and_mark(days[idx], original, ["drills"])
        elif name == "remove_session":
            days[idx]["drills"] = []
            _rebase_and_mark(days[idx], original, ["drills"])
        elif name == "modify_session":
            field = args["field"]
            if field not in TEAM_DAY_EDITABLE_FIELDS:
                return False
            days[idx][field] = args["new_value"]
            _rebase_and_mark(days[idx], original, [field])
        else:
            return False
        return True

    # player target
    players = player_plan.get("players", [])
    original_players = player_plan.get("_original_players", [])
    p_idx = next((i for i, p in enumerate(players) if str(p.get("player_id")) == str(target)), None)
    if p_idx is None:
        return False
    sessions = players[p_idx].setdefault("sessions", [])
    orig_sessions = (original_players[p_idx].get("sessions", []) if p_idx < len(original_players) else [])

    if name == "add_session":
        new_session = {"day": args["day"], "title": args["title"], "note": args["note"], "tag": "Chat-added"}
        sessions.append(new_session)
        if p_idx < len(original_players):
            original_players[p_idx].setdefault("sessions", []).append(copy.deepcopy(new_session))
            _rebase_and_mark(sessions[-1], original_players[p_idx]["sessions"][-1], ["title", "note", "tag"])
    elif name == "remove_session":
        s_idx = _find_day_index([s.get("day", "") for s in sessions], args["day"])
        if s_idx is None:
            return False
        sessions.pop(s_idx)
        if s_idx < len(orig_sessions):
            orig_sessions.pop(s_idx)
    elif name == "modify_session":
        s_idx = _find_day_index([s.get("day", "") for s in sessions], args["day"])
        if s_idx is None:
            return False
        field = args["field"]
        if field not in PLAYER_SESSION_EDITABLE_FIELDS:
            return False
        sessions[s_idx][field] = args["new_value"]
        original = orig_sessions[s_idx] if s_idx < len(orig_sessions) else {}
        _rebase_and_mark(sessions[s_idx], original, [field])
    else:
        return False
    return True


# ==========================================
# UI (render_chatbot_tab) — called from app.py's 5th tab
# ==========================================

def _render_source_tags(tags):
    """Plain-text pills, per the spec (not clickable). Technical-report tags
    (Part 5) get a visibly distinct color/icon from match-data tags, reusing
    this app's existing purple accent (THEME_COLORSCALE_PURPLE's #9b5fe0 in
    app.py) rather than inventing a new color - so a claim about the pipeline
    itself is never visually confusable with a claim about this match's
    data."""
    if not tags:
        return
    pills = []
    for t in tags:
        if t.startswith("technical report"):
            pills.append(
                '<span style="display:inline-block;font-size:11px;color:#c9a6f0;'
                'background:rgba(155,95,224,0.12);border:1px solid rgba(155,95,224,0.4);'
                f'padding:2px 8px;border-radius:999px;margin-right:6px;">📄 {t}</span>'
            )
        else:
            pills.append(
                '<span style="display:inline-block;font-size:11px;color:#8a93a6;'
                'background:rgba(79,140,255,0.08);border:1px solid rgba(79,140,255,0.25);'
                f'padding:2px 8px;border-radius:999px;margin-right:6px;">📊 {t}</span>'
            )
    st.markdown(" ".join(pills), unsafe_allow_html=True)


def render_chatbot_tab(df, stats_json, team_a, team_b, ai_report_text, api_key, source, key, save_training_plan_fn):
    st.subheader("💬 Ask the Assistant")
    st.caption(
        "Answers are grounded in this match's real tracking data and coach report only — "
        "the assistant won't guess at anything outside the analyzed window."
    )
    if not api_key:
        st.error("No Gemini API key configured.")
        return

    # Session-only history/pending-edit state, per the spec (no new
    # persistence). Ephemeral matches (no stable source/key, e.g. Demo Mode
    # CSV) get a random per-session id instead of sharing one collection, so
    # two different unsaved matches viewed in the same session never mix
    # retrieval results together.
    if not source:
        if "chatbot_ephemeral_key" not in st.session_state:
            import uuid
            st.session_state.chatbot_ephemeral_key = uuid.uuid4().hex[:12]
        source, key = "session", st.session_state.chatbot_ephemeral_key

    identity = (source, key)
    if st.session_state.get("chatbot_identity") != identity:
        st.session_state.chatbot_history = []
        st.session_state.chatbot_pending_edit = None
        st.session_state.chatbot_identity = identity
    st.session_state.setdefault("chatbot_history", [])
    st.session_state.setdefault("chatbot_pending_edit", None)
    st.session_state.setdefault("chatbot_debug_last", [])

    client = genai.Client(api_key=api_key)
    collection = get_chroma_collection(source, key)
    with st.spinner("Indexing this match's data for search…"):
        build_collection_if_needed(collection, api_key, df, ai_report_text, team_a, team_b)
    # Match-independent - built once total, not per match (count()>0 guard
    # below skips rebuilding it every session, same pattern as above).
    project_info_collection = get_project_info_collection()
    build_project_info_collection_if_needed(project_info_collection, api_key)

    rail_col, chat_col = st.columns([1, 3])

    with rail_col:
        st.markdown("**Suggested questions**")
        suggestions = []
        highlights, _ = get_tactical_event_highlights(stats_json, top_n=1)
        if highlights:
            h = highlights[0]
            suggestions.append(f"What made the {h['type']} event by player {h.get('player_id')} significant?")
        if stats_json and stats_json.get("players"):
            fastest = max(stats_json["players"], key=lambda p: p.get("top_speed_kmh") or 0)
            suggestions.append(f"What was P{fastest['player_id']}'s top speed?")
        suggestions.append(f"When did {team_a} look most vulnerable?")
        suggestions.append("Add a recovery session for the team on Thursday")
        for i, sug in enumerate(suggestions):
            if st.button(sug, key=f"chat_sugg_{i}", use_container_width=True):
                st.session_state.chatbot_pending_submit = sug
                st.rerun()

    with chat_col:
        for msg in st.session_state.chatbot_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                _render_source_tags(msg.get("tags"))

        pending = st.session_state.chatbot_pending_edit
        if pending:
            with st.chat_message("assistant"):
                st.markdown("⚠ **PROPOSED — not yet saved**")
                st.markdown(describe_proposal(pending["name"], pending["args"]))
                c1, c2 = st.columns(2)
                if c1.button("✅ Confirm", key="chat_edit_confirm", use_container_width=True):
                    draft = st.session_state.get("training_plan_draft")
                    with st.spinner("Saving to training plan…"):
                        ok = bool(draft) and apply_edit(draft, pending["name"], pending["args"])
                        if ok:
                            st.session_state.training_plan_draft = draft
                            if source != "session":
                                save_training_plan_fn(source, key, draft)
                    note = "✅ Applied to the training plan." if ok else "⚠️ Couldn't apply that edit (target/day not found)."
                    st.session_state.chatbot_history.append({"role": "assistant", "content": note, "tags": []})
                    st.session_state.chatbot_pending_edit = None
                    st.rerun()
                if c2.button("Cancel", key="chat_edit_cancel", use_container_width=True):
                    st.session_state.chatbot_pending_edit = None
                    st.rerun()

        question = st.chat_input("Ask about this match, or ask me to edit the training plan…")
        if not question and st.session_state.get("chatbot_pending_submit"):
            question = st.session_state.pop("chatbot_pending_submit")

        if question:
            debug_rows = []
            st.session_state.chatbot_history.append({"role": "user", "content": question, "tags": []})
            tags = []

            with st.status("Checking what kind of question this is…", expanded=False) as status:
                route = route_question(client, question)
                debug_rows.append(f"Route: {route}")

                if route == "EDIT":
                    status.update(label="Checking this can be grounded in real data…")
                    draft = st.session_state.get("training_plan_draft")
                    if not draft:
                        answer = "Generate a training plan first (Training Plan tab) before I can propose changes to it."
                    else:
                        ctx = build_grounding_context(df, stats_json)
                        kind, payload = propose_edit(client, question, ctx, df, stats_json)
                        if kind == "decline":
                            answer = payload
                        else:
                            st.session_state.chatbot_pending_edit = payload
                            answer = "Here's what I'd change — see the proposal below."
                elif route == "STRUCTURED":
                    status.update(label="Looking up the answer…")
                    answer, tag = run_structured_lookup(client, question, df, stats_json, st.session_state.get("training_plan_draft"))
                    if answer is None:
                        route = "SEMANTIC"
                    else:
                        tags = [tag] if tag else []
                elif route == "PROJECT_INFO":
                    status.update(label="Searching the technical report…")
                    retrieved = query_project_info(project_info_collection, api_key, question)
                    debug_rows.append(
                        "Retrieved (project info): " + ", ".join(r["metadata"].get("section", "") for r in retrieved)
                    )
                    status.update(label="Writing the answer…")
                    answer, tags = project_info_answer(api_key, question, retrieved)

                if route == "SEMANTIC":
                    status.update(label="Searching match data…")
                    retrieved = query_collection(collection, api_key, question)
                    debug_rows.append(
                        "Retrieved: " + ", ".join(
                            r["metadata"].get("section") or f"minute {r['metadata'].get('minute')}" for r in retrieved
                        )
                    )
                    status.update(label="Writing the answer…")
                    answer, tags = semantic_answer(api_key, question, retrieved, team_a, team_b)

                status.update(label="Done", state="complete")

            st.session_state.chatbot_history.append({"role": "assistant", "content": answer, "tags": tags})
            st.session_state.chatbot_debug_last = debug_rows
            st.rerun()

    with st.expander("🔍 Debug: routing"):
        if st.session_state.chatbot_debug_last:
            for row in st.session_state.chatbot_debug_last:
                st.text(row)
        else:
            st.caption("Ask a question to see routing/retrieval details here.")
