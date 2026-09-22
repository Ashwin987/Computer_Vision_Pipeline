"""
player_labels.py — manual tracking-ID -> real player name labeling, CV Deep
Analysis tab.

Storage: a per-match ID->name mapping, layered non-destructively on top of
the CV pipeline's own stats.json at READ time only - same discipline as
corner_kicks.py / training_plan.py (a separate file under CACHE_DIR, never
written into stats.json or bundle.json; see app.py's CACHE_DIR comment for
why this needs to be the writable system-temp dir, not the git-cloned
source tree).

Substitution is via explicit call sites (player_label() for one known ID,
substitute_player_labels() for free text that might mention a player as
"P123"/"Player 123") - never a blind app-wide text scan. This mirrors
_cv_team_label being called explicitly at every team-label render site,
rather than chatbot.py's substitute_team_tokens' literal-placeholder-token
approach: blind substitution is safe for {TEAM_A}/{TEAM_B} because those are
unambiguous synthetic tokens burned into text at generation time, but a raw
player-id number has no such guarantee (it could collide with an unrelated
stat value) - so it's only ever replaced immediately after a 'P'/'Player'
prefix, matching how every render site already spells out a player id.

Scope, explicit: this labels the LIVE UI's text/data only - the Window
Stats readout, training plans, chatbot answers/citations, stat cards. It
does NOT retroactively relabel already-rendered CV overlay videos: those
have raw IDs burned into pixels by the CV pipeline's own rendering step,
which this dashboard-side feature has no way to touch after the fact.
"""
import json
import os
import re
from pathlib import Path


def _labels_path(cache_dir, match_key):
    return Path(cache_dir) / "player_labels" / f"{match_key}.json"


def load_labels(cache_dir, match_key):
    """{} (never None) so every call site can use the result directly
    without an extra None-check - an unlabeled match is a normal, common
    state, not an error."""
    if not match_key:
        return {}
    path = _labels_path(cache_dir, match_key)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {str(k): v for k, v in data.items() if v}
    except (json.JSONDecodeError, OSError):
        return {}


def save_labels(cache_dir, match_key, labels):
    """Only non-blank names are persisted - clearing a name field back to
    blank removes that player's entry rather than saving an empty-string
    override that would otherwise shadow the real P<id> fallback forever."""
    path = _labels_path(cache_dir, match_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {str(k): str(v).strip() for k, v in labels.items() if v and str(v).strip()}
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)
    os.replace(tmp, path)
    return cleaned


def player_label(player_id, team_label, labels):
    """The one formatting call every render site should use in place of its
    own f"P{player_id} ({team_label})" - the real name once labeled (still
    with the team in parens, so existing team-scoping is never lost), the
    original ID-based label otherwise."""
    name = (labels or {}).get(str(player_id))
    base = name if name else f"P{player_id}"
    return f"{base} ({team_label})" if team_label else base


_ID_MENTION_RE = re.compile(r'\bP(\d+)\b|\bPlayer\s+(\d+)\b', re.IGNORECASE)


def substitute_player_labels(text, labels):
    """Bounded substitution for free text (Gemini prose, chat answers) that
    mentions a player as 'P123' or 'Player 123' - requires that exact
    prefix, so an unrelated number elsewhere in the same sentence (e.g. a
    speed or distance value) is never mistaken for a player-id mention."""
    if not text or not labels:
        return text

    def _repl(m):
        pid = m.group(1) or m.group(2)
        name = labels.get(pid)
        return name if name else m.group(0)

    return _ID_MENTION_RE.sub(_repl, text)
