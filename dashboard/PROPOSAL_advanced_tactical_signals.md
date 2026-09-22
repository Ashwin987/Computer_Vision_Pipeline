# Proposal: Pressing Triggers, Buildup Analysis, and Other Advanced Tactical Signals

**Status: investigation + proposal only. Nothing in this document has been implemented.**

Scope of this investigation: what the pipeline (per `Real-Time_Soccer_Analytics_Pipeline_v4.pdf`,
current as of this writing) and dashboard already compute that could honestly ground new
tactical-report / training-plan content, what would need new computation, and what should
simply not be attempted yet.

---

## 1. Pressing-trigger detection

### 1a. Team compactness delta (recommended — build first)

**What it is:** `corner_kicks.py`'s `_compactness()` (mean pairwise distance between a team's
outfield players) already exists and is validated — it's the same metric behind the corner-kick
module's "defending compactness" number. A pressing trigger, in the threshold-based spirit of the
existing `tactical_events` module (Sprint/Press/Recovery etc. — Section 8), would be: **a
team's own defensive compactness value drops sharply (players collapse toward each other)
within a short window**, using a rising-edge trigger + cooldown, exactly like the existing ten
tactical events (Section 8.2 documents why that discipline — rising-edge, minimum-duration,
noise-guard on out-of-bounds positions — matters; the first cut of that module over-fired 977
events on one clip before this discipline was applied).

**What's already computed vs. new:** The *formula* is already written and real (position-based,
no role inference, matching this project's corner-kick discipline). What's **not** already
computed is per-frame real-world positions for the *main analysis window* — `player_positions.json`
only exists today for windows that have been run through `reconstruct_positions.py`, and I
confirmed directly that the main window's own CV output directory (`liverpool_psg_verified`) has
no such file — only corner marks that were explicitly reconstructed do. `reconstruct_positions.py`
is also an **offline script that requires cv_pipeline's real dependencies (torch/ultralytics)**,
which the dashboard's own environment deliberately does not install (see that script's own
docstring). So this is not a small dashboard-side addition — it requires a pipeline-side batch
step (extending `reconstruct_positions.py`'s reach to the main window, or generating that output
as part of the normal CV run) before the dashboard could read it.

**Implementation size: medium.** Threshold/trigger logic is small and can reuse the existing
tactical-events pattern; the real cost is the pipeline-side position-reconstruction step for the
main window, which is a new capability, not a config change.

### 1b. Defensive-line height/depth change (recommended — build alongside 1a)

**What it is:** Same data dependency as 1a (needs per-frame real positions for the main window).
The corner-kick module already computes "distance from deepest defender to own goal line" for a
single corner snapshot (Section 13.3); a pressing-trigger version would track that same number
*over time* across the main window and flag a rapid rise (a team stepping up its line — often a
deliberate high-press trigger) or fall (dropping off — often a trigger to sit deep and counter).

**Implementation size: medium**, for the same reason as 1a — same data dependency, same
threshold-design cost, and the two are cheap to ship together once per-frame main-window
positions exist.

### 1c. Distance-to-ball-carrier (optional — ship only gated, lower confidence)

**What it is:** "How close was the nearest defender to the ball carrier" is a classic pressing-intensity
signal, but it requires a reliable **ball position**, not just player positions — a materially
different reliability bar than 1a/1b, which are ball-free by design (the same reason the
corner-kick module could be built at all: Section 13.1 sidesteps ball detection entirely).

For `liverpool_psg_verified`, the pipeline's own numbers: ball detection **96.4%** combined
final rate, but only **35.1%** from the primary detector before the fallback recovers it (`ball`
block in `stats.json`); calibration mean confidence **0.90**, but **9.8%** of frames rated `low`
and **19.8%** `medium` (`calibration` block). Critically, Section 2.4.8 of the report found that
the *existing* calibration-confidence number can itself be high while the underlying homography
is wrong by tens of metres (a penalty-arc/centre-circle keypoint confusion, confirmed present on
~18% of frames in one of two segments checked) — and the fix for that specific failure mode is
**not yet implemented**. So even gating this signal on today's confidence number is a partial,
not complete, safeguard.

**Recommendation:** build this only after 1a/1b, gate it per-window behind both
`ball.final_detection_rate_pct` and `calibration.pct_low`/`pct_medium` thresholds (reusing the
exact "shown with an explicit notice rather than a computed value" pattern the corner-kick module
already established in Section 13.4 for exactly this situation), and label it lower-confidence
than 1a/1b in the UI.

**Implementation size: medium-large** — same position-reconstruction dependency as 1a/1b, plus
new ball-reliability gating logic, plus UI work to honestly express "this number wasn't computed
for this window" as a first-class state rather than a missing/zero value.

---

## 2. Buildup-play analysis (passing sequences, progression through thirds)

**Not reliably feasible today. Recommend not building it yet.**

Two independent pieces of evidence, both already in this codebase, point the same direction:

1. **The report's own documented history.** Section 8.1 states plainly that the *original* design
   of the tactical-events module tried exactly this — inferring passes/interceptions/shots/tackles
   from ball proximity and speed — and abandoned it specifically because ball detection in
   broadcast footage is "small, fast-moving, and frequently occluded," and "any event logic built
   on top of that noise inherits it." The module was deliberately redesigned around ball-free,
   position-only signals for this reason (Section 8.1), and Section 11 lists reliable on-ball event
   recognition as **future work that would first require a substantially more reliable ball-tracking
   model than broadcast footage currently permits with the present detector.**

2. **A concrete, currently-unused artifact that would tempt exactly this mistake.** I found that
   `cv_pipeline/match_events/rule_based_events.py` — a ball-proximity/speed heuristic detector for
   PASS/INTERCEPTION/SHOT/TACKLE/DRIBBLE — still runs and still writes real output: `stats.json`'s
   `match_events` block (`{"INTERCEPTION": 91, "PASS": 101, "total": 192}` for `liverpool_psg`) and
   `transitions` (`{"count": 80}`). **This data is never read anywhere in the dashboard today** (I
   grepped `app.py`, `training_plan.py`, `chatbot.py`, `corner_kicks.py` — zero references). It
   is exactly the same ball-proximity+speed-threshold approach (`BALL_PASS_SPD = 8 px/frame`,
   `NEAR_BALL_PX = 60px`) the technical report explicitly diagnosed as unreliable and pivoted away
   from for the *tactical_events* module. Building buildup-play analysis on `match_events`/
   `transitions` today would silently reintroduce the exact failure mode the report already
   found and moved past — a "PASS: 101" count sitting right there in the data is a real trap, not
   a hypothetical one.

**Bottom line:** genuine buildup-play analysis (passing sequences, progression through thirds)
needs a materially more reliable ball-tracking signal than this pipeline currently has for full
open play. That's a ball-detection-model improvement, not a dashboard feature — out of scope for
this dashboard-side proposal. I'm flagging it as infeasible rather than proposing a "buildup"
feature that would end up citing `match_events`/`transitions` and presenting fabricated
confidence.

---

## 3. Other realistically-buildable signals

### 3a. Space-control trend for a specific player/zone (small-medium)
Module 3 (Voronoi space control) and Module 4 (team pitch control) already compute real,
validated space-control data during video *rendering* — but per `training_plan.py`'s own
documented investigation (see its `_player_cv_signals` docstring: "space control, movement trails,
and stamina are each computed transiently during video RENDERING elsewhere in the pipeline but
never written to stats.json"), none of it is persisted. Surfacing it for training-plan grounding
would require writing a rendering-time aggregate (e.g., "average Voronoi area per player over the
window") to `stats.json`, similar in spirit to how `tactical_events.highlights` already gets
persisted today. **Implementation size: small-medium** — no new detection/tracking, just persisting
an aggregate of a computation that already runs.

### 3b. Stamina-drop flag, window-scoped honestly (small)
Module 2 (stamina) is also transient-only today. A single, honestly-scoped stat — stamina level
at the *end* of the analyzed window relative to its start, for players tracked continuously
through it — could be persisted the same way as 3a, **as long as it is captioned the same way
`training_plan.py`'s `PLAYER_PLAN_SCOPE_NOTE` already captions player data**: this window only,
never a full-match fatigue curve. This is exactly the mistake the module's docstring says the
original mockup made and this codebase deliberately avoids — any stamina signal must repeat that
same discipline.

### 3c. Tracking-coverage / data-reliability as its own persistent signal (already shipped)
This one doesn't need building — `training_plan.py`'s `generate_cv_insights` already does exactly
this (`_player_cv_signals`' `coverage_pct`/`confidence`, `NOTABLE_COVERAGE_PCT_THRESHOLD`).
Mentioned here only to confirm it's not a gap.

---

## 4. Summary table

| Signal | Data needed | Already computed/cached? | New computation required | Size |
|---|---|---|---|---|
| Compactness-delta press trigger (1a) | Per-frame real positions, main window | Formula yes (corner_kicks); main-window positions no | Pipeline-side position reconstruction for main window + trigger logic | Medium |
| Defensive-line depth change (1b) | Same as 1a | Same as 1a | Same as 1a (ship together) | Medium |
| Distance-to-ball-carrier press trigger (1c) | Per-frame positions + reliable ball position | Ball detection-rate/calibration numbers yes; per-frame ball position for main window no | Same position reconstruction + ball-reliability gating + "withheld, not computed" UI state | Medium-large |
| Buildup-play / passing sequences | Reliable ball tracking across a full possession | No — report documents this was tried and abandoned | A better ball-detection model (out of scope here) | Not recommended now |
| Space-control trend (3a) | Voronoi output, already computed during rendering | Computed but not persisted | Persist an aggregate to stats.json | Small-medium |
| Window-scoped stamina drop (3b) | Stamina module output, already computed during rendering | Computed but not persisted | Persist an aggregate to stats.json, with the same honest scope note pattern as player plans | Small |
| Tracking coverage / reliability signal (3c) | stats.json player + video frame counts | Yes, already shipped | None | Done |

---

## 5. Feeding into the Team Training Plan (item 1)

Today's team plan (`compute_team_source_stats`) grounds `why_stat` in five real per-minute
fields: gegenpress minutes, counter-attack minutes, defensive/overlapping fullback minutes, and
drop-deep minutes — all from the per-minute Gemini-tagged `raw_data`, not the CV pipeline. If
built, the two ball-free signals from Section 1 would add genuinely new, CV-grounded material to
that same `why_stat` sentence, not just more of the same kind of number:

- **Compactness-delta press triggers (1a)** → directly grounds a day like "Pressing Trigger
  Recognition," citing a real count of detected compactness collapses instead of only the
  Gemini-tagged `gegenpress` minute count already used today — two independently-sourced signals
  agreeing (or disagreeing, which would itself be worth surfacing) rather than one restated twice.
- **Defensive-line depth change (1b)** → directly grounds a day like "Defensive Line Discipline,"
  complementing the existing `drop_deep` minute count with an actual measured distance trend.
- **Distance-to-ball-carrier (1c)**, once gated and shipped, would ground "closing-down speed"
  style drills — but only for windows where it passed its reliability gate, with player plans
  already establishing the UI precedent for "this data wasn't reliable enough to show" as an
  honest, non-blocking state.
- **Space-control trend (3a)** and **window-scoped stamina drop (3b)** would extend the *player*
  plan (`generate_player_plan`) the same way `generate_cv_insights` already does today — as an
  additive layer, never replacing the existing speed/distance-grounded sessions.

None of these should be wired into the training plan until they're built and independently
verified against real match data, the same discipline items 1 and 2 in this session were held to.
