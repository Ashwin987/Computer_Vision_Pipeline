# Known Issues

## Calibration confidence does not reliably predict correct homography

**Status:** understood, not yet fixed. Tracked here as a known, scheduled issue —
not urgent-blocking as of the commit that added this note.

**The bug:** the pitch-keypoint pose model can misclassify a penalty-box D-arc point
as a center-circle point (or vice versa) — both are curved white lines on grass, and
under narrow/zoomed camera framing the model loses the wider context it needs to tell
them apart. When this happens, RANSAC (`cv2.findHomography(..., cv2.RANSAC, 8.0)` in
`cv_pipeline/pitch_calibrator.py`) finds a self-consistent cluster of these
*wrongly*-classified points and accepts it as the homography's inlier set — silently
rejecting the genuinely correct points as "outliers" because they disagree with the
larger wrong cluster. The result is a homography that's internally coherent (tight
reprojection residuals) but globally wrong, mapping real, visible landmarks (e.g. the
center circle) tens of meters from where they actually are on screen.

**Why existing checks miss it:** `pitch_calibrator._validate()` only checks that the
frame's center pixel lands inside a loosely padded pitch box. Separately,
`view_transformer.compute_frame_confidence()` only checks what fraction of tracked
players' *already-transformed* positions land inside `[0,105]x[0,68]`. Neither checks
whether the mapping itself is geometrically correct — a wrong-but-bounded homography
scores exactly as well as a correct one on both checks. A secondary, smaller finding
from the same investigation: the RANSAC reprojection threshold (`8.0`) is applied in
world-space meters, not pixels, despite the parameter's name/docstring — an
extremely loose tolerance for a 105x68m pitch.

**Confirmed scope** (via direct back-projection of known landmarks against real
homography stubs, cross-checked visually against real video frames):
- Triggered by tight/zoomed camera framing, **not corner kicks specifically**.
- `liverpool_psg_verified`: correct for ~82% of its analyzed window (frames ~0–614,
  wide shot, center circle in view); wrong for ~18% (frames ~620–752, where the
  broadcast camera zooms into a box-only shot during open play) — that entire
  zoomed segment reports 0.92–1.00 confidence throughout despite being wrong.
- `barca_madrid_pt1_verified`: no evidence found — camera stays wide throughout the
  sampled window.
- All 3 corner-kick segments (`corner1/2/3_liverpool_psg`): wrong for effectively the
  entire segment, since a corner-kick shot is always tight/zoomed. `corner1`/`corner2`
  are affected on ~100% of frames; `corner3` is mostly correct (~96% of frames) — its
  lower reported confidence comes from an unrelated, already-known, already-mitigated
  wild-outlier class (see `dashboard/corner_kicks.py`'s bounds-filtering).

**Measured impact on affected frames:** moderate, not catastrophic — the homography's
local pixel-to-meter scale in the bad `liverpool_psg_verified` segment measured ~1.3x
the match's own median, so frame-to-frame delta measurements (speed, distance covered)
are only moderately distorted there. Metrics that read *absolute* pitch position
directly (block-height/zone classification, space-control heatmaps, the corner-kick
feature's box-scoped metrics) are the most exposed, since "which part of the pitch"
is exactly what this bug gets wrong.

**Fix — implemented, partial.** `pitch_calibrator.py`'s `_solve_homography` now: (1)
fits world→pixel with RANSAC (not pixel→world) so the reprojection threshold is
genuinely applied in pixel space as documented, instead of silently being an 8-*metre*
tolerance; (2) after RANSAC accepts an inlier cluster, re-fits a homography restricted
to just the rejected ("outlier") points — if that rejected set ALSO fits its own clean
homography (`inliers2 >= min_keypoints` and `inliers2/len(outliers) >= 0.6`), the frame
is flagged `'ambiguous_cluster'` and rejected outright (falls back to the nearest
calibrated neighbor, the same existing fallback every other calibration failure already
uses) rather than confidently accepting one of two indistinguishable interpretations.

**Scope of this fix, verified by real-coordinate back-projection (not by trusting the
new logic's own report) across multiple frames per segment — only corner-kick segments
were touched; `liverpool_psg_verified`/`barca_madrid_pt1_verified` are untouched and
checksum-verified identical before/after:**
- `corner3_barca_madrid`: genuinely fixed. Box corners, goal posts, and the center
  circle now land in visually correct positions across every checked frame.
  Box-tracked frames: 8.7% → 72.4%. **Adopted** — its shipped `player_positions.json`
  and homography stub now reflect the recalibrated data, `calibration_status.json`
  flipped to `reliable: true`.
- `corner1_liverpool_psg`: improved but inconsistent — 2 of 3 manually-checked frames
  now project correctly, 1 does not. Box-tracked frames rose from 0% to 36% (passes
  the frame-count gate), but this is not confirmed reliable to the same bar as
  `corner3_barca_madrid`. **Not adopted** — left flagged `reliable: false`, pending a
  more thorough per-frame check than this pass had time for.
- `corner2_liverpool_psg`: still wrong on every manually-checked frame (box corners
  project onto open midfield grass, nowhere near the actual box). **Not adopted** —
  unchanged, still `reliable: false`.
- `corner2_barca_madrid`: still wrong on the checked frame, same failure shape as
  `corner2_liverpool_psg`. **Not adopted** — unchanged, still `reliable: false`.
- `corner1_barca_madrid`: previously the one segment flagged *reliable* (for its
  near-box positions). Re-verifying it under the new logic was explicitly required
  before trusting it — and it should NOT be trusted: multiple checked frames now show
  box corners projecting onto open grass, a regression from its prior state. **Not
  adopted** — its original (pre-fix) calibration data is kept as shipped, since it was
  the better of the two and remains flagged `reliable: true` based on that original,
  separately-verified data, not the new fit.

**Bottom line:** the fix demonstrably works for the case it targets (two competing,
internally-consistent keypoint clusters), and fully resolved 1 of the 5 previously-
checked corner segments. It's a real improvement, not a complete one — 4 of 5 segments
still need either the existing honest-fallback UI treatment (already in place and
correctly gating all of them) or further calibration work, most likely something
beyond cluster-agreement checking alone: at least one failure mode observed here
(`corner1_liverpool_psg`, frame 90) is a small-but-self-consistent inlier set that
fits itself with near-zero residual yet is too spatially clustered to safely
extrapolate a full projective transform beyond its own immediate neighborhood — a
different problem from the two-competing-clusters signature this fix targets, and not
yet addressed.

**Update — `barca_madrid_pt1`'s 3 new corner segments** (same real-coordinate
back-projection method, same per-segment `calibration_status.json` sidecar the
corner-kicks feature now reads): `corner1_barca_madrid` (4:45–4:55) is reliable for
the near-box positions its metrics actually use (80% of frames have a defender
tracked inside the real penalty box) despite the same far-field D-arc/center-circle
confusion also showing up there — a useful nuance this investigation hadn't seen
before: the bug's *far-field extrapolation* being wrong doesn't necessarily mean the
*near-field* positions a corner-kick metric depends on are also wrong; check the
box-zone frame count directly rather than assuming one implies the other.
`corner2_barca_madrid` (16:17–16:26, 9.3% box-tracked) and `corner3_barca_madrid`
(20:50–21:01, originally 8.7% box-tracked, degrading partway through its own window)
were both unreliable at the time this was written, matching the `corner1/2_liverpool_psg`
pattern. **`corner3_barca_madrid`'s status has since changed** — see the "Fix —
implemented, partial" section above; it was recalibrated and is now `reliable: true`.
`corner2_barca_madrid` remains unreliable, unchanged.

## Corner-kicks feature: `resolve_corner_team_mapping` assumed reference team1 = team_a

**Status:** fixed. `dashboard/corner_kicks.py`'s `resolve_corner_team_mapping`
hardcoded that a match's own main-window `team_resolution.team_colors_bgr["team1"]`
is always real `team_a` — true for `liverpool_psg` only by coincidence (its
`cv_team_mapping` happens to be `{"1":"team_a","2":"team_b"}`), but false for
`barca_madrid_pt1` (whose `cv_team_mapping` was never confirmed via the main
dashboard's one-time swatch-confirmation UI, and whose own team1 color is the darker
Maroon/Barcelona kit, i.e. team_b, not team_a's White). Found while adding
`barca_madrid_pt1`'s 3 corner marks: metrics would have silently labeled Real
Madrid's numbers as Barcelona's and vice versa. Fixed by adding an explicit
`reference_team1_is_team_a` flag, resolved once per match (from real jersey/on-screen
evidence, not guessed) and stored on each mark itself — never touches `bundle.json`,
never hardcodes a team name in the metric-display code.
