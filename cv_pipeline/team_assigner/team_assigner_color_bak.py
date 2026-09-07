import cv2
import numpy as np
from sklearn.cluster import KMeans

# Fixed seed for get_player_color's per-crop cv2.kmeans call (reset via
# cv2.setRNGSeed immediately before every call -- see the comment at that
# call site). Arbitrary constant; only fixedness matters, not the value.
JERSEY_COLOR_KMEANS_SEED = 42

# Max Euclidean BGR distance to the nearest centroid for a confident assignment.
# Pixels not within this distance of either team centre are labelled Unknown (0).
#
# Empirically tuned from 40.0: on two test clips, permanently-unresolved
# players' best-ever color distance showed a clean bimodal split — a cluster
# at 40-47 (genuine matches barely missing the old cutoff) and a gap before
# the next cluster at 54+ (occluded/tiny-bbox crops, not threshold-fixable).
# 50.0 recovers the first cluster while staying below half of the tighter of
# the two clips' centroid separations (103.7 / 2 = 51.85), so it introduces
# no risk of a color falling within threshold of BOTH centroids at once.
CONFIDENCE_THRESHOLD = 50.0

# Frames sampled for the KMeans fit. Expanded from [0,10,20,30,40] (98 total
# samples, only 6 of them confident+reliable for Team 2 in the first 100
# frames — an under-sampled, less stable fit) to every 10th frame across the
# first 300 (589 total samples). Empirically verified: the fitted centroids
# barely move (a few BGR units) between the two, but the larger sample is a
# more statistically robust basis, especially for Team 2 which is naturally
# under-represented in any small sample.
CLUSTER_FRAMES = list(range(0, 300, 10))

# Warn if the two team centroids are less than this far apart — indicates the
# filter still isn't separating jersey colours cleanly.
SEPARATION_WARNING = 50.0

# get_player_color falls back to the UNFILTERED torso-crop mean when fewer
# than this many pixels survive the grass/shadow/grey HSV filter. That
# fallback color is not a jersey reading at all — on a contaminated frame
# (e.g. legs spread mid-stride, torso rotated away from the fixed crop
# window) it's dominated by background. A real case in this codebase: a
# player's first ~140 frames all had 0% pixel survival (100% grass) and
# read confidently but WRONGLY as the other team; only once the crop
# started catching real jersey pixels (95-100% survival) did it read
# correctly. Samples below this threshold are excluded from per-player
# locking (NOT from the initial cluster fit — tried excluding them there
# too; with a small sample pool it nearly halved centroid separation, so
# unreliable samples get diluted by the rest of the pool instead. Re-tested
# at the larger CLUSTER_FRAMES sample size below and the same conclusion
# holds — reliability-filtering the fit still degrades it).
MIN_RELIABLE_JERSEY_PIXELS = 10

# A player's bounding box overlapping another tracked player's box is a
# strong sign the wide-torso crop (15-85% width, 15-65% height of the
# bbox — see get_player_color) is catching some of the OTHER player's
# jersey, not just this one's. Confirmed on this codebase's own data: two
# players (pids 48, 21) genuinely wear green but were frequently locked in
# tight duels with red opponents; their overlap-frame color samples read
# confidently RED (distances 35-48, well inside CONFIDENCE_THRESHOLD)
# while their zero-overlap samples read cleanly GREEN (distances 1-8).
# Calibrated empirically: every known-contaminated frame checked had
# overlap fraction >= 0.24; every known-clean frame had exactly 0.0 — this
# threshold sits with margin below the lowest contaminated case.
# Fraction is overlap-area / this player's OWN bbox area (not IOU — a
# small box mostly covered by a bigger box is the contamination case we
# care about, regardless of the bigger box's own size).
OCCLUSION_OVERLAP_THRESHOLD = 0.10

# A single reliable, confident sample is not enough to permanently lock a
# player's team — a self-consistent run of early misreads (like the case
# above) would still lock cleanly on its own terms. Require several
# reliable+confident samples that mostly agree before locking.
MIN_SAMPLES_TO_LOCK = 3
AGREEMENT_THRESHOLD = 0.70

# Sanity check on the model's own is_goalkeeper vote: a real goalkeeper
# spends essentially all match time in (or very near) their own defensive
# third — the model can still misfire on a visually "solo, distinctly-
# colored figure" that isn't actually a keeper (confirmed: the match
# referee, in a solid neon kit, misclassified as a goalkeeper on a
# different clip/camera — 9 distinct "goalkeeper" track fragments found
# in one minute of footage, all confined to pitch-center, when a real
# match has exactly 2). PITCH_LENGTH_M is the standard FIFA pitch length
# already assumed throughout this codebase (speed_and_distance_estimator's
# PITCH_X_MAX) — genuinely universal, not tuned to any one video. A track
# that NEVER enters either defensive third across its whole recorded
# position history gets its is_goalkeeper vote overridden.
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0    # standard FIFA pitch width, same convention as PITCH_LENGTH_M
DEFENSIVE_THIRD_M = PITCH_LENGTH_M / 3.0
GOALKEEPER_POSITION_MIN_SAMPLES = 5   # need at least this many valid position
                                      # samples to trust the check either way —
                                      # too little data means "insufficient
                                      # evidence to override", not "reject"
# Bound how much history counts toward the vote so a very long career
# doesn't require permanent unanimity; keeps the check responsive to a
# genuine early split rather than needing to win by attrition.
PENDING_SAMPLE_WINDOW = 10

# Sanity check on the model's own is_goalkeeper vote, #2: catches a static
# object misdetected as a goalkeeper (confirmed real case: a corner flag
# pole sitting right at the pitch corner, which is geometrically INSIDE a
# defensive third, so the position check above can't catch it) that
# _reject_non_goalkeeper_positions structurally cannot.
#
# Requires BOTH near-zero movement rate AND an extreme (tall/thin) bbox
# aspect ratio -- NEITHER alone is safe, verified against real goalkeeper
# tracks across all 3 known-flagged clips, not just the one the flag came
# from:
#   - Movement rate alone: a confirmed corner-flag misdetection (pid913,
#     BarcaMadridPT1_w529, n=58 frames) moved only 0.13px/frame. But a
#     real, CONFIRMED goalkeeper in a calmer clip (AlgeriaArgentia_w685
#     pid67, n=180 frames) moved just 0.10px/frame -- LOWER than the flag
#     -- while simply standing in position through a quiet defensive
#     spell. A goalkeeper's own activity level varies enough clip-to-clip
#     that "barely moves" alone is not evidence of "not a person".
#   - Aspect ratio alone: the flag reads 3.29 (very tall/thin), but a real
#     goalkeeper crop can run nearly as extreme (AlgeriaArgentia_w685
#     pid81: 2.81) with only a thin margin below the flag -- not a safe
#     solo cutoff either.
#   - Combined: no real goalkeeper fragment across all 3 clips (9 checked,
#     each n>=20 frames) has BOTH rate<0.3 px/frame AND aspect>3.0 at
#     once -- the closest, pid81, clears the aspect gate by 0.19 and the
#     rate gate by a wide margin the other direction (its own rate is
#     actually well under 0.3, but its aspect of 2.81 stays under 3.0).
#     The flag clears both simultaneously (0.13 and 3.29). This is a
#     conjunctive, deliberately conservative test -- it will miss a
#     misdetection that's only static OR only oddly-shaped, not both, but
#     that's the right tradeoff over risking a real, calm goalkeeper.
#
# GOALKEEPER_MOTION_MIN_FRAMES gates this the same way
# GOALKEEPER_POSITION_MIN_SAMPLES gates the position check above: several
# genuine short keeper fragments (4-14 frames) show LESS raw pixel range
# than the 58-frame flag simply because they haven't had time to move yet,
# not because they're static objects -- judging a rate over too few frames
# is noise, not evidence, so tracks under this length are exempt
# ("insufficient evidence", not "reject") exactly like the position check.
GOALKEEPER_MOTION_MIN_FRAMES = 20
GOALKEEPER_MOTION_MAX_RATE_PX_PER_FRAME = 0.3
GOALKEEPER_MAX_ASPECT_RATIO = 3.0


class TeamAssigner:
    def __init__(self):
        # 0 = Unknown/referee, 3 = goalkeeper. Goalkeepers get a fixed,
        # visually distinct color rather than a jersey-derived one — their
        # kit is deliberately a third color, so there is nothing meaningful
        # to cluster them against. Cyan/sky-blue: earlier used a
        # yellow-green (0,255,180) meant to sit apart from referee-yellow,
        # but that shade turned out too close in hue to team 2's actual
        # fitted green centroid to read as visually distinct at render
        # size (confirmed on a real screenshot — a goalkeeper and a team-2
        # player were indistinguishable at a glance). Cyan is far in hue
        # from red (team 1), green (team 2), AND referee-yellow all at
        # once. (Purple was also tried earlier and reverted — not
        # revisiting that.)
        self.team_colors      = {0: np.array([128.0, 128.0, 128.0]),
                                  3: np.array([255.0, 220.0, 0.0])}   # BGR cyan/sky-blue
        self.player_team_dict = {}   # pid → cached team id (0 / 1 / 2 / 3)
        self.locked_colors    = {}   # {1: BGR, 2: BGR} — set once by assign_team_color
        self.pending_samples  = {}   # pid → list of team guesses (1/2) from reliable+confident frames, awaiting enough agreement to lock
        self.color_vote_frames = {}  # pid → list of frame numbers, index-parallel to pending_samples.
                                      # Populated by resolve_all_teams alongside pending_samples so a
                                      # caller needing frame-ordered (fn, guess) pairs (e.g. an ID-swap
                                      # scan) can zip these together instead of re-sampling every
                                      # player's colour a second time across the whole clip. Kept as a
                                      # SEPARATE list rather than changing pending_samples' own element
                                      # type to (fn, guess) tuples — pending_samples is read elsewhere
                                      # as a plain list of ints (majority/count logic), and tuples would
                                      # silently break that (every (fn, guess) pair is unique, so
                                      # set()/count() would stop finding a majority at all).
        self.pending_gk_samples = {}  # pid → list of is_goalkeeper booleans, awaiting enough agreement to lock to team 3
        self.best_unresolved_distance = {}  # pid → (dist, team_guess, color) best-ever seen (reliable or not) while still unresolved
        self.fallback_assigned = {}  # pid → {'distance':..., 'team':...} for players resolved via finalize_fallback_assignments, not the normal path
        self.merged_pids = set()     # pids whose final team came from merge_fragmented_tracks's propagation, not a direct lock

    # ── Jersey colour sampler ─────────────────────────────────────────────────

    def get_player_color(self, frame, bbox, return_reliability=False):
        """
        Sample the torso region of a player bounding box and return the BGR
        colour of the jersey-like pixels.

        Crop: 15-85% of bbox width, 15-65% of bbox height — a wide
        torso-inclusive region. Widened from an earlier center-40%/20-55%
        window: that narrower crop assumed an upright, bbox-centered player
        and, verified on real frames, produced as little as 45% jersey
        coverage for a leaning/mid-stride pose (the rest grass) because the
        actual jersey wasn't where the fixed window expected it to be. The
        wider window makes it much more likely the jersey is captured
        SOMEWHERE inside it, at the cost of also being more likely to catch
        some grass/skin/shadow alongside it — handled below.

        HSV filter applied inside the crop:
          - Grass / yellow-green pixels removed  (H 35–85 in OpenCV 0-179 scale)
          - Shadow / very dark pixels removed    (V < 40)
          - Desaturated / near-grey pixels removed (S < 40; white lines, pitch blur)

        Dominant-cluster extraction: rather than averaging every surviving
        pixel together (which blends jersey with whatever contamination the
        wider crop also let through — skin, shorts, shadow edges — into a
        muddy intermediate color), the surviving pixels are split with
        k-means (k=2) and the LARGER cluster's centroid is used. The torso
        should dominate a well-centered wide crop, so the majority cluster
        is the jersey; a minority cluster (skin/shorts/etc.) gets excluded
        rather than blended in.

        Falls back to the full torso crop mean if fewer than
        MIN_RELIABLE_JERSEY_PIXELS pixels survive the filter, or the
        surviving pixels don't have a clear dominant cluster (roughly even
        split — a sign the crop is still ambiguous even after filtering).
        That fallback color is NOT a trustworthy jersey reading.

        return_reliability=True additionally returns whether the dominant-
        cluster path was used, so callers can avoid trusting an unreliable
        sample even if it happens to land close to a centroid.
        """
        def _bail(color):
            return (color, False) if return_reliability else color

        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            fh, fw = frame.shape[:2]
            x1 = max(0, x1);  x2 = min(fw - 1, x2)
            y1 = max(0, y1);  y2 = min(fh - 1, y2)
            bw = x2 - x1;    bh = y2 - y1

            if bw < 4 or bh < 4:
                return _bail(np.array([128.0, 128.0, 128.0]))

            # Wide torso box
            cx1 = x1 + int(bw * 0.15)
            cx2 = x1 + int(bw * 0.85)
            cy1 = y1 + int(bh * 0.15)
            cy2 = y1 + int(bh * 0.65)
            cx2 = max(cx1 + 1, min(cx2, fw - 1))
            cy2 = max(cy1 + 1, min(cy2, fh - 1))

            crop_bgr = frame[cy1:cy2, cx1:cx2]
            if crop_bgr.size == 0:
                return _bail(np.array([128.0, 128.0, 128.0]))

            # HSV filter — keep only jersey-like pixels
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            H = crop_hsv[:, :, 0]   # 0-179 in OpenCV
            S = crop_hsv[:, :, 1]   # 0-255
            V = crop_hsv[:, :, 2]   # 0-255

            # Turf and a GREEN JERSEY share almost the same hue and
            # saturation (measured on this clip: turf H median=45 S
            # median=123; jersey H median=45 S median=123 — literally
            # indistinguishable on hue/sat alone). What differs is
            # brightness: turf V median=118 (p90=170) vs jersey V
            # median=202 (p10=136) — turf is matte, jersey fabric is
            # brighter/more reflective. So only treat a grass-hue pixel as
            # turf (removable) if it's ALSO not too bright; a bright pixel
            # in the same hue range is kept as a jersey candidate instead
            # of being discarded outright. This is a real tradeoff, not a
            # clean split (some sunlit turf is this bright too, some
            # jersey pixels in shadow are this dark) — calibrated to
            # retain ~70% of true jersey pixels while still removing ~86%
            # of true turf pixels.
            GRASS_MAX_VALUE_TO_REMOVE = 150
            grass_mask  = (H >= 35) & (H <= 85) & (V < GRASS_MAX_VALUE_TO_REMOVE)
            shadow_mask = V < 40                   # very dark (shadows)
            grey_mask   = S < 40                   # near-white / desaturated

            keep = ~(grass_mask | shadow_mask | grey_mask)

            pixels = crop_bgr.reshape(-1, 3).astype(np.float32)
            kept   = pixels[keep.ravel()]

            if len(kept) < MIN_RELIABLE_JERSEY_PIXELS:
                # Fallback: unfiltered torso crop — not reliable
                color = pixels.mean(axis=0)
                return (color, False) if return_reliability else color

            if len(kept) >= 20:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                # cv2.kmeans's KMEANS_PP_CENTERS init draws from OpenCV's
                # global cv::theRNG(), which is never reset between calls --
                # it just keeps advancing across every kmeans call in the
                # process. Confirmed real impact: for a clip whose 3-way
                # team-color cluster fit has a near-tied 2nd/3rd population
                # split (a coin-flip for which cluster gets dropped as
                # "referee"), calling this same code on identical cached
                # data gave cluster_separation 80.7 on the very first call
                # in a fresh process and a stable-but-different 44.2 on
                # every call after that -- i.e. results depended on
                # whatever ran earlier in the process, not just this
                # clip's own data. Reseeding immediately before every call
                # (verified empirically: makes repeated calls bit-for-bit
                # identical regardless of intervening kmeans calls) removes
                # that dependency. The constant itself is arbitrary -- only
                # fixedness matters, not the specific value.
                cv2.setRNGSeed(JERSEY_COLOR_KMEANS_SEED)
                _, labels, centers = cv2.kmeans(
                    kept, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                counts = np.bincount(labels.ravel(), minlength=2)
                dominant = int(counts.argmax())
                dominant_frac = counts[dominant] / counts.sum()

                # A near-even k-means split in raw BGR doesn't necessarily
                # mean two different MATERIALS — real jersey fabric has
                # shading (folds, highlights) that alone can split evenly
                # into a darker and a lighter cluster while staying the
                # same hue. Only treat a split as real contamination
                # (jersey + something else) if the two clusters actually
                # differ in HUE; if they're just different brightnesses of
                # the same color, merge them back into one reading.
                centers_u8 = np.clip(centers, 0, 255).reshape(-1, 1, 3).astype(np.uint8)
                centers_hsv = cv2.cvtColor(centers_u8, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                hue_diff = abs(int(centers_hsv[0][0]) - int(centers_hsv[1][0]))
                hue_diff = min(hue_diff, 180 - hue_diff)   # H is circular in OpenCV (0-179)

                if hue_diff <= 15:
                    color = kept.mean(axis=0).astype(np.float64)
                    reliable = True
                else:
                    color = centers[dominant].astype(np.float64)
                    # A near-even split of genuinely different hues means the
                    # crop is still ambiguous even after filtering — don't
                    # trust it as a confident reading.
                    reliable = dominant_frac >= 0.55
            else:
                # Too few surviving pixels for a meaningful k-means split;
                # just average them (already passed the reliability floor).
                color = kept.mean(axis=0).astype(np.float64)
                reliable = True

            return (color, reliable) if return_reliability else color

        except Exception:
            return _bail(np.array([128.0, 128.0, 128.0]))

    # ── Occlusion detection ───────────────────────────────────────────────────

    @staticmethod
    def _bbox_overlap_frac(bbox, other_bbox):
        """Fraction of bbox's own area covered by other_bbox."""
        x1, y1, x2, y2 = bbox
        ox1, oy1, ox2, oy2 = other_bbox
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return inter / area if area > 0 else 0.0

    def is_occluded(self, player_id, bbox, frame_dets):
        """True if another tracked player's bbox in this same frame covers
        >= OCCLUSION_OVERLAP_THRESHOLD of this player's own bbox area —
        i.e. the wide-torso crop get_player_color samples is likely
        catching some of that other player's jersey. frame_dets is the
        {pid: det} dict for this one frame (all_player_tracks[fn])."""
        for other_id, other_det in frame_dets.items():
            if other_id == player_id:
                continue
            other_bbox = other_det.get('bbox')
            if other_bbox is None:
                continue
            if self._bbox_overlap_frac(bbox, other_bbox) >= OCCLUSION_OVERLAP_THRESHOLD:
                return True
        return False

    # ── Multi-frame cluster fitting ───────────────────────────────────────────

    def assign_team_color(self, video_frames, all_player_tracks):
        """
        Collect filtered jersey colours from multiple frames, fit KMeans(3),
        and lock the two team cluster centroids permanently.

        Parameters
        ----------
        video_frames       : list of BGR frames (full video)
        all_player_tracks  : list of per-frame player-track dicts
                             (tracks['players'] from the tracker)
        """
        sample_fns = [fn for fn in CLUSTER_FRAMES
                      if fn < len(video_frames) and fn < len(all_player_tracks)]

        # NOTE: unlike per-player locking (get_player_team), this pools ~100
        # samples across many players — a handful of unreliable (grass-
        # fallback) crops get diluted by the rest and don't meaningfully
        # skew KMeans, so they're kept in. (Tried excluding them too: with
        # only 5 sample frames the reliable-sample pool shrank enough to
        # nearly halve centroid separation — worse, not better. The
        # reliability gate matters at the per-player level, where a single
        # player has no other players' samples to average against.)
        colors = []
        n_gk_excluded = 0
        for fn in sample_fns:
            for _, det in all_player_tracks[fn].items():
                if det.get('is_goalkeeper'):
                    n_gk_excluded += 1
                    continue
                bbox = det.get('bbox')
                if bbox is None:
                    continue
                c = self.get_player_color(video_frames[fn], bbox)
                colors.append(c)
        if n_gk_excluded:
            print(f"[team_assigner] excluded {n_gk_excluded} goalkeeper detection(s) "
                  f"from team-color clustering")

        if len(colors) < 3:
            c0 = colors[0] if len(colors) > 0 else np.array([220.0, 50.0, 50.0])
            c1 = colors[1] if len(colors) > 1 else np.array([50.0, 50.0, 220.0])
            self.team_colors[1] = c0
            self.team_colors[2] = c1
            self.locked_colors  = {1: c0.copy(), 2: c1.copy()}
            print("[team_assigner] WARNING: too few detections for reliable clustering.")
            return

        colors_arr = np.array(colors, dtype=np.float32)
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(colors_arr)

        # Two largest clusters → teams; smallest → referees / unknowns
        counts     = np.bincount(kmeans.labels_, minlength=3)
        sorted_idx = np.argsort(counts)[::-1]
        t1c = kmeans.cluster_centers_[sorted_idx[0]]
        t2c = kmeans.cluster_centers_[sorted_idx[1]]

        self.team_colors[1] = t1c
        self.team_colors[2] = t2c
        self.locked_colors  = {1: t1c.copy(), 2: t2c.copy()}

        sep = float(np.linalg.norm(t1c - t2c))
        sep_label = "OK" if sep >= SEPARATION_WARNING else f"WARNING: only {sep:.1f} — teams may be misassigned"

        print()
        print("=" * 60)
        print(f"[team_assigner] KMeans fit on {len(colors)} samples "
              f"from frames {sample_fns}")
        print(f"  Team 1  BGR: B={t1c[0]:5.1f}  G={t1c[1]:5.1f}  R={t1c[2]:5.1f}")
        print(f"  Team 2  BGR: B={t2c[0]:5.1f}  G={t2c[1]:5.1f}  R={t2c[2]:5.1f}")
        print(f"  Centroid separation: {sep:.1f}  [{sep_label}]")
        print(f"  Cluster populations: {[counts[i] for i in sorted_idx]}")
        print("=" * 60)
        print()

    # ── Batch (two-pass) team resolution ──────────────────────────────────────
    #
    # This is the pipeline entry point main.py / fast_common.py use. It
    # replaces sequential online locking (get_player_team below, kept for
    # standalone diagnostic scripts that still call it directly) with an
    # offline two-pass design:
    #   Pass 1 (resolve_all_teams): read every frame once, gather each
    #     player's evidence across their ENTIRE screen time, then decide one
    #     final team per player from the complete evidence.
    #   Pass 2 (apply_final_teams): write that one final decision into every
    #     frame the player appears in.
    #
    # Sequential online locking made every early frame's rendered colour a
    # function of "how much evidence had accumulated by that point in
    # processing order" — for a player who locked late (e.g. frame 400 of a
    # 750-frame clip), every frame before 400 rendered gray even though the
    # player's team was never actually in doubt once the whole clip was
    # considered. Verified concretely: pid 10 gray for 43% of its own
    # screen time, pid 82 for 89%, despite both eventually locking
    # correctly — that's a property of processing order, not evidence
    # quality, and offline video has no reason to pay that cost.

    def resolve_all_teams(self, video_frames, all_player_tracks, all_referee_tracks=None):
        """
        PASS 1. Fits the team centroids (assign_team_color, unchanged), then
        makes a single read through every frame, collecting for every
        player: (a) every is_goalkeeper vote, and (b) every reliable,
        confident colour sample (same reliability gate and
        CONFIDENCE_THRESHOLD as the online path). Only after that full read
        does it decide each player's final team — same
        MIN_SAMPLES_TO_LOCK/AGREEMENT_THRESHOLD majority-agreement rule as
        before, just applied to the complete evidence instead of a
        recent-history window (PENDING_SAMPLE_WINDOW existed to bound an
        *online* algorithm's responsiveness; a batch pass has the whole
        clip up front, so there's no reason to discard older evidence).

        A player is only left unresolved (absent from player_team_dict,
        meaning still team 0) if their complete evidence never clears the
        same bar the online path required — this is a real, intended
        outcome, not a bug: finalize_fallback_assignments() and
        merge_fragmented_tracks() get a chance to resolve some of those
        afterward, and whatever's left after both stays genuinely unknown.

        Does not write into all_player_tracks — call apply_final_teams()
        once all resolution steps (this + fallback + merge) are done.
        """
        self.assign_team_color(video_frames, all_player_tracks)

        gk_votes = {}      # pid -> [bool, ...] across every frame it appears
        color_votes = {}   # pid -> [1/2, ...] from reliable+confident frames only
        color_vote_frames = {}   # pid -> [fn, ...], index-parallel to color_votes

        n_frames = min(len(video_frames), len(all_player_tracks))
        for fn in range(n_frames):
            frame = video_frames[fn]
            for player_id, det in all_player_tracks[fn].items():
                bbox = det.get('bbox')
                if bbox is None:
                    continue

                gk_votes.setdefault(player_id, []).append(bool(det.get('is_goalkeeper', False)))

                if not self.locked_colors:
                    continue

                # NOTE: an occlusion filter (is_occluded(), calibrated
                # against known-contaminated frames — see
                # OCCLUSION_OVERLAP_THRESHOLD) was tried here and reverted.
                # It correctly excluded contaminated samples for pids
                # 48/21/11 (40-87% of their "confident" evidence), but did
                # NOT fix their misclassification and regressed 3 other
                # players (30, 168, 180) by starving them of votes.
                # Root cause turned out to be upstream of occlusion: these
                # players' jersey hue overlaps the grass-removal HSV range
                # in get_player_color, so their OWN jersey pixels get
                # filtered out alongside turf, leaving them chronically
                # "unreliable" (58-73% of frames) rather than confidently
                # green — occlusion was a real but secondary contributor.
                # is_occluded()/_bbox_overlap_frac() are kept below for
                # whatever fix addresses the grass-filter issue directly.

                color, reliable = self.get_player_color(frame, bbox, return_reliability=True)
                c1 = self.locked_colors.get(1, np.zeros(3))
                c2 = self.locked_colors.get(2, np.zeros(3))
                d1 = float(np.linalg.norm(color - c1))
                d2 = float(np.linalg.norm(color - c2))
                d = min(d1, d2)
                guess = 1 if d1 <= d2 else 2

                best = self.best_unresolved_distance.get(player_id)
                if best is None or d < best[0]:
                    self.best_unresolved_distance[player_id] = (d, guess, color)

                if reliable and d < CONFIDENCE_THRESHOLD:
                    color_votes.setdefault(player_id, []).append(guess)
                    color_vote_frames.setdefault(player_id, []).append(fn)

        for player_id, votes in gk_votes.items():
            if len(votes) >= MIN_SAMPLES_TO_LOCK:
                agreement = sum(votes) / len(votes)
                if agreement >= AGREEMENT_THRESHOLD:
                    self.player_team_dict[player_id] = 3
                    continue   # goalkeeper decision wins outright, skip colour

            samples = color_votes.get(player_id)
            if samples and len(samples) >= MIN_SAMPLES_TO_LOCK:
                majority = max(set(samples), key=samples.count)
                agreement = samples.count(majority) / len(samples)
                if agreement >= AGREEMENT_THRESHOLD:
                    self.player_team_dict[player_id] = majority

        # Kept around (not consumed elsewhere) for introspection/debugging
        # parity with the old per-player pending_samples/pending_gk_samples.
        self.pending_samples = color_votes
        self.color_vote_frames = color_vote_frames
        self.pending_gk_samples = gk_votes

        self._reject_non_goalkeeper_positions(all_player_tracks)
        self._reject_static_goalkeeper_detections(all_player_tracks)
        if all_referee_tracks is not None:
            self._reject_referee_colored_players(video_frames, all_player_tracks, all_referee_tracks)

    def _reject_referee_colored_players(self, video_frames, all_player_tracks, all_referee_tracks):
        """Move a track_id's 'players'-labeled frames into
        all_referee_tracks (in place) when it's really a referee whose
        detection CLASS flips to 'player' on some frames — the same
        physical, continuously-tracked entity, not a tracking gap.

        Replaces an earlier attempt (formerly in Tracker, run before team
        colors existed) that used majority vote: whichever label ('player'
        or 'referee') won more of the entity's own frames. That looked
        solid on 3 known referees (80-91% referee-labeled) until a 4th
        confirmed referee on this same clip (track_id 27) tested at only
        26% referee-labeled — the detector was flat-out wrong on this one
        MORE often than right, so no vote-share cutoff can catch it.

        What's reliable instead: a real referee's kit is a third, visually
        distinct color chosen specifically to not be confused with either
        team's -- so genuine referees confidently sit far from BOTH locked
        team centroids, while real players (even ones with noisy per-frame
        class flicker) sit close to one of them. Verified against this
        clip's own confirmed cases: real referees' average reliable jersey
        color landed >=57.6 from the nearer centroid; real players landed
        <=43.9 -- a clean gap straddling the codebase's existing
        CONFIDENCE_THRESHOLD=50.0 (the same bar already used everywhere
        else in this file to decide "is this color a confident match").

        Requires self.locked_colors (populated by assign_team_color,
        called earlier in resolve_all_teams) -- returns immediately if
        team colors were never fit (too few detections).

        Two-step gate per shared track_id, both required:
          1. Continuity: the id's combined 'players'+'referees' timeline,
             time-ordered, must be spatially/temporally continuous
             end-to-end (reusing MERGE_MAX_GAP_FRAMES/MERGE_MAX_DIST_PX,
             merge_fragmented_tracks's own "is this plausibly one physical
             entity" bar) -- guards against two different entities
             coincidentally sharing a recycled ByteTrack id. A timeline
             with any break is left completely untouched.
          2. Color: average this entity's RELIABLE jersey-color samples
             (get_player_color's dominant-cluster path, same reliability
             gate used everywhere else in this file) across its
             'players'-labeled frames, then require its distance to the
             NEARER locked team centroid to exceed CONFIDENCE_THRESHOLD --
             i.e. it doesn't confidently look like either team.
        """
        if not self.locked_colors:
            return

        referee_ids = set()
        for frame in all_referee_tracks:
            referee_ids.update(frame.keys())
        if not referee_ids:
            return

        shared_ids = referee_ids & {
            tid for frame in all_player_tracks for tid in frame.keys()
        }
        if not shared_ids:
            return

        def _center(bbox):
            return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        c1 = self.locked_colors.get(1, np.zeros(3))
        c2 = self.locked_colors.get(2, np.zeros(3))
        n_frames = min(len(video_frames), len(all_player_tracks))

        moved_ids = set()
        moved_frames = 0
        skipped_discontinuous = set()
        skipped_no_evidence = set()
        skipped_color_matches_team = set()
        for tid in shared_ids:
            timeline = []   # (frame_num, source, bbox)
            for fn, frame in enumerate(all_player_tracks):
                if tid in frame:
                    timeline.append((fn, 'players', frame[tid]['bbox']))
            for fn, frame in enumerate(all_referee_tracks):
                if tid in frame:
                    timeline.append((fn, 'referees', frame[tid]['bbox']))
            timeline.sort(key=lambda t: t[0])

            continuous = True
            for (fn_a, _, bbox_a), (fn_b, _, bbox_b) in zip(timeline, timeline[1:]):
                if fn_b - fn_a > self.MERGE_MAX_GAP_FRAMES:
                    continuous = False
                    break
                ca, cb = _center(bbox_a), _center(bbox_b)
                dist = ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5
                if dist > self.MERGE_MAX_DIST_PX:
                    continuous = False
                    break
            if not continuous:
                skipped_discontinuous.add(tid)
                continue

            reliable_colors = []
            for fn, source, bbox in timeline:
                if source != 'players' or fn >= n_frames:
                    continue
                color, reliable = self.get_player_color(video_frames[fn], bbox, return_reliability=True)
                if reliable:
                    reliable_colors.append(color)
            if not reliable_colors:
                skipped_no_evidence.add(tid)
                continue

            avg_color = np.mean(reliable_colors, axis=0)
            d1 = float(np.linalg.norm(avg_color - c1))
            d2 = float(np.linalg.norm(avg_color - c2))
            min_dist = min(d1, d2)
            if min_dist <= CONFIDENCE_THRESHOLD:
                skipped_color_matches_team.add(tid)
                continue

            for fn, source, _ in timeline:
                if source == 'players':
                    all_referee_tracks[fn][tid] = all_player_tracks[fn].pop(tid)
                    moved_frames += 1
            self.player_team_dict.pop(tid, None)
            moved_ids.add(tid)

        if moved_ids:
            print(f"[team_assigner] reconciled {moved_frames} frame(s) across {len(moved_ids)} "
                  f"track id(s) confirmed (continuous entity, jersey color far from both team "
                  f"centroids) to be a referee misclassified as 'player' on some frames (moved "
                  f"to all_referee_tracks so they never get a team color or numeric ID): "
                  f"{sorted(moved_ids)}")
        if skipped_discontinuous:
            print(f"[team_assigner] left {len(skipped_discontinuous)} shared referee/player id(s) "
                  f"untouched -- timeline wasn't continuous, treated as id-recycling coincidence "
                  f"between two different entities: {sorted(skipped_discontinuous)}")
        if skipped_color_matches_team:
            print(f"[team_assigner] left {len(skipped_color_matches_team)} shared referee/player "
                  f"id(s) untouched -- continuous but jersey color confidently matches a team "
                  f"(occasional referee-class-flicker noise on what's really a player): "
                  f"{sorted(skipped_color_matches_team)}")

    def _reject_non_goalkeeper_positions(self, all_player_tracks):
        """Override the model's is_goalkeeper vote for any team=3 track
        whose position history doesn't show genuine defensive-third
        presence — see PITCH_LENGTH_M/DEFENSIVE_THIRD_M above for why this
        check is general and clip-independent. Demotes to team 0
        (unresolved/unknown role) rather than attempting a color-based
        team guess: a rejected track is most likely a match official, who
        genuinely isn't on either team, not a misread outfield player.

        Two robustness requirements, both needed (verified against real
        data — an earlier version using only x<=35 first, plus 105 the
        overall x-extent both under- and over-rejected):
          1. Only positions within the plausible pitch rectangle
             (0-105m x 0-68m) count as evidence at all. A single
             badly-calibrated frame can report a position hundreds of
             metres off the real pitch — if that noise happens to land on
             the same side as a defensive third, an extent-only check
             (x<=35 or x>=70 anywhere in the track) would wrongly treat it
             as real goalkeeper positioning.
          2. Requires at least MIN_SAMPLES_TO_LOCK (the same "enough
             evidence to act on" bar this file already uses for locking a
             team decision, not a new one-off number) genuine in-range
             samples inside a defensive third — not just one. A handful of
             samples sitting right at the x=35/70 boundary (measurement
             jitter around the line, not real presence) shouldn't singly
             overturn the model's own is_goalkeeper vote.
        """
        gk_pids = [pid for pid, team in self.player_team_dict.items() if team == 3]
        rejected = []
        for player_id in gk_pids:
            xs = []
            for frame in all_player_tracks:
                det = frame.get(player_id)
                if det is None:
                    continue
                pos = det.get('position_transformed')
                if pos is None:
                    continue
                try:
                    x, y = float(pos[0]), float(pos[1])
                except (TypeError, IndexError, ValueError):
                    continue
                if x == 0.0 and y == 0.0:
                    continue   # (0,0) placeholder, not a real sample
                if not (0.0 <= x <= PITCH_LENGTH_M and 0.0 <= y <= PITCH_WIDTH_M):
                    continue   # outside the plausible pitch rectangle -- calibration
                              # noise, not real position evidence either way
                xs.append(x)

            if len(xs) < GOALKEEPER_POSITION_MIN_SAMPLES:
                continue   # not enough evidence to override either way

            in_defensive_third = sum(
                1 for x in xs
                if x <= DEFENSIVE_THIRD_M or x >= (PITCH_LENGTH_M - DEFENSIVE_THIRD_M))
            if in_defensive_third < MIN_SAMPLES_TO_LOCK:
                self.player_team_dict[player_id] = 0
                rejected.append(player_id)

        if rejected:
            print(f"[team_assigner] rejected {len(rejected)} is_goalkeeper track(s) whose "
                  f"position history never entered a defensive third (likely a match "
                  f"official, not a real goalkeeper): {rejected}")

    def _reject_static_goalkeeper_detections(self, all_player_tracks):
        """Override the model's is_goalkeeper vote for any team=3 track
        that shows BOTH near-zero movement AND a non-person bbox aspect
        ratio over its own lifetime -- see the GOALKEEPER_MOTION_MIN_FRAMES
        / MAX_RATE_PX_PER_FRAME / MAX_ASPECT_RATIO comment above for the
        real-data derivation and why this needs both signals at once, not
        either alone. Catches what _reject_non_goalkeeper_positions can't:
        a static object that happens to sit geometrically inside a
        defensive third (e.g. a corner flag pole), which passes the
        position check on location alone since that check only asks "is
        this near a goal?", not "is this a goalkeeper at all?". Demotes to
        team 0 (unresolved), same as the position check -- most likely a
        misdetected static object, not a misread outfield player.
        """
        gk_pids = [pid for pid, team in self.player_team_dict.items() if team == 3]
        rejected = []
        for player_id in gk_pids:
            cxs, cys, widths, heights = [], [], [], []
            for frame in all_player_tracks:
                det = frame.get(player_id)
                if det is None:
                    continue
                x1, y1, x2, y2 = det['bbox']
                cxs.append((x1 + x2) / 2.0)
                cys.append((y1 + y2) / 2.0)
                widths.append(x2 - x1)
                heights.append(y2 - y1)

            if len(cxs) < GOALKEEPER_MOTION_MIN_FRAMES:
                continue   # not enough evidence to judge motion/shape either way

            cx_range = max(cxs) - min(cxs)
            cy_range = max(cys) - min(cys)
            total_range = (cx_range ** 2 + cy_range ** 2) ** 0.5
            rate = total_range / len(cxs)

            mean_w = sum(widths) / len(widths)
            mean_h = sum(heights) / len(heights)
            aspect = mean_h / max(mean_w, 1e-6)

            if rate < GOALKEEPER_MOTION_MAX_RATE_PX_PER_FRAME and aspect > GOALKEEPER_MAX_ASPECT_RATIO:
                self.player_team_dict[player_id] = 0
                rejected.append(player_id)

        if rejected:
            print(f"[team_assigner] rejected {len(rejected)} is_goalkeeper track(s) with "
                  f"near-zero position movement AND a non-person bbox aspect ratio over "
                  f"their own lifetime (likely a static object misdetected as a goalkeeper, "
                  f"e.g. a corner flag): {rejected}")

    def apply_final_teams(self, all_player_tracks):
        """
        PASS 2. Writes each player's FINAL team decision (from
        resolve_all_teams + finalize_fallback_assignments +
        merge_fragmented_tracks, whatever combination of those actually
        resolved them) into every frame that player appears in — so a
        player's colour is visually consistent across their entire
        appearance instead of gray-then-coloured partway through. Call
        this once, after all resolution steps are finished.
        """
        for frame in all_player_tracks:
            for player_id, det in frame.items():
                team = self.player_team_dict.get(player_id, 0)
                det['team'] = team
                det['team_color'] = self.team_colors[team]
                if player_id in self.fallback_assigned:
                    det['team_fallback'] = True
                if player_id in self.merged_pids:
                    det['team_merged'] = True

    # ── Per-player team assignment (legacy online/sequential path) ─────────────
    # Retained for standalone diagnostic scripts (test_team_classification.py,
    # the diag_*.py / analyze_*.py tools) that call it directly. The live
    # rendering pipeline (main.py, fast_common.py) uses resolve_all_teams +
    # apply_final_teams above instead — see that section's docstring for why.

    def get_player_team(self, frame, player_bbox, player_id, is_goalkeeper=False):
        """
        Compare the player's filtered jersey colour against the locked team
        centroids.  Returns 1 or 2 once enough reliable, confident samples
        agree; else 0 (Unknown / referee, still being evaluated).

        Goalkeepers (is_goalkeeper=True, from the detection model's own
        goalkeeper class) are never colour-compared against the outfield
        centroids at all — their kit is deliberately a third, distinct
        colour, so jersey-colour distance is meaningless for them. Locking
        to team 3 requires the same MIN_SAMPLES_TO_LOCK/AGREEMENT_THRESHOLD
        consistency check as team 1/2 — a single is_goalkeeper=True frame is
        not enough (a real case: one player was flagged goalkeeper on only
        2 of 565 frames, a stray misdetection that used to permanently lock
        them to team 3 on the spot). While goalkeeper evidence is still
        pending, the player falls through to the normal colour-based
        evaluation below, so a misdetected outfield player keeps getting a
        real chance to resolve to their actual team instead of stalling.

        Locking requires MIN_SAMPLES_TO_LOCK reliable+confident samples
        (get_player_color's HSV filter actually found jersey-like pixels,
        AND the resulting colour is within CONFIDENCE_THRESHOLD of a
        centroid) with at least AGREEMENT_THRESHOLD of them agreeing on the
        same team — not just a single lucky/unlucky frame. This guards
        against a self-consistent run of bad-crop frames (e.g. a player
        whose early appearances are all mid-stride with legs spread, so the
        fixed torso-crop window repeatedly catches background instead of
        jersey) locking confidently onto the wrong team before a good crop
        ever appears — a single-sample lock has no way to tell "consistently
        right" from "consistently wrong" apart.

        Confident assignments (1, 2, or 3) are cached permanently so the same
        player is never re-evaluated after a good lock is obtained.
        """
        # Return permanently locked assignment. Unknown (0) is never cached
        # so the player is re-evaluated every frame until enough confident
        # crops are obtained.
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if is_goalkeeper:
            gk_samples = self.pending_gk_samples.setdefault(player_id, [])
            gk_samples.append(True)
            if len(gk_samples) > PENDING_SAMPLE_WINDOW:
                del gk_samples[:-PENDING_SAMPLE_WINDOW]
            if len(gk_samples) >= MIN_SAMPLES_TO_LOCK:
                agreement = sum(gk_samples) / len(gk_samples)
                if agreement >= AGREEMENT_THRESHOLD:
                    self.player_team_dict[player_id] = 3   # lock permanently
                    del self.pending_gk_samples[player_id]
                    return 3
            # Not enough goalkeeper evidence yet — fall through to the normal
            # colour-based evaluation below rather than stalling; if this is
            # a real misdetection the player's actual jersey colour is right
            # there in this same frame.
        elif player_id in self.pending_gk_samples:
            # Had some goalkeeper votes pending; this frame says otherwise —
            # record it so a stray earlier True gets diluted rather than
            # sitting there waiting to be reinforced by chance.
            gk_samples = self.pending_gk_samples[player_id]
            gk_samples.append(False)
            if len(gk_samples) > PENDING_SAMPLE_WINDOW:
                del gk_samples[:-PENDING_SAMPLE_WINDOW]

        if not self.locked_colors:
            return 0

        color, reliable = self.get_player_color(frame, player_bbox, return_reliability=True)

        c1 = self.locked_colors.get(1, np.zeros(3))
        c2 = self.locked_colors.get(2, np.zeros(3))
        d1 = float(np.linalg.norm(color - c1))
        d2 = float(np.linalg.norm(color - c2))
        d = min(d1, d2)
        guess = 1 if d1 <= d2 else 2

        # Track the best-ever distance seen for this still-unresolved player,
        # reliable or not — a conservative end-of-clip fallback
        # (finalize_fallback_assignments) uses this for players who
        # structurally never pass the reliability gate (e.g. a jersey hue
        # the grass filter always strips, however wide the crop).
        best = self.best_unresolved_distance.get(player_id)
        if best is None or d < best[0]:
            self.best_unresolved_distance[player_id] = (d, guess, color)

        if not reliable:
            return 0   # fallback (unfiltered) crop color — not trustworthy for the normal locking path

        if d >= CONFIDENCE_THRESHOLD:
            return 0   # not close enough to either centroid this frame
        samples = self.pending_samples.setdefault(player_id, [])
        samples.append(guess)
        if len(samples) > PENDING_SAMPLE_WINDOW:
            del samples[:-PENDING_SAMPLE_WINDOW]

        if len(samples) >= MIN_SAMPLES_TO_LOCK:
            majority = max(set(samples), key=samples.count)
            agreement = samples.count(majority) / len(samples)
            if agreement >= AGREEMENT_THRESHOLD:
                self.player_team_dict[player_id] = majority   # lock permanently
                del self.pending_samples[player_id]
                return majority

        return 0   # still accumulating evidence

    # ── Conservative end-of-clip fallback ─────────────────────────────────────

    FALLBACK_DISTANCE_THRESHOLD = 15.0

    def finalize_fallback_assignments(self, fallback_threshold=None):
        """
        Call once, after processing every frame of the clip. Some players
        never pass the reliability gate on a single frame — e.g. a jersey
        hue that the grass-removal filter strips out no matter how the
        crop is framed — and stay Unknown (0) forever under the normal
        path even when their best-ever (unreliable) color reading is an
        excellent, unambiguous match.

        Assigns team 1/2 to any STILL-unresolved player whose best-ever
        distance to a centroid is under fallback_threshold (default
        FALLBACK_DISTANCE_THRESHOLD=15.0 — deliberately tight, well inside
        CONFIDENCE_THRESHOLD=50.0, so only near-certain matches get a
        fallback guess). A real case: 5 players with 85-741 frames of
        screen time each never once passed the reliability gate, yet their
        best-ever distance was 1.3-6.8 — clearly one team, just never
        confirmed. Genuinely ambiguous players (best distance in the
        50-100 range) are left unresolved rather than forced to a guess.

        Returns {pid: team} for players newly assigned this way. Each is
        also recorded in self.fallback_assigned (with its distance) so a
        caller can flag it as lower-confidence downstream — e.g. tag the
        track with a 'team_fallback': True marker — rather than being
        indistinguishable from a normally-locked player.
        """
        threshold = fallback_threshold if fallback_threshold is not None else self.FALLBACK_DISTANCE_THRESHOLD
        newly_assigned = {}
        for pid, (dist, guess, color) in self.best_unresolved_distance.items():
            if pid in self.player_team_dict:
                continue   # already resolved through the normal path
            if dist < threshold:
                self.player_team_dict[pid] = guess
                self.fallback_assigned[pid] = {'distance': dist, 'team': guess}
                newly_assigned[pid] = guess
        return newly_assigned

    # ── Track fragmentation merging ───────────────────────────────────────────

    MERGE_MAX_GAP_FRAMES = 30
    MERGE_MAX_DIST_PX = 100.0
    MERGE_APPEARANCE_THRESHOLD = 40.0

    # Candidacy bar for the OVERLAPPING-track case (two track ids active at
    # the same time, e.g. a ByteTrack duplicate-ID glitch), as mean
    # _bbox_overlap_frac across every frame both tracks share -- proximity
    # evidence parallel to max_dist_px above, just for tracks that don't
    # have a sequential gap at all. Derived from this codebase's own data,
    # not guessed: a confirmed duplicate-ID pair (AlgeriaArgentia_w685
    # pid67/pid81) held 99.6% mean overlap across their full 31-frame
    # shared window. The closest thing to a false-positive risk -- genuine
    # outfield duels/near-collisions checked across 4 clips, which DO spike
    # to 0.49-0.69 on a single contact frame -- averaged only 0.16-0.20
    # across their own full shared window once you look past that one
    # frame. 0.5 sits with a >2x margin below the true positive and >2x
    # above the highest genuine-duel mean found.
    MERGE_OVERLAP_MIN_MEAN_FRAC = 0.5

    def merge_fragmented_tracks(self, video_frames, all_player_tracks,
                                 max_gap_frames=None, max_dist_px=None,
                                 appearance_threshold=None, exclude_pairs=None):
        """
        Detect and merge ByteTrack ID fragments of the same physical
        player — OUTFIELD or GOALKEEPER alike (see the note above
        MERGE_OVERLAP_MIN_MEAN_FRAC: goalkeepers used to be excluded from
        this entirely, which meant a real keeper fragmented by ByteTrack
        had no way to ever be relinked) — e.g. after a brief occlusion,
        the tracker loses and re-acquires the same person under a new
        track_id, leaving two (or more) separate, individually-short
        tracks that never individually accumulate enough evidence to
        lock, even though a combined view of them would.

        A candidate pair (A, B) requires BOTH:
          1. Proximity, either of:
             a. Sequential: B starts within max_gap_frames of A's last
                frame, within max_dist_px of A's last position.
             b. Overlapping: A and B are both active at the same time
                (e.g. a duplicate-ID glitch, not a gap) and their boxes
                cover >= MERGE_OVERLAP_MIN_MEAN_FRAC of each other, on
                average, across every frame they share.
          2. Appearance agreement: A's and B's own average reliable jersey
             color (sampled independently across each segment) must be
             within appearance_threshold of each other.
        Proximity alone is not trustworthy — verified on this codebase's
        own data: several proximity-only candidates showed CONTRADICTORY
        resolved teams between predecessor and successor (e.g. one train
        of pure spatial coincidence linked a Team 1 track to a Team 2
        track), i.e. two different players who simply happened to be near
        each other at a handover moment. The appearance check catches
        those -- and is exactly what keeps a real goalkeeper fragment from
        merging into an unrelated nearby misdetection (e.g. a corner flag
        or a diving outfield player caught in the same goal-mouth scramble)
        even though both would pass the proximity/overlap test.

        For an accepted pair where exactly one side already has a
        resolved team (1/2/3, from the normal path or the fallback), that
        team is propagated to every frame of the OTHER (still-unresolved)
        side, and those frames get 'team_merged': True so the propagation
        stays distinguishable from a normal lock. Pairs where both sides
        are already resolved but DISAGREE are rejected, not overridden.
        Pairs where neither side is resolved are rejected too (nothing to
        propagate) even if their appearance agrees.

        exclude_pairs : iterable of (pid, pid) pairs, optional
            Pairs to never consider as merge candidates regardless of how
            well they'd otherwise score. Needed for a pair that was just
            produced by splitting one ByteTrack id into two (see
            trackers.Tracker.applied_splits) — the split-off fragment is
            by construction immediately adjacent (zero gap) to the
            original, so it would otherwise look like an ideal merge
            candidate on proximity grounds; merging it back together
            would silently undo the split, since the whole reason for
            splitting was that the two sides' appearance disagrees.

        Returns (merged, rejected) — lists of dicts describing each
        candidate pair's outcome and why, for reporting.
        """
        max_gap_frames = max_gap_frames if max_gap_frames is not None else self.MERGE_MAX_GAP_FRAMES
        max_dist_px = max_dist_px if max_dist_px is not None else self.MERGE_MAX_DIST_PX
        appearance_threshold = (appearance_threshold if appearance_threshold is not None
                                 else self.MERGE_APPEARANCE_THRESHOLD)
        exclude_pairs = {frozenset(p) for p in exclude_pairs} if exclude_pairs else set()

        # NOTE: goalkeeper (team=3) tracks are NOT excluded here (they used
        # to be, unconditionally) -- a real goalkeeper fragments under
        # ByteTrack exactly like an outfield player does (confirmed: 3
        # known clips where a single real keeper split into 7-9 track IDs,
        # all clustered in the same goal-area position range, all visually
        # the same person), and excluding them meant those fragments could
        # NEVER be relinked. The same two-part evidence bar below (temporal/
        # spatial adjacency + appearance agreement) applies to them
        # unchanged -- no goalkeeper-specific heuristic. _reject_non_
        # goalkeeper_positions (called from resolve_all_teams, before this
        # method runs) already removed the non-goalkeeper misdetections it
        # can catch (position never near a goal); a separate static-object
        # check (_reject_static_goalkeeper_detections) catches the ones it
        # can't (e.g. a corner flag sitting right at the pitch corner,
        # which IS geometrically inside a defensive third).
        track_info = {}
        for fn, frame in enumerate(all_player_tracks):
            for pid, det in frame.items():
                bbox = det['bbox']
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if pid not in track_info:
                    track_info[pid] = {'first_fn': fn, 'first_pos': (cx, cy),
                                        'last_fn': fn, 'last_pos': (cx, cy)}
                else:
                    track_info[pid]['last_fn'] = fn
                    track_info[pid]['last_pos'] = (cx, cy)

        candidates = []
        pids = list(track_info.keys())
        for a in pids:
            a_end_fn, a_end_pos = track_info[a]['last_fn'], track_info[a]['last_pos']
            for b in pids:
                if a == b:
                    continue
                if frozenset((a, b)) in exclude_pairs:
                    continue
                b_start_fn = track_info[b]['first_fn']
                if b_start_fn <= a_end_fn:
                    continue
                gap = b_start_fn - a_end_fn
                if gap > max_gap_frames:
                    continue
                b_start_pos = track_info[b]['first_pos']
                dist = ((a_end_pos[0] - b_start_pos[0]) ** 2
                        + (a_end_pos[1] - b_start_pos[1]) ** 2) ** 0.5
                if dist < max_dist_px:
                    candidates.append((a, b, gap, dist))

        # Overlapping-track candidates: a duplicate-ID glitch (ByteTrack
        # assigns a SECOND id to the same physical detection for a run of
        # frames, rather than losing and reacquiring it) produces two
        # tracks that are simultaneously active -- the sequential loop
        # above can never find these (it requires b to start strictly
        # after a ends). Reuses _bbox_overlap_frac, the same primitive
        # already validated for within-frame occlusion detection, as the
        # proximity signal here instead of inventing a new one; the
        # appearance-agreement gate below applies identically either way.
        seen_overlap_pairs = set()
        for a in pids:
            a_first_fn, a_last_fn = track_info[a]['first_fn'], track_info[a]['last_fn']
            for b in pids:
                if a == b:
                    continue
                pair_key = frozenset((a, b))
                if pair_key in exclude_pairs or pair_key in seen_overlap_pairs:
                    continue
                b_first_fn, b_last_fn = track_info[b]['first_fn'], track_info[b]['last_fn']
                overlap_lo, overlap_hi = max(a_first_fn, b_first_fn), min(a_last_fn, b_last_fn)
                if overlap_lo > overlap_hi:
                    continue   # no temporal overlap -- already covered above
                seen_overlap_pairs.add(pair_key)

                fracs = []
                for fn in range(overlap_lo, overlap_hi + 1):
                    det_a = all_player_tracks[fn].get(a)
                    det_b = all_player_tracks[fn].get(b)
                    if det_a is None or det_b is None:
                        continue
                    fracs.append(self._bbox_overlap_frac(det_a['bbox'], det_b['bbox']))
                if not fracs:
                    continue
                mean_overlap = sum(fracs) / len(fracs)
                if mean_overlap >= self.MERGE_OVERLAP_MIN_MEAN_FRAC:
                    candidates.append((a, b, 0, 0.0))

        def _avg_color(pid, lo_fn, hi_fn, stride=3):
            """Average reliable jersey color across [lo_fn, hi_fn]. If the
            track never produces a single reliable sample, fall back to
            self.best_unresolved_distance's stored BEST (least-noisy)
            single sample rather than a flat average of mostly-contaminated
            fallback-crop reads — averaging noise dilutes the one good
            signal a track might have, exactly the problem
            finalize_fallback_assignments's best-distance approach exists
            to avoid; consistent to reuse that same philosophy (and the
            already-computed value) here. Returns (color, used_reliable)."""
            reliable_colors = []
            for fn in range(lo_fn, hi_fn + 1, stride):
                if fn >= len(all_player_tracks) or fn >= len(video_frames):
                    break
                det = all_player_tracks[fn].get(pid)
                if det is None:
                    continue
                color, reliable = self.get_player_color(video_frames[fn], det['bbox'],
                                                          return_reliability=True)
                if reliable:
                    reliable_colors.append(color)
            if reliable_colors:
                return np.mean(reliable_colors, axis=0), True
            best = self.best_unresolved_distance.get(pid)
            if best is not None:
                return best[2], False   # (dist, team_guess, color)
            return None, False

        merged, rejected = [], []
        for a, b, gap, dist in candidates:
            a_color, a_rel = _avg_color(a, track_info[a]['first_fn'], track_info[a]['last_fn'])
            b_color, b_rel = _avg_color(b, track_info[b]['first_fn'], track_info[b]['last_fn'])

            if a_color is None or b_color is None:
                rejected.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                  'reason': 'insufficient appearance data'})
                continue

            # Stricter bar when either side's estimate is itself built from
            # unreliable (fallback-crop) samples only — noisier evidence
            # needs a tighter match before it's trusted.
            threshold = appearance_threshold if (a_rel and b_rel) else appearance_threshold * 0.5
            appearance_dist = float(np.linalg.norm(a_color - b_color))
            if appearance_dist > threshold:
                rejected.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                  'reason': f'appearance mismatch (dist={appearance_dist:.1f}, '
                                            f'threshold={threshold:.1f}, reliable={a_rel and b_rel})'})
                continue

            a_team = self.player_team_dict.get(a, 0)
            b_team = self.player_team_dict.get(b, 0)

            if a_team != 0 and b_team == 0:
                self._propagate_team(all_player_tracks, b, a_team,
                                      track_info[b]['first_fn'], track_info[b]['last_fn'])
                merged.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                'appearance_dist': appearance_dist,
                                'action': f'propagated A(team={a_team}) -> B'})
            elif b_team != 0 and a_team == 0:
                self._propagate_team(all_player_tracks, a, b_team,
                                      track_info[a]['first_fn'], track_info[a]['last_fn'])
                merged.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                'appearance_dist': appearance_dist,
                                'action': f'propagated B(team={b_team}) -> A'})
            elif a_team != 0 and b_team != 0 and a_team == b_team:
                merged.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                'appearance_dist': appearance_dist,
                                'action': f'already agree (team={a_team})'})
            elif a_team != 0 and b_team != 0 and a_team != b_team:
                rejected.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                  'reason': f'contradictory resolved teams (A={a_team}, B={b_team})'})
            else:
                rejected.append({'a': a, 'b': b, 'gap': gap, 'dist': dist,
                                  'reason': 'both unresolved, nothing to propagate'})

        return merged, rejected

    def _propagate_team(self, all_player_tracks, pid, team, lo_fn, hi_fn):
        """Write a merged-in team assignment across every frame pid appears
        in within [lo_fn, hi_fn], flagged 'team_merged': True.

        Records the decision in player_team_dict/merged_pids either way;
        the direct per-frame write below is now redundant with the caller
        following up with apply_final_teams() (which covers the pid's full
        appearance range, not just [lo_fn, hi_fn]) but is kept so any
        caller that doesn't call apply_final_teams still gets correct
        results, same as before this pid was two-pass."""
        self.player_team_dict[pid] = team
        self.merged_pids.add(pid)
        for fn in range(lo_fn, hi_fn + 1):
            if fn >= len(all_player_tracks):
                break
            det = all_player_tracks[fn].get(pid)
            if det is not None:
                det['team'] = team
                det['team_color'] = self.team_colors[team]
                det['team_merged'] = True
