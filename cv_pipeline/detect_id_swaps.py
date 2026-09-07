"""
Diagnostic-only script (not part of the pipeline). Scans every player
track for a sharp, sustained team-color discontinuity partway through its
life -- a signature consistent with ByteTrack silently reassigning one
track ID to a different physical player (rather than occlusion noise,
which produces scattered/mixed misreads, not a clean two-segment split).

STEP 1: for every pid, gather confident (reliable + d<CONFIDENCE_THRESHOLD)
color samples in frame order, then scan all candidate split points for the
one that best separates the sequence into two internally-pure, oppositely-
majority segments. Flag pids where splitting meaningfully beats the
whole-track majority purity.

STEP 2: for each flagged pid, locate the frame gap spanning the split and
check bbox continuity across it (does the track actually drop frames right
around the swap, or does it stay continuously present -- i.e. did
ByteTrack's own tracking ever lose the object, or did an IOU association
silently pick the wrong neighboring detection while never losing the ID).

Usage:
    python detect_id_swaps.py
    python detect_id_swaps.py --video Match_videos/other.mp4 --stub stubs/other_track_stub.pkl
    python detect_id_swaps.py --start 0 --end 1500   # scan only this frame window

--start/--end bound both which frames get scanned AND how much of the
video is ever read into memory (streamed via cv2.VideoCapture, not
utils.read_video()'s full-list load) -- required to run against a long
clip (e.g. the fastpipe_ stubs' source video) without exhausting RAM the
same way fast_setup.py's own streaming design avoids it.
"""

import argparse
import pickle

import cv2
import numpy as np

from team_assigner import TeamAssigner
from team_assigner.team_assigner_color_bak import CONFIDENCE_THRESHOLD

DEFAULT_TRACK_STUB = 'stubs/track_stubs_121364_ball_fallback.pkl'
DEFAULT_VIDEO_PATH = 'Match_videos/121364_0.mp4'

MIN_TRACK_SAMPLES = 10     # need at least this many confident samples to even consider
MIN_SEGMENT = 5            # each side of a candidate split needs >= this many samples
SEGMENT_PURITY_MIN = 0.88  # each side must be at least this pure

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--video', default=DEFAULT_VIDEO_PATH,
                    help='Path to the source video (default: original tuned clip)')
parser.add_argument('--stub', default=DEFAULT_TRACK_STUB,
                    help='Path to the track stub with players/referees/ball (default: original tuned clip)')
parser.add_argument('--start', type=int, default=0,
                    help='First frame (inclusive) to scan')
parser.add_argument('--end', type=int, default=None,
                    help='Last frame (exclusive); default = full stub length')
args = parser.parse_args()
VIDEO_PATH, TRACK_STUB = args.video, args.stub
is_default_clip = (VIDEO_PATH == DEFAULT_VIDEO_PATH and TRACK_STUB == DEFAULT_TRACK_STUB)

with open(TRACK_STUB, 'rb') as f:
    full_tracks = pickle.load(f)

stub_len = len(full_tracks['players'])
start = max(0, args.start)
end = stub_len if args.end is None else min(args.end, stub_len)
if start >= end:
    raise ValueError(f"start ({start}) must be < end ({end})")

tracks = {'players': full_tracks['players'][start:end]}
print(f"Scanning window [{start}:{end}) of {stub_len} frames in {TRACK_STUB}")

print(f"Reading video frames [{start}:{end}) from {VIDEO_PATH} (streamed, not the whole file)...")
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, start)
video_frames = []
for _ in range(end - start):
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
cap.release()

ta = TeamAssigner()
ta.assign_team_color(video_frames, tracks['players'])
c1, c2 = ta.locked_colors[1], ta.locked_colors[2]

# ── Gather confident samples per pid, excluding goalkeepers ────────────────
all_pids = set()
for fd in tracks['players']:
    all_pids.update(fd.keys())

gk_frac = {}
for fd in tracks['players']:
    for pid, det in fd.items():
        g, t = gk_frac.get(pid, (0, 0))
        gk_frac[pid] = (g + (1 if det.get('is_goalkeeper') else 0), t + 1)
majority_gk = {pid for pid, (g, t) in gk_frac.items() if t and g / t > 0.5}

n_frames = min(len(video_frames), len(tracks['players']))
samples_by_pid = {pid: [] for pid in all_pids}
present_frames_by_pid = {pid: [] for pid in all_pids}
# local_fn indexes the windowed lists; abs_fn (= start + local_fn) is what
# gets stored/printed, so reported frame numbers always match the real
# video's own numbering regardless of --start.
for local_fn in range(n_frames):
    abs_fn = start + local_fn
    for pid, det in tracks['players'][local_fn].items():
        bbox = det.get('bbox')
        if bbox is None:
            continue
        present_frames_by_pid[pid].append(abs_fn)
        if pid in majority_gk:
            continue
        color, reliable = ta.get_player_color(video_frames[local_fn], bbox, return_reliability=True)
        d1 = float(np.linalg.norm(color - c1))
        d2 = float(np.linalg.norm(color - c2))
        if reliable and min(d1, d2) < CONFIDENCE_THRESHOLD:
            guess = 1 if d1 <= d2 else 2
            samples_by_pid[pid].append((abs_fn, guess))

print(f"\nScanned {len(all_pids)} player tracks ({len(majority_gk)} excluded as majority-goalkeeper)")


def best_split(samples):
    """samples: list of (fn, guess) in frame order. Returns the split point
    that maximises the WORSE of the two segment purities (so a real clean
    break -- both sides internally consistent -- wins over a split that
    merely maximises the larger segment while leaving the smaller one
    messy). Ties broken by preferring more evidence on the smaller side.
    Returns (k, purityA, purityB, majA, majB, weighted_purity) or None."""
    n = len(samples)
    guesses = [g for _, g in samples]
    best = None
    best_key = None
    for k in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        segA, segB = guesses[:k], guesses[k:]
        majA = max(set(segA), key=segA.count)
        majB = max(set(segB), key=segB.count)
        if majA == majB:
            continue
        purA = segA.count(majA) / len(segA)
        purB = segB.count(majB) / len(segB)
        if purA < SEGMENT_PURITY_MIN or purB < SEGMENT_PURITY_MIN:
            continue
        weighted = (purA * len(segA) + purB * len(segB)) / n
        key = (min(purA, purB), min(len(segA), len(segB)))
        if best is None or key > best_key:
            best = (k, purA, purB, majA, majB, weighted)
            best_key = key
    return best


flagged = []
for pid in sorted(all_pids):
    samples = samples_by_pid[pid]
    if len(samples) < MIN_TRACK_SAMPLES:
        continue
    guesses = [g for _, g in samples]
    whole_maj = max(set(guesses), key=guesses.count)
    whole_purity = guesses.count(whole_maj) / len(guesses)

    result = best_split(samples)
    if result is None:
        continue
    k, purA, purB, majA, majB, weighted = result
    improvement = weighted - whole_purity

    split_fn_before = samples[k - 1][0]   # last sample of segment A
    split_fn_after = samples[k][0]        # first sample of segment B
    flagged.append({
        'pid': pid, 'n_samples': len(samples), 'k': k,
        'whole_purity': whole_purity, 'purA': purA, 'purB': purB,
        'majA': majA, 'majB': majB, 'weighted': weighted,
        'improvement': improvement,
        'split_fn_before': split_fn_before, 'split_fn_after': split_fn_after,
        'segA_len': k, 'segB_len': len(samples) - k,
        'first_fn': samples[0][0], 'last_fn': samples[-1][0],
    })

print(f"\n=== STEP 1: {len(flagged)} track(s) flagged as ID-swap candidates ===\n")
for f in sorted(flagged, key=lambda x: -x['improvement']):
    print(f"pid {f['pid']:>4}: {f['n_samples']} confident samples "
          f"[{f['first_fn']}..{f['last_fn']}], whole-track purity={f['whole_purity']:.0%}")
    print(f"    best split: seg A ({f['segA_len']} samples, team {f['majA']}, "
          f"purity {f['purA']:.0%}) | seg B ({f['segB_len']} samples, team {f['majB']}, "
          f"purity {f['purB']:.0%})  -- weighted purity {f['weighted']:.0%} "
          f"(+{f['improvement']:.0%} over whole-track)")
    print(f"    split gap: last seg-A confident sample at fn={f['split_fn_before']}, "
          f"first seg-B confident sample at fn={f['split_fn_after']}")

if is_default_clip:
    validated = {f['pid'] for f in flagged}
    print(f"\nKnown-positive validation: pid 4 flagged = {4 in validated}, "
          f"pid 11 flagged = {11 in validated}")

# ── STEP 2: gap / continuity analysis around each split ────────────────────
print("\n\n=== STEP 2: continuity analysis around each flagged split ===\n")
for f in sorted(flagged, key=lambda x: x['pid']):
    pid = f['pid']
    lo, hi = f['split_fn_before'], f['split_fn_after']
    present = present_frames_by_pid[pid]
    present_set = set(present)
    frames_in_window = [fn for fn in range(lo, hi + 1)]
    missing_in_window = [fn for fn in frames_in_window if fn not in present_set]

    bbox_before = tracks['players'][lo - start].get(pid, {}).get('bbox')
    bbox_after = tracks['players'][hi - start].get(pid, {}).get('bbox')
    center_before = None
    center_after = None
    dist = None
    if bbox_before is not None:
        center_before = ((bbox_before[0] + bbox_before[2]) / 2, (bbox_before[1] + bbox_before[3]) / 2)
    if bbox_after is not None:
        center_after = ((bbox_after[0] + bbox_after[2]) / 2, (bbox_after[1] + bbox_after[3]) / 2)
    if center_before and center_after:
        dist = float(np.hypot(center_after[0] - center_before[0], center_after[1] - center_before[1]))

    print(f"pid {pid}: window fn[{lo}..{hi}] ({hi - lo + 1} frames)")
    print(f"    frames MISSING from track in this window: {len(missing_in_window)} "
          f"{'(gap present -- track lost then re-acquired)' if missing_in_window else '(NO gap -- continuously tracked straight through)'}")
    if missing_in_window and len(missing_in_window) <= 20:
        print(f"    missing frames: {missing_in_window}")
    print(f"    bbox center at fn={lo}: {center_before}   at fn={hi}: {center_after}")
    print(f"    center displacement across window: {dist:.1f}px "
          f"over {hi - lo} frame(s)" if dist is not None else "    (bbox missing on one side)")
