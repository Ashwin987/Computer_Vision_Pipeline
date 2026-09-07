"""
test_ball_fallback_integration.py — tests the REAL targeted-fallback
integration (ball_fallback.py + the call sites added to
trackers/tracker.py and fast_setup.py's detect_and_track), not an
isolated model test.

Ground truth for "did the primary model miss this frame" comes straight
from the real cached fastpipe primary-detection stub
(stubs/fastpipe_track_stubs.pkl) — the actual output of fast_setup.py's
detection loop for this video, at fastpipe's real every-3rd-frame
cadence. This script does not re-run primary detection (that part of
the pipeline is unchanged and its result is already known); it only
exercises the new fallback trigger condition and the real
BallFallback.try_recover() call on the real frame pixels, exactly as
fast_setup.py's detect_and_track() would when use_ball_fallback=True.

Two windows, both from the diagnosis:
  - gap47:  frames 39301-39347, the 47-frame shot gap. Small enough
    (15 attempted-and-missed frames) to run EVERY real trigger — no
    sampling, no extrapolation.
  - fastpipe9000: frames 33475-42475. 2136 real primary misses among
    3000 attempted frames — running the real fallback on all of them
    would take ~9-12 hours based on the per-trigger latency already
    measured in test_roboflow_ball.py. Instead this runs the real
    fallback on a representative stride-sampled subset (which always
    includes every gap47 frame) and reports both the raw sampled
    numbers AND the extrapolation to the full window, clearly labeled
    as an extrapolation, not a full run.

A single BallFallback instance processes the combined sample in
ascending frame order (one real, sequential fallback pass), matching
how the real integrated pipeline would encounter these frames.
"""
import json
import pickle
import time

from test_roboflow_ball import stream_frames
from ball_fallback import BallFallback

FAST_VIDEO = 'Sample_Videos/LiverpoolPSG_short.mp4'
FAST_STUB = 'stubs/fastpipe_track_stubs.pkl'
DETECT_EVERY = 3

GAP47_LO, GAP47_HI = 39301, 39348
WINDOW_LO, WINDOW_HI = 33475, 42475

SAMPLE_STRIDE = 24  # ~89 of the 2136 real misses in the 9000-frame window

RESULTS_PATH = 'test_ball_fallback_integration_results.json'


def load_stub_ball():
    with open(FAST_STUB, 'rb') as f:
        tracks = pickle.load(f)
    return tracks['ball']


def attempted_and_split(ball, lo, hi):
    attempted = [i for i in range(lo, hi) if i % DETECT_EVERY == 0]
    hit_set = {i for i in attempted if i < len(ball) and ball[i].get(1) is not None}
    hits = [i for i in attempted if i in hit_set]
    misses = [i for i in attempted if i not in hit_set]
    return attempted, hits, misses


def run_real_fallback(indices, label):
    """One real BallFallback instance, one real sequential video pass
    over exactly these frame indices. Returns (fallback_obj, {frame: recovered_bool})."""
    fb = BallFallback()
    results = {}
    t0 = time.time()
    n_done = 0
    for fn, frame in stream_frames(FAST_VIDEO, indices):
        bbox = fb.try_recover(frame)
        results[fn] = bbox is not None
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {n_done}/{len(indices)}  elapsed={elapsed:.0f}s  "
                  f"avg={elapsed / n_done:.2f}s/frame", flush=True)
    return fb, results


def main():
    ball = load_stub_ball()

    g_attempted, g_hits, g_misses = attempted_and_split(ball, GAP47_LO, GAP47_HI)
    w_attempted, w_hits, w_misses = attempted_and_split(ball, WINDOW_LO, WINDOW_HI)

    print(f"gap47: attempted={len(g_attempted)} primary_hits={len(g_hits)} "
          f"primary_misses={len(g_misses)}")
    print(f"fastpipe9000: attempted={len(w_attempted)} primary_hits={len(w_hits)} "
          f"primary_misses={len(w_misses)}")

    sample = sorted(set(w_misses[::SAMPLE_STRIDE]) | set(g_misses))
    print(f"\nRunning ONE real, sequential fallback pass over {len(sample)} frames "
          f"(all {len(g_misses)} gap47 misses + a stride-{SAMPLE_STRIDE} sample of "
          f"the {len(w_misses)} fastpipe9000-window misses)...\n")

    fb, results = run_real_fallback(sample, 'fallback_pass')
    fb_summary = fb.summary()

    # ---- gap47: exact, every real trigger measured ----
    g_recovered = sum(1 for i in g_misses if results[i])
    g_new_hits = len(g_hits) + g_recovered
    gap47_report = {
        'window_frames': GAP47_HI - GAP47_LO,
        'attempted': len(g_attempted),
        'primary_hits': len(g_hits),
        'primary_hit_rate_of_attempted': len(g_hits) / len(g_attempted),
        'fallback_triggered': len(g_misses),
        'fallback_recovered': g_recovered,
        'fallback_recovery_rate': g_recovered / len(g_misses) if g_misses else None,
        'new_hits_total': g_new_hits,
        'new_raw_detection_rate': g_new_hits / (GAP47_HI - GAP47_LO),
        'new_pct_of_attempted': g_new_hits / len(g_attempted),
        'still_missed_frames': [i for i in g_misses if not results[i]],
        'measurement': 'EXACT — every one of the 15 real misses was run through the real fallback, no sampling',
    }

    # ---- fastpipe9000: exact miss count (from stub), sampled real fallback ----
    n_sampled = len(sample)
    n_sampled_recovered = sum(results.values())
    recovery_rate = n_sampled_recovered / n_sampled if n_sampled else None
    avg_sec_per_trigger = fb_summary['avg_sec_per_trigger']
    extrapolated_added_sec = avg_sec_per_trigger * len(w_misses) if avg_sec_per_trigger else None
    new_hits_estimate = len(w_hits) + (recovery_rate * len(w_misses) if recovery_rate is not None else 0)

    window_report = {
        'window_frames': WINDOW_HI - WINDOW_LO,
        'attempted': len(w_attempted),
        'primary_hits': len(w_hits),
        'primary_hit_rate_of_attempted': len(w_hits) / len(w_attempted),
        'primary_misses_exact': len(w_misses),
        'note_on_82_90pct_expectation': (
            f"Real miss rate among ATTEMPTED frames is {len(w_misses)/len(w_attempted):.1%} "
            f"({len(w_misses)}/{len(w_attempted)}), below the 82-90% guess — that guess likely "
            f"assumed a denominator of all frames or drew on the smaller earlier sample. As a "
            f"fraction of ALL {WINDOW_HI-WINDOW_LO} frames in the window (attempted + skipped), "
            f"triggered frames are {len(w_misses)/(WINDOW_HI-WINDOW_LO):.1%}."
        ),
        'real_sampled_triggers': n_sampled,
        'real_sampled_recovered': n_sampled_recovered,
        'measured_recovery_rate': recovery_rate,
        'real_measured_total_fallback_sec_for_sample': fb_summary['total_fallback_sec'],
        'real_measured_avg_sec_per_trigger': avg_sec_per_trigger,
        'extrapolated_total_added_sec_for_full_window': extrapolated_added_sec,
        'extrapolated_total_added_hours_for_full_window': (
            extrapolated_added_sec / 3600 if extrapolated_added_sec else None),
        'estimated_new_hits_total': new_hits_estimate,
        'estimated_new_raw_detection_rate': new_hits_estimate / (WINDOW_HI - WINDOW_LO),
        'estimated_new_pct_of_attempted': new_hits_estimate / len(w_attempted),
        'measurement': (
            f'EXTRAPOLATED — primary_misses_exact ({len(w_misses)}) is exact (from the real cached '
            f'stub), but only {n_sampled} of those misses were actually run through the real '
            f'fallback in this session; the rest are projected using the measured avg_sec_per_trigger '
            f'and measured recovery_rate from that real sample.'
        ),
    }

    all_results = {'gap47': gap47_report, 'fastpipe9000_window': window_report,
                    'raw_fallback_summary_for_sample': fb_summary}
    print("\n" + json.dumps(all_results, indent=2))
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == '__main__':
    main()
