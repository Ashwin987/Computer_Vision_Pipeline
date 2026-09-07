"""
ball_fallback.py — targeted per-frame ball-detection fallback.

When the primary model (models/best.pt via trackers/tracker.py) fails to
detect the ball on a frame it actually attempted, this runs the tiled
Roboflow model (test_roboflow_ball.py's InferenceSlicer + BallTracker
approach) ONLY on that one frame to try to recover it. It is never run
on frames the primary model already succeeded on, and never run on
frames a pipeline didn't attempt detection on in the first place (e.g.
fast_setup.py's skipped every-3rd-frame gaps).

Reuses test_roboflow_ball.py's load_model()/build_slicer()/BallTracker
as-is (imported, not duplicated) — that file remains the standalone
isolated-test entry point; this module is the integration-facing
wrapper trackers/tracker.py and fast_setup.py call into.

The Roboflow model + slicer are loaded lazily (only on the first actual
fallback trigger) so importing this module, or running a pipeline with
the fallback disabled, costs nothing and needs no API key.
"""
import time

from test_roboflow_ball import load_model, build_slicer, BallTracker


class BallFallback:
    """One instance per pipeline run. Call try_recover(frame) only on
    frames where the primary model already missed the ball."""

    def __init__(self, verbose=True, progress_every=10):
        self._model = None
        self._slicer = None
        self._tracker = BallTracker(buffer_size=10)
        self.n_triggered = 0
        self.n_recovered = 0
        self.total_fallback_sec = 0.0
        self.verbose = verbose
        self.progress_every = progress_every
        self._run_start = None

    def _ensure_ready(self, frame_shape):
        if self._model is None:
            self._model = load_model()
        if self._slicer is None:
            self._slicer = build_slicer(self._model, frame_shape)

    def try_recover(self, frame):
        """frame: full-resolution BGR frame (np.ndarray) at native video
        resolution (NOT half-res/resized) — the same frame the primary
        model would have seen if it hadn't been downscaled.

        Returns a [x1,y1,x2,y2] bbox on success, or None if the fallback
        also failed to find the ball. Always updates the instrumentation
        counters, so a caller can report the real cost/yield afterward
        regardless of outcome."""
        self._ensure_ready(frame.shape)
        if self._run_start is None:
            self._run_start = time.time()
        self.n_triggered += 1

        t0 = time.time()
        dets = self._slicer(frame)
        tracked = self._tracker.update(dets)
        self.total_fallback_sec += time.time() - t0

        recovered = len(tracked) > 0
        if recovered:
            self.n_recovered += 1

        if self.verbose and self.n_triggered % self.progress_every == 0:
            elapsed = time.time() - self._run_start
            print(f"  [ball_fallback] {self.n_triggered} triggered so far "
                  f"({self.n_recovered} recovered)  elapsed={elapsed:.0f}s  "
                  f"avg={elapsed / self.n_triggered:.2f}s/trigger", flush=True)

        return tracked.xyxy[0].tolist() if recovered else None

    def summary(self):
        return {
            'n_triggered': self.n_triggered,
            'n_recovered': self.n_recovered,
            'recovery_rate': (self.n_recovered / self.n_triggered) if self.n_triggered else None,
            'total_fallback_sec': self.total_fallback_sec,
            'avg_sec_per_trigger': (self.total_fallback_sec / self.n_triggered) if self.n_triggered else None,
        }
