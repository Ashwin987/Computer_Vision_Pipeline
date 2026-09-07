# -*- coding: utf-8 -*-
"""
Diagnostic: why is position_transformed out of bounds when calibration=95.6%?

Checks the pipeline:
  bbox -> position -> position_adjusted -> (H) -> position_transformed
"""

import sys, os, pickle
import numpy as np
import cv2

sys.path.insert(0, r'c:\UCLA\yolo model 2')
from utils import get_foot_position

TRACK_STUB     = 'stubs/track_stubs_121364.pkl'
CAM_STUB       = 'stubs/camera_movement_stub_121364.pkl'
HOG_STUB       = 'stubs/homography_stub_121364.pkl'
PIXEL_VERTICES = np.array([[0,1080],[0,0],[1920,0],[1920,1080]], dtype=np.float32)
TARGET_FRAMES  = [0, 50, 100, 200, 400]
PITCH_X_MAX    = 105.0
PITCH_Y_MAX    = 68.0

# ── Load stubs ────────────────────────────────────────────────────────────────
print("Loading stubs...")
with open(TRACK_STUB, 'rb') as f: tracks  = pickle.load(f)
with open(CAM_STUB,   'rb') as f: cam_mv  = pickle.load(f)
with open(HOG_STUB,   'rb') as f: hom     = pickle.load(f)

hom_none  = sum(1 for v in hom.values() if v is None)
hom_valid = len(hom) - hom_none
print(f"  track frames : {len(tracks['players'])}")
print(f"  cam_mv frames: {len(cam_mv)}")
print(f"  hom frames   : {len(hom)}  valid H: {hom_valid}  None: {hom_none}"
      f"  ({100*hom_valid/max(len(hom),1):.1f}% non-None)")
print(f"  hom sample keys: {sorted(hom.keys())[:5]}")
print()

# Fixed fallback H (ViewTransformer default)
court_length = 23.32
court_width  = 68.0
pv = np.array([[0,1080],[0,0],[1920,0],[1920,1080]], dtype=np.float32)
tv = np.array([[0,court_width],[0,0],[court_length,0],[court_length,court_width]],
               dtype=np.float32)
H_fixed = cv2.getPerspectiveTransform(pv, tv)
print(f"Fixed fallback H maps full frame to 0-{court_length}m x 0-{court_width}m")
print(f"  (NOT the full 105m pitch -- covers only {court_length}m of length)")
print()

# ── Per-frame analysis ────────────────────────────────────────────────────────
for fn in TARGET_FRAMES:
    SEP = "=" * 68
    print(SEP)
    print(f"FRAME {fn}")
    print(SEP)

    # Check H availability
    raw_h = hom.get(fn)
    if raw_h is None:
        print(f"  [H] hom[{fn}] = None  (replaced with fallback in main.py)")
        H = None
    else:
        H = np.array(raw_h[0], dtype=np.float64)
        print(f"  [H] Valid calibrated H found in stub")
        print(f"      row0: {H[0].round(5).tolist()}")
        print(f"      row1: {H[1].round(5).tolist()}")
        print(f"      row2: {H[2].round(6).tolist()}")

    # Camera movement
    cm = cam_mv[fn] if fn < len(cam_mv) else [0, 0]
    cmx, cmy = float(cm[0]), float(cm[1])
    print(f"  [cam] camera_movement = ({cmx:.2f}, {cmy:.2f}) px")

    if fn >= len(tracks['players']):
        print(f"  frame {fn} out of tracks range -- skipping")
        continue

    player_frame = tracks['players'][fn]
    in_bounds    = 0
    out_bounds   = 0
    fail_inside  = 0
    rows = []

    for pid, info in player_frame.items():
        pos     = get_foot_position(info['bbox'])
        pos_adj = (pos[0] - cmx, pos[1] - cmy)

        inside = cv2.pointPolygonTest(
            PIXEL_VERTICES, (int(pos_adj[0]), int(pos_adj[1])), False) >= 0

        if not inside:
            fail_inside += 1
            out_bounds  += 1
            rows.append((pid, pos, pos_adj, False, None, "is_inside_FAIL"))
            continue

        h_use = H if H is not None else H_fixed
        arr   = np.array([[[pos_adj[0], pos_adj[1]]]], dtype=np.float32)
        w     = cv2.perspectiveTransform(arr, h_use.astype(np.float32))
        wx, wy = float(w[0,0,0]), float(w[0,0,1])

        ok = (0.0 <= wx <= PITCH_X_MAX and 0.0 <= wy <= PITCH_Y_MAX)
        if ok:
            in_bounds += 1
            rows.append((pid, pos, pos_adj, True, (wx, wy), "OK"))
        else:
            out_bounds += 1
            rows.append((pid, pos, pos_adj, True, (wx, wy),
                         f"OUT({wx:.0f},{wy:.0f})"))

    print(f"\n  Players in frame   : {len(player_frame)}")
    print(f"  is_inside FAILED   : {fail_inside}   (pos_adjusted outside frame bounds)")
    print(f"  in  bounds [0-105,0-68]: {in_bounds}")
    print(f"  OUT of bounds          : {out_bounds - fail_inside}")

    # Table
    print(f"\n  {'PID':>4}  {'orig':>12}  {'adjusted':>12}  "
          f"{'in':>5}  {'world_xy':>14}  result")
    print(f"  {'-'*65}")
    for pid, pos, adj, ins, wpt, rsn in sorted(rows):
        w_str = f"({wpt[0]:.1f},{wpt[1]:.1f})" if wpt else "None"
        print(f"  {pid:>4}  ({pos[0]:.0f},{pos[1]:.0f})"
              f"  ({adj[0]:.0f},{adj[1]:.0f})"
              f"  {str(ins):>5}  {w_str:>14}  {rsn}")

    # Step-by-step for one player where H is valid but result is out of bounds
    if H is not None:
        bad_rows = [(pid,pos,adj,wpt) for pid,pos,adj,ins,wpt,rsn in rows
                    if ins and wpt and not (0<=wpt[0]<=PITCH_X_MAX
                                            and 0<=wpt[1]<=PITCH_Y_MAX)]
        if bad_rows:
            pid, pos, adj, wpt = bad_rows[0]
            print(f"\n  MANUAL STEP-BY-STEP for PID {pid} (H valid, result bad):")
            print(f"    bbox foot original pixel : ({pos[0]:.1f}, {pos[1]:.1f})")
            print(f"    camera_movement          : ({cmx:.2f}, {cmy:.2f})")
            print(f"    position_adjusted        : ({adj[0]:.1f}, {adj[1]:.1f})")
            print(f"    is_inside [frame bounds] : True")

            num_x = H[0,0]*adj[0] + H[0,1]*adj[1] + H[0,2]
            num_y = H[1,0]*adj[0] + H[1,1]*adj[1] + H[1,2]
            den   = H[2,0]*adj[0] + H[2,1]*adj[1] + H[2,2]
            print(f"    H @ position_adjusted:")
            print(f"      numerator_x  = {H[0,0]:.5f}*{adj[0]:.1f}"
                  f" + {H[0,1]:.5f}*{adj[1]:.1f} + {H[0,2]:.3f} = {num_x:.3f}")
            print(f"      numerator_y  = {H[1,0]:.5f}*{adj[0]:.1f}"
                  f" + {H[1,1]:.5f}*{adj[1]:.1f} + {H[1,2]:.3f} = {num_y:.3f}")
            print(f"      denominator  = {H[2,0]:.6f}*{adj[0]:.1f}"
                  f" + {H[2,1]:.6f}*{adj[1]:.1f} + {H[2,2]:.6f} = {den:.6f}")
            print(f"      wx = {num_x:.3f} / {den:.6f} = {num_x/den:.2f}"
                  f"  wy = {num_y:.3f} / {den:.6f} = {num_y/den:.2f}  --> OUT OF BOUNDS")

            # Same H on the ORIGINAL pixel (before camera correction)
            num_xo = H[0,0]*pos[0] + H[0,1]*pos[1] + H[0,2]
            num_yo = H[1,0]*pos[0] + H[1,1]*pos[1] + H[1,2]
            den_o  = H[2,0]*pos[0] + H[2,1]*pos[1] + H[2,2]
            wxo    = num_xo / den_o
            wyo    = num_yo / den_o
            valid_orig = (0 <= wxo <= 105 and 0 <= wyo <= 68)
            print(f"\n    CONTRAST -- same H on ORIGINAL pixel ({pos[0]:.1f},{pos[1]:.1f}):")
            print(f"      wx = {wxo:.2f}  wy = {wyo:.2f}"
                  f"  --> {'VALID' if valid_orig else 'ALSO BAD'}")

    print()

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 68)
print("QUESTION 4: Where / how is position_transformed computed?")
print("=" * 68)
print("""
  File   : view_transformer/view_transformer.py
  Method : add_transformed_position_to_tracks(tracks, homography_per_frame)

  For each frame:
    H_cal = homography_per_frame[frame_num][0]   # per-frame calibrated H
            else self.persepctive_trasnformer     # fixed fallback

    position = track_info['position_ADJUSTED']   # <-- camera-corrected pixels

    is_inside = cv2.pointPolygonTest(
        [[0,1080],[0,0],[1920,0],[1920,1080]], position, False) >= 0

    result = cv2.perspectiveTransform(position, H_cal)

  DISCONNECT ANALYSIS:
  --------------------
  (A) H was calibrated against ORIGINAL frame pixels (not adjusted).
      position_adjusted = original_pixel - camera_movement.
      Applying H to adjusted pixels gives wrong world coords when cam moves.

  (B) is_inside uses [[0,0]...[1920,1080]] full-frame bounds.
      When camera_movement != 0, adjusted coords can go negative or >1920/1080
      -> is_inside = False -> position_transformed = None.

  (C) Fixed fallback H only covers 0-23.32m x 0-68m (court_length=23.32),
      not the full 105m pitch.  Any player outside that sub-region gets
      an out-of-bounds world coord even with H 'working'.

  WHY TEST_CALIBRATOR REPORTED 95.6%:
  ------------------------------------
  test_calibrator.py counts frames where H != None.
  It only spot-checks frame CENTER pixel -> world coords.
  It does NOT test whether player positions (especially adjusted ones)
  project into valid world bounds.  95.6% = 'H was computed', not
  'player positions are correctly transformed'.
""")
