#!/usr/bin/env python3
"""
label_keypoints.py — Interactive PITCH_KP_WORLD labeler.

LEFT panel : frame 0 from the match video with detected keypoints as
             numbered dots (yellow = confident, grey = low conf).
RIGHT panel: standard FIFA pitch diagram you click to assign world coords.

Workflow
--------
1. Click a dot on the LEFT panel to select a keypoint (turns red).
2. Click on the RIGHT pitch diagram to assign its world (x_m, y_m).
   The dot turns green and a label appears on the pitch.
3. Repeat for all keypoints you want to fix.
4. Press  S  to write the updated PITCH_KP_WORLD into pitch_calibrator.py.
5. Press  U  to undo the last assignment.
6. Press  Backspace  to clear the selected keypoint's assignment.
7. Press  Tab  to cycle to the next keypoint.
8. Press  Q / Esc  to quit without saving.

All 48 keypoints are pre-populated from the current PITCH_KP_WORLD so you
only need to click the ones that are wrong.
"""

import cv2
import numpy as np
import re

VIDEO_PATH = 'Match_videos/121364_0.mp4'
MODEL_PATH = 'pose/pitch_keypoints_v3/weights/best.pt'
CALIBRATOR = 'pitch_calibrator.py'
CONF_THR   = 0.15

# ── Pitch geometry (metres) ───────────────────────────────────────────────
PM_W = 105.0   # x: 0 = left goal line, 105 = right goal line
PM_H =  68.0   # y: 0 = far touchline,   68 = near touchline

# ── Window layout ─────────────────────────────────────────────────────────
VID_W, VID_H  = 960, 540
PITCH_DRAW_W  = 750
PITCH_DRAW_H  = int(PITCH_DRAW_W * PM_H / PM_W)   # ≈ 486
MARGIN        = 50
STATUS_H      = 50                                  # space above pitch for status text
PITCH_PANEL_W = PITCH_DRAW_W + 2 * MARGIN
PITCH_PANEL_H = STATUS_H + PITCH_DRAW_H + MARGIN
WIN_H         = max(VID_H, PITCH_PANEL_H)
WIN_W         = VID_W + PITCH_PANEL_W

# Pitch drawing origin within the right panel
PITCH_OX = MARGIN
PITCH_OY = STATUS_H

# ── Colors (BGR) ──────────────────────────────────────────────────────────
C_VISIBLE  = (  0, 220, 255)   # yellow-ish — detected, unmodified
C_SELECTED = (  0,   0, 255)   # red — currently selected
C_ASSIGNED = (  0, 200,   0)   # green — user has clicked to assign
C_LOWCONF  = (120, 120, 120)   # grey — conf < threshold
C_PITCH_BG = ( 34, 139,  34)   # grass green
C_LINE     = (255, 255, 255)   # white pitch lines
C_DOT_DEF  = (200, 200,   0)   # cyan-ish — default (pre-populated) dot on pitch
C_DOT_USER = (  0, 200,   0)   # green — user-assigned dot on pitch
C_DOT_SEL  = (  0,   0, 255)   # red — selected dot on pitch


# ── Coordinate helpers ────────────────────────────────────────────────────

def w2p(wx, wy):
    """World metres → pixel in right panel pitch drawing."""
    px = PITCH_OX + int(wx / PM_W * PITCH_DRAW_W)
    py = PITCH_OY + int(wy / PM_H * PITCH_DRAW_H)
    return px, py


def p2w(px, py):
    """Pixel in right panel → world metres (clamped)."""
    wx = (px - PITCH_OX) / PITCH_DRAW_W * PM_W
    wy = (py - PITCH_OY) / PITCH_DRAW_H * PM_H
    return max(0.0, min(PM_W, wx)), max(0.0, min(PM_H, wy))


# ── Pitch drawing ─────────────────────────────────────────────────────────

def make_pitch_base():
    """Return the static pitch diagram (no keypoint dots)."""
    panel = np.full((PITCH_PANEL_H, PITCH_PANEL_W, 3), C_PITCH_BG, dtype=np.uint8)
    lw = 2

    # Outer boundary
    cv2.rectangle(panel, w2p(0, 0), w2p(PM_W, PM_H), C_LINE, lw)

    # Halfway line
    cv2.line(panel, w2p(52.5, 0), w2p(52.5, PM_H), C_LINE, lw)

    # Centre circle (r = 9.15 m)
    r_px = int(9.15 / PM_W * PITCH_DRAW_W)
    cx, cy = w2p(52.5, 34.0)
    cv2.circle(panel, (cx, cy), r_px, C_LINE, lw)
    cv2.circle(panel, w2p(52.5, 34.0), 4, C_LINE, -1)

    # Left penalty area & goal area
    cv2.rectangle(panel, w2p(0, 13.84), w2p(16.5, 54.16), C_LINE, lw)
    cv2.rectangle(panel, w2p(0, 24.84), w2p(5.5, 43.16), C_LINE, lw)
    cv2.circle(panel, w2p(11.0, 34.0), 4, C_LINE, -1)

    # Right penalty area & goal area
    cv2.rectangle(panel, w2p(88.5, 13.84), w2p(PM_W, 54.16), C_LINE, lw)
    cv2.rectangle(panel, w2p(99.5, 24.84), w2p(PM_W, 43.16), C_LINE, lw)
    cv2.circle(panel, w2p(94.0, 34.0), 4, C_LINE, -1)

    # X-axis tick labels (top edge)
    for xm in [0, 16.5, 52.5, 88.5, 105]:
        ppx, _ = w2p(xm, 0)
        cv2.putText(panel, f'{xm:.0f}', (ppx - 10, PITCH_OY - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_LINE, 1)

    # Y-axis tick labels (left edge)
    for ym in [0, 13.84, 34, 54.16, 68]:
        _, ppy = w2p(0, ym)
        cv2.putText(panel, f'{ym:.0f}', (2, ppy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_LINE, 1)

    return panel


# ── Data loading ──────────────────────────────────────────────────────────

def load_data():
    """Read frame 0, run YOLO, return (scaled_frame, kp_list, original_world)."""
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f'Cannot read {VIDEO_PATH}')

    from pitch_calibrator import SoccerNetCalibrator, PITCH_KP_WORLD
    print('Loading YOLO model ...')
    sc = SoccerNetCalibrator(MODEL_PATH, conf_threshold=0.0)
    results = sc._model.predict(source=frame, conf=0.05, imgsz=1280,
                                save=False, verbose=False)
    xy   = results[0].keypoints.xy[0].cpu().numpy()    # (48, 2) pixel
    conf = results[0].keypoints.conf[0].cpu().numpy()  # (48,)

    sx = VID_W / frame.shape[1]
    sy = VID_H / frame.shape[0]
    scaled_frame = cv2.resize(frame, (VID_W, VID_H))

    kp_list = []
    for i, (pt, c) in enumerate(zip(xy, conf)):
        if i >= len(PITCH_KP_WORLD):
            continue
        # Skip truly undetected (model outputs 0,0 with 0 conf)
        if pt[0] < 1 and pt[1] < 1 and c < 0.01:
            continue
        kp_list.append({
            'idx':  i,
            'px':   int(pt[0] * sx),
            'py':   int(pt[1] * sy),
            'conf': float(c),
        })

    return scaled_frame, kp_list, PITCH_KP_WORLD


# ── Rendering ─────────────────────────────────────────────────────────────

def render(base_frame, base_pitch, kp_list, assignments, user_assigned,
           selected_idx):
    left  = base_frame.copy()
    right = base_pitch.copy()

    # ── Draw keypoints on video panel ──────────────────────────────────
    for kp in kp_list:
        i      = kp['idx']
        px, py = kp['px'], kp['py']

        if i == selected_idx:
            color, r = C_SELECTED, 10
        elif i in user_assigned:
            color, r = C_ASSIGNED, 7
        elif i in assignments:
            color, r = C_VISIBLE, 7
        elif kp['conf'] >= CONF_THR:
            color, r = C_VISIBLE, 7
        else:
            color, r = C_LOWCONF, 4

        cv2.circle(left, (px, py), r, color, -1)
        cv2.putText(left, str(i), (px + r + 2, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # ── Draw assigned dots on pitch panel ──────────────────────────────
    for i, (wx, wy) in assignments.items():
        ppx, ppy = w2p(wx, wy)
        if i == selected_idx:
            dc = C_DOT_SEL
        elif i in user_assigned:
            dc = C_DOT_USER
        else:
            dc = C_DOT_DEF
        cv2.circle(right, (ppx, ppy), 5, dc, -1)
        cv2.putText(right, str(i), (ppx + 6, ppy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, dc, 1, cv2.LINE_AA)

    # ── Status line in right panel ─────────────────────────────────────
    if selected_idx is not None:
        kp = next((k for k in kp_list if k['idx'] == selected_idx), None)
        conf_str = f'conf={kp["conf"]:.2f}' if kp else 'not detected'
        if selected_idx in assignments:
            wx, wy = assignments[selected_idx]
            tag = 'USER' if selected_idx in user_assigned else 'orig'
            status = (f'kp{selected_idx}  {conf_str}  [{tag}] -> '
                      f'({wx:.1f}, {wy:.1f})  click pitch to reassign')
        else:
            status = f'kp{selected_idx}  {conf_str}  UNASSIGNED — click pitch'
    else:
        status = 'Click a dot on the LEFT to select, then click pitch to assign'

    cv2.putText(right, status, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

    n_user = len(user_assigned)
    n_tot  = len(kp_list)
    cv2.putText(left, f'User-assigned: {n_user}/{n_tot}  S=save U=undo Tab=next Q=quit',
                (8, VID_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── Compose full window ────────────────────────────────────────────
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:VID_H,          :VID_W]           = left
    canvas[:PITCH_PANEL_H,  VID_W:VID_W + PITCH_PANEL_W] = right
    cv2.line(canvas, (VID_W, 0), (VID_W, WIN_H), (180, 180, 180), 2)
    return canvas


# ── Save ──────────────────────────────────────────────────────────────────

def save_to_calibrator(assignments):
    with open(CALIBRATOR, 'r') as f:
        src = f.read()

    lines = ['PITCH_KP_WORLD = {\n']
    for i in sorted(assignments.keys()):
        wx, wy = assignments[i]
        lines.append(f'    {i:2d}: ({wx:6.2f}, {wy:5.2f}),\n')
    lines.append('}')
    new_block = ''.join(lines)

    # Locate and replace the existing dict block
    start = src.find('PITCH_KP_WORLD = {')
    if start == -1:
        raise ValueError('PITCH_KP_WORLD not found in pitch_calibrator.py')
    close = src.find('\n}', start)
    if close == -1:
        raise ValueError('Closing brace of PITCH_KP_WORLD not found')
    end = close + 2   # include '\n}'

    new_src = src[:start] + new_block + src[end:]
    with open(CALIBRATOR, 'w') as f:
        f.write(new_src)
    print(f'Saved {len(assignments)} entries to {CALIBRATOR}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print('Loading ...')
    base_frame, kp_list, orig_world = load_data()
    base_pitch = make_pitch_base()

    # Pre-populate from current PITCH_KP_WORLD (so only wrong ones need fixing)
    assignments  = {kp['idx']: orig_world[kp['idx']] for kp in kp_list
                    if kp['idx'] in orig_world}
    user_assigned = set()   # indices the user explicitly clicked
    undo_stack    = []      # list of (idx, old_val_or_None)
    selected_idx  = None

    kp_indices = [kp['idx'] for kp in kp_list]

    WIN = 'Keypoint Labeler  [LEFT=select  RIGHT=assign  S=save  U=undo  Tab=next  Q=quit]'
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H)

    def on_mouse(event, x, y, flags, param):
        nonlocal selected_idx
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if x < VID_W:
            # Select nearest keypoint on video panel
            best_i, best_d = None, 30
            for kp in kp_list:
                d = ((kp['px'] - x) ** 2 + (kp['py'] - y) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_i = d, kp['idx']
            if best_i is not None:
                selected_idx = best_i
        else:
            # Assign world coords on pitch panel
            if selected_idx is None:
                return
            rx = x - VID_W
            ry = y
            if (PITCH_OX <= rx <= PITCH_OX + PITCH_DRAW_W and
                    PITCH_OY <= ry <= PITCH_OY + PITCH_DRAW_H):
                wx, wy = p2w(rx, ry)
                wx = round(wx, 2)
                wy = round(wy, 2)
                old = assignments.get(selected_idx)
                undo_stack.append((selected_idx, old,
                                   selected_idx in user_assigned))
                assignments[selected_idx] = (wx, wy)
                user_assigned.add(selected_idx)
                print(f'  kp{selected_idx:2d} -> ({wx:.2f}, {wy:.2f})')

    cv2.setMouseCallback(WIN, on_mouse)
    print(f'Window ready. {len(kp_list)} keypoints detected.')

    while True:
        canvas = render(base_frame, base_pitch, kp_list, assignments,
                        user_assigned, selected_idx)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print('Quit without saving.')
            break

        elif key in (ord('s'), ord('S')):
            save_to_calibrator(assignments)
            print('Saved. Press Q to exit.')

        elif key in (ord('u'), ord('U')):
            if undo_stack:
                idx, old_val, was_user = undo_stack.pop()
                if old_val is None:
                    assignments.pop(idx, None)
                else:
                    assignments[idx] = old_val
                if not was_user:
                    user_assigned.discard(idx)
                print(f'  Undo kp{idx}')

        elif key == 8:   # Backspace — clear selected assignment
            if selected_idx is not None and selected_idx in assignments:
                old = assignments.pop(selected_idx)
                in_user = selected_idx in user_assigned
                user_assigned.discard(selected_idx)
                undo_stack.append((selected_idx, old, in_user))
                print(f'  Cleared kp{selected_idx}')

        elif key == 9:   # Tab — next keypoint
            if not kp_indices:
                pass
            elif selected_idx is None:
                selected_idx = kp_indices[0]
            else:
                pos = kp_indices.index(selected_idx) if selected_idx in kp_indices else -1
                selected_idx = kp_indices[(pos + 1) % len(kp_indices)]

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
