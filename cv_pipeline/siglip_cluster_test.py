"""
siglip_cluster_test.py — Step 1 of the SigLIP+UMAP+KMeans team-clustering
evaluation (see project memory / conversation for Step 0 context).

Standalone script. Does NOT touch main.py, fast_common.py, or the
team_assigner package. Only reads the existing production stub +
source video for the short demo clip.

Pipeline tested here:
    1. Sample 1-3 reasonably-sized/centered crops per (non-goalkeeper)
       player track ID from stubs/track_stubs_121364_ball_fallback.pkl
       + Match_videos/121364_0.mp4.
    2. Embed each crop with SigLIP (google/siglip-base-patch16-224).
    3. Reduce embeddings with UMAP.
    4. Cluster with KMeans(k=2).
    5. Report cluster sizes + ambiguous crops (close to both centroids)
       + timing for each stage.

Goalkeepers are excluded from the fit population, same as the old
color-heuristic's assign_team_color() did — not new logic, just
carrying over the established reason (GK kits use a 3rd color and
contaminate a k=2 fit meant to separate the two outfield teams).
Goalkeeper *resolution* itself is explicitly out of scope this session.
"""

import pickle
import time

import cv2
import numpy as np

STUB_PATH = 'stubs/track_stubs_121364_ball_fallback.pkl'
VIDEO_PATH = 'Match_videos/121364_0.mp4'
SIGLIP_MODEL = 'google/siglip-base-patch16-224'

MIN_CROP_HEIGHT = 45     # ~median-ish height in this clip (p10=44, median=58)
EDGE_MARGIN_PX = 3       # reject boxes clipped against the frame edge
MAX_SAMPLES_PER_PID = 3
FRAME_W, FRAME_H = 1920, 1080


def collect_candidate_crops():
    with open(STUB_PATH, 'rb') as f:
        d = pickle.load(f)
    players = d['players']

    per_pid = {}
    for fn, fd in enumerate(players):
        for pid, info in fd.items():
            if info.get('is_goalkeeper'):
                continue
            bbox = info.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            w = x2 - x1
            if h < MIN_CROP_HEIGHT:
                continue
            if x1 < EDGE_MARGIN_PX or y1 < EDGE_MARGIN_PX or \
               x2 > FRAME_W - EDGE_MARGIN_PX or y2 > FRAME_H - EDGE_MARGIN_PX:
                continue
            per_pid.setdefault(pid, []).append((fn, x1, y1, x2, y2, h))

    # Pick up to MAX_SAMPLES_PER_PID per pid, spread across the track's
    # lifetime (first / middle / last qualifying frame) for diversity.
    picks = []  # (pid, frame_num, bbox)
    for pid, cands in per_pid.items():
        cands.sort(key=lambda c: c[0])
        n = len(cands)
        if n == 0:
            continue
        if n <= MAX_SAMPLES_PER_PID:
            chosen = cands
        else:
            idxs = sorted(set(
                round(i * (n - 1) / (MAX_SAMPLES_PER_PID - 1))
                for i in range(MAX_SAMPLES_PER_PID)
            ))
            chosen = [cands[i] for i in idxs]
        for fn, x1, y1, x2, y2, h in chosen:
            picks.append((pid, fn, (int(x1), int(y1), int(x2), int(y2))))

    return picks


def read_needed_frames(picks):
    needed = sorted(set(fn for _, fn, _ in picks))
    frames = {}
    cap = cv2.VideoCapture(VIDEO_PATH)
    for fn in needed:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret:
            frames[fn] = frame
    cap.release()
    return frames


def main():
    t_start = time.time()

    print("Collecting candidate crops from stub...")
    picks = collect_candidate_crops()
    n_pids = len(set(pid for pid, _, _ in picks))
    print(f"  -> {len(picks)} crops across {n_pids} non-goalkeeper track IDs "
          f"(target <= {MAX_SAMPLES_PER_PID}/pid)")

    t0 = time.time()
    frames = read_needed_frames(picks)
    t_frames = time.time() - t0
    print(f"Read {len(frames)} unique source frames from video in {t_frames:.1f}s")

    crops = []
    crop_meta = []
    for pid, fn, (x1, y1, x2, y2) in picks:
        frame = frames.get(fn)
        if frame is None:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(crop_rgb)
        crop_meta.append((pid, fn))

    print(f"Prepared {len(crops)} valid crops for embedding")

    print("Loading SigLIP model...")
    t0 = time.time()
    import torch
    from transformers import SiglipModel, SiglipImageProcessor
    model = SiglipModel.from_pretrained(SIGLIP_MODEL)
    processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL)
    model.eval()
    t_load = time.time() - t0
    print(f"  -> model+processor ready in {t_load:.1f}s")

    print("Embedding crops with SigLIP (CPU)...")
    t0 = time.time()
    embeddings = []
    BATCH = 16
    with torch.no_grad():
        for i in range(0, len(crops), BATCH):
            batch = crops[i:i + BATCH]
            inputs = processor(images=batch, return_tensors="pt")
            out = model.get_image_features(**inputs)
            # transformers 5.x returns BaseModelOutputWithPooling here, not
            # a bare tensor (API changed from earlier versions this code
            # was originally written against).
            feats = out.pooler_output if hasattr(out, 'pooler_output') else out
            embeddings.append(feats.numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    t_embed = time.time() - t0
    print(f"  -> embedded {len(crops)} crops in {t_embed:.1f}s "
          f"({t_embed / max(len(crops),1)*1000:.0f} ms/crop), dim={embeddings.shape[1]}")

    print("Reducing with UMAP...")
    t0 = time.time()
    import umap
    reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced = reducer.fit_transform(embeddings)
    t_umap = time.time() - t0
    print(f"  -> UMAP done in {t_umap:.1f}s, shape={reduced.shape}")

    print("Clustering with KMeans(k=2)...")
    t0 = time.time()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(reduced)
    dists = km.transform(reduced)  # distance to each of the 2 centroids
    t_kmeans = time.time() - t0
    print(f"  -> KMeans done in {t_kmeans:.2f}s")

    # ── Report ──────────────────────────────────────────────────────
    t_total = time.time() - t_start

    cluster_sizes = {c: int((labels == c).sum()) for c in (0, 1)}
    print("\n=== CLUSTER SIZES (per crop) ===")
    print(cluster_sizes)

    # Per-pid majority vote + agreement fraction, to see how clean
    # separation is at the PLAYER level (not just crop level).
    from collections import defaultdict
    pid_labels = defaultdict(list)
    for (pid, fn), lab in zip(crop_meta, labels):
        pid_labels[pid].append(lab)

    print(f"\n=== PER-PLAYER CLUSTER AGREEMENT ({len(pid_labels)} players) ===")
    disagreeing_pids = []
    for pid, labs in sorted(pid_labels.items()):
        labs = np.array(labs)
        maj = np.bincount(labs).argmax()
        agree = (labs == maj).mean()
        if agree < 1.0:
            disagreeing_pids.append((pid, labs.tolist(), agree))
    print(f"Players with 100% internal agreement across their samples: "
          f"{len(pid_labels) - len(disagreeing_pids)}/{len(pid_labels)}")
    if disagreeing_pids:
        print("Players whose own samples split across clusters:")
        for pid, labs, agree in disagreeing_pids:
            print(f"    pid {pid}: labels={labs} (agreement={agree:.0%})")

    # Ambiguous crops: distance to nearest centroid is close to distance
    # to the other centroid (ratio near 1.0 = right on the boundary).
    d_sorted = np.sort(dists, axis=1)
    ambiguity_ratio = d_sorted[:, 0] / np.maximum(d_sorted[:, 1], 1e-9)
    AMBIG_THRESHOLD = 0.85  # nearest-centroid dist >= 85% of far-centroid dist
    ambig_idxs = np.where(ambiguity_ratio >= AMBIG_THRESHOLD)[0]
    print(f"\n=== AMBIGUOUS CROPS (nearest/farthest centroid dist ratio >= {AMBIG_THRESHOLD}) ===")
    print(f"{len(ambig_idxs)}/{len(crops)} crops flagged ambiguous")
    for idx in ambig_idxs:
        pid, fn = crop_meta[idx]
        print(f"    pid {pid}, frame {fn}: ratio={ambiguity_ratio[idx]:.2f}, "
              f"cluster={labels[idx]}, dists={dists[idx].round(2)}")

    print("\n=== TIMING (CPU) ===")
    print(f"  frame read:        {t_frames:.1f}s")
    print(f"  model load:        {t_load:.1f}s")
    print(f"  SigLIP embedding:  {t_embed:.1f}s  ({len(crops)} crops)")
    print(f"  UMAP fit:          {t_umap:.1f}s")
    print(f"  KMeans fit:        {t_kmeans:.2f}s")
    print(f"  TOTAL:             {t_total:.1f}s")

    np.savez('siglip_cluster_test_output.npz',
              embeddings=embeddings, reduced=reduced, labels=labels,
              dists=dists, pids=np.array([m[0] for m in crop_meta]),
              frames=np.array([m[1] for m in crop_meta]))
    print("\nSaved raw arrays to siglip_cluster_test_output.npz")


if __name__ == '__main__':
    main()
