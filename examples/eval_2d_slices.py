"""Run mwatershed per-z-slice (Z affs killed) and evaluate F1 per slice."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

NEIGHBORHOOD_3D = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 12, 0], [0, 0, 12],
]
CHANNELS_2D = [i for i, off in enumerate(NEIGHBORHOOD_3D) if off[0] == 0]
OFFSETS_2D = [[off[1], off[2]] for off in [NEIGHBORHOOD_3D[i] for i in CHANNELS_2D]]
SHORT_CHANNELS = [0, 1]
LONG_CHANNELS = [2, 3]

# Best 2D params
S_SHORT, B_SHORT = 2.0, -1.6
S_LONG, B_LONG = 1.5, -1.6

BLOCK_SHAPE = [100, 100]
CONTEXT = [12, 12]

print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

affs_vol = np.ascontiguousarray(affs_ds[affs_ds.roi][CHANNELS_2D], dtype=np.float64)
labels_vol = labels_ds[labels_ds.roi]
_, Z, H, W = affs_vol.shape
print(f"Volume: affs={affs_vol.shape}, labels={labels_vol.shape}")


def sparse_f1(gt, seg):
    mask = gt > 0
    gt_m = gt[mask]
    seg_m = seg[mask]
    if gt_m.size == 0:
        return None, 0, 0

    gt_ids = np.unique(gt_m)
    seg_ids = np.unique(seg_m)

    gt_map = {g: i for i, g in enumerate(gt_ids)}
    seg_map = {s: i for i, s in enumerate(seg_ids)}

    contingency = np.zeros((len(gt_ids), len(seg_ids)), dtype=np.int64)
    for g, s in zip(gt_m, seg_m):
        contingency[gt_map[g], seg_map[s]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    total_tp = sum(contingency[r, c] for r, c in zip(row_ind, col_ind))
    n = gt_m.size
    f1 = total_tp / n if n > 0 else 0
    return f1, len(gt_ids), len(seg_ids)


f1_scores = []
t0 = time.perf_counter()

for z in range(Z):
    gt = labels_vol[z]
    if not np.any(gt > 0):
        continue

    affs = affs_vol[:, z].copy()
    affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * S_SHORT + B_SHORT
    affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * S_LONG + B_LONG

    seg = mwatershed.agglom_blockwise(
        affs, OFFSETS_2D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )

    f1, n_gt, n_seg = sparse_f1(gt, seg)
    if f1 is not None:
        f1_scores.append((z, f1, n_gt, n_seg))

elapsed = time.perf_counter() - t0

print(f"\nProcessed {len(f1_scores)} slices in {elapsed:.1f}s\n")

print(f"{'z':>5}  {'F1':>8}  {'#gt':>5}  {'#seg':>6}")
print("-" * 30)
for z, f1, n_gt, n_seg in f1_scores:
    print(f"{z:5d}  {f1:8.4f}  {n_gt:5d}  {n_seg:6d}")

arr = np.array([x[1] for x in f1_scores])
print(f"\nF1 summary: mean={arr.mean():.4f}, std={arr.std():.4f}, "
      f"min={arr.min():.4f}, max={arr.max():.4f}, median={np.median(arr):.4f}")
