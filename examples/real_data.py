"""Load real 3D affinities, run mwatershed on a 2D slice, compute sparse F1."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

# Full 3D neighborhood (from watershed_blockwise.py)
NEIGHBORHOOD_3D = [
    [1, 0, 0],   # ch 0: Z short
    [0, 1, 0],   # ch 1: Y short
    [0, 0, 1],   # ch 2: X short
    [2, 0, 0],   # ch 3: Z long
    [0, 12, 0],  # ch 4: Y long
    [0, 0, 12],  # ch 5: X long
]

# For a 2D YX slice, keep only channels with zero Z offset
CHANNELS_2D = [i for i, off in enumerate(NEIGHBORHOOD_3D) if off[0] == 0]
OFFSETS_2D = [[off[1], off[2]] for off in [NEIGHBORHOOD_3D[i] for i in CHANNELS_2D]]

print(f"2D channels: {CHANNELS_2D}")
print(f"2D offsets: {OFFSETS_2D}")

# Load data
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

Z_SLICE = 300

# Extract 2D slice: affs shape (C, Z, Y, X) -> (C_2d, Y, X)
affs_2d = affs_ds[affs_ds.roi][CHANNELS_2D, Z_SLICE, :, :]
labels_2d = labels_ds[labels_ds.roi][Z_SLICE, :, :]

# mwatershed expects affinities centered at 0 (positive=merge, negative=split)
affs_2d = np.ascontiguousarray(affs_2d, dtype=np.float64) * 2.0 - 1.0

print(f"Affinities 2D: shape={affs_2d.shape}, range=[{affs_2d.min():.3f}, {affs_2d.max():.3f}]")
print(f"Labels 2D: shape={labels_2d.shape}, unique={len(np.unique(labels_2d))}, "
      f"nonzero={np.count_nonzero(labels_2d)}/{labels_2d.size}")

import mwatershed

t0 = time.perf_counter()
seg = mwatershed.agglom(affs_2d, OFFSETS_2D)
elapsed = time.perf_counter() - t0
print(f"\nmwatershed.agglom: {elapsed:.3f}s, {len(np.unique(seg))} labels")


def sparse_f1(gt, seg):
    """Compute F1 score masked to gt > 0, using Hungarian matching."""
    mask = gt > 0
    gt_m = gt[mask]
    seg_m = seg[mask]

    gt_ids = np.unique(gt_m)
    seg_ids = np.unique(seg_m)

    # Build contingency matrix
    gt_map = {g: i for i, g in enumerate(gt_ids)}
    seg_map = {s: i for i, s in enumerate(seg_ids)}

    contingency = np.zeros((len(gt_ids), len(seg_ids)), dtype=np.int64)
    for g, s in zip(gt_m, seg_m):
        contingency[gt_map[g], seg_map[s]] += 1

    # Hungarian matching on -overlap to maximize total overlap
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Per-matched-pair: precision & recall
    total_tp = 0
    total_gt = 0
    total_seg = 0
    gt_sizes = contingency.sum(axis=1)
    seg_sizes = contingency.sum(axis=0)

    for r, c in zip(row_ind, col_ind):
        tp = contingency[r, c]
        total_tp += tp
        total_gt += gt_sizes[r]
        total_seg += seg_sizes[c]

    # Also count unmatched gt/seg sizes
    total_gt = gt_m.size  # all labeled pixels
    total_seg = gt_m.size  # denominator is same (masked region)

    precision = total_tp / total_seg if total_seg > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


f1, prec, rec = sparse_f1(labels_2d, seg)
print(f"Sparse F1: {f1:.4f} (precision={prec:.4f}, recall={rec:.4f})")
