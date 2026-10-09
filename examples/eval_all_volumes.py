"""Segment all 3 training volumes with best params, evaluate F1/VOI, save fragments."""

import time
import os
import shutil
import numpy as np
from funlib.persistence import open_ds, prepare_ds
from scipy.optimize import linear_sum_assignment
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr"
VOLUMES = ["gel_2_crop_0", "gel_2_crop_1", "gel_2_crop_3"]

OFFSETS_3D = [
    [1, 0, 0],   # ch 0: Z short
    [0, 1, 0],   # ch 1: Y short
    [0, 0, 1],   # ch 2: X short
    [2, 0, 0],   # ch 3: Z long
    [0, 12, 0],  # ch 4: Y long
    [0, 0, 12],  # ch 5: X long
]

XY_SHORT = [1, 2]
XY_LONG = [4, 5]
Z_SHORT = [0]
Z_LONG = [3]

# Best XY params from coordinate descent (optimized on 2D XY slices of gel_2_crop_0)
S_XY_SHORT, B_XY_SHORT = 0.8, -0.35
S_XY_LONG, B_XY_LONG = 2.4, -2.8
# Z params
S_Z, B_Z = 1.0, -2.0

BLOCK_SHAPE = [64, 64, 64]
CONTEXT = [12, 12, 12]


def sparse_f1(gt, seg):
    """F1 score on labelled (gt > 0) voxels using Hungarian matching."""
    mask = gt > 0
    gt_m, seg_m = gt[mask], seg[mask]
    if gt_m.size == 0:
        return 0.0, 0, 0
    gt_ids = np.unique(gt_m)
    seg_ids = np.unique(seg_m)
    gt_map = {g: i for i, g in enumerate(gt_ids)}
    seg_map = {s: i for i, s in enumerate(seg_ids)}
    contingency = np.zeros((len(gt_ids), len(seg_ids)), dtype=np.int64)
    for g, s in zip(gt_m, seg_m):
        contingency[gt_map[g], seg_map[s]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    total_tp = sum(contingency[r, c] for r, c in zip(row_ind, col_ind))
    return total_tp / gt_m.size, len(gt_ids), len(seg_ids)


def voi(gt, seg):
    """Variation of Information on labelled (gt > 0) voxels."""
    mask = gt > 0
    gt_m, seg_m = gt[mask], seg[mask]
    n = gt_m.size
    if n == 0:
        return 0.0
    contingency = {}
    for g, s in zip(gt_m, seg_m):
        contingency[(g, s)] = contingency.get((g, s), 0) + 1
    gt_counts = {}
    seg_counts = {}
    for (g, s), c in contingency.items():
        gt_counts[g] = gt_counts.get(g, 0) + c
        seg_counts[s] = seg_counts.get(s, 0) + c
    h_gt = -sum((c / n) * np.log2(c / n) for c in gt_counts.values())
    h_seg = -sum((c / n) * np.log2(c / n) for c in seg_counts.values())
    h_joint = -sum((c / n) * np.log2(c / n) for c in contingency.values())
    return (h_joint - h_seg) + (h_joint - h_gt)


print("=" * 70)
print("Evaluating 3 volumes with best params")
print(f"  XY: s_short={S_XY_SHORT}, b_short={B_XY_SHORT}, s_long={S_XY_LONG}, b_long={B_XY_LONG}")
print(f"  Z:  s={S_Z}, b={B_Z}")
print(f"  block_shape={BLOCK_SHAPE}, context={CONTEXT}")
print("=" * 70)

for vol_name in VOLUMES:
    vol_path = f"{ZARR_BASE}/{vol_name}"
    print(f"\n{'─' * 70}")
    print(f"Volume: {vol_name}")
    print(f"{'─' * 70}")

    # Load affinities
    t0 = time.perf_counter()
    affs_ds = open_ds(f"{vol_path}/pseudo_affs", mode="r")
    affs = np.ascontiguousarray(affs_ds[affs_ds.roi], dtype=np.float64)
    t_load = time.perf_counter() - t0
    print(f"  Loaded affs: {affs.shape} ({affs.nbytes / 1e9:.2f} GB) in {t_load:.1f}s")

    # Load labels
    labels_ds = open_ds(f"{vol_path}/labels", mode="r")
    labels = labels_ds[labels_ds.roi]
    n_gt = len(np.unique(labels[labels > 0]))
    print(f"  Labels: {labels.shape}, {n_gt} GT objects")

    # Apply scale/bias per channel group
    affs[XY_SHORT] = affs[XY_SHORT] * S_XY_SHORT + B_XY_SHORT
    affs[XY_LONG] = affs[XY_LONG] * S_XY_LONG + B_XY_LONG
    affs[Z_SHORT] = affs[Z_SHORT] * S_Z + B_Z
    affs[Z_LONG] = affs[Z_LONG] * S_Z + B_Z

    # Run blockwise mwatershed
    print(f"  Running mwatershed (lossy, block={BLOCK_SHAPE}, ctx={CONTEXT})...", flush=True)
    t0 = time.perf_counter()
    seg = mwatershed.agglom_blockwise(
        affs, OFFSETS_3D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )
    t_mws = time.perf_counter() - t0
    n_frags = len(np.unique(seg))
    print(f"  Done in {t_mws:.1f}s, {n_frags} fragments")

    # Slicewise evaluation
    print(f"  Evaluating slicewise...", flush=True)
    print(f"  {'z':>5}  {'F1':>8}  {'VOI':>8}  {'GT':>5}  {'Seg':>5}")
    print(f"  {'-'*40}")
    f1_all, voi_all = [], []
    for z in range(labels.shape[0]):
        gt_slice = labels[z]
        if not np.any(gt_slice > 0):
            continue
        seg_slice = seg[z]
        f1_z, n_gt_z, n_seg_z = sparse_f1(gt_slice, seg_slice)
        voi_z = voi(gt_slice, seg_slice)
        f1_all.append(f1_z)
        voi_all.append(voi_z)
        print(f"  {z:5d}  {f1_z:8.4f}  {voi_z:8.4f}  {n_gt_z:5d}  {n_seg_z:5d}")
    mean_f1 = np.mean(f1_all) if f1_all else 0.0
    mean_voi = np.mean(voi_all) if voi_all else 0.0
    print(f"  {'-'*40}")
    print(f"  {'MEAN':>5}  {mean_f1:8.4f}  {mean_voi:8.4f}")
    print(f"  Slices evaluated: {len(f1_all)}")

    # Save fragments
    frag_path = f"{vol_path}/fragments"
    if os.path.exists(frag_path):
        shutil.rmtree(frag_path)
    out_ds = prepare_ds(
        frag_path,
        shape=seg.shape,
        offset=labels_ds.offset,
        voxel_size=labels_ds.voxel_size,
        axis_names=labels_ds.axis_names,
        units=labels_ds.units,
        dtype=seg.dtype,
    )
    out_ds[out_ds.roi] = seg
    print(f"  Saved fragments to {frag_path}")

print(f"\n{'=' * 70}")
print("Done!")
print("=" * 70)
