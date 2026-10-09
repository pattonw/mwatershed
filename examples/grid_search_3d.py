"""Grid search on center 2/3rds of volume, focus on long-range splitting."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

OFFSETS_3D = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 12, 0], [0, 0, 12],
]

XY_SHORT = [1, 2]
XY_LONG = [4, 5]
Z_SHORT = [0]
Z_LONG = [3]

BLOCK_SHAPE = [64, 64, 64]
CONTEXT = [12, 12, 12]

# Load center 2/3rds
print("Loading center 2/3rds...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

S = slice(100, 500)
affs_raw = np.ascontiguousarray(
    affs_ds[affs_ds.roi][:, S, S, S], dtype=np.float64
)
gt = labels_ds[labels_ds.roi][S, S, S]

n_gt = len(np.unique(gt[gt > 0]))
print(f"Affinities: {affs_raw.shape}")
print(f"GT: {gt.shape}, {n_gt} objects, {np.count_nonzero(gt)}/{gt.size} nonzero")

S_SHORT = 2.0
S_LONG = 1.5

# Fixed params
B_XY_SHORT = -1.6
B_Z_SHORT = -1.7


def sparse_f1(gt, seg):
    mask = gt > 0
    gt_m = gt[mask]
    seg_m = seg[mask]
    if gt_m.size == 0:
        return 0.0, 0

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
    return f1, len(seg_ids)


def run(affs_raw, b_xy_l, b_z_l):
    affs = affs_raw.copy()
    affs[XY_SHORT] = affs[XY_SHORT] * S_SHORT + B_XY_SHORT
    affs[XY_LONG] = affs[XY_LONG] * S_LONG + b_xy_l
    affs[Z_SHORT] = affs[Z_SHORT] * S_SHORT + B_Z_SHORT
    affs[Z_LONG] = affs[Z_LONG] * S_LONG + b_z_l
    return mwatershed.agglom_blockwise(
        affs, OFFSETS_3D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )


# Grid: push b_xy_long more negative for more splitting
biases_xy_long = [-2.6, -2.4, -2.2, -2.0, -1.8, -1.6]
biases_z_long = [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8]

total = len(biases_xy_long) * len(biases_z_long)
print(f"\nGrid: {len(biases_xy_long)} x {len(biases_z_long)} = {total} combos")
print(f"Fixed: s_short={S_SHORT}, s_long={S_LONG}, "
      f"b_xy_short={B_XY_SHORT}, b_z_short={B_Z_SHORT}")
print(f"Block shape: {BLOCK_SHAPE}, context: {CONTEXT}, lossy=True\n")

results = []
t_start = time.perf_counter()

for i, b_xy_l in enumerate(biases_xy_long):
    for b_z_l in biases_z_long:
        t0 = time.perf_counter()
        seg = run(affs_raw, b_xy_l, b_z_l)
        f1, n = sparse_f1(gt, seg)
        dt = time.perf_counter() - t0
        results.append((f1, n, b_xy_l, b_z_l))
        print(f"  b_xy_l={b_xy_l:.1f}, b_z_l={b_z_l:.1f}: "
              f"F1={f1:.4f}, #seg={n}, {dt:.1f}s")

elapsed = time.perf_counter() - t_start

results.sort(key=lambda x: -x[0])

print(f"\n{'F1':>8}  {'#seg':>6}  {'b_xy_long':>10}  {'b_z_long':>10}")
print("-" * 40)
for f1, n, b_xy_l, b_z_l in results:
    print(f"{f1:8.4f}  {n:6d}  {b_xy_l:10.2f}  {b_z_l:10.2f}")

print(f"\nTotal time: {elapsed:.1f}s ({elapsed/total:.1f}s/combo)")
best = results[0]
print(f"\nBest: F1={best[0]:.4f}, b_xy_long={best[2]:.2f}, b_z_long={best[3]:.2f}, #seg={best[1]}")
