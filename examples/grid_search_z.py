"""Grid search over Z-axis scale/bias for mutex watershed on 3D slabs."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

# Full 3D neighborhood
OFFSETS_3D = [
    [1, 0, 0],   # ch 0: Z short
    [0, 1, 0],   # ch 1: Y short
    [0, 0, 1],   # ch 2: X short
    [2, 0, 0],   # ch 3: Z long
    [0, 12, 0],  # ch 4: Y long
    [0, 0, 12],  # ch 5: X long
]

# Channel groups
XY_SHORT = [1, 2]   # Y short, X short
XY_LONG = [4, 5]    # Y long, X long
Z_SHORT = [0]       # Z short
Z_LONG = [3]        # Z long

# Fixed XY params (best from 2D search)
S_XY_SHORT, B_XY_SHORT = 2.0, -1.6
S_XY_LONG, B_XY_LONG = 1.5, -1.6

BLOCK_SHAPE = [5, 100, 100]
CONTEXT = [2, 12, 12]

# Load data
print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

affs_vol = np.ascontiguousarray(affs_ds[affs_ds.roi], dtype=np.float64)
labels_vol = labels_ds[labels_ds.roi]
print(f"Volume: affs={affs_vol.shape}, labels={labels_vol.shape}")

# Stats from gt_pseudo_aff_stats.json:
# Z short [1,0,0]: same_mean=0.974, diff_mean=0.915
# Z long  [2,0,0]: same_mean=0.952, diff_mean=0.833
print("\nZ channel stats (from gt_pseudo_aff_stats.json):")
print("  Z short [1,0,0]: same=0.974, diff=0.915, gap=0.059")
print("  Z long  [2,0,0]: same=0.952, diff=0.833, gap=0.119")

# Test slabs: sample several z positions across the volume
SLAB_THICKNESS = 5
SLAB_CENTERS = [100, 200, 300, 400, 500]
slabs = []
for zc in SLAB_CENTERS:
    z0 = zc - SLAB_THICKNESS // 2
    z1 = z0 + SLAB_THICKNESS
    gt_slab = labels_vol[z0:z1]
    if np.any(gt_slab > 0):
        slabs.append((z0, z1))
        n_gt = len(np.unique(gt_slab[gt_slab > 0]))
        print(f"  Slab z={z0}:{z1}, {n_gt} GT objects")

print(f"\nUsing {len(slabs)} slabs")


def sparse_f1(gt, seg):
    """F1 score masked to gt > 0, Hungarian matching."""
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


def run_slab(affs_slab, s_z_short, b_z_short, s_z_long, b_z_long):
    affs = affs_slab.copy()
    affs[XY_SHORT] = affs[XY_SHORT] * S_XY_SHORT + B_XY_SHORT
    affs[XY_LONG] = affs[XY_LONG] * S_XY_LONG + B_XY_LONG
    affs[Z_SHORT] = affs[Z_SHORT] * s_z_short + b_z_short
    affs[Z_LONG] = affs[Z_LONG] * s_z_long + b_z_long
    return mwatershed.agglom_blockwise(
        affs, OFFSETS_3D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )


# Grid search: Z bias terms, using same scales as XY
S_Z_SHORT = 2.0
S_Z_LONG = 1.5
offsets = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
biases_z_short = [-1.6 + d for d in offsets]
biases_z_long = [-1.6 + d for d in offsets]

total = len(biases_z_short) * len(biases_z_long)
print(f"\nGrid: {len(biases_z_short)} x {len(biases_z_long)} = {total} combos")
print(f"s_z_short={S_Z_SHORT}, s_z_long={S_Z_LONG}")
print(f"Block shape: {BLOCK_SHAPE}, context: {CONTEXT}, lossy=True\n")

results = []
t_start = time.perf_counter()

for b_zs in biases_z_short:
    for b_zl in biases_z_long:
        f1s = []
        n_segs = []
        for z0, z1 in slabs:
            affs_slab = affs_vol[:, z0:z1, :, :]
            gt_slab = labels_vol[z0:z1]
            seg = run_slab(affs_slab, S_Z_SHORT, b_zs, S_Z_LONG, b_zl)
            f1, n = sparse_f1(gt_slab, seg)
            f1s.append(f1)
            n_segs.append(n)
        mean_f1 = np.mean(f1s)
        mean_segs = int(np.mean(n_segs))
        results.append((mean_f1, mean_segs, b_zs, b_zl))

elapsed = time.perf_counter() - t_start

results.sort(key=lambda x: -x[0])

print(f"{'F1':>8}  {'#seg':>6}  {'b_z_short':>10}  {'b_z_long':>10}")
print("-" * 42)
for f1, n, b_zs, b_zl in results:
    print(f"{f1:8.4f}  {n:6d}  {b_zs:10.2f}  {b_zl:10.2f}")

print(f"\nTotal time: {elapsed:.1f}s ({elapsed/total:.2f}s per combo)")
best = results[0]
print(f"\nBest: F1={best[0]:.4f}, b_z_short={best[2]:.2f}, b_z_long={best[3]:.2f}, "
      f"avg #seg={best[1]}")
