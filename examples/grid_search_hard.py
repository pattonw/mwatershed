"""Grid search bias/shift params on hard z-slices (93-97)."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import mwatershed
from itertools import product

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

NEIGHBORHOOD_3D = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 12, 0], [0, 0, 12],
]
CHANNELS_2D = [i for i, off in enumerate(NEIGHBORHOOD_3D) if off[0] == 0]
OFFSETS_2D = [[off[1], off[2]] for off in [NEIGHBORHOOD_3D[i] for i in CHANNELS_2D]]
SHORT_CHANNELS = [0, 1]
LONG_CHANNELS = [2, 3]

BLOCK_SHAPE = [100, 100]
CONTEXT = [12, 12]

Z_RANGE = range(93, 98)  # z=93..97 inclusive

print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

affs_vol = np.ascontiguousarray(affs_ds[affs_ds.roi][CHANNELS_2D], dtype=np.float64)
labels_vol = labels_ds[labels_ds.roi]
_, Z, H, W = affs_vol.shape
print(f"Volume: affs={affs_vol.shape}, labels={labels_vol.shape}")

# Preload slices
slices = []
for z in Z_RANGE:
    gt = labels_vol[z]
    if np.any(gt > 0):
        slices.append((z, affs_vol[:, z].copy(), gt))
print(f"Loaded {len(slices)} slices: z={[s[0] for s in slices]}")


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
    return total_tp / n if n > 0 else 0, len(gt_ids), len(seg_ids)


def evaluate(s_short, b_short, s_long, b_long):
    """Return mean F1 across hard slices for given params."""
    f1s = []
    for z, raw_affs, gt in slices:
        affs = raw_affs.copy()
        affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * s_short + b_short
        affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * s_long + b_long
        seg = mwatershed.agglom_blockwise(
            affs, OFFSETS_2D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
        )
        f1, n_gt, n_seg = sparse_f1(gt, seg)
        if f1 is not None:
            f1s.append(f1)
    return np.mean(f1s) if f1s else 0.0


# Grid search ranges
S_SHORT_VALS = [1.5, 2.0, 2.5, 3.0, 3.5]
B_SHORT_VALS = [-2.5, -2.0, -1.6, -1.2, -0.8, -0.4]
S_LONG_VALS = [1.0, 1.5, 2.0, 2.5, 3.0]
B_LONG_VALS = [-2.5, -2.0, -1.6, -1.2, -0.8, -0.4]

combos = list(product(S_SHORT_VALS, B_SHORT_VALS, S_LONG_VALS, B_LONG_VALS))
print(f"\nGrid: {len(combos)} combinations "
      f"({len(S_SHORT_VALS)}x{len(B_SHORT_VALS)}x{len(S_LONG_VALS)}x{len(B_LONG_VALS)})")

# Baseline
t0 = time.perf_counter()
baseline = evaluate(2.0, -1.6, 1.5, -1.6)
t_one = time.perf_counter() - t0
print(f"Baseline (s_s=2.0, b_s=-1.6, s_l=1.5, b_l=-1.6): F1={baseline:.4f}  ({t_one:.2f}s/eval)")
print(f"Estimated total: {t_one * len(combos) / 60:.1f} min\n")

best_f1 = 0.0
best_params = None
results = []

t_start = time.perf_counter()
for i, (ss, bs, sl, bl) in enumerate(combos):
    f1 = evaluate(ss, bs, sl, bl)
    results.append((f1, ss, bs, sl, bl))
    if f1 > best_f1:
        best_f1 = f1
        best_params = (ss, bs, sl, bl)
        print(f"  [{i+1}/{len(combos)}] NEW BEST: F1={f1:.4f}  "
              f"s_short={ss}, b_short={bs}, s_long={sl}, b_long={bl}")
    if (i + 1) % 50 == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i+1}/{len(combos)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
              f"best so far: F1={best_f1:.4f}")

total = time.perf_counter() - t_start

# Top 10
results.sort(reverse=True)
print(f"\n{'='*70}")
print(f"Grid search complete: {len(combos)} combos in {total:.1f}s")
print(f"\nTop 10 parameter sets (z={Z_RANGE.start}-{Z_RANGE.stop-1}):")
print(f"{'Rank':>4}  {'F1':>8}  {'s_short':>7}  {'b_short':>7}  {'s_long':>6}  {'b_long':>6}")
print("-" * 50)
for rank, (f1, ss, bs, sl, bl) in enumerate(results[:10], 1):
    marker = " <-- baseline" if (ss, bs, sl, bl) == (2.0, -1.6, 1.5, -1.6) else ""
    print(f"{rank:4d}  {f1:8.4f}  {ss:7.1f}  {bs:7.1f}  {sl:6.1f}  {bl:6.1f}{marker}")

# Per-slice breakdown for best
print(f"\nBest params per-slice breakdown:")
ss, bs, sl, bl = best_params
for z, raw_affs, gt in slices:
    affs = raw_affs.copy()
    affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * ss + bs
    affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * sl + bl
    seg = mwatershed.agglom_blockwise(
        affs, OFFSETS_2D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )
    f1, n_gt, n_seg = sparse_f1(gt, seg)
    print(f"  z={z}: F1={f1:.4f}  #gt={n_gt}  #seg={n_seg}")
