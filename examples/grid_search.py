"""Grid search over per-channel scale/bias for mutex watershed on real 2D data."""

import os
import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

NEIGHBORHOOD_3D = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 12, 0], [0, 0, 12],
]
CHANNELS_2D = [i for i, off in enumerate(NEIGHBORHOOD_3D) if off[0] == 0]
OFFSETS_2D = [[off[1], off[2]] for off in [NEIGHBORHOOD_3D[i] for i in CHANNELS_2D]]
SHORT_CHANNELS = [0, 1]  # [0,1,0] and [0,0,1]
LONG_CHANNELS = [2, 3]   # [0,12,0] and [0,0,12]

# Load data
print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

Z_SLICE = 300
affs_raw = np.ascontiguousarray(
    affs_ds[affs_ds.roi][CHANNELS_2D, Z_SLICE, :, :], dtype=np.float64
)
labels_2d = labels_ds[labels_ds.roi][Z_SLICE, :, :]

n_gt = len(np.unique(labels_2d[labels_2d > 0]))
print(f"Raw affinities: shape={affs_raw.shape}, range=[{affs_raw.min():.3f}, {affs_raw.max():.3f}]")
print(f"Labels: {n_gt} objects, {np.count_nonzero(labels_2d)}/{labels_2d.size} nonzero")

BLOCK_SHAPE = [100, 100]
CONTEXT = [12, 12]

# Stats from gt_pseudo_aff_stats.json (2D channels only)
# Short: same_mean~0.97, diff_mean~0.87, midpoint~0.92
# Long:  same_mean~0.81, diff_mean~0.35, midpoint~0.58
#
# Target priority: |long_split| > |short_merge| > |short_split| > |long_merge|
# bias_short ~ -0.92 * s_s  (zero-crossing at short midpoint)
# bias_long  ~ -0.81 * s_l  (zero-crossing near long same_mean, pushing long_merge small)


def sparse_f1(gt, seg):
    """F1 score masked to gt > 0, Hungarian matching."""
    mask = gt > 0
    gt_m = gt[mask]
    seg_m = seg[mask]

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
    precision = total_tp / n if n > 0 else 0
    recall = total_tp / n if n > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, len(np.unique(seg))


def run_one(affs_raw, s_s, b_s, s_l, b_l):
    affs = affs_raw.copy()
    affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * s_s + b_s
    affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * s_l + b_l
    return mwatershed.agglom_blockwise(
        affs, OFFSETS_2D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )


# Current best parameters
S_SHORT = 2.0
S_LONG = 1.5
biases_short = [-1.6]
biases_long = [-1.6]

total = len(biases_short) * len(biases_long)
print(f"\nRunning best params: s_short={S_SHORT}, b_short={biases_short[0]}, "
      f"s_long={S_LONG}, b_long={biases_long[0]}")
print(f"Block shape: {BLOCK_SHAPE}, context: {CONTEXT}, lossy=True\n")

PLOT_DIR = "examples/grid_search_plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def random_label_cmap(n=256, seed=42):
    """Random colormap with black for background (label 0)."""
    rng = np.random.default_rng(seed)
    colors = rng.random((n, 3))
    colors[0] = 0  # background black
    return ListedColormap(colors)


def save_plot(seg, labels, f1, n_labels, b_s, b_l, outdir):
    cmap = random_label_cmap(max(seg.max(), labels.max()) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(labels, cmap=cmap, interpolation="nearest")
    axes[0].set_title(f"Ground truth ({n_gt} objects)")
    axes[0].axis("off")

    axes[1].imshow(seg, cmap=cmap, interpolation="nearest")
    axes[1].set_title(f"Prediction ({n_labels} seg, F1={f1:.4f})")
    axes[1].axis("off")

    fig.suptitle(f"s_short={S_SHORT}, b_short={b_s:.2f}, s_long={S_LONG}, b_long={b_l:.2f}")
    fig.tight_layout()
    fname = f"bs{b_s:+.2f}_bl{b_l:+.2f}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


results = []
t_start = time.perf_counter()

for b_s in biases_short:
    for b_l in biases_long:
        seg = run_one(affs_raw, S_SHORT, b_s, S_LONG, b_l)
        f1, n_labels = sparse_f1(labels_2d, seg)
        results.append((f1, n_labels, S_SHORT, b_s, S_LONG, b_l))
        save_plot(seg, labels_2d, f1, n_labels, b_s, b_l, PLOT_DIR)

elapsed = time.perf_counter() - t_start

results.sort(key=lambda x: -x[0])

print(f"\n{'F1':>8}  {'#seg':>6}  {'s_short':>8}  {'b_short':>8}  {'s_long':>8}  {'b_long':>8}")
print("-" * 60)
for f1, n, s_s, b_s, s_l, b_l in results:
    print(f"{f1:8.4f}  {n:6d}  {s_s:8.2f}  {b_s:8.2f}  {s_l:8.2f}  {b_l:8.2f}")

print(f"\nPlots saved to {PLOT_DIR}/")
print(f"Total time: {elapsed:.1f}s ({elapsed/total:.3f}s per combo)")
best = results[0]
print(f"\nBest: F1={best[0]:.4f}, s_short={best[2]}, b_short={best[3]}, "
      f"s_long={best[4]}, b_long={best[5]}, #seg={best[1]}")
