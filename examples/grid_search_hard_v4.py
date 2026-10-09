"""Tight grid search recentered on ss=0.8, bs=-0.35, sl=2.4, bl=-2.8. ~256 combos."""

import time
import numpy as np
from funlib.persistence import open_ds
from scipy.optimize import linear_sum_assignment
import mwatershed
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib

matplotlib.use("Agg")

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
Z_RANGE = range(93, 98)

print("Loading data...", flush=True)
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")
affs_vol = np.ascontiguousarray(affs_ds[affs_ds.roi][CHANNELS_2D], dtype=np.float64)
labels_vol = labels_ds[labels_ds.roi]
print(f"Volume: affs={affs_vol.shape}, labels={labels_vol.shape}", flush=True)

slices = []
for z in Z_RANGE:
    gt = labels_vol[z]
    if np.any(gt > 0):
        slices.append((z, affs_vol[:, z].copy(), gt))
print(f"Loaded {len(slices)} slices: z={[s[0] for s in slices]}", flush=True)


def make_random_cmap(n=1024):
    rng = np.random.RandomState(42)
    colors = rng.rand(n, 3)
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)

RAND_CMAP = make_random_cmap()


def sparse_f1(gt, seg):
    mask = gt > 0
    gt_m, seg_m = gt[mask], seg[mask]
    if gt_m.size == 0:
        return None, 0, 0
    gt_ids, seg_ids = np.unique(gt_m), np.unique(seg_m)
    gt_map = {g: i for i, g in enumerate(gt_ids)}
    seg_map = {s: i for i, s in enumerate(seg_ids)}
    contingency = np.zeros((len(gt_ids), len(seg_ids)), dtype=np.int64)
    for g, s in zip(gt_m, seg_m):
        contingency[gt_map[g], seg_map[s]] += 1
    row_ind, col_ind = linear_sum_assignment(-contingency)
    total_tp = sum(contingency[r, c] for r, c in zip(row_ind, col_ind))
    return total_tp / gt_m.size, len(gt_ids), len(seg_ids)


def voi(gt, seg):
    mask = gt > 0
    gt_m, seg_m = gt[mask], seg[mask]
    n = gt_m.size
    if n == 0:
        return None
    gt_ids, seg_ids = np.unique(gt_m), np.unique(seg_m)
    gt_map = {g: i for i, g in enumerate(gt_ids)}
    seg_map = {s: i for i, s in enumerate(seg_ids)}
    contingency = np.zeros((len(gt_ids), len(seg_ids)), dtype=np.float64)
    for g, s in zip(gt_m, seg_m):
        contingency[gt_map[g], seg_map[s]] += 1
    p_gt = contingency.sum(axis=1) / n
    p_seg = contingency.sum(axis=0) / n
    p_joint = contingency / n
    h_gt = -np.sum(p_gt[p_gt > 0] * np.log2(p_gt[p_gt > 0]))
    h_seg = -np.sum(p_seg[p_seg > 0] * np.log2(p_seg[p_seg > 0]))
    nz = p_joint > 0
    h_joint = -np.sum(p_joint[nz] * np.log2(p_joint[nz]))
    return (h_joint - h_seg) + (h_joint - h_gt)


def run_seg(raw_affs, s_short, b_short, s_long, b_long):
    affs = raw_affs.copy()
    affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * s_short + b_short
    affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * s_long + b_long
    return mwatershed.agglom_blockwise(affs, OFFSETS_2D, BLOCK_SHAPE, lossy=True, context=CONTEXT)


def evaluate(s_short, b_short, s_long, b_long):
    f1s, vois = [], []
    for z, raw_affs, gt in slices:
        seg = run_seg(raw_affs, s_short, b_short, s_long, b_long)
        f1, n_gt, n_seg = sparse_f1(gt, seg)
        v = voi(gt, seg)
        if f1 is not None:
            f1s.append(f1)
        if v is not None:
            vois.append(v)
    return np.mean(f1s) if f1s else 0.0, np.mean(vois) if vois else 999.0


# 4 values per param => 256 combos
S_SHORT_VALS = [0.6, 0.8, 1.0, 1.2]
B_SHORT_VALS = [-0.5, -0.4, -0.3, -0.2]
S_LONG_VALS = [2.2, 2.4, 2.6, 2.8]
B_LONG_VALS = [-3.0, -2.8, -2.6, -2.4]

combos = list(product(S_SHORT_VALS, B_SHORT_VALS, S_LONG_VALS, B_LONG_VALS))
print(f"\nGrid: {len(combos)} combinations "
      f"({len(S_SHORT_VALS)}x{len(B_SHORT_VALS)}x{len(S_LONG_VALS)}x{len(B_LONG_VALS)})", flush=True)

t0 = time.perf_counter()
bl_f1, bl_voi = evaluate(0.8, -0.35, 2.4, -2.8)
t_one = time.perf_counter() - t0
print(f"Center (0.8, -0.35, 2.4, -2.8): F1={bl_f1:.4f}  VOI={bl_voi:.4f}  ({t_one:.2f}s/eval)")
print(f"Estimated total: {t_one * len(combos) / 60:.1f} min\n", flush=True)

best_f1 = 0.0; best_f1_params = None
best_voi = 999.0; best_voi_params = None
best_combo = -999.0; best_combo_params = None
results = []

t_start = time.perf_counter()
for i, (ss, bs, sl, bl) in enumerate(combos):
    f1, v = evaluate(ss, bs, sl, bl)
    results.append((f1, v, ss, bs, sl, bl))
    combo_score = f1 - v
    updated = []
    if f1 > best_f1:
        best_f1 = f1; best_f1_params = (ss, bs, sl, bl)
        updated.append(f"F1={f1:.4f}")
    if v < best_voi:
        best_voi = v; best_voi_params = (ss, bs, sl, bl)
        updated.append(f"VOI={v:.4f}")
    if combo_score > best_combo:
        best_combo = combo_score; best_combo_params = (ss, bs, sl, bl)
        updated.append(f"F1-VOI={combo_score:.4f}")
    if updated:
        print(f"  [{i+1}/{len(combos)}] NEW BEST {', '.join(updated)}  "
              f"ss={ss}, bs={bs}, sl={sl}, bl={bl}", flush=True)
    if (i + 1) % 50 == 0:
        elapsed = time.perf_counter() - t_start
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i+1}/{len(combos)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining  "
              f"best: F1={best_f1:.4f}  VOI={best_voi:.4f}  F1-VOI={best_combo:.4f}", flush=True)

total = time.perf_counter() - t_start
print(f"\n{'='*70}")
print(f"Grid search complete: {len(combos)} combos in {total:.1f}s\n")

for label, key_fn in [
    ("Top 10 by F1 (higher=better)", lambda r: r[0]),
    ("Top 10 by VOI (lower=better)", lambda r: -r[1]),
    ("Top 10 by F1-VOI (higher=better)", lambda r: r[0] - r[1]),
]:
    ranked = sorted(results, key=key_fn, reverse=True)
    print(f"\n{label}:")
    print(f"{'Rank':>4}  {'F1':>8}  {'VOI':>8}  {'F1-VOI':>8}  {'s_short':>7}  {'b_short':>7}  {'s_long':>6}  {'b_long':>6}")
    print("-" * 70)
    for rank, (f1, v, ss, bs, sl, bl) in enumerate(ranked[:10], 1):
        print(f"{rank:4d}  {f1:8.4f}  {v:8.4f}  {f1 - v:8.4f}  {ss:7.2f}  {bs:7.2f}  {sl:6.2f}  {bl:6.2f}")


def plot_result(params, metric_name, metric_val, filename):
    ss, bs, sl, bl = params
    n_slices = len(slices)
    fig, axes = plt.subplots(n_slices, 5, figsize=(25, 5 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]
    col_titles = [
        "Segments (full)", "GT labels", "Eval frags (seg masked to gt>0)",
        "Eval frags on >1 GT", "GT frags with >1 eval frag",
    ]
    for row, (z, raw_affs, gt) in enumerate(slices):
        seg = run_seg(raw_affs, ss, bs, sl, bl)
        gt_mask = gt > 0
        axes[row, 0].imshow(seg % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 0].set_ylabel(f"z={z}", fontsize=12)
        axes[row, 0].set_title(f"{len(np.unique(seg))} frags")
        axes[row, 1].imshow(gt % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 1].set_title(f"{len(np.unique(gt[gt > 0]))} GT labels")
        eval_frags = np.where(gt_mask, seg, 0)
        axes[row, 2].imshow(eval_frags % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 2].set_title(f"{len(np.unique(eval_frags[eval_frags > 0]))} eval frags")
        eval_ids = np.unique(eval_frags[eval_frags > 0])
        multi_gt = {eid for eid in eval_ids if len(np.unique(gt[eval_frags == eid])[np.unique(gt[eval_frags == eid]) > 0]) > 1}
        col3 = np.where(np.isin(eval_frags, list(multi_gt)), eval_frags, 0)
        axes[row, 3].imshow(col3 % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 3].set_title(f"{len(multi_gt)} frags on >1 GT")
        gt_ids = np.unique(gt[gt > 0])
        multi_seg = {gid for gid in gt_ids if len(np.unique(eval_frags[gt == gid])[np.unique(eval_frags[gt == gid]) > 0]) > 1}
        col4 = np.where(np.isin(gt, list(multi_seg)), gt, 0)
        axes[row, 4].imshow(col4 % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 4].set_title(f"{len(multi_seg)} GT with >1 frag")
    if n_slices > 0:
        for c, title in enumerate(col_titles):
            axes[0, c].set_title(title + "\n" + axes[0, c].get_title(), fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Best {metric_name}={metric_val:.4f}  |  ss={ss}, bs={bs}, sl={sl}, bl={bl}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename, dpi=150); plt.close(fig)
    print(f"Saved {filename}", flush=True)

f1_val, _ = evaluate(*best_f1_params)
_, voi_val = evaluate(*best_voi_params)
f1c, vc = evaluate(*best_combo_params)
plot_result(best_f1_params, "F1", f1_val, "examples/best_f1_v4.png")
plot_result(best_voi_params, "VOI", voi_val, "examples/best_voi_v4.png")
plot_result(best_combo_params, "F1-VOI", f1c - vc, "examples/best_f1_minus_voi_v4.png")
