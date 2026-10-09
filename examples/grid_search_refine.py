"""Iterative coordinate descent to find local optimum, then confirm with +-0.1 grid."""

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
print(f"Loaded {len(slices)} slices: z={[s[0] for s in slices]}\n", flush=True)


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


# Cache to avoid re-evaluating the same point
_cache = {}

def evaluate(params):
    key = tuple(round(p, 4) for p in params)
    if key in _cache:
        return _cache[key]
    ss, bs, sl, bl = key
    f1s, vois = [], []
    for z, raw_affs, gt in slices:
        seg = run_seg(raw_affs, ss, bs, sl, bl)
        f1, n_gt, n_seg = sparse_f1(gt, seg)
        v = voi(gt, seg)
        if f1 is not None:
            f1s.append(f1)
        if v is not None:
            vois.append(v)
    result = (np.mean(f1s) if f1s else 0.0, np.mean(vois) if vois else 999.0)
    _cache[key] = result
    return result


def score(params):
    """Combined score: F1 - VOI (higher is better)."""
    f1, v = evaluate(params)
    return f1 - v


PARAM_NAMES = ["s_short", "b_short", "s_long", "b_long"]

# ============================================================
# Phase 1: Coarse coordinate descent (step=0.5) to find region
# ============================================================
print("=" * 60)
print("PHASE 1: Coarse search (step=0.5)")
print("=" * 60, flush=True)

center = [0.8, -0.35, 2.4, -2.8]
step = 0.5
n_evals = 0

f1, v = evaluate(center)
print(f"Start: {dict(zip(PARAM_NAMES, center))}  F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)

for iteration in range(20):
    improved = False
    for dim in range(4):
        best_score = score(center)
        best_val = center[dim]
        for direction in [-step, +step]:
            candidate = list(center)
            candidate[dim] = round(center[dim] + direction, 4)
            s = score(candidate)
            n_evals += 1
            if s > best_score + 1e-6:
                best_score = s
                best_val = candidate[dim]
                improved = True
        if best_val != center[dim]:
            old = center[dim]
            center[dim] = best_val
            f1, v = evaluate(center)
            print(f"  iter {iteration}: {PARAM_NAMES[dim]} {old:.2f} -> {best_val:.2f}  "
                  f"F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)
    if not improved:
        print(f"Converged at step={step} after {iteration+1} iterations ({n_evals} evals)", flush=True)
        break

# ============================================================
# Phase 2: Medium coordinate descent (step=0.2)
# ============================================================
print(f"\n{'=' * 60}")
print("PHASE 2: Medium search (step=0.2)")
print("=" * 60, flush=True)

step = 0.2
f1, v = evaluate(center)
print(f"Start: {dict(zip(PARAM_NAMES, center))}  F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)

for iteration in range(20):
    improved = False
    for dim in range(4):
        best_score = score(center)
        best_val = center[dim]
        for direction in [-step, +step]:
            candidate = list(center)
            candidate[dim] = round(center[dim] + direction, 4)
            s = score(candidate)
            n_evals += 1
            if s > best_score + 1e-6:
                best_score = s
                best_val = candidate[dim]
                improved = True
        if best_val != center[dim]:
            old = center[dim]
            center[dim] = best_val
            f1, v = evaluate(center)
            print(f"  iter {iteration}: {PARAM_NAMES[dim]} {old:.2f} -> {best_val:.2f}  "
                  f"F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)
    if not improved:
        print(f"Converged at step={step} after {iteration+1} iterations ({n_evals} evals)", flush=True)
        break

# ============================================================
# Phase 3: Fine coordinate descent (step=0.1)
# ============================================================
print(f"\n{'=' * 60}")
print("PHASE 3: Fine search (step=0.1)")
print("=" * 60, flush=True)

step = 0.1
f1, v = evaluate(center)
print(f"Start: {dict(zip(PARAM_NAMES, center))}  F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)

for iteration in range(30):
    improved = False
    for dim in range(4):
        best_score = score(center)
        best_val = center[dim]
        for direction in [-step, +step]:
            candidate = list(center)
            candidate[dim] = round(center[dim] + direction, 4)
            s = score(candidate)
            n_evals += 1
            if s > best_score + 1e-6:
                best_score = s
                best_val = candidate[dim]
                improved = True
        if best_val != center[dim]:
            old = center[dim]
            center[dim] = best_val
            f1, v = evaluate(center)
            print(f"  iter {iteration}: {PARAM_NAMES[dim]} {old:.2f} -> {best_val:.2f}  "
                  f"F1={f1:.4f}  VOI={v:.4f}  F1-VOI={f1-v:.4f}", flush=True)
    if not improved:
        print(f"Converged at step={step} after {iteration+1} iterations ({n_evals} evals)", flush=True)
        break

# ============================================================
# Phase 4: Confirmation grid — center ± 0.1 for all 4 params
# ============================================================
print(f"\n{'=' * 60}")
print("PHASE 4: Confirmation grid (center ± 0.1)")
print("=" * 60, flush=True)

center = [round(c, 2) for c in center]
print(f"Center: {dict(zip(PARAM_NAMES, center))}", flush=True)

vals = []
for c in center:
    vals.append([round(c - 0.1, 2), round(c, 2), round(c + 0.1, 2)])

combos = list(product(*vals))
print(f"Grid: {len(combos)} combinations (3x3x3x3 = 81)\n", flush=True)

results = []
for i, combo in enumerate(combos):
    f1, v = evaluate(list(combo))
    results.append((f1, v, *combo))
    n_evals += 1

print(f"Total evaluations across all phases: {n_evals}")
print(f"Cache hits saved: {n_evals - len(_cache)} re-computations\n")

# Verify center is best
center_f1, center_voi = evaluate(center)
center_combo = center_f1 - center_voi

all_better = True
print(f"{'Params':>45}  {'F1':>8}  {'VOI':>8}  {'F1-VOI':>8}  {'vs center':>10}")
print("-" * 90)

results_sorted = sorted(results, key=lambda r: r[0] - r[1], reverse=True)
for f1, v, ss, bs, sl, bl in results_sorted:
    combo_s = f1 - v
    is_center = (ss, bs, sl, bl) == tuple(center)
    marker = " <-- CENTER" if is_center else ""
    delta = combo_s - center_combo
    if not is_center and combo_s > center_combo + 1e-6:
        all_better = False
    print(f"  ss={ss:5.2f} bs={bs:5.2f} sl={sl:5.2f} bl={bl:5.2f}  "
          f"{f1:8.4f}  {v:8.4f}  {combo_s:8.4f}  {delta:+10.4f}{marker}")

print()
if all_better:
    print("CONFIRMED: Center is the local optimum — all ±0.1 neighbors are equal or worse.")
else:
    print("WARNING: Center is NOT the local optimum — a neighbor scored higher.")
print(f"\nFinal best: {dict(zip(PARAM_NAMES, center))}")
print(f"  F1={center_f1:.4f}  VOI={center_voi:.4f}  F1-VOI={center_combo:.4f}", flush=True)


# ============================================================
# Visualization
# ============================================================
def make_random_cmap(n=1024):
    rng = np.random.RandomState(42)
    colors = rng.rand(n, 3)
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)

RAND_CMAP = make_random_cmap()


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
        multi_gt = set()
        for eid in eval_ids:
            emask = eval_frags == eid
            gt_hits = np.unique(gt[emask])
            gt_hits = gt_hits[gt_hits > 0]
            if len(gt_hits) > 1:
                multi_gt.add(eid)
        col3 = np.where(np.isin(eval_frags, list(multi_gt)), eval_frags, 0)
        axes[row, 3].imshow(col3 % 1024, cmap=RAND_CMAP, interpolation="nearest")
        axes[row, 3].set_title(f"{len(multi_gt)} frags on >1 GT")
        gt_ids = np.unique(gt[gt > 0])
        multi_seg = set()
        for gid in gt_ids:
            gmask = gt == gid
            seg_hits = np.unique(eval_frags[gmask])
            seg_hits = seg_hits[seg_hits > 0]
            if len(seg_hits) > 1:
                multi_seg.add(gid)
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


# Find best by each metric from confirmation grid
best_f1_row = max(results, key=lambda r: r[0])
best_voi_row = min(results, key=lambda r: r[1])
best_combo_row = max(results, key=lambda r: r[0] - r[1])

plot_result(best_f1_row[2:], "F1", best_f1_row[0], "examples/best_f1_final.png")
plot_result(best_voi_row[2:], "VOI", best_voi_row[1], "examples/best_voi_final.png")
f1c, vc = best_combo_row[0], best_combo_row[1]
plot_result(best_combo_row[2:], "F1-VOI", f1c - vc, "examples/best_f1_minus_voi_final.png")
