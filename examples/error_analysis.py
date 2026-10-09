"""Analyze false merges and false splits across all z-slices."""

import time
import numpy as np
from funlib.persistence import open_ds
from collections import defaultdict
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

S_SHORT, B_SHORT = 2.0, -1.6
S_LONG, B_LONG = 1.5, -1.6
BLOCK_SHAPE = [100, 100]
CONTEXT = [12, 12]

# Load full volume
print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")

affs_vol = np.ascontiguousarray(
    affs_ds[affs_ds.roi][CHANNELS_2D], dtype=np.float64
)
labels_vol = labels_ds[labels_ds.roi]

_, Z, H, W = affs_vol.shape
print(f"Volume: affs={affs_vol.shape}, labels={labels_vol.shape}")


def analyze_slice(affs_raw, gt, offsets):
    """Analyze a single z-slice. Returns per-pair stats or None if no GT."""
    mask = gt > 0
    if not np.any(mask):
        return None

    # Run mwatershed
    affs = affs_raw.copy()
    affs[SHORT_CHANNELS] = affs[SHORT_CHANNELS] * S_SHORT + B_SHORT
    affs[LONG_CHANNELS] = affs[LONG_CHANNELS] * S_LONG + B_LONG
    seg = mwatershed.agglom_blockwise(
        affs, offsets, BLOCK_SHAPE, lossy=True, context=CONTEXT,
    )

    h, w = gt.shape

    # Map GT <-> seg
    gt_to_segs = defaultdict(set)
    seg_to_gt = defaultdict(set)
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                gt_to_segs[gt[y, x]].add(seg[y, x])
                seg_to_gt[seg[y, x]].add(gt[y, x])

    false_merge_segs = {s: gts for s, gts in seg_to_gt.items() if len(gts) > 1}
    false_split_gts = {g: segs for g, segs in gt_to_segs.items() if len(segs) > 1}

    n_gt = len(np.unique(gt[mask]))
    n_pred = len(np.unique(seg[mask]))

    results = {}
    for ignore_zero in [False, True]:
        suffix = "_no0" if ignore_zero else ""

        merge_pair_affs = defaultdict(lambda: {"short": [], "long": []})
        split_pair_affs = defaultdict(lambda: {"short": [], "long": []})

        for ch_idx, offset in enumerate(offsets):
            group = "short" if ch_idx in [0, 1] else "long"
            raw_ch = affs_raw[ch_idx]
            dy, dx = offset

            for y in range(h):
                for x in range(w):
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not mask[y, x] or not mask[ny, nx]:
                        continue

                    aff_val = raw_ch[y, x]
                    if ignore_zero and aff_val == 0.0:
                        continue

                    g1, g2 = gt[y, x], gt[ny, nx]
                    s1, s2 = seg[y, x], seg[ny, nx]

                    if s1 == s2 and g1 != g2:
                        pair = (min(g1, g2), max(g1, g2))
                        merge_pair_affs[pair][group].append(aff_val)

                    if g1 == g2 and s1 != s2:
                        pair = (g1, min(s1, s2), max(s1, s2))
                        split_pair_affs[pair][group].append(aff_val)

        fm_short, fm_long = [], []
        for pair, ad in merge_pair_affs.items():
            if ad["short"]:
                fm_short.append(np.min(ad["short"]))
            if ad["long"]:
                fm_long.append(np.min(ad["long"]))

        fs_short, fs_long = [], []
        for pair, ad in split_pair_affs.items():
            if ad["short"]:
                fs_short.append(np.max(ad["short"]))
            if ad["long"]:
                fs_long.append(np.max(ad["long"]))

        results[f"false_merge_short{suffix}"] = fm_short
        results[f"false_merge_long{suffix}"] = fm_long
        results[f"false_split_short{suffix}"] = fs_short
        results[f"false_split_long{suffix}"] = fs_long

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_false_merge_segs": len(false_merge_segs),
        "n_false_split_gts": len(false_split_gts),
        "affs": results,
    }


# Aggregate across all z-slices
KEYS = [
    "false_merge_short", "false_merge_long",
    "false_split_short", "false_split_long",
    "false_merge_short_no0", "false_merge_long_no0",
    "false_split_short_no0", "false_split_long_no0",
]

agg = {k: [] for k in KEYS}
total_gt = 0
total_pred = 0
total_false_merge_segs = 0
total_false_split_gts = 0
n_slices = 0

t0 = time.perf_counter()
for z in range(Z):
    affs_raw = affs_vol[:, z, :, :]
    gt = labels_vol[z, :, :]

    if not np.any(gt > 0):
        continue

    res = analyze_slice(affs_raw, gt, OFFSETS_2D)
    if res is None:
        continue

    n_slices += 1
    total_gt += res["n_gt"]
    total_pred += res["n_pred"]
    total_false_merge_segs += res["n_false_merge_segs"]
    total_false_split_gts += res["n_false_split_gts"]

    for k in KEYS:
        agg[k].extend(res["affs"][k])

    if n_slices % 50 == 0:
        print(f"  z={z}, {n_slices} slices processed...")

elapsed = time.perf_counter() - t0

print(f"\nProcessed {n_slices} z-slices with GT in {elapsed:.1f}s")
print(f"Total GT objects (summed across slices): {total_gt}")
print(f"Total pred fragments in GT mask: {total_pred}")
print(f"Total false merge segments: {total_false_merge_segs}")
print(f"Total false split GT objects: {total_false_split_gts}")

for label, ignore_zero in [("All edges", False), ("Ignoring 0-affinity edges", True)]:
    suffix = "_no0" if ignore_zero else ""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for key, desc in [
        (f"false_merge_short{suffix}", "False merges - min SHORT aff between merged GT pairs"),
        (f"false_merge_long{suffix}",  "False merges - min LONG aff between merged GT pairs"),
        (f"false_split_short{suffix}", "False splits - max SHORT aff between split seg pairs"),
        (f"false_split_long{suffix}",  "False splits - max LONG aff between split seg pairs"),
    ]:
        vals = agg[key]
        if vals:
            arr = np.array(vals)
            print(f"\n  {desc}:")
            print(f"    count={len(arr)}, mean={arr.mean():.4f}, "
                  f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
        else:
            print(f"\n  {desc}:")
            print(f"    (no cases)")
