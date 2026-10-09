"""Compare serial, lossless blockwise, and lossy blockwise on real 2D data."""

import time
import numpy as np
from funlib.persistence import open_ds
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

NEIGHBORHOOD_3D = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [2, 0, 0], [0, 12, 0], [0, 0, 12],
]
CHANNELS_2D = [i for i, off in enumerate(NEIGHBORHOOD_3D) if off[0] == 0]
OFFSETS_2D = [[off[1], off[2]] for off in [NEIGHBORHOOD_3D[i] for i in CHANNELS_2D]]

print("Loading data...")
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
affs_raw = np.ascontiguousarray(
    affs_ds[affs_ds.roi][CHANNELS_2D, 300, :, :], dtype=np.float64
)
affs = affs_raw * 2.0 - 1.0
print(f"Affinities: {affs.shape}\n")

BLOCK_SHAPE = [100, 100]
CONTEXT = [12, 12]
N_WARMUP = 1
N_RUNS = 3


def bench(name, fn):
    # warmup
    for _ in range(N_WARMUP):
        result = fn()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    n_labels = len(np.unique(result))
    print(f"{name:40s}  {median:.4f}s  {n_labels:6d} labels")
    return result, median


serial_seg, t_serial = bench(
    "serial (agglom)",
    lambda: mwatershed.agglom(affs, OFFSETS_2D),
)

lossless_seg, t_lossless = bench(
    "blockwise lossless, context=12",
    lambda: mwatershed.agglom_blockwise(
        affs, OFFSETS_2D, BLOCK_SHAPE, context=CONTEXT, lossy=False,
    ),
)

lossy_seg, t_lossy = bench(
    "blockwise lossy, context=12",
    lambda: mwatershed.agglom_blockwise(
        affs, OFFSETS_2D, BLOCK_SHAPE, context=CONTEXT, lossy=True,
    ),
)

print()
print(f"Speedup lossless vs serial: {t_serial / t_lossless:.2f}x")
print(f"Speedup lossy vs serial:    {t_serial / t_lossy:.2f}x")

# Compare results
print()
print(f"serial vs lossless identical: {np.array_equal(serial_seg, lossless_seg)}")
print(f"serial vs lossy identical:    {np.array_equal(serial_seg, lossy_seg)}")

# If not identical, measure agreement
for name, seg in [("lossless", lossless_seg), ("lossy", lossy_seg)]:
    if np.array_equal(serial_seg, seg):
        continue
    # Pixel-level: what fraction of pixels have the same label-pair relationships?
    # Sample random pairs and check if they agree on same/different segment
    rng = np.random.default_rng(42)
    n_samples = 100_000
    flat_serial = serial_seg.ravel()
    flat_other = seg.ravel()
    idx = rng.integers(0, flat_serial.size, size=(n_samples, 2))
    same_serial = flat_serial[idx[:, 0]] == flat_serial[idx[:, 1]]
    same_other = flat_other[idx[:, 0]] == flat_other[idx[:, 1]]
    rand_agree = np.mean(same_serial == same_other)
    print(f"serial vs {name} Rand agreement: {rand_agree:.6f}")
