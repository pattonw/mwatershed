"""Run mutex watershed on the full 3D volume and save fragments."""

import time
import numpy as np
from funlib.persistence import open_ds, prepare_ds
import mwatershed

ZARR_BASE = "/home/will/Code/Scripts/will-training-experiments/self_supervised_embedding/exploration/ss_data.zarr/gel_2_crop_0"

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

# Best params (from 3D grid search)
S_XY_SHORT, B_XY_SHORT = 2.0, -1.6
S_XY_LONG, B_XY_LONG = 1.5, -1.8
S_Z_SHORT, B_Z_SHORT = 2.0, -1.7
S_Z_LONG, B_Z_LONG = 1.5, -1.2

BLOCK_SHAPE = [64, 64, 64]
CONTEXT = [12, 12, 12]

# Load
print("Loading affinities...")
t0 = time.perf_counter()
affs_ds = open_ds(f"{ZARR_BASE}/pseudo_affs", mode="r")
affs = np.ascontiguousarray(affs_ds[affs_ds.roi], dtype=np.float64)
t_load = time.perf_counter() - t0
print(f"Loaded: {affs.shape}, {affs.nbytes / 1e9:.2f} GB in {t_load:.1f}s")

# Apply scale/bias
print("Applying scale/bias...")
affs[XY_SHORT] = affs[XY_SHORT] * S_XY_SHORT + B_XY_SHORT
affs[XY_LONG] = affs[XY_LONG] * S_XY_LONG + B_XY_LONG
affs[Z_SHORT] = affs[Z_SHORT] * S_Z_SHORT + B_Z_SHORT
affs[Z_LONG] = affs[Z_LONG] * S_Z_LONG + B_Z_LONG

# Run lossy blockwise
print(f"Running mwatershed (lossy, block={BLOCK_SHAPE}, context={CONTEXT})...")
t0 = time.perf_counter()
seg = mwatershed.agglom_blockwise(
    affs, OFFSETS_3D, BLOCK_SHAPE, lossy=True, context=CONTEXT,
)
t_mws = time.perf_counter() - t0

n_labels = len(np.unique(seg))
print(f"Done in {t_mws:.1f}s, {n_labels} fragments")

# Save
print("Saving fragments...")
labels_ds = open_ds(f"{ZARR_BASE}/labels", mode="r")
import shutil, os
frag_path = f"{ZARR_BASE}/fragments"
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
print(f"Saved to {ZARR_BASE}/fragments")
print(f"\nTotal: load={t_load:.1f}s, mwatershed={t_mws:.1f}s")
