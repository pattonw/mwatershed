import numpy as np

import mwatershed

import time

shape = [100, 100, 100]
offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
nodes = np.zeros(shape, dtype=np.uint64)
affinities = np.random.randn(3, *shape)

t1 = time.time()
components = mwatershed.agglom(affinities, offsets)
t2 = time.time()

print(f"MWATERSHED: {len(np.unique(components))} components in {t2-t1} seconds")

from affogato.segmentation.mws import compute_mws_segmentation_from_affinities

t3 = time.time()
components = compute_mws_segmentation_from_affinities(affinities, offsets, 0)
t4 = time.time()

print(f"AFFOGATO: {len(np.unique(components))} components in {t4-t3} seconds")
