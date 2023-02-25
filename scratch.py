import numpy as np
import pytest

import mwatershed


nodes = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint64)
offsets = [(0, 1), (1, 0)]
affinities = np.array(
    [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0], [0, 1, 0]]],
    dtype=float,
)
# 9 nodes. connecting edges:
# 2-3, 5-6, 8-9, 4-7, 5-8, 6-9
# components: [(1,),(2,3),(4,7),(5,6,8,9)]

components = mwatershed.agglom(affinities, offsets, seeds=nodes)
