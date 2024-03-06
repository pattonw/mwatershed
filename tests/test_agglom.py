import numpy as np

import mwatershed


def test_agglom_2d_with_strides():
    offsets = [(0, 1), (1, 0)]
    strides = [(2, 1), (1, 2)]
    affinities = (
        np.array(
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]],
            dtype=float,
        )
        - 0.5
    )
    # 9 nodes. connecting edges:
    # 2-3, 5-6, 8-9, 4-7, 5-8, 6-9
    # components: [(1,),(2,3),(4,7),(5,6,8,9)]

    components = mwatershed.agglom(affinities, offsets, strides=strides)

    _, counts = np.unique(components, return_counts=True)
    counts = sorted(counts)
    assert len(counts) == 4, components

    assert counts[0] == 2
    assert counts[1] == 2
    assert counts[2] == 2
    assert counts[3] == 3
