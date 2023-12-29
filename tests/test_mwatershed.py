import numpy as np

import mwatershed


def test_agglom_3d():
    offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    affinities = (
        np.array(
            [
                [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
                [[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
                [[[1, 0], [0, 1]], [[0, 0], [0, 0]]],
            ],
            dtype=float,
        )
        - 0.5
    )
    # 8 nodes. connecting edges:
    # 1-2, 1-3, 1-7, 4-8, 6-8, 7-8
    # components: [(1,2,3,7),(4,6,7,8)]

    components = mwatershed.agglom(affinities, offsets)

    _, counts = np.unique(components, return_counts=True)
    counts = sorted(counts)
    assert len(np.unique(components)) == 2

    assert counts[0] == 4
    assert counts[1] == 4


def test_agglom_2d():
    offsets = [(0, 1), (1, 0)]
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

    components = mwatershed.agglom(affinities, offsets)

    _, counts = np.unique(components, return_counts=True)
    counts = sorted(counts)
    assert len(counts) == 4

    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[2] == 2
    assert counts[3] == 4


def test_agglom_2d_negative_offsets():
    offsets = [(0, -1), (-1, 0)]
    affinities = (
        np.array(
            [[[0, 0, 1], [0, 0, 1], [0, 0, 1]], [[0, 0, 0], [0, 0, 0], [1, 1, 1]]],
            dtype=float,
        )
        - 0.5
    )
    # 9 nodes. connecting edges:
    # 2-3, 5-6, 8-9, 4-7, 5-8, 6-9
    # components: [(1,),(2,3),(4,7),(5,6,8,9)]

    components = mwatershed.agglom(affinities, offsets)

    _, counts = np.unique(components, return_counts=True)
    counts = sorted(counts)
    assert len(counts) == 4

    assert counts[0] == 1
    assert counts[1] == 2
    assert counts[2] == 2
    assert counts[3] == 4


def test_agglom_2d_with_extra_edges():
    nodes = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 9]], dtype=np.uint64)
    offsets = [(0, 1), (1, 0)]
    affinities = (
        np.array(
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]],
            dtype=float,
        )
        - 0.5
    )
    edges = [(True, 9, 1)]
    # 9 nodes. connecting edges:
    # 2-3, 5-6, 8-9, 4-7, 5-8, 6-9
    # components: [(1,),(2,3),(4,7),(5,6,8,9)]

    components = mwatershed.agglom(affinities, offsets, seeds=nodes, edges=edges)

    _, counts = np.unique(components, return_counts=True)
    counts = sorted(counts)
    assert len(counts) == 3

    assert counts[0] == 2
    assert counts[1] == 2
    assert counts[2] == 5
