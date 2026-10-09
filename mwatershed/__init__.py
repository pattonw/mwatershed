__author__ = """William Patton"""
__email__ = """wllmpttn24@gmail.com"""
__version__ = """0.5.3"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from collections.abc import Sequence

import numpy as np

from .mwatershed import agglom_rs, cluster, cluster_weighted

LINKAGES = ("abs_max", "sum", "mean", "max", "min")
"""Supported linkage criteria.

* ``"abs_max"`` -- the edge with the largest magnitude decides (mutex watershed).
* ``"sum"`` -- sum of connecting edge weights (GAEC / HCC-Sum).
* ``"mean"`` -- mean of connecting edge weights (average linkage, HC-Avg / HCC-Avg).
* ``"max"`` -- maximum connecting edge weight (single linkage).
* ``"min"`` -- minimum connecting edge weight (complete linkage).

Aliases ``"absmax"``, ``"mutex"``, ``"average"``, ``"avg"``, ``"single"`` and
``"complete"`` are also accepted.
"""


def agglom(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    seeds: np.ndarray | None = None,
    edges: Sequence[tuple[bool, int, int]] | None = None,
    strides: Sequence[Sequence[int]] | None = None,
    randomized_strides: bool = False,
    linkage: str = "abs_max",
    cannot_link: bool = True,
) -> np.ndarray:
    """
    Perform affinity-based agglomeration using the GASP framework (Bailoni et al.
    2022). With the default ``linkage="abs_max"`` this is the mutex watershed. Given
    affinities and their associated offsets, this function will return a label array
    containing a segmentation of the input affinities array.

    :param affinities: Affinities to be segmented. Boundaries between objects should be
        negative, and affinities within objects should be positive. Mutex is a greedy
        algorithm that will always process edges in descending order of magnitude.
    :param offsets: List of offsets to be used for the affinities. Each offset should
        be a list of integers, representing the offset per dimension. For example for
        3D affinities, offsets are typically something like: `[[1, 0, 0], [0, 1, 0],
        [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2]]`.
    :param seeds: Optional seed array. If provided, any non-zero values in the seed
        array are guaranteed to stay unchanged in the output label array. This is useful
        for preserving certain ids in the output label array. The seed array should be
        the same shape as the affinities array.
    :param edges: Optional list of edges used to guarantee specific merge or split pixels.
    :param strides: Optional strides to be used for the affinities. Each stride should
        be a list of integers, representing the stride per dimension. For example for
        3D affinities, strides are typically something like: `[[1, 1, 1], [1, 1, 1],
        [1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]`. Long range edges are usually split
        biased, and providing a stride lets us avoid excessive splitting.
    :param randomized_strides: If True, the strides will be randomized by just turning
        the stride into a probability of selecting a specific affinity via `1/prod(stride)`.
    :param linkage: Linkage criterion used to compute the interaction between two
        clusters from the affinities connecting them. One of :data:`LINKAGES`:
        ``"abs_max"`` (default, mutex watershed), ``"sum"``, ``"mean"``, ``"max"`` or
        ``"min"``.
    :param cannot_link: If True (default), a repulsive interaction between two
        clusters adds a cannot-link constraint so they are not merged while the
        algorithm is processing interactions by magnitude; remaining attractive
        pairs are merged afterwards. Has no effect on the result for ``"abs_max"``.
    """
    return agglom_rs(
        affinities,
        offsets,
        seeds,
        edges,
        strides,
        randomized_strides,
        linkage,
        cannot_link,
    )


def cluster_edges(
    edges: Sequence[tuple[float, int, int]],
    linkage: str = "abs_max",
    cannot_link: bool = True,
) -> list[tuple[int, int]]:
    """
    Perform affinity-based agglomeration on a weighted graph. With the default
    ``linkage="abs_max"`` this is the mutex watershed.

    :param edges: List of edges to be used for the affinities. Each edge should be a
        tuple of (weight, node1, node2). The weight is the affinity between the two
        nodes. The nodes are the integer ids of the fragments in the graph.
    :param linkage: Linkage criterion, see :func:`agglom`.
    :param cannot_link: Whether to add cannot-link constraints, see :func:`agglom`.

    :return: List of tuples (fragment_id: int, segment_id: int) mapping fragments to
        segments. The segment id is the smallest fragment id in the segment.
    """
    return cluster_weighted(
        [(float(w), int(u), int(v)) for w, u, v in edges], linkage, cannot_link
    )


__all__ = ["LINKAGES", "agglom", "cluster", "cluster_edges", "cluster_weighted"]
