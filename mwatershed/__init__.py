__author__ = """William Patton"""
__email__ = """wllmpttn24@gmail.com"""
__version__ = """0.5.3"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .mwatershed import agglom_rs, cluster
import numpy as np
from typing import Optional
from collections.abc import Sequence


def agglom(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    seeds: Optional[np.ndarray] = None,
    edges: Optional[Sequence[tuple[bool, int, int]]] = None,
    strides: Optional[Sequence[Sequence[int]]] = None,
    randomized_strides: bool = False,
) -> np.ndarray:
    """
    Perform affinity-based agglomeration using mutex watershed. Given affinities and
    their associated offsets, this function will return a label array containing a
    segmentation of the input affinities array.

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
    """
    return agglom_rs(affinities, offsets, seeds, edges, strides, randomized_strides)


def cluster_edges(
    edges: Sequence[tuple[float, int, int]],
) -> list[tuple[int, int]]:
    """
    Perform affinity-based agglomeration using mutex watershed on a set of edges.

    :param edges: List of edges to be used for the affinities. Each edge should be a
        tuple of (weight, node1, node2). The weight is the affinity between the two
        nodes. The nodes are the integer ids of the fragments in the graph.

    :return: List of tuples (fragment_id: int, segment_id: int) mapping fragments to segments.
    """
    sorted_edges = sorted(edges, reverse=True, key=lambda x: abs(x[0]))
    bool_edges = [(weight > 0, node1, node2) for weight, node1, node2 in sorted_edges]
    return cluster(bool_edges)


__all__ = ["agglom", "cluster_edges", "cluster"]
