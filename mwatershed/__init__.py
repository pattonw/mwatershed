__author__ = """William Patton"""
__email__ = """wllmpttn24@gmail.com"""
__version__ = """0.5.3"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .mwatershed import agglom_rs, agglom_blockwise_rs, cluster
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


def agglom_blockwise(
    affinities: np.ndarray,
    offsets: Sequence[Sequence[int]],
    block_shape: Sequence[int],
    seeds: Optional[np.ndarray] = None,
    edges: Optional[Sequence[tuple[bool, int, int]]] = None,
    strides: Optional[Sequence[Sequence[int]]] = None,
    randomized_strides: bool = False,
    context: Optional[Sequence[int]] = None,
    lossy: bool = False,
) -> np.ndarray:
    """
    Blockwise parallel affinity-based agglomeration using mutex watershed.

    Splits the volume into spatial blocks, processes each block in parallel,
    then resolves cross-block edges sequentially.

    :param affinities: Affinities to be segmented (same as :func:`agglom`).
    :param offsets: List of offsets (same as :func:`agglom`).
    :param block_shape: Spatial block size per dimension, e.g. ``[64, 64]`` for 2D
        or ``[32, 32, 32]`` for 3D. Must have the same number of elements as the
        spatial dimensions of the affinities array.
    :param seeds: Optional seed array (same as :func:`agglom`).
    :param edges: Optional extra edges (same as :func:`agglom`).
    :param strides: Optional strides (same as :func:`agglom`).
    :param randomized_strides: Whether to randomize strides (same as :func:`agglom`).
    :param context: Optional additional context per dimension (e.g. ``[5, 5, 5]``).
        Each block reads this many extra voxels on each side, allowing more edges
        to be resolved locally in parallel. ``None`` means no extra context.
    :param lossy: If ``True``, cross-block edges are deferred but interior edges
        are never frozen — all interior decisions are made locally. This can
        dramatically reduce the serial workload at the cost of potentially
        different (but practically very similar) results. If ``False`` (default),
        freeze propagation ensures identical results to sequential processing.
    """
    return agglom_blockwise_rs(
        affinities, offsets, list(block_shape), seeds, edges, strides, randomized_strides,
        list(context) if context is not None else None, lossy,
    )


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


__all__ = ["agglom", "agglom_blockwise", "cluster_edges", "cluster"]
