"""Type stub for the compiled `mwatershed.mwatershed` extension.

Hand-written; keep in sync with the `#[pyfunction]` signatures in
`src/lib.rs`. `usize` on the Rust side is `np.uintp` here.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

def agglom_rs(
    affinities: npt.NDArray[np.float64],
    offsets: Sequence[Sequence[int]],
    seeds: npt.NDArray[np.uintp] | None = None,
    edges: Sequence[tuple[bool, int, int]] | None = None,
    strides: Sequence[Sequence[int]] | None = None,
    randomized_strides: bool | None = None,
    linkage: str | None = None,
    cannot_link: bool | None = None,
) -> npt.NDArray[np.uintp]: ...
def cluster(edges: Sequence[tuple[bool, int, int]]) -> list[tuple[int, int]]: ...
def cluster_weighted(
    edges: Sequence[tuple[float, int, int]],
    linkage: str | None = None,
    cannot_link: bool | None = None,
) -> list[tuple[int, int]]: ...
def cluster_arrays(
    u: npt.NDArray[np.uint64],
    v: npt.NDArray[np.uint64],
    weights: npt.NDArray[np.float64],
    nodes: npt.NDArray[np.uint64] | None = None,
    sorted: bool = False,
    prefix: tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.bool_]]
    | None = None,
    linkage: str | None = None,
    cannot_link: bool | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]: ...
