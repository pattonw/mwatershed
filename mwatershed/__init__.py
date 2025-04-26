# -*- coding: utf-8 -*-
"""Top-level package for mwatershed."""

__author__ = """William Patton"""
__email__ = """wllmpttn24@gmail.com"""
__version__ = """0.5.2"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .mwatershed import agglom_rs, cluster
import numpy as np
from typing import Optional


def agglom(
    affinities: np.ndarray,
    offsets: list[list[int]],
    seeds: Optional[np.ndarray] = None,
    edges: Optional[list[tuple[bool, int, int]]] = None,
    strides: Optional[list[list[int]]] = None,
    randomized_strides: bool = False,
):
    return agglom_rs(affinities, offsets, seeds, edges, strides, randomized_strides)


__all__ = ["agglom", "cluster"]
