# -*- coding: utf-8 -*-
"""Top-level package for mwatershed."""

__author__ = """William Hunter Patton"""
__email__ = """pattonw@hhmi.org"""
__version__ = """0.1.0"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .mwatershed import agglom_rs, cluster
import numpy as np
from typing import Optional


def agglom(
    affinities: np.ndarray,
    offsets: list[list[int]],
    seeds: Optional[np.ndarray] = None,
    edges: Optional[list[tuple[bool, int, int]]] = None,
):
    return agglom_rs(affinities, offsets, seeds, edges)


__all__ = ["agglom", "cluster"]
