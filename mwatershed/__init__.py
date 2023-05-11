# -*- coding: utf-8 -*-
"""Top-level package for mwatershed."""

__author__ = """William Hunter Patton"""
__email__ = """pattonw@hhmi.org"""
__version__ = """0.1.0"""
__version_info__ = tuple(int(n) for n in __version__.split("."))

from .mwatershed import agglom, cluster

__all__ = ["agglom", "cluster"]
