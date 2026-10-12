"""Cross-check the Rust GASP engine against a naive Python reference.

The reference recomputes every cluster interaction from scratch at each step,
so it is slow but obviously correct with respect to Algorithm 1 of Bailoni et
al. (2022). Random small multigraphs are clustered with every linkage, with and
without cannot-link constraints, and the partitions must agree.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import mwatershed

LINKAGE_FN = {
    "abs_max": lambda ws: max(ws, key=abs),
    "sum": sum,
    "mean": lambda ws: sum(ws) / len(ws),
    "max": max,
    "min": min,
}


def reference_gasp(n_nodes, edges, linkage, cannot_link):
    cluster_of = list(range(n_nodes))
    members = {i: {i} for i in range(n_nodes)}
    constrained: set[frozenset] = set()
    dismissed: dict[frozenset, float] = {}
    next_id = n_nodes
    f = LINKAGE_FN[linkage]

    def interactions():
        pair_edges: dict[frozenset, list[float]] = {}
        for w, u, v in edges:
            a, b = cluster_of[u], cluster_of[v]
            if a != b:
                pair_edges.setdefault(frozenset((a, b)), []).append(w)
        return {pair: f(ws) for pair, ws in pair_edges.items()}

    def merge(pair):
        nonlocal next_id
        a, b = tuple(pair)
        new = next_id
        next_id += 1
        members[new] = members.pop(a) | members.pop(b)
        for node in members[new]:
            cluster_of[node] = new
        # constraints follow the merged cluster
        for c in list(constrained):
            if a in c or b in c:
                constrained.discard(c)
                rest = c - {a, b}
                if rest:  # a constraint with a third cluster follows the merge
                    constrained.add(frozenset((new, next(iter(rest)))))

    # phase one: by magnitude, constrain repulsive pairs
    while True:
        inter = interactions()
        eligible = [
            (abs(w), pair)
            for pair, w in inter.items()
            if pair not in constrained and dismissed.get(pair) != w
        ]
        if not eligible:
            break
        _, pair = max(eligible, key=lambda x: x[0])
        w = inter[pair]
        if w > 0:
            merge(pair)
        elif cannot_link:
            constrained.add(pair)
        else:
            dismissed[pair] = w

    # phase two: lift constraints, merge attractive pairs by signed value
    if cannot_link:
        while True:
            inter = interactions()
            positive = [(w, pair) for pair, w in inter.items() if w > 0]
            if not positive:
                break
            _, pair = max(positive, key=lambda x: x[0])
            merge(pair)

    return [cluster_of[i] for i in range(n_nodes)]


def same_partition(a, b):
    fwd, rev = {}, {}
    for x, y in zip(a, b):
        if fwd.setdefault(x, y) != y or rev.setdefault(y, x) != x:
            return False
    return True


@pytest.mark.parametrize("linkage", list(LINKAGE_FN))
@pytest.mark.parametrize("cannot_link", [True, False])
def test_engine_matches_reference(linkage, cannot_link):
    rng = np.random.default_rng(hash((linkage, cannot_link)) % 2**32)
    for trial in range(40):
        n = int(rng.integers(4, 14))
        m = int(rng.integers(n, 4 * n))
        u = rng.integers(0, n, size=m)
        v = rng.integers(0, n, size=m)
        keep = u != v
        u, v = u[keep], v[keep]
        w = rng.normal(size=len(u))
        edges = list(zip(w.tolist(), u.tolist(), v.tolist()))
        # multi-edges between the same pair are part of the test
        expected = reference_gasp(n, edges, linkage, cannot_link)
        mapping = dict(
            mwatershed.cluster_edges(edges, linkage=linkage, cannot_link=cannot_link)
        )
        got = [mapping.get(i, -1 - i) for i in range(n)]
        assert same_partition(expected, got), (
            f"{linkage} cannot_link={cannot_link} trial {trial}: edges={edges} "
            f"expected={expected} got={got}"
        )


def test_reference_sanity():
    # a-b strongly repulsive, a-c and b-c attractive: sum keeps b apart, max merges all
    edges = [(-3.0, 0, 1), (2.0, 0, 2), (1.5, 1, 2)]
    assert len(set(reference_gasp(3, edges, "sum", True))) == 2
    assert len(set(reference_gasp(3, edges, "max", True))) == 1
    for linkage, cannot_link in itertools.product(LINKAGE_FN, [True, False]):
        mapping = dict(
            mwatershed.cluster_edges(edges, linkage=linkage, cannot_link=cannot_link)
        )
        assert same_partition(
            reference_gasp(3, edges, linkage, cannot_link),
            [mapping[i] for i in range(3)],
        )
