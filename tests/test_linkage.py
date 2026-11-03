import numpy as np
import pytest

import mwatershed

LINKAGES = ["abs_max", "sum", "mean", "max", "min"]


def chain_affinities():
    """1D chain with offsets [1] and [2].

    0-1: +0.6, 1-2: +0.6, 2-3: +0.6 (offset 1)
    0-2: -0.7, 1-3: +0.5             (offset 2)
    """
    offsets = [(1,), (2,)]
    affinities = np.array([[0.6, 0.6, 0.6, 0.0], [-0.7, 0.5, 0.0, 0.0]])
    return affinities, offsets


def test_default_is_mutex_watershed():
    affinities, offsets = chain_affinities()
    default = mwatershed.agglom(affinities, offsets)
    explicit = mwatershed.agglom(affinities, offsets, linkage="abs_max")
    assert np.array_equal(default, explicit)
    assert len(np.unique(default)) == 2
    assert default[0] == default[1]
    assert default[2] == default[3]


@pytest.mark.parametrize(
    "linkage,num_clusters",
    [("abs_max", 2), ("min", 2), ("sum", 1), ("mean", 1), ("max", 1)],
)
@pytest.mark.parametrize("cannot_link", [True, False])
def test_linkages_on_chain(linkage, num_clusters, cannot_link):
    affinities, offsets = chain_affinities()
    labels = mwatershed.agglom(
        affinities, offsets, linkage=linkage, cannot_link=cannot_link
    )
    assert len(np.unique(labels)) == num_clusters
    assert np.all(labels != 0)


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("absmax", "abs_max"),
        ("mutex", "abs_max"),
        ("average", "mean"),
        ("avg", "mean"),
        ("single", "max"),
        ("complete", "min"),
        ("SUM", "sum"),
    ],
)
def test_linkage_aliases(alias, canonical):
    affinities, offsets = chain_affinities()
    a = mwatershed.agglom(affinities, offsets, linkage=alias)
    b = mwatershed.agglom(affinities, offsets, linkage=canonical)
    assert np.array_equal(a, b)


def test_unknown_linkage_raises():
    affinities, offsets = chain_affinities()
    with pytest.raises(ValueError, match="Unknown linkage"):
        mwatershed.agglom(affinities, offsets, linkage="median")
    with pytest.raises(ValueError, match="Unknown linkage"):
        mwatershed.cluster_edges([(1.0, 0, 1)], linkage="median")


@pytest.mark.parametrize("linkage", LINKAGES)
def test_2d_grid_with_seeds(linkage):
    offsets = [(0, 1), (1, 0)]
    affinities = (
        np.array(
            [[[0, 1, 0], [0, 1, 0], [0, 1, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]],
            dtype=float,
        )
        - 0.5
    )
    seeds = np.array([[1, 2, 0], [4, 0, 0], [0, 0, 0]], dtype=np.uint64)
    # components: [(1,),(2,3),(4,7),(5,6,8,9)] for every linkage because every
    # pair of clusters is only connected by repulsive edges.
    labels = mwatershed.agglom(affinities, offsets, seeds=seeds, linkage=linkage)
    assert labels[0, 0] == 1
    assert labels[0, 1] == 2 and labels[0, 2] == 2
    assert labels[1, 0] == 4 and labels[2, 0] == 4
    assert len(np.unique(labels)) == 4
    assert not np.any(labels == 0)
    _, counts = np.unique(labels, return_counts=True)
    assert sorted(counts) == [1, 2, 2, 4]


@pytest.mark.parametrize("linkage", LINKAGES)
@pytest.mark.parametrize("cannot_link", [True, False])
def test_3d_random_is_deterministic_and_keeps_seeds(linkage, cannot_link):
    rng = np.random.default_rng(0)
    offsets = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 3), (0, 3, 0), (3, 0, 0)]
    affinities = rng.normal(size=(len(offsets), 10, 10, 10))
    seeds = np.zeros((10, 10, 10), dtype=np.uint64)
    seeds[0, 0, 0] = 1000
    seeds[9, 9, 9] = 2000
    seeds[5, 5, 5] = 3000
    a = mwatershed.agglom(
        affinities, offsets, seeds=seeds, linkage=linkage, cannot_link=cannot_link
    )
    b = mwatershed.agglom(
        affinities, offsets, seeds=seeds, linkage=linkage, cannot_link=cannot_link
    )
    assert np.array_equal(a, b)
    assert a[0, 0, 0] == 1000
    assert a[9, 9, 9] == 2000
    assert a[5, 5, 5] == 3000
    # seeds never merge with each other
    assert len({a[0, 0, 0], a[9, 9, 9], a[5, 5, 5]}) == 3


@pytest.mark.parametrize("linkage", LINKAGES)
def test_extremes(linkage):
    offsets = [(0, 1), (1, 0)]
    attractive = np.full((2, 6, 6), 0.8)
    repulsive = np.full((2, 6, 6), -0.8)
    one = mwatershed.agglom(attractive, offsets, linkage=linkage)
    assert len(np.unique(one)) == 1 and one[0, 0] != 0
    none = mwatershed.agglom(repulsive, offsets, linkage=linkage)
    assert np.all(none == 0)


def test_sum_linkage_accumulates_weak_attractions():
    # Two 1D segments {0,1} and {2,3} joined by one strong repulsion and two
    # weak attractions. abs_max keeps them apart, sum merges them.
    offsets = [(1,), (2,), (3,)]
    affinities = np.zeros((3, 4))
    affinities[0, 0] = 1.0  # 0-1
    affinities[0, 2] = 1.0  # 2-3
    affinities[0, 1] = -0.5  # 1-2
    affinities[1, 0] = 0.3  # 0-2
    affinities[2, 0] = 0.3  # 0-3
    absmax = mwatershed.agglom(affinities, offsets, linkage="abs_max")
    assert len(np.unique(absmax)) == 2
    summed = mwatershed.agglom(affinities, offsets, linkage="sum")
    assert len(np.unique(summed)) == 1


def test_cluster_edges_linkages():
    edges = [
        (0.6, 10, 11),
        (0.6, 11, 12),
        (0.6, 12, 13),
        (-0.7, 10, 12),
        (0.5, 11, 13),
    ]
    absmax = dict(mwatershed.cluster_edges(edges))
    assert absmax[10] == absmax[11]
    assert absmax[12] == absmax[13]
    assert absmax[10] != absmax[12]
    # segment ids are the smallest fragment id of each segment
    assert absmax[10] == 10 and absmax[12] == 12

    summed = dict(mwatershed.cluster_edges(edges, linkage="sum"))
    assert len(set(summed.values())) == 1 and summed[13] == 10

    complete = dict(mwatershed.cluster_edges(edges, linkage="min"))
    assert complete == absmax

    # the legacy boolean entry point still works
    sorted_edges = sorted(edges, key=lambda e: -abs(e[0]))
    legacy = dict(mwatershed.cluster([(w > 0, u, v) for w, u, v in sorted_edges]))
    assert legacy == absmax


def test_no_label_collisions_with_background_or_seeds():
    # A singleton cluster at node 0 (never merged, but with a positive edge that
    # is blocked by a mutex) must not be labelled 0, and a cluster whose
    # representative id equals a seed value must not take that seed's label.
    offsets = [(1,), (2,)]
    affinities = np.array([[0.5, 0.9, 0.0], [-1.0, 0.0, 0.0]])
    # 0-1: +0.5, 1-2: +0.9, 0-2: -1.0  -> {1,2} merged, 0 singleton
    labels = mwatershed.agglom(affinities, offsets)
    assert labels[0] != 0
    assert labels[1] == labels[2] != labels[0]

    seeds = np.array([0, 0, 1], dtype=np.uint64)  # seed value 1 == node id 1
    affinities = np.array([[0.5, -0.9, 0.0], [-1.0, 0.0, 0.0]])
    # 0-1: +0.5, 1-2: -0.9, 0-2: -1.0 -> {0,1}, {2 (seed 1)}
    labels = mwatershed.agglom(affinities, offsets, seeds=seeds)
    assert labels[2] == 1
    assert labels[0] == labels[1] != 1
    assert labels[0] != 0


def _reference(u, v, w, nodes):
    """``cluster`` on the same edges, stable-sorted by |w|, as a per-node label."""
    order = np.argsort(-np.abs(w), kind="stable")
    lut = dict(
        mwatershed.cluster([(bool(w[i] > 0), int(u[i]), int(v[i])) for i in order])
    )
    return np.array([lut.get(int(n), int(n)) for n in nodes], dtype=np.uint64)


def test_cluster_arrays_matches_cluster():
    rng = np.random.default_rng(0)
    n, m = 200, 2000
    u = rng.integers(0, n, m)
    v = rng.integers(0, n, m)
    w = rng.normal(size=m)
    nodes, labels = mwatershed.cluster_arrays(u, v, w, nodes=np.arange(n))
    np.testing.assert_array_equal(nodes, np.arange(n))
    np.testing.assert_array_equal(labels, _reference(u, v, w, nodes))
    # presorted input gives the same result
    order = np.argsort(-np.abs(w), kind="stable")
    _, presorted = mwatershed.cluster_arrays(
        u[order], v[order], w[order], nodes=np.arange(n), sorted=True
    )
    np.testing.assert_array_equal(presorted, labels)


def test_cluster_arrays_arbitrary_ids():
    rng = np.random.default_rng(1)
    ids = rng.choice(np.arange(10**12, 10**12 + 10**6), size=300, replace=False)
    u = rng.choice(ids, 3000)
    v = rng.choice(ids, 3000)
    w = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 3000)  # plenty of ties
    nodes, labels = mwatershed.cluster_arrays(u, v, w)
    np.testing.assert_array_equal(nodes, np.unique(np.r_[u, v]))
    np.testing.assert_array_equal(labels, _reference(u, v, w, nodes))
    # every label is the smallest id of its cluster
    for label in np.unique(labels):
        assert label == nodes[labels == label].min()
    # explicit nodes: extra nodes stay singletons, labels align with them
    explicit = np.unique(np.r_[ids, 7, 10**13])
    nodes2, labels2 = mwatershed.cluster_arrays(u, v, w, nodes=explicit)
    np.testing.assert_array_equal(nodes2, explicit)
    np.testing.assert_array_equal(labels2, _reference(u, v, w, explicit))
    assert labels2[nodes2 == 7] == 7 and labels2[nodes2 == 10**13] == 10**13


def test_cluster_arrays_tie_order():
    # 0-1 attract, 1-2 repel, 0-2 attract, all with |w| = 1: input order decides
    u, v, w = np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([1.0, -1.0, 1.0])
    _, forward = mwatershed.cluster_arrays(u, v, w)
    _, backward = mwatershed.cluster_arrays(u[::-1], v[::-1], w[::-1])
    assert forward.tolist() == [0, 0, 2]
    assert backward.tolist() == [0, 1, 0]
    np.testing.assert_array_equal(forward, _reference(u, v, w, [0, 1, 2]))
    np.testing.assert_array_equal(
        backward, _reference(u[::-1], v[::-1], w[::-1], [0, 1, 2])
    )


def test_cluster_arrays_prefix():
    u, v, w = np.array([10, 20]), np.array([20, 30]), np.array([5.0, -1.0])
    # without a prefix the strong edge merges 10 and 20
    _, plain = mwatershed.cluster_arrays(u, v, w)
    assert plain.tolist() == [10, 10, 30]
    # a prefix split is a mutex the strong attractive edge cannot override
    _, split = mwatershed.cluster_arrays(u, v, w, prefix=([10], [20], [False]))
    assert split.tolist() == [10, 20, 30]
    # a prefix merge joins 20 and 30 although their only edge repels
    _, merged = mwatershed.cluster_arrays(u, v, w, prefix=([20], [30], [True]))
    assert merged.tolist() == [10, 10, 10]
    # prefix edges are processed in their own order, before the weighted edges
    _, both = mwatershed.cluster_arrays(
        u, v, w, prefix=([10, 20], [20, 30], [False, True])
    )
    assert both.tolist() == [10, 20, 20]
    # a prefix endpoint is a node even without weighted edges
    nodes, labels = mwatershed.cluster_arrays(u, v, w, prefix=([30], [40], [True]))
    assert nodes.tolist() == [10, 20, 30, 40]
    assert labels.tolist() == [10, 10, 30, 30]
    # the prefix matches cluster() with the same edges in front
    nodes, labels = mwatershed.cluster_arrays(
        u, v, w, prefix=([10, 20], [20, 30], [False, True])
    )
    lut = dict(mwatershed.cluster([(False, 10, 20), (True, 20, 30), (True, 10, 20)]))
    assert labels.tolist() == [lut[int(n)] for n in nodes]


def test_cluster_arrays_rejects_bad_input():
    with pytest.raises(ValueError, match="equal length"):
        mwatershed.cluster_arrays([0, 1], [1], [1.0, 1.0])
    with pytest.raises(ValueError, match="not a node"):
        mwatershed.cluster_arrays([0], [5], [1.0], nodes=[0, 1])
    with pytest.raises(ValueError, match="sorted and unique"):
        mwatershed.cluster_arrays([0], [1], [1.0], nodes=[1, 0])
    with pytest.raises(ValueError, match="sorted and unique"):
        mwatershed.cluster_arrays([0], [1], [1.0], nodes=[0, 1, 1])
    with pytest.raises(ValueError, match="NaN"):
        mwatershed.cluster_arrays([0], [1], [np.nan])
    with pytest.raises(ValueError, match="negative"):
        mwatershed.cluster_arrays([-1], [1], [1.0])
    with pytest.raises(ValueError, match="prefix"):
        mwatershed.cluster_arrays([0], [1], [1.0], prefix=([0], [1, 2], [True]))
    with pytest.raises(ValueError, match="not a node"):
        mwatershed.cluster_arrays(
            [0], [1], [1.0], nodes=[0, 1], prefix=([0], [9], [True])
        )


def test_cluster_arrays_empty():
    nodes, labels = mwatershed.cluster_arrays([], [], [])
    assert len(nodes) == 0 and len(labels) == 0
    nodes, labels = mwatershed.cluster_arrays([], [], [], nodes=[3, 5])
    assert nodes.tolist() == [3, 5] and labels.tolist() == [3, 5]


@pytest.mark.parametrize("linkage", ["sum", "mean", "max", "min", "abs_max"])
@pytest.mark.parametrize("cannot_link", [True, False])
def test_cluster_arrays_linkages_match_cluster_edges(linkage, cannot_link):
    rng = np.random.default_rng(2)
    ids = rng.choice(np.arange(10**9, 10**9 + 10**5), size=60, replace=False)
    u = rng.choice(ids, 400)
    v = rng.choice(ids, 400)
    w = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], 400)  # plenty of ties
    nodes, labels = mwatershed.cluster_arrays(
        u, v, w, linkage=linkage, cannot_link=cannot_link
    )
    lut = dict(
        mwatershed.cluster_edges(
            list(zip(w.tolist(), u.tolist(), v.tolist())), linkage, cannot_link
        )
    )
    assert labels.tolist() == [lut[int(n)] for n in nodes]


def test_cluster_arrays_prefix_with_linkage():
    # 10-11 and 11-12 attract, 10-12 repels weakly: sum linkage merges all three
    u, v, w = np.array([10, 11, 10]), np.array([11, 12, 12]), np.array([1.0, 1.0, -0.5])
    _, plain = mwatershed.cluster_arrays(u, v, w, linkage="sum")
    assert plain.tolist() == [10, 10, 10]
    # a prefix split between 10 and 12 is a hard cannot-link: 12 stays apart
    _, split = mwatershed.cluster_arrays(
        u, v, w, prefix=([10], [12], [False]), linkage="sum"
    )
    assert split.tolist() == [10, 10, 12]
    # a prefix merge joins 12 and 13 although nothing else connects them
    nodes, merged = mwatershed.cluster_arrays(
        u, v, w, prefix=([12], [13], [True]), linkage="sum"
    )
    assert nodes.tolist() == [10, 11, 12, 13]
    assert merged.tolist() == [10, 10, 10, 10]
    # a prefix split beats a later prefix merge of the same pair
    _, both = mwatershed.cluster_arrays(
        u, v, w, prefix=([10, 10], [12, 12], [False, True]), linkage="mean"
    )
    assert both.tolist() == [10, 10, 12]


def test_cluster_arrays_rejects_unknown_linkage():
    with pytest.raises(ValueError):
        mwatershed.cluster_arrays([0], [1], [1.0], linkage="median")
