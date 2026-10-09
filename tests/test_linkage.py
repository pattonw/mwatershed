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
