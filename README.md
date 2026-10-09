# mwatershed
[![pytest](https://github.com/pattonw/mwatershed/actions/workflows/pytest.yaml/badge.svg)](https://github.com/pattonw/mwatershed/actions/workflows/pytest.yaml)
[![ruff](https://github.com/pattonw/mwatershed/actions/workflows/ruff.yaml/badge.svg)](https://github.com/pattonw/mwatershed/actions/workflows/ruff.yaml)
[![cargo](https://github.com/pattonw/mwatershed/actions/workflows/cargo.yaml/badge.svg)](https://github.com/pattonw/mwatershed/actions/workflows/cargo.yaml)

[![pypi](https://img.shields.io/pypi/pyversions/mwatershed.svg)](https://pypi.python.org/pypi/mwatershed)


## A rusty mutex watershed

See docs [here](https://pattonw.github.io/mwatershed)

### Installation

`pip install mwatershed`

### Features

* A mutex watershed implementation for affinities
* Should handle up to 6 dimensional arrays of fragments (7 dimensions if you include offsets in affinities) but only tested on 2 and 3 dimensions

### Usage

```python
components = mwatershed.agglom(
    affinities: NDArray[np.float64],
    offsets: list[list[int]],
    seeds: NDArray[np.uint64],
    edges: Optional[list[tuple[usize, usize, f64]]] = None,
)
```
where:
* `affinities` is a `k+1` dimensional array of non `nan` affinities with leading dimension having size `n`
* `offsets` is a list of length `n` of offset tuples of `k` integers
* `seeds` is a `k` dimensional array of fragment ids. Note `seeds.shape` must be equal to `affinities.shape[1:]`. Any entry not equal to 0 is guaranteed to stay that way, any entry equal to zero has no priors.
* `edges` is a list of `(u, v, aff)` tuples to insert arbitrary extra affinities between fragment ids

### Linkage criteria (GASP)

`agglom` and `cluster_edges` accept a `linkage` argument that selects how the
interaction between two clusters is computed from the affinities connecting
them, following the GASP framework of
[Bailoni et al. (2022)](https://arxiv.org/abs/1906.11713):

| `linkage`   | Interaction between two clusters            | Known as                     |
|-------------|---------------------------------------------|------------------------------|
| `"abs_max"` | the connecting edge with the largest magnitude (default) | Mutex watershed |
| `"sum"`     | sum of the connecting affinities            | GAEC / HCC-Sum               |
| `"mean"`    | mean of the connecting affinities           | average linkage (HC-Avg / HCC-Avg) |
| `"max"`     | largest connecting affinity                 | single linkage               |
| `"min"`     | smallest connecting affinity                | complete linkage             |

```python
components = mwatershed.agglom(affinities, offsets, linkage="mean")
```

Pairs of clusters are visited in order of decreasing interaction magnitude.
An attractive pair is merged; a repulsive pair receives a cannot-link
constraint when `cannot_link=True` (the default). Once every remaining pair is
repulsive or constrained, the constraints are lifted and any pairs that are
still attractive are merged. With `"abs_max"` the constraints never change the
result, which is exactly the mutex watershed. Seeds and the `edges` argument
always act as hard constraints regardless of the linkage.

The generic linkages use a contracted-graph priority queue and are several
times slower than the dedicated mutex watershed path; see
[`LINKAGE_REPORT.md`](LINKAGE_REPORT.md) for timings
(`cargo run --release --example benchmark_linkages`) or run
`cargo bench` for the Criterion comparison.
