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
