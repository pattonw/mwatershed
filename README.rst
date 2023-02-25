==========
mwatershed
==========


.. image:: https://img.shields.io/pypi/pyversions/mwatershed.svg
        :target: https://pypi.python.org/pypi/mwatershed

.. image:: https://img.shields.io/pypi/v/mwatershed.svg
        :target: https://pypi.python.org/pypi/mwatershed

.. image:: https://readthedocs.org/projects/mwatershed/badge/?version=latest
        :target: https://mwatershed.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
        
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



A rusty mutex watershed


* Free software: MIT License
* Documentation: https://mwatershed.readthedocs.io.

Installation
------------

pip install git+https://github.com/pattonw/mwatershed

Features
--------

A mutex watershed implementation

Usage
-----

```
components = mwatershed.agglom(
    affinities: NDArray[np.float64],
    offsets: list[list[int]],
    seeds: NDArray[np.uint64],
    edges: Optional[list[tuple[usize, usize, f64]]] = None,
)
```
where:
`affinities` is a `k+1` dimensional array of non `nan` affinities with leading dimension having size `n`
`offsets` is a list of length `n` of offset tuples of `k` integers
`seeds` is a `k` dimensional array of fragment ids. Note `seeds.shape` must be equal to `affinities.shape[1:]`. The simplest case is assigning a unique int to each element of seeds.
`edges` is a list of `(u, v, aff)` tuples to insert arbitrary extra affinities between fragment ids

Credits
-------

This package was created with Cookiecutter_ and the `pattonw/cookiecutter-rust-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`pattonw/cookiecutter-rust-pypackage`: https://github.com/pattonw/cookiecutter-rust-pypackage
