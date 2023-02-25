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
    bias: Optional[f64] = None,
    seeds: Optional[NDArray[np.uint64]] = None,
    edges: Optional[list[tuple[usize, usize, f64]]] = None,
)
```
where:
`affinities` is a `k+1` dimensional array of non `nan` affinities
`offsets` is a list of length `k` tuples of integer offsets
`bias` is a float determining the bias towards merging or fragmenting
`seeds` is an array of fragment ids
`edges` is a list of `(u, v, aff)` tuples to insert arbitrary extra affinities between fragment ids

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
