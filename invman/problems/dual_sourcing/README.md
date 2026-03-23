# Dual Sourcing

This package implements the small-scale dual-sourcing benchmark family used by Gijsbrechts et al. (2022) for the Veeraraghavan-Scheller-Wolf settings:

- regular supplier lead time `lr in {2, 3, 4}`
- expedited lead time `le = 0`
- discrete uniform demand on `{0, 1, 2, 3, 4}`
- holding cost `h = 5`
- backlog cost `b = 495`
- regular cost `c_r = 100`
- expedited cost `c_e in {105, 110}`

Implemented heuristic families:

- single-index
- dual-index
- capped dual-index
- tailored base-surge

The package also includes a bounded dynamic-programming solver over the reduced `lr`-dimensional state representation for correctness checks on the small-scale instances.
