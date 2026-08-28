# Repository for Intrinsic Viscous Liquid Dynamics (Randium 3D)

This repository contain code and data for the Randium model of viscous liquid dynamics.
The companion paper is 

> *Intrinsic viscous liquid dynamics* by Ulf R. Pedersen (Roskilde University, Denmark), [arXiv.org/abs/2511.02991](http://arXiv.org/abs/2511.02991)


This is the output of the work done in the 2026 "Glass and Time" Roskilde summer school by
Guglielmo Mennella (GM).

Specifically, this is the implementation of Randium in 3D.

Main difference from the original Randium code:

- All files:
    - Removal of duplicated and legacy code.

- backend.py:
    - Implementation of neighbours lists to handle different lattice geometries.
    - Implementation of a 'get neighbours' function to determine a site's specific
        available neighbours.

- run.py:
    - Generalisation of the equilibrium criterion to 2/z * u / beta, where
        z is the coordination number and beta is the inverse temperature.

Notes:
- versione 1 and 2 are legacy.