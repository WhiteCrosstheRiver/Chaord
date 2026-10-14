"""Collective variables: one definition per quantity.

Each CV is registered once and used three ways: measured by lift, restrained by
build, and checked by `assert`. Measures take a Frame plus the dialect (every
threshold comes from the dialect by name — never hard-coded).
"""
from .registry import CVS, CVDef, register, measure, missing_measure  # noqa: F401
from . import local  # noqa: F401  (registers the local-structure CVs)
from . import rich  # noqa: F401  (species-pair g(r), bond angles, density profiles)
from . import field  # noqa: F401  (field observables: dipole orientation P1, radial profile)
