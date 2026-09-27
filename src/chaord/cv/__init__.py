"""Collective variables: one definition per quantity.

Each CV is registered once and used three ways: measured by lift, restrained by
build, and checked by `assert`. Measures take a Frame plus the dialect (every
threshold comes from the dialect by name — never hard-coded).
"""
from .registry import CVS, CVDef, register, measure, missing_measure  # noqa: F401
from . import local  # noqa: F401  (registers the local-structure CVs)
