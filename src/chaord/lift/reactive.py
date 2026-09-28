"""Reactive census: dissociation statements from the fragment population."""
from __future__ import annotations

from ..build.molecules import molecule_census
from ..lang.ir import Arrow, At, Name, Plus, Quantity, Statement

# census formulas follow Hill order (HO); these display names are the
# conventional species names used by the template registry and the grammar
DISPLAY_NAMES = {"HO": "OH"}


def display_name(formula: str) -> str:
    """Conventional species name for a census formula (HO -> OH)."""
    return DISPLAY_NAMES.get(formula, formula)


def dissociation_from_census(census: dict[str, int],
                             reactant: str = "H2O") -> Statement | None:
    """`dissociate H2O -> OH + H count N` when both fragments are present.

    The count is the number of dissociation events the census can explain:
    min(count(OH), count(H)); species without fragments are ignored. The
    census may come from any frame region (e.g. only a surface overlayer)."""
    # census names follow Hill order (HO); display uses the conventional OH
    n_oh = census.get("OH", census.get("HO", 0))
    n_h = census.get("H", 0)
    n = min(n_oh, n_h)
    if n == 0:
        return None
    return Statement(
        kind="build", key="dissociate",
        values=[Name(text=reactant), Arrow(), Name(text="OH"), At(),
                Name(text="surface"), Plus(), Name(text="H"), At(),
                Name(text="surface"), Name(text="count"), Quantity(num=str(n))])


def dissociation_statement(frame, dialect, reactant="H2O") -> Statement | None:
    """dissociation_from_census over the molecular census of a whole frame."""
    try:
        census = molecule_census(frame, dialect)
    except Exception:
        return None
    return dissociation_from_census(census, reactant)
