"""Reactive census: dissociation statements from the fragment population."""
from __future__ import annotations

from ..build.molecules import molecule_census
from ..lang.ir import Arrow, Name, Quantity, Statement, At, Plus


def dissociation_statement(frame, dialect, reactant="H2O") -> Statement | None:
    """`dissociate H2O -> OH + H count N` when both fragments are present.

    The count is the number of dissociation events the census can explain:
    min(count(OH), count(H)); species without fragments are ignored."""
    try:
        census = molecule_census(frame, dialect)
    except Exception:
        return None
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
