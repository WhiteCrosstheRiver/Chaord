"""Field and electrolyte observables of the research case: what an external
field changes that can be measured, and the solvation structure of the
confined electrolyte.

The reviewer's binding (docs/reviews/open_items_v2.md, section 5, item 2:
"After the gate: the first research use"): the field statement comes "plus
the measurable consequences: a dipole-orientation order parameter and the
radial density profile".  These two observables are what separates a
field-on from a field-off program in the round trip of item 3, so they
follow the cv convention exactly: take (frame, dialect), return a dict of
named scalars/arrays, read every number from the dialect by name, and fail
closed.

  * dipole_orientation   — P1 = <cos theta>, theta the angle between each
                           molecule's dipole moment and the field axis,
                           averaged over the molecules of the species named
                           by `field_dipole_species` (bond-graph molecules;
                           site charges from the frame when the reader
                           carried them, else the dialect's published
                           site-charge table)
  * radial_density_profile — cylindrical number density n(r) about the
                           axis through the cell centre (or a caller
                           centre), minimum-image wrapped, binned to the
                           largest inscribed cylinder or to a caller r_max
                           (Y5: the nanotube region passes its tube radius)
  * solvation_shell      — the Li+ first solvation shell BY SPECIES on the
                           dialect's named sites, and the ion-pair classes
                           SSIP/CIP/AGG (D36, open_items_v4.md Y5 item 2)
  * species_radial_profiles — per-species cylindrical number-density
                           profiles of the molecule centres inside the
                           tube radius (Y5 item 3)
  * field_observables / field_distance — the composer and the distance for
                           a future noise-floor computation on these
                           observables (chaord.cv.noise.distance knows the
                           legacy key families only, so the field set ships
                           its own pair; both raw arrays AND summary
                           scalars are exposed -- see field_distance)

Thresholds live in the research dialect (dialects/research.yaml): the
species list, the site-charge table with its citations, the charge
tolerance, the zero-dipole floor, the radial bin count and the inner/outer
split fraction, the solvation site rules and their g(r)-derived cutoffs,
the species-profile bin width and the pore's wall-exclusion depth.
"""
from __future__ import annotations

import numpy as np

from ..io.frames import Frame
from ..lang.errors import ChaordError
from .registry import register

_AXIS_ALIASES = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis) -> int:
    """The index of a named cell axis: 'x'/'y'/'z' (or 0/1/2)."""
    if isinstance(axis, str):
        key = axis.strip().lower()
        if key not in _AXIS_ALIASES:
            raise ChaordError(
                f"unknown axis {axis!r}; name it 'x', 'y' or 'z'")
        return _AXIS_ALIASES[key]
    axis = int(axis)
    if not 0 <= axis < 3:
        raise ChaordError(f"profile axis must be 0, 1 or 2, got {axis!r}")
    return axis


def _named_species(dialect) -> list:
    """The census formulas P1 averages over (`field_dipole_species`)."""
    species = dialect.threshold("field_dipole_species")
    if isinstance(species, str):
        species = [species]
    species = [str(s) for s in species]
    if not species:
        raise ChaordError("field_dipole_species names no species")
    return species


def _molecule_labels(frame: Frame, dialect) -> np.ndarray:
    """Atom -> molecule index: connected components of the covalent-radius
    bond graph (the same rule the census and the partial g(r) use)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..build.molecules import bond_graph
    edges = bond_graph(frame, dialect)
    n = len(frame)
    if not edges:
        return np.arange(n)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    _, labels = connected_components(
        coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)),
        directed=False)
    return labels


def _frame_charges(frame: Frame) -> np.ndarray | None:
    """Per-atom site charges in e when the reader carried them.

    The convention: a `charges` entry in frame.info, one finite number per
    atom, in elementary charge units (the io layer records what the input
    file stated -- the LAMMPS data/dump reader of the research case fills
    it from the file's charge column).  Anything malformed refuses (fail
    closed)."""
    if "charges" not in frame.info:
        return None
    q = np.asarray(frame.info["charges"], float).ravel()
    if len(q) != len(frame):
        raise ChaordError(
            f"frame.info['charges'] has {len(q)} entries for {len(frame)} "
            "atoms; per-atom site charges are all or nothing")
    if not np.isfinite(q).all():
        raise ChaordError("frame.info['charges'] holds non-finite values; "
                          "refusing the frame")
    return q


def _table_charges(frame: Frame, dialect, labels, species) -> np.ndarray:
    """Per-atom site charges in e from the dialect's site-charge table,
    resolved by census formula (the fallback when the frame carries none).

    Every table entry is validated once (an element table); the neutrality
    requirement is enforced per molecule in dipole_orientation, where the
    element multiplicities are known: the site charges of one molecule must
    sum to zero within `field_charge_tolerance` -- the orientation of a
    charged species depends on the arbitrary origin, so the order parameter
    is defined on neutral molecules only, and a typo'd charge model refuses
    there rather than measuring nonsense."""
    table = dialect.threshold("field_dipole_site_charges")
    if not isinstance(table, dict):
        raise ChaordError("field_dipole_site_charges must be a table keyed "
                          "by census formula (see dialects/research.yaml)")
    for formula, entry in table.items():
        if not isinstance(entry, dict):
            raise ChaordError(
                f"site-charge entry for {formula!r} is not an element table")
    missing = [s for s in species if s not in table]
    if missing:
        raise ChaordError(
            f"no site-charge model for {', '.join(missing)} in "
            "field_dipole_site_charges, and the frame carries no charges "
            "(frame.info['charges']) -- cannot compute a dipole")
    from ..build.molecules import _formula
    q = np.zeros(len(frame))
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        formula = _formula([frame.symbols[i] for i in idx])
        if formula not in species:
            continue
        entry = table[formula]
        unknown = sorted({frame.symbols[i] for i in idx} - set(entry))
        if unknown:
            raise ChaordError(
                f"the site-charge model for {formula!r} has no charge for "
                f"{', '.join(unknown)}")
        for i in idx:
            q[i] = float(entry[frame.symbols[i]])
    return q


def dipole_orientation(frame: Frame, dialect, axis="z") -> dict:
    """Dipole-orientation order parameter P1 = <cos theta> against `axis`.

    Molecules are the connected components of the bond graph (the census
    rule); the averaged set is the species named by `field_dipole_species`
    (census formulas).  Each molecule's dipole moment is

        mu = sum_i q_i (r_i - r_ref)

    with minimum-image unwrapped offsets from the molecule's first atom
    (translation- and wrap-safe; the reference drops out of the direction
    for the neutral species the order parameter is defined on).  Site
    charges resolve in this order: the frame's own per-atom charges
    (``frame.info["charges"]``, e -- what the input file stated) win; else
    the dialect's ``field_dipole_site_charges`` table keyed by census
    formula, with published, cited models.  A molecule whose |mu| falls at
    or below ``field_dipole_moment_floor`` has no orientation and refuses.

    Returns ``{"P1": float, "n_molecules": int}``.
    """
    ax = _axis_index(axis)
    e = np.zeros(3)
    e[ax] = 1
    species = _named_species(dialect)
    labels = _molecule_labels(frame, dialect)
    q = _frame_charges(frame)
    if q is None:
        q = _table_charges(frame, dialect, labels, species)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    floor = float(dialect.threshold("field_dipole_moment_floor"))
    from ..build.molecules import _formula
    cosines: list[float] = []
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        formula = _formula([frame.symbols[i] for i in idx])
        if formula not in species:
            continue
        d = pos[idx] - pos[idx[0]]
        d -= L * np.round(d / L)
        qtot = float(q[idx].sum())
        if abs(qtot) > float(dialect.threshold("field_charge_tolerance")):
            raise ChaordError(
                f"a {formula} molecule's site charges sum to {qtot:+.4f} e; "
                "the dipole-orientation order parameter is defined on "
                "neutral molecules (its dipole would depend on the origin)")
        mu = (q[idx][:, None] * d).sum(axis=0)
        norm = float(np.linalg.norm(mu))
        if norm <= floor:
            raise ChaordError(
                f"a {formula} molecule has |mu| = {norm:.3e} e x length, at "
                "or below field_dipole_moment_floor: no orientation to "
                "average (name a polar species or fix its charge model)")
        cosines.append(float(mu @ e) / norm)
    if not cosines:
        raise ChaordError(
            f"no molecule of species {', '.join(species)} in this frame "
            f"(bond-graph census found none); refusing to average an empty "
            "set")
    return {"P1": float(np.mean(cosines)), "n_molecules": len(cosines)}


def radial_density_profile(frame: Frame, dialect, axis="z", centre=None,
                           bins=None, r_max=None) -> dict:
    """Cylindrical number-density profile n(r) about `axis`.

    The axis runs through the cell centre by default, or through the
    transverse coordinates a caller passes as `centre` (e.g. the axis of a
    nanotube lifted from the carbon positions).  Every atom's radial
    coordinate is its minimum-image distance from the axis, so atoms near a
    boundary wrap to their nearest image (a molecule straddling the cell
    edge still bins correctly).  Bins run from 0 to

        r_max = min(L_perp) / 2

    (half the shorter transverse cell edge): for the cell-centred axis that
    is exactly the largest inscribed cylinder, and in a periodic box no
    circular shell overlaps its own image below it -- beyond it the corners
    would double-count and the shell normalisation would break.  A caller
    may pass a smaller `r_max` -- the nanotube composition passes its
    tube's radius (Y5 item 1) -- never a larger one.  Atoms with r > r_max
    (the box corners, or the vacuum outside a pore wall) are excluded and
    reported as ``n_outside``.

    Bin count comes from `radial_profile_bins` unless the caller overrides
    (tests, resolution studies).  Returns ``{"profile": (bins,) number
    density in atoms per volume unit of the system, "r_centers": bin
    centres, "r_max": float, "n_outside": int}``.
    """
    ax = _axis_index(axis)
    bins = int(bins if bins is not None
               else dialect.threshold("radial_profile_bins"))
    if bins < 2:
        raise ChaordError(f"radial_profile_bins must be at least 2, got {bins}")
    L = frame.cell_diag
    perp = [i for i in range(3) if i != ax]
    c = (np.array([L[perp[0]] / 2, L[perp[1]] / 2]) if centre is None
         else np.asarray(centre, float).ravel()[:2])
    limit = float(min(L[perp[0]], L[perp[1]])) / 2
    if r_max is None:
        r_max = limit
    else:
        r_max = float(r_max)
        if not 0 < r_max <= limit:
            raise ChaordError(
                f"r_max must lie in (0, {limit:g}] (half the shorter "
                f"transverse cell edge; a larger cylinder overlaps its own "
                f"periodic image): got {r_max:g}")
    pos = np.mod(frame.pos, L)
    d = pos[:, perp] - c
    d -= L[perp] * np.round(d / L[perp])
    r = np.linalg.norm(d, axis=1)
    edges = np.linspace(0, r_max, bins + 1)
    h, _ = np.histogram(r[r <= r_max], edges)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * L[ax]
    return {"profile": h / shell,
            "r_centers": (edges[1:] + edges[:-1]) / 2,
            "r_max": r_max,
            "n_outside": int((r > r_max).sum())}


def inner_outer_ratio(radial: dict, dialect) -> float:
    """Summary scalar of a radial profile: mean inner-shell density over
    mean outer-shell density, split at `radial_profile_inner_fraction` x
    r_max (inner bins have r_centre <= the split).

    1 for a homogeneous system, > 1 when confinement or a solvation shell
    piles atoms near the axis.  A fully depleted outer region (a narrow
    pore with vacuum at the wall) is NOT measurable: the ratio would divide
    by an empty shell, so it refuses with the documented not-measurable
    error instead of a ZeroDivisionError (Y5 item 1, open_items_v4.md).
    This is the distance()-friendly scalar; the raw profile array ships
    alongside it in field_observables."""
    frac = float(dialect.threshold("radial_profile_inner_fraction"))
    inner = radial["r_centers"] <= frac * radial["r_max"]
    if not inner.any() or inner.all():
        raise ChaordError(
            "radial_profile_inner_fraction leaves the inner or the outer "
            "shell empty; choose a fraction strictly inside (0, 1)")
    outer_mean = float(radial["profile"][~inner].mean())
    if outer_mean <= 0.0:  # dialect-exempt: numerical-guard: an empty shell is exactly zero
        raise ChaordError(
            "the radial profile's outer shell is empty (mean density 0 "
            f"beyond {frac:g} x r_max = {frac * radial['r_max']:.3f}): the "
            "inner/outer ratio is not measurable -- pass r_max from the "
            "confining region (the tube radius) or widen the outer shell")
    return float(radial["profile"][inner].mean()) / outer_mean


def field_observables(frame: Frame, dialect, axis="z", centre=None,
                      bins=None) -> dict:
    """Compose the field observables of one frame (the noise-floor input).

    Keys: the order parameter and its statistics (``field_P1``,
    ``field_n_molecules``), the raw radial profile with its geometry
    (``radial_profile``, ``radial_r_centers``, ``radial_r_max``,
    ``radial_n_outside``) and the summary scalar
    ``radial_inner_outer_ratio``.  Both raw arrays AND the summary scalar
    are exposed: chaord.cv.noise.distance compares only the legacy
    observable families by hard-coded key patterns, so the field set ships
    its own distance (field_distance) instead of growing noise.py's --
    the integration choice is stated here, where it can be found."""
    dip = dipole_orientation(frame, dialect, axis=axis)
    rad = radial_density_profile(frame, dialect, axis=axis, centre=centre,
                                 bins=bins)
    return {"field_P1": dip["P1"],
            "field_n_molecules": dip["n_molecules"],
            "radial_profile": rad["profile"],
            "radial_r_centers": rad["r_centers"],
            "radial_r_max": rad["r_max"],
            "radial_n_outside": rad["n_outside"],
            "radial_inner_outer_ratio": inner_outer_ratio(rad, dialect)}


# ------------------------------------------------- the electrolyte set (Y5) --
# D36 (docs/reviews/open_items_v4.md Y5): the Li+ solvation shell by
# species, the SSIP/CIP/AGG ion-pair classes and the species radial
# number-density profiles of the confined electrolyte.  These follow the
# cv convention exactly (take (frame, dialect), return named scalars or
# arrays, read every number from the dialect, fail closed) but stay module
# functions, not registry CVs: the registry is names in / floats out and
# these are per-species tables (the registry pattern has no slot for a
# species-indexed family).  One definition per quantity still holds: the
# lift's statements are printed FROM these functions.


def _molecule_centres(frame: Frame, labels) -> tuple[np.ndarray, np.ndarray]:
    """(centres, formulas): per bond-graph component the minimum-image mean
    position (the lift's molecule-centre rule, lift/fluid.py _centers: one
    definition, stated here) and its census formula."""
    from ..build.molecules import _formula
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    centres = np.zeros((labels.max() + 1, 3))
    formulas = np.empty(labels.max() + 1, dtype=object)
    for g in range(labels.max() + 1):
        idx = np.where(labels == g)[0]
        rel = pos[idx] - pos[idx[0]]
        rel -= L * np.round(rel / L)
        centres[g] = pos[idx[0]] + rel.mean(axis=0)
        formulas[g] = _formula([frame.symbols[i] for i in idx])
    return centres, formulas


def _shell_sites(frame: Frame, dialect, labels, formulas) -> dict:
    """{census formula: indices of its first-shell sites} for the species
    the dialect's `li_shell_sites` names and the frame holds.

    Site rules are structural, read from the dialect by name: `carbonyl_O`
    is the carbonate carbonyl oxygen (build.molecules
    carbonyl_oxygen_sites); a bare element symbol selects every atom of
    that element in the named species' molecules (PF6-: every F)."""
    from ..build.molecules import carbonyl_oxygen_sites
    from ase.data import chemical_symbols
    rules = dialect.threshold("li_shell_sites")
    if not isinstance(rules, dict) or not rules:
        raise ChaordError(
            "li_shell_sites must be a non-empty table keyed by census "
            "formula (see dialects/research.yaml)")
    present = set(formulas)
    picked: dict[str, np.ndarray] = {}
    carbonyl = None
    for formula, rule in rules.items():
        if str(formula) not in present:
            continue
        rule = str(rule)
        members = np.where(formulas == str(formula))[0]
        atoms_of = np.where(np.isin(labels, members))[0]
        if rule == "carbonyl_O":
            if carbonyl is None:
                carbonyl = carbonyl_oxygen_sites(frame, dialect)
            idx = np.intersect1d(carbonyl, atoms_of)
        elif rule in chemical_symbols and rule != "X":
            idx = np.array([i for i in atoms_of
                            if frame.symbols[i] == rule], int)
        else:
            raise ChaordError(
                f"li_shell_sites names unknown site rule {rule!r} for "
                f"{formula!r}: expected 'carbonyl_O' or an element symbol")
        picked[str(formula)] = idx
    return picked


def solvation_shell(frame: Frame, dialect) -> dict | None:
    """The Li+ first solvation shell by species and the ion-pair classes
    (Y5 item 2, D36).

    The cation is the dialect's `li_shell_cation` census element (the bond
    graph never bonds through it, so each ion is its own census
    component); the shell of each species named by `li_shell_sites` is
    counted on its named sites within `li_shell_cutoffs` (the first
    g(r) minimum of the reference zero-field runs -- provenance in the
    research dialect, never measured on the frames under test).  Ion pairs
    on the `li_shell_anion` species: SSIP (no anion site within the
    cutoff), CIP (an anion contacted by exactly this one cation), AGG (an
    anion shared by two or more cations); the classes partition the cations.

    Returns ``{"n_li": int, "per_species": {formula: (mean, sd over the
    cations)}, "ion_pairs": {"SSIP": n, "CIP": n, "AGG": n}, "cutoffs":
    {formula: r}}`` or None when the frame holds no cation (no shell to
    average; the caller stays silent rather than averaging an empty set)."""
    cation = str(dialect.threshold("li_shell_cation"))
    anion = str(dialect.threshold("li_shell_anion"))
    cutoffs = dialect.threshold("li_shell_cutoffs")
    if not isinstance(cutoffs, dict):
        raise ChaordError("li_shell_cutoffs must be a table keyed by census "
                          "formula (see dialects/research.yaml)")
    labels = _molecule_labels(frame, dialect)
    _, formulas = _molecule_centres(frame, labels)
    li = np.array([i for i, s in enumerate(frame.symbols) if s == cation],
                  int)
    if not len(li):
        return None
    if anion not in set(formulas):
        raise ChaordError(
            f"the frame holds no molecule of the li_shell_anion species "
            f"{anion!r}: the ion-pair classes are undefined without the "
            "counter-ion (fix the census or the dialect)")
    sites = _shell_sites(frame, dialect, labels, formulas)
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    per_species: dict[str, list[float]] = {}
    anion_members = np.where(formulas == anion)[0]
    anion_sites = sites.get(anion)
    if anion_sites is None or not len(anion_sites):
        raise ChaordError(
            f"the anion species {anion!r} has no sites selected by "
            "li_shell_sites: the ion-pair classes cannot be counted")
    contacts: list[set] = []
    for i in li:
        contacted: set = set()
        for formula in sorted(sites):
            if formula not in cutoffs:
                raise ChaordError(
                    f"li_shell_cutoffs has no entry for {formula!r}: every "
                    "li_shell_sites species needs its cutoff (see "
                    "dialects/research.yaml)")
            cut = float(cutoffs[formula])
            idx = sites[formula]
            if not len(idx):
                per_species.setdefault(formula, []).append(0.0)  # dialect-exempt: numerical-guard: an absent species contributes no shell members
                continue
            d = pos[idx] - pos[i]
            d -= L * np.round(d / L)
            within = np.linalg.norm(d, axis=1) <= cut
            per_species.setdefault(formula, []).append(float(within.sum()))
            if formula == anion:
                contacted = set(labels[idx[within]].tolist())
        contacts.append(contacted)
    ion_pairs = {"SSIP": 0, "CIP": 0, "AGG": 0}
    for a, mine in enumerate(contacts):
        if not mine:
            ion_pairs["SSIP"] += 1
        elif any(mine & other for b, other in enumerate(contacts) if b != a):
            ion_pairs["AGG"] += 1
        else:
            ion_pairs["CIP"] += 1
    assert sum(ion_pairs.values()) == len(li)   # dialect-exempt: numerical-guard: the classes partition the cations
    return {"n_li": int(len(li)),
            "per_species": {f: (float(np.mean(v)), float(np.std(v)))
                            for f, v in sorted(per_species.items())},
            "ion_pairs": ion_pairs,
            "cutoffs": {f: float(cutoffs[f]) for f in sorted(sites)},
            "n_anion_molecules": int(len(anion_members))}


def species_radial_profiles(frame: Frame, dialect, axis="z", centre=None,
                            r_max=None) -> dict:
    """Per-species cylindrical number-density profiles n_k(r) of the
    MOLECULE CENTRES (Y5 item 3): for every census species, molecule
    centres binned by minimum-image distance from `axis` through `centre`,
    from r = 0 to the caller's `r_max` (the nanotube composition passes
    the tube radius), divided by the shell volumes pi (r_out^2 - r_in^2)
    L_axis.  Bin width is the dialect's `species_nr_bin_width`; the last
    bin ends at floor(r_max / width) x width so every bin lies fully
    inside the region.

    Returns ``{census formula: {"profile": (n_bins,) number density in
    molecules per volume unit, "r_centers": bin centres, "n_outside": int,
    "count": int}}``."""
    ax = _axis_index(axis)
    L = frame.cell_diag
    perp = [i for i in range(3) if i != ax]
    limit = float(min(L[perp[0]], L[perp[1]])) / 2
    if r_max is None:
        r_max = limit
    else:
        r_max = float(r_max)
        if not 0 < r_max <= limit:
            raise ChaordError(
                f"r_max must lie in (0, {limit:g}] (half the shorter "
                f"transverse cell edge): got {r_max:g}")
    width = float(dialect.threshold("species_nr_bin_width"))
    if not width > 0:
        raise ChaordError("species_nr_bin_width must be positive")
    n_bins = int(r_max // width)
    if n_bins < 1:
        raise ChaordError(
            f"the region radius {r_max:g} is narrower than one "
            f"species_nr_bin_width ({width:g}): no radial profile")
    c = (np.array([L[perp[0]] / 2, L[perp[1]] / 2]) if centre is None
         else np.asarray(centre, float).ravel()[:2])
    labels = _molecule_labels(frame, dialect)
    centres, formulas = _molecule_centres(frame, labels)
    d = centres[:, perp] - c
    d -= L[perp] * np.round(d / L[perp])
    r = np.linalg.norm(d, axis=1)
    edges = np.linspace(0, n_bins * width, n_bins + 1)
    shell = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * L[ax]
    out: dict = {}
    for formula in sorted(set(formulas)):
        sel = formulas == formula
        h, _ = np.histogram(r[sel][r[sel] <= n_bins * width], edges)
        out[str(formula)] = {"profile": h / shell,
                             "r_centers": (edges[1:] + edges[:-1]) / 2,
                             "n_outside": int((r[sel] > n_bins * width).sum()),
                             "count": int(sel.sum())}
    return out


def field_distance(o1: dict, o2: dict) -> dict:
    """Distances between two field observable sets (see field_observables):
    |dP1|, the radial-profile RMS and the inner/outer ratio difference --
    the three numbers a future field noise floor (two independent frames of
    one simulation, as chaord.cv.noise.noise_floor does for the legacy set)
    is built from.  Mismatched bin counts or missing keys refuse, mirroring
    noise.distance's fail-closed contract."""
    need = ("field_P1", "radial_profile", "radial_inner_outer_ratio")
    bad = [k for k in need if (k in o1) != (k in o2)]
    if bad:
        raise ChaordError(
            f"observable sets disagree on {', '.join(map(repr, bad))} "
            "(build both with field_observables)")
    for k in need:
        if k not in o1:
            raise ChaordError(
                f"observable sets lack {k!r}; build them with "
                "field_observables")
    if len(o1["radial_profile"]) != len(o2["radial_profile"]):
        raise ChaordError("radial profiles have different bin counts")
    return {"field_p1_delta": float(abs(o1["field_P1"] - o2["field_P1"])),
            "field_radial_rms": float(np.sqrt(np.mean(
                (o1["radial_profile"] - o2["radial_profile"]) ** 2))),
            "field_ratio_delta": float(abs(
                o1["radial_inner_outer_ratio"]
                - o2["radial_inner_outer_ratio"]))}


# ------------------------------------------------------------ registry --
# Rule 7 (one definition per quantity): the headline scalars are registered
# CVs so lift/build/assert can dispatch by name; the array observables stay
# module functions (the registry is names in, floats out).

def _measure_p1(frame: Frame, dialect, **kwargs) -> float:
    return dipole_orientation(frame, dialect, **kwargs)["P1"]


def _measure_ratio(frame: Frame, dialect, **kwargs) -> float:
    return inner_outer_ratio(
        radial_density_profile(frame, dialect, **kwargs), dialect)


register("dipole_p1", None,
         "dipole-orientation order parameter <cos theta> of the named "
         "solvent species against the analysis axis -- the field's axis "
         "when one is given, else the tube's, else the dialect default "
         "(research case; blind per D29, open_items_v4.md)",
         thresholds=("field_dipole_species", "field_dipole_site_charges",
                     "field_charge_tolerance", "field_dipole_moment_floor",
                     "dipole_axis_default"),
         dialect="research", measure=_measure_p1)
register("radial_inner_outer_ratio", None,
         "mean inner-shell over outer-shell cylindrical number density "
         "about the field axis (research case)",
         thresholds=("radial_profile_bins", "radial_profile_inner_fraction"),
         dialect="research", measure=_measure_ratio)
