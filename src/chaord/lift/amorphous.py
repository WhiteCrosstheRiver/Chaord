"""Amorphous lifter: disordered network -> amorphous region program."""
from __future__ import annotations

import numpy as np

from ..cv.glass import ring_distribution
from ..io.frames import Frame
from ..lang.ir import (
    GeoChain, Name, PhysicsBlock, Program, ProvenanceBlock, Quantity,
    RegionBlock, ResidualBlock, ShAll, Statement, StrVal, SystemBlock, Tol,
)


def _strictly_wrapped(frame: Frame) -> Frame:
    """A copy of the frame wrapped into [0, L) with a strict upper edge.

    ``np.mod`` can return exactly L (or a hair above), and every KD tree in
    the classification paths (``typical_neighbor_distance``, the lattice fit,
    ``qbar``'s pair query) builds with ``boxsize=L`` and rejects such points
    outright.  Lifting tolerates marginally unwrapped input everywhere else;
    the phase gates must too (the failure used to be swallowed by the old
    blanket except clauses W1 deleted)."""
    L = frame.cell_diag
    return Frame(pos=np.minimum(np.mod(frame.pos, L), L * (1 - 1e-9)),  # dialect-exempt: numerical-guard: strict upper edge for KD trees
                 cell=frame.cell, symbols=frame.symbols, pbc=frame.pbc)


def network_edges(frame: Frame, dialect):
    """Bonds for the amorphous test: covalent when species are elements,
    geometric (a fraction of the typical NN distance) for placeholder species."""
    from ..build.molecules import bond_graph
    from ase.data import chemical_symbols
    if all(s in chemical_symbols and s != "X" for s in frame.symbols):
        return bond_graph(frame, dialect)
    from scipy.spatial import cKDTree
    from ..build.defects import typical_neighbor_distance
    L = frame.cell_diag
    pos = np.minimum(np.mod(frame.pos, L), L * (1 - 1e-9))  # dialect-exempt: numerical-guard: strict upper edge for KD trees
    rc = float(dialect.threshold("amorphous_bond_factor")) * typical_neighbor_distance(
        _strictly_wrapped(frame))
    return [tuple(e) for e in cKDTree(pos, boxsize=L).query_pairs(rc, output_type="ndarray")]


def crystal_like_fraction(frame: Frame, dialect) -> float:
    """Fraction of crystal-like atoms by the unit-free local-order rule (W1,
    Review 8): neighbours within `q_cutoff_factor` x the frame's own median
    d_NN; an atom with fewer than `q_min_neighbours` such neighbours is never
    crystal-like (a gas atom with one or two neighbours has q6bar near 1);
    crystal-like means q6bar > `q6_solid` OR q4bar > `q4_solid`.

    Every radius scales with the median d_NN `typical_neighbor_distance`
    measures on the frame, so the rule reads Angstrom crystals and LJ sigma
    frames identically -- there is no absolute cutoff left to get wrong
    (reviewer's matrix: prototype crystals 0.65-1.00, glasses/liquids/gases
    <= 0.12)."""
    from ..build.defects import typical_neighbor_distance
    from .passes import qbar
    frame = _strictly_wrapped(frame)
    L = frame.cell_diag
    pos = frame.pos
    rc = float(dialect.threshold("q_cutoff_factor")) * typical_neighbor_distance(frame)
    q6, cnt, _ = qbar(pos, L, l=6, rc=rc)
    q4, _, _ = qbar(pos, L, l=4, rc=rc)
    counted = cnt >= int(dialect.threshold("q_min_neighbours"))
    solid = ((q6 > float(dialect.threshold("q6_solid")))
             | (q4 > float(dialect.threshold("q4_solid"))))
    return float((counted & solid).mean())


# every threshold the lattice fit (`lift.defects.fit_crystal`, its helpers and
# the cell snap) reads; is_amorphous resolves them BEFORE the fit so a missing
# dialect key raises as itself, never disguised as the fit's no-fit verdict
_CRYSTAL_FIT_KEYS = (
    "site_match_tol_fraction", "defect_stoichiometry_slack",
    "lattice_fit_tol_fraction", "site_plausibility_floor",
    "lattice_scan_factor_lo", "lattice_scan_factor_hi", "lattice_scan_steps",
    "lattice_fit_gate_min", "lattice_refine_half", "lattice_refine_passes",
    "lattice_anchor_improvement", "stoichiometric_preference_min",
    "lattice_match_tolerance",
)


def _crystal_fit_coverage(frame: Frame, dialect):
    """Site coverage of the best crystal-prototype lattice fit, or None when
    no prototype fits the frame (W1 step 1).

    The fit is unit-free (every key is a fraction of the frame's d_NN; the
    keys are core.yaml defaults since W1, metal.yaml overrides), so it runs
    under every dialect stack.  Fail-closed: after the pre-check above, the
    only ChaordError `fit_crystal` can still raise is its designed
    "no cubic prototype fits the frame" refusal -- a missing threshold is a
    configuration error and must never be read as "disordered".

    Placeholder species (W7: the Kob-Andersen 'A'/'B' mixture) are not
    elements, and the prototype fitter maps species to atomic numbers for
    its stoichiometry test ('X' is ASE's dummy and maps to 0; 'A'/'B' do
    not map at all).  A prototype fit is not defined for such frames: the
    coverage is None and is_amorphous falls through to the unit-free
    local-order rule, which is species-agnostic."""
    from ase.data import chemical_symbols as _chem
    if not all(s in _chem for s in set(frame.symbols)):
        return None
    from ..lang.errors import ChaordError
    from .defects import fit_crystal
    for key in _CRYSTAL_FIT_KEYS:
        dialect.threshold(key)
    try:
        _name, _a, _slot_species, gate, _sites, _site_species = \
            fit_crystal(_strictly_wrapped(frame), dialect)
    except ChaordError:
        return None                       # the fit's explicit no-fit verdict
    return float(gate)


def is_amorphous(frame: Frame, dialect) -> bool:
    """Single phase, everywhere disordered, but with a bonded network.

    W1 (Review 8): unit-free and fail-closed.  The crystal lattice fit gates
    first -- a frame any prototype covers at or above
    `crystal_site_coverage_min` is a crystal, not a glass (perovskite and
    diamond read crystal-like below the local-order threshold and need this
    gate; hcp in a non-orthohexagonal box fits no cubic prototype and is
    caught by the local-order rule instead).  Disorder is then the
    unit-free local-order rule: crystal-like fraction below
    `amorphous_solid_frac_max`.  A missing threshold raises (the old
    ``except Exception: disordered = True`` is deleted)."""
    coverage = _crystal_fit_coverage(frame, dialect)
    if coverage is not None and coverage >= float(
            dialect.threshold("crystal_site_coverage_min")):
        return False
    edges = network_edges(frame, dialect)
    n = len(frame)
    # a network: most atoms have 3-6 covalent neighbours
    if not edges:
        return False
    deg = np.zeros(n, int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    network = float((deg >= 3).mean())  # any extended bonded structure
    disordered = (crystal_like_fraction(frame, dialect)
                  < float(dialect.threshold("amorphous_solid_frac_max")))
    return network > float(dialect.threshold('amorphous_network_min')) and disordered


def infer_mixture_model(frame: Frame, dialect):
    """Infer a named LJ-mixture model from the frame (W7 step 5, D9).

    The review's rule, verbatim: the composition and the sigma_AB/sigma_AA
    ratio read off the FIRST PARTIAL g(r) PEAKS name the model -- the
    potential itself is never measurable from one configuration, so the
    caller must print the model as ASSUMED.  Returns
    (model_name, detail_dict) or None when the frame is single-species, the
    dialect defines no lj_mixtures table, or no entry matches within the
    dialect's tolerances."""
    from ..lang.errors import ChaordError
    species = sorted(set(frame.symbols))
    if len(species) < 2:
        return None
    try:
        table = dialect.threshold("lj_mixtures")
        frac_tol = float(dialect.threshold("lj_mixture_fraction_tol"))
        ratio_tol = float(dialect.threshold("lj_mixture_sigma_ratio_tol"))
    except ChaordError:
        return None
    n = len(frame.symbols)
    fractions = {s: frame.symbols.count(s) / n for s in species}
    # sigma_AB/sigma_AA measured as the first-peak ratio of the two partials
    from ..cv.rich import partial_gr
    peaks = {}
    for a, b in ((species[0], species[0]), (species[0], species[1])):
        g, r = partial_gr(frame, dialect, a, b)
        peaks[f"{a}{b}"] = float(r[int(np.argmax(g))])
    ratio = peaks[f"{species[0]}{species[1]}"] / peaks[f"{species[0]}{species[0]}"]
    for name, entry in sorted(table.items()):
        if sorted(entry["species"]) != species:
            continue
        if any(abs(fractions[s] - float(entry["fractions"][s])) > frac_tol
               for s in species):
            continue
        sig = entry["sigma"]
        want = float(sig[_pairkey(species[0], species[1])]) / \
            float(sig[_pairkey(species[0], species[0])])
        if abs(ratio - want) > ratio_tol:
            continue
        return name, {"fractions": fractions, "peak_ratio": ratio,
                      "sigma_ratio": want}
    return None


def _pairkey(a: str, b: str) -> str:
    return a + b if a <= b else b + a


def _partial_coordination(frame: Frame, a: str, b: str, cutoff: float) -> float:
    """Mean number of b-neighbours within cutoff around each a-atom."""
    from scipy.spatial import cKDTree
    L = frame.cell_diag
    pos = np.mod(frame.pos, L)
    syms = np.asarray(frame.symbols)
    ta = cKDTree(pos[syms == a], boxsize=L)
    tb = cKDTree(pos[syms == b], boxsize=L)
    idx = ta.query_ball_tree(tb, cutoff)
    counts = np.array([len(x) for x in idx], float)
    if a == b:                       # unordered pairs: self and double count
        counts = np.maximum(counts - 1, 0)
    return float(counts.mean())


def lift_amorphous(frame: Frame, dialect, backend="lj") -> Program:
    counts: dict[str, int] = {}
    for s in frame.symbols:
        counts[s] = counts.get(s, 0) + 1
    conserve_values = []
    for s in sorted(counts):
        conserve_values += [Name(text=s), Quantity(num=str(counts[s]))]

    L = frame.cell_diag
    rho = len(frame.pos) / float(np.prod(L))
    region_stmts = []
    if len(counts) == 1:
        region_stmts.append(Statement(kind="build", key="composition",
                                      values=[Name(text=next(iter(counts)))]))
    else:
        vals = []
        for s in sorted(counts):
            vals += [Name(text=s), Quantity(num=str(counts[s]))]
        region_stmts.append(Statement(kind="build", key="composition", values=vals))
    region_stmts.append(Statement(kind="state", key="density",
                                  values=[Quantity(num=f"{rho:.4f}")]))
    # the protocol is the shortest description of how a glass is made: the
    # dialect's default melt-quench, in the backend's reduced units (LJ only).
    # These parameters are the dialect defaults, not anything measured from
    # the frame: the provenance note says the protocol was assumed.
    assumed_history = False
    if backend == "lj":
        from ..lang.ir import Arrow
        t_melt = float(dialect.threshold("glass_melt_T"))
        t_q = float(dialect.threshold("glass_quench_T"))
        t_a = float(dialect.threshold("glass_anneal_T"))
        # diffusive size scaling (same law as the reference generator): erase
        # the melt memory of an N-atom cell takes t_mix ~ N^(2/3) steps, so
        # the dialect's base counts (calibrated at glass_protocol_ref_n) are
        # scaled and the SCALED numbers stated in the history line -- the
        # rebuild then runs the protocol the frame's size requires
        ref_n = float(dialect.threshold("glass_protocol_ref_n"))
        scale = (len(frame) / ref_n) ** (2.0 / 3.0)  # dialect-exempt: exact-geometry: diffusive t_mix ~ N^(2/3) scaling law
        n_melt = int(round(int(dialect.threshold("glass_melt_steps")) * scale))
        n_q = int(round(int(dialect.threshold("glass_quench_steps")) * scale))
        n_a = int(round(int(dialect.threshold("glass_anneal_steps")) * scale))
        rate = round((t_melt - t_q) / n_q, 6)  # dialect-exempt: numerical-guard: printable cooling rate, T per step
        region_stmts.append(Statement(
            kind="history", key="melt",
            values=[Quantity(num=f"{t_melt:g}"), Name(text="for"),
                    Quantity(num=str(n_melt)), Arrow(),
                    Name(text="quench"), Name(text="to"), Quantity(num=f"{t_q:g}"),
                    Name(text="at"), Quantity(num=f"{rate:g}"), Arrow(),
                    Name(text="anneal"), Quantity(num=f"{t_a:g}"), Name(text="for"),
                    Quantity(num=str(n_a))]))
        assumed_history = True
    # held-out network statistics as asserts; placeholder species (X) use a
    # geometric cutoff (fraction of d_NN), real elements the dialect rule
    from ..cv.local import angle_mean, cn_mean
    from ase.data import chemical_symbols
    from ..build.defects import typical_neighbor_distance
    try:
        if all(s in chemical_symbols and s != "X" for s in frame.symbols):
            cut = float(dialect.threshold("cn_cutoff"))
            q_cut = Quantity(num=f"{cut:.2f}", unit="A")
        else:
            cut = float(dialect.threshold("amorphous_bond_factor")) * typical_neighbor_distance(frame)
            q_cut = Quantity(num=f"{cut:.2f}")
        cn = cn_mean(frame, dialect, cutoff=cut)
        ang = angle_mean(frame, dialect, cutoff=cut)
        region_stmts.append(Statement(
            kind="assert", key="cn",
            values=[Quantity(num=f"{cn:.2f}"), Tol(value=Quantity(num="0.30")),  # dialect-exempt: numerical-guard: canonical printed tolerance
                    Name(text="cutoff"), q_cut]))
        region_stmts.append(Statement(
            kind="assert", key="angle_mean",
            values=[Quantity(num=f"{ang:.1f}"), Tol(value=Quantity(num="3.0")),  # dialect-exempt: numerical-guard: canonical printed tolerance
                    Name(text="deg")]))
    except Exception:
        pass
    # W7 step 5 (D9): a multi-species amorphous region states its held-out
    # PARTIAL structure -- one g(r) first-peak and one partial coordination
    # assert per unordered species pair, measured at the same cutoffs
    if len(counts) > 1:
        from itertools import combinations_with_replacement
        from ..cv.rich import partial_gr
        from ..build.defects import typical_neighbor_distance as _tnd
        from ase.data import chemical_symbols as _chem
        real = all(s in _chem and s != "X" for s in frame.symbols)
        for a, b in combinations_with_replacement(sorted(counts), 2):
            try:
                cut_p = (float(dialect.threshold("cn_cutoff")) if real
                         else float(dialect.threshold("amorphous_bond_factor"))
                         * _tnd(frame))
                cn_ab = _partial_coordination(frame, a, b, cut_p)
                g, r = partial_gr(frame, dialect, a, b)
            except Exception:
                continue
            ipk = int(np.argmax(g))
            region_stmts.append(Statement(
                kind="assert", key="gr_peak",
                values=[Quantity(num=f"{r[ipk]:.2f}"), Name(text="height"),
                        Quantity(num=f"{g[ipk]:.2f}"), Name(text="pair"),
                        Name(text=f"{a}-{b}")]))
            region_stmts.append(Statement(
                kind="assert", key="cn",
                values=[Quantity(num=f"{cn_ab:.2f}"),
                        Tol(value=Quantity(num="0.30")),  # dialect-exempt: numerical-guard: canonical printed tolerance
                        Name(text="cutoff"),
                        Quantity(num=f"{cut_p:.2f}"), Name(text="pair"),
                        Name(text=f"{a}-{b}")]))
    rings = ring_distribution(frame, dialect, edges=network_edges(frame, dialect))
    if rings:
        dom = max(rings, key=lambda k: rings[k])
        region_stmts.append(Statement(
            kind="assert", key="ring_mode",
            values=[Quantity(num=str(dom))]))

    system = SystemBlock(statements=[
        Statement(kind="build", key="cell", values=[
            Quantity(num=f"{L[i]:.3f}") for i in range(3)]),
        Statement(kind="build", key="pbc", values=[Name(text="xyz")]),
        Statement(kind="conserve", key="atoms", values=conserve_values),
    ])
    # W7 step 5: the physics block names the mixture model, INFERRED from the
    # composition and the partial g(r) peak ratio and printed as assumed (a
    # potential is not measurable from one frame)
    model = infer_mixture_model(frame, dialect)
    physics_stmts = [Statement(kind="build", key="backend",
                               values=[Name(text=backend)])]
    model_notes = []
    if model is not None:
        name, det = model
        physics_stmts.append(Statement(kind="build", key="model",
                                       values=[Name(text=name)]))
        model_notes.append(
            f"model {name} assumed: inferred from composition "
            + " ".join(f"{s} {det['fractions'][s]:.3f}" for s in sorted(det["fractions"]))
            + f" and partial g(r) first-peak ratio r_AB/r_AA "
            f"{det['peak_ratio']:.2f} vs sigma_AB/sigma_AA "
            f"{det['sigma_ratio']:.2f} (the potential itself is never "
            "measurable from one frame)")
    region = RegionBlock(phase="amorphous", name="glass",
                         geometry=GeoChain(parts=[ShAll()], ops=[]),
                         statements=region_stmts)
    return Program(
        version="0.1", dialects=list(dialect.names),  # dialect-exempt: numerical-guard: language version constant
        blocks=[system,
                PhysicsBlock(statements=physics_stmts),
                region, ResidualBlock(none=True),
                ProvenanceBlock(statements=[
                    Statement(kind="build", key="dialects",
                              values=[StrVal(text=dialect.version_string)]),
                    Statement(kind="build", key="lift_version",
                              values=[StrVal(text="0.1.0")]),
                    *([Statement(kind="build", key="note",
                                  values=[StrVal(
                                      text="assumed default protocol from "
                                           "dialect")])]
                      if assumed_history else []),
                    *(Statement(kind="build", key="note",
                                values=[StrVal(text=note)])
                      for note in model_notes),
                ])])
