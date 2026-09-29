"""ASE realization backend: built-in ASE calculators behind the LJ interface.

Review 2 asked for an MD relaxation backend for the reference molecular cases
(water_tip4p, nacl_aq), which lift to `backend classical`. This module wraps
ASE's own calculators where one exists and re-implements published pair
kernels vectorized where ASE's would not finish inside the acceptance budget:

* water-water  : the TIP4P water kernel of ``ase.calculators.tip4p`` --
  same constants (its ``qH``/``sigma0``/``epsilon0``), same virtual-site
  construction, same per-molecule O-O shift and smooth truncation, same
  M-site force redistribution -- computed as one vectorized pass over a
  KD-tree pair list instead of ASE's per-molecule Python loop. The
  equivalence is pinned to machine precision by the tests (the water-only
  flavor is compared against ``ase.calculators.tip4p.TIP4P`` head to head);
  only the summation order (computation organization) differs. The classical
  flavors also realize rigid SPC/E water (three sites, O-site LJ and charge,
  published parameters from the dialect's ``water_models`` table,
  Berendsen/Grigera/Straatsma 1987): the model is resolved from the water
  geometry itself -- rigid MD conserves the O-H bond length, the same
  median-OH measurement and dialect windows the fluid lift classifies with
  (one definition, rule 7) -- or stated explicitly through ``model=``.
* water-ion and ion-ion terms of solutions: published Joung-Cheatham
  parameters with the same smooth O-distance truncation scheme as ASE's
  TIP4P (Lorentz-Berthelot LJ + Coulomb, virtual M site included, M-site
  force redistributed with ASE's formulas),
* molecular fluids without water (gases such as Ar, N2): site-wise LJ with
  published parameters from the dialect's ``classical_pair_potentials``
  table (monatomic species: one LJ site per atom; two-site species: one LJ
  site per atom, intramolecular pair excluded, bond held rigid), smooth
  truncation and Lorentz-Berthelot cross terms as everywhere here,
* tabulated EAM: ``ase.calculators.eam.EAM`` on the shipped
  ``bench/reference/potentials/Cu_u3.eam``,
* generic pair LJ: ``ase.calculators.lj.LennardJones``.

Two interfaces:

* the LJ-class one (``build(r)`` / ``forces(r)`` / ``energy(r)``, eV and A),
  so constraint-free flavors plug into ``realize.lj.run_md`` unchanged;
* ``relax(r, rng, ...)`` -- a deterministic FIRE + Langevin protocol
  (physical units, seeded rng). Rigid water is propagated with a vectorized
  FixBondLengths-equivalent constraint (``RigidWater`` below): SHAKE for
  positions, RATTLE for momenta/forces, the same correction formulas as
  ``ase.constraints.FixBondLengths`` but swept over bond types instead of
  individual bonds (ASE's per-bond Python loop costs ~1 s/step at the
  1600-atom solution scale; this costs milliseconds). Rigid diatomics (N2)
  ride the same sweep through ``RigidBonds``.

Calculator choice for a backend name comes from the dialect's optional
``ase_backend`` mapping (backend name -> flavor); without it the composition
decides (pure water -> TIP4P, water + parameterized ions -> the solution
calculator, no water and all species in the LJ table -> the LJ-fluid
calculator). MD protocol numbers likewise come from the optional ``ase_md``
dialect threshold; the molecular dialect ships one (its relax friction and
step count are the shipped MD references' own recorded protocol, sized to
the per-case acceptance budget -- see molecular.yaml), other dialects use
the module defaults below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.eam import EAM
from ase.calculators.lj import LennardJones
from ase.calculators.tip4p import epsilon0, qH, sigma0
from ase.constraints import FixConstraint
from ase.md.langevin import Langevin
from ase.optimize import FIRE

from ..lang.errors import ChaordError

KCAL = units.kcal / units.mol          # eV per (kcal/mol)   # dialect-exempt: unit conversion via ASE
KC = units.Hartree * units.Bohr        # Coulomb constant    # dialect-exempt: unit conversion via ASE
OM_DISTANCE = 0.15                     # dialect-exempt: published TIP4P O-M distance, A (Jorgensen 1983)

# Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), Table 3, SPC/E set:
# sigma in A, epsilon in kcal/mol. Exact published data (dialect-exempt per
# AGENTS.md, like the molecule template registry) and identical to the
# parameters recorded in bench/reference/nacl_aq/provenance.json.
# dialect-exempt-begin: exact-geometry (published Joung-Cheatham parameters)
ION_LJ = {
    "Na": (2.159542, 0.3526418),
    "Cl": (4.830486, 0.0300147),
}
# dialect-exempt-end
ION_CHARGE = {"Na": 1, "Cl": -1}       # dialect-exempt: integer ionic charges, e

# Water charges and LJ parameters are ASE's own TIP4P module constants
# (qH, sigma0, epsilon0); nothing is restated here.

# MD protocol defaults until a dialect defines an `ase_md` threshold
# (mirrors the recorded protocol of the shipped reference cases -- FIRE
# relaxation, Langevin NVT with dt 1 fs, friction 0.05/fs).
# dialect-exempt-begin: protocol-defaults (dialect ase_md section pending)
_MD_DEFAULTS = {
    "rc": 8.0,                 # A, pair cutoff on O/ion distances
    "width": 1.0,              # A, smooth truncation width
    "rc_margin": 0.05,         # A, safety gap below L/2 (ASE asserts 2 rc <= L)
    "fire_steps": 100,         # capped minimization of the packed start
    "fire_fmax": 0.5,          # eV/A
    "fast_steps": 120,         # strain-heat dump stage
    "fast_dt_fs": 0.5,
    "fast_gamma_per_fs": 0.05,
    "relax_steps": 600,        # production relaxation (md_steps overrides)
    "relax_dt_fs": 1.0,
    "relax_gamma_per_fs": 0.01,
    "reference_T_K": 300.0,    # stated when the program has no state T
}
# dialect-exempt-end


def _md_params(dialect) -> dict:
    """Protocol numbers: the dialect's `ase_md` threshold when it defines
    one, the module defaults otherwise."""
    params = dict(_MD_DEFAULTS)
    if dialect is not None:
        try:
            overrides = dialect.threshold("ase_md")
        except ChaordError:
            overrides = None
        if overrides:
            params.update({k: v for k, v in overrides.items() if k in params})
    return params


def _default_eam_potential() -> Path | None:
    """The shipped tabulated Cu potential (Foiles 1986, universal 3)."""
    p = (Path(__file__).resolve().parents[3] / "bench" / "reference"
         / "potentials" / "Cu_u3.eam")
    return p if p.exists() else None


def split_water_ions(symbols, dialect=None) -> tuple[np.ndarray, np.ndarray]:
    """Parse OHH water triples and monatomic ions out of a symbol list.

    Waters must appear as contiguous O,H,H triples (the packer's order);
    remaining symbols must be ions with parameters. Anything else is a
    static error naming the species the classical backend cannot realize
    (the message lists the LJ molecular species too when the dialect's
    ``classical_pair_potentials`` table defines any). Returns (water mask,
    ion mask)."""
    syms = list(symbols)
    water = np.zeros(len(syms), bool)
    ions = np.zeros(len(syms), bool)
    unknown: set[str] = set()
    i = 0
    while i < len(syms):
        if (syms[i] == "O" and i + 2 < len(syms)
                and syms[i + 1] == "H" and syms[i + 2] == "H"):
            water[i:i + 3] = True
            i += 3
        elif syms[i] in ION_LJ:
            ions[i] = True
            i += 1
        else:
            unknown.add(syms[i])
            i += 1
    if unknown:
        have = ["H2O (TIP4P/SPC/E)"] + [
            f"{s}{'+' if ION_CHARGE[s] > 0 else '-'} (Joung-Cheatham)"
            for s in sorted(ION_LJ)]
        have += [f"{el} (site-wise LJ)" for el in sorted(classical_pair_table(dialect))]
        raise ChaordError(
            f"no ASE classical potential for {', '.join(sorted(unknown))}; "
            f"parameterized: {', '.join(have)} (build with physics=False for "
            f"the packed frame alone; water does not mix with the LJ species)")
    return water, ions


def classical_pair_table(dialect) -> dict[str, tuple[float, float, int]]:
    """{element: (sigma [A], epsilon [eV], sites)} from the dialect's
    ``classical_pair_potentials`` threshold (empty when undefined).

    Published parameters live only in the dialect (AGENTS.md rule); epsilon
    is stated as epsilon/kB in K and converted here through ase units."""
    if dialect is None:
        return {}
    try:
        table = dialect.threshold("classical_pair_potentials")
    except ChaordError:
        return {}
    if not isinstance(table, dict):
        return {}
    return {el: (float(spec["sigma_A"]),
                 float(spec["epsilon_K"]) * units.kB,
                 int(spec.get("sites", 1)))
            for el, spec in table.items()}


def split_lj_species(symbols, dialect):
    """Per-atom LJ parameters and molecule grouping of a classical molecular
    fluid (the molecular dialect's ``classical_pair_potentials`` table).

    Monatomic species: one LJ site per atom, each atom its own molecule.
    Two-site species: contiguous equal-element pairs (the packer's order),
    one LJ site per atom, the intramolecular pair excluded by the molecule
    grouping, the bond held rigid by the backend's constraint. Returns
    (sigma, epsilon, molecule id per atom, (bond i, bond j) index arrays);
    raises a static error naming the elements no published parameters exist
    for (never silently)."""
    table = classical_pair_table(dialect)
    syms = list(symbols)
    sigma = np.zeros(len(syms))
    eps = np.zeros(len(syms))
    mol = np.zeros(len(syms), int)
    bond_i: list[int] = []
    bond_j: list[int] = []
    unknown: set[str] = set()
    i = 0
    mid = 0
    while i < len(syms):
        s = syms[i]
        if s not in table:
            unknown.add(s)
            i += 1
        elif table[s][2] == 1:
            sigma[i], eps[i] = table[s][0], table[s][1]
            mol[i] = mid
            mid += 1
            i += 1
        elif table[s][2] == 2:
            if i + 1 >= len(syms) or syms[i + 1] != s:
                raise ChaordError(
                    f"two-site classical species {s!r} needs contiguous {s},{s} "
                    f"pairs (the packer's order); symbol {i} is unpaired")
            sigma[i:i + 2] = table[s][0]
            eps[i:i + 2] = table[s][1]
            mol[i:i + 2] = mid
            bond_i.append(i)
            bond_j.append(i + 1)
            mid += 1
            i += 2
        else:
            raise ChaordError(
                f"classical pair table declares {table[s][2]} sites for {s!r}; "
                f"the ASE backend realizes 1 (atom-centred) or 2 (diatomic)")
    if unknown:
        raise ChaordError(
            f"no classical pair potential for {', '.join(sorted(unknown))}; "
            f"parameterized: "
            f"{', '.join(sorted(table))} (build with physics=False for the "
            f"packed frame alone)")
    return sigma, eps, mol, (np.array(bond_i, int), np.array(bond_j, int))


def _water_geometry(pos, widx, L):
    """Whole-molecule water geometry: O, H1, H2, the TIP4P virtual M site and
    the ASE redistribution factors (n-hat, gamma). H's are repaired to their
    O's image (virtual sites need whole molecules); the molecule geometry
    follows ase.calculators.tip4p (b = OM_DISTANCE on the HOH bisector)."""
    O = pos[widx[:, 0]]
    H1 = O + _mic(pos[widx[:, 1]] - O, L)
    H2 = O + _mic(pos[widx[:, 2]] - O, L)
    nvec = 0.5 * (H1 + H2) - O      # dialect-exempt: exact-geometry (HOH bisector)
    norm = np.linalg.norm(nvec, axis=1)
    nhat = nvec / norm[:, None]
    M = O + OM_DISTANCE * nhat
    gamma = OM_DISTANCE / norm      # ASE M-site redistribution factor
    return O, H1, H2, M, nhat, gamma


def _mic(d, L):
    return d - L * np.round(d / L)


def _cutoff(r, rc, width):
    """ASE TIP4P's smooth pair truncation: t(r) and dt/dr. width = 0 is the
    sharp cut (unit factor inside, zero derivative)."""
    r = np.asarray(r, float)
    t = np.zeros_like(r)
    dtdd = np.zeros_like(r)
    inside = r < rc
    switch = (r > rc - width) & inside
    t[inside] = 1.0  # dialect-exempt: exact-geometry (unit cutoff factor)
    if width > 0:  # sharp cut when the width is zero
        y = (r[switch] - rc + width) / width
        t[switch] -= y * y * (3.0 - 2.0 * y)          # dialect-exempt: exact-geometry (ASE scheme)
        dtdd[switch] -= 6.0 / width * y * (1.0 - y)   # dialect-exempt: exact-geometry (ASE scheme)
    return t, dtdd


def _scatter_add(target, idx, vec):
    """target[idx] += vec, accumulated with np.bincount: same sums as
    np.add.at (a different summation order, hence different last bits, but
    deterministic) at a fraction of its cost on thousands of rows."""
    for ax in range(3):
        target[:, ax] += np.bincount(idx, weights=vec[:, ax],
                                     minlength=len(target))


def _scatter_sub(target, idx, vec):
    """target[idx] -= vec (the _scatter_add mirror)."""
    for ax in range(3):
        target[:, ax] -= np.bincount(idx, weights=vec[:, ax],
                                     minlength=len(target))


class TIP4PSolution(Calculator):
    """Rigid TIP4P or SPC/E water + monatomic ions, fully vectorized.

    Water-water energy and forces are the formulas of
    ``ase.calculators.tip4p.TIP4P`` (its constants, its virtual-site
    construction, its per-molecule O-O lattice shift and smooth truncation,
    its M-site force redistribution), computed as one vectorized pass over a
    KD-tree pair list instead of the ASE calculator's per-molecule Python
    loop; a system without ions reproduces the ASE calculator to machine
    precision (pinned by the tests).

    With ``model="spce"`` (or a frame whose water geometry the dialect's
    ``water_models`` table classifies as SPC/E) the water terms are the
    published three-site SPC/E potential instead: LJ and charge on O,
    charge on H, no M site (parameters from the dialect table; Berendsen,
    Grigera & Straatsma, J. Phys. Chem. 91, 6269 (1987)). The model is
    resolved once, from the explicit ``model=`` argument or from the
    configuration's own median O-H distance (a conserved quantity under
    rigid MD -- the same measurement the fluid lift classifies frames
    with), and cached per run: the builder's ASEBackend call sites hand
    over packed positions, whose template geometry already names the model.

    Ion-ion and ion-water interactions follow the reference potential family
    of the shipped cases: Lennard-Jones with Lorentz-Berthelot mixing and
    Coulomb electrostatics on the water model's site charges (for TIP4P
    including the virtual M site, its force redistributed to O/H by ASE's
    formulas), smoothly truncated on the defining O/ion distance exactly
    like ase.calculators.tip4p."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, rc=None, width=None, dialect=None, model=None):
        Calculator.__init__(self)
        self.rc = float(rc) if rc is not None else float(_MD_DEFAULTS["rc"])
        self.width = (float(width) if width is not None
                      else float(_MD_DEFAULTS["width"]))
        self.dialect = dialect
        self._model = self._validate_model(model)
        self._model_resolved = self._model is not None
        self._masks = None    # (water, ion) masks, cached per composition
        self._dsf_atoms = None  # (sigma, eps, charge, molecule id), same cache

    # ------------------------------------------------------ water model --
    def _validate_model(self, model):
        if model is None or model == "tip4p":
            return model
        from ..build.molecules import water_model_table
        try:
            table = water_model_table(self.dialect)
        except ChaordError:
            table = {}
        known = sorted(k for k, v in table.items()
                       if isinstance(v, dict) and "sigma_O_A" in v)
        if model not in known:
            raise ChaordError(
                f"unknown water model {model!r}; parameterized in the "
                f"dialect's water_models table: {', '.join(known) or 'none'}"
                f" (TIP4P is the built-in default)")
        return model

    def _measure_oh_median(self, pos, widx, L):
        """Median minimum-image O-H length over the configuration's water
        triples (the frame invariant the dialect's water_models windows are
        defined on)."""
        O = pos[widx[:, 0]]
        d = np.concatenate([
            np.linalg.norm(_mic(pos[widx[:, 1]] - O, L), axis=1),
            np.linalg.norm(_mic(pos[widx[:, 2]] - O, L), axis=1)])
        return float(np.median(d)) if len(d) else None

    def _resolve_model(self, pos, widx, L) -> str:
        """Model selection: explicit ``model=`` wins; otherwise the dialect
        table classifies the measured median O-H (default on no table, no
        water or no match -- consistent with the whole chain's tip4p
        default). The table resolves under any dialect combination (the
        molecular table is the single source; see molecules.water_model_table)."""
        if self._model is not None:
            return self._model
        from ..build.molecules import classify_water_model, default_water_model
        if self.dialect is None:
            return "tip4p"
        med = self._measure_oh_median(pos, widx, L)
        if med is None:
            return default_water_model(self.dialect)
        try:
            model = classify_water_model(self.dialect, med)
        except ChaordError:
            model = None
        return model if model is not None else default_water_model(self.dialect)

    @property
    def model_name(self) -> str:
        """The resolved water model (tip4p before any water configuration
        has been seen, unless stated explicitly)."""
        if self._model_resolved:
            return self._model
        return "tip4p"

    def _water_params(self):
        """(site charges, O-LJ sigma, O-LJ epsilon, n_sites) of the model.

        TIP4P's constants are ASE's tip4p module's own (nothing restated);
        every other model's come from the water-model table (the molecular
        dialect's, resolved under any dialect combination)."""
        if self._model == "tip4p":
            return (np.array([0.0, qH, qH, -2.0 * qH]),  # dialect-exempt: TIP4P site charges via ASE's qH
                    sigma0, epsilon0, 4)
        from ..build.molecules import water_model_table
        spec = water_model_table(self.dialect)[self._model]
        return (np.array([float(spec["q_O_e"]), float(spec["q_H_e"]),
                          float(spec["q_H_e"])]),
                float(spec["sigma_O_A"]),
                float(spec["epsilon_O_K"]) * units.kB, 3)

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        pos = at.positions
        L = np.asarray(at.cell.lengths(), float)
        n = len(at)
        if self._masks is None:    # the composition never changes mid-run
            self._masks = split_water_ions(at.get_chemical_symbols(),
                                           self.dialect)
        water, ion_mask = self._masks
        forces = np.zeros((n, 3))
        energy = 0.0  # dialect-exempt: numerical-guard: zero accumulator

        # --------------------------------------------- water-water: TIP4P --
        if water.any():
            widx = np.where(water)[0].reshape(-1, 3)
            if not self._model_resolved:
                self._model = self._resolve_model(pos, widx, L)
                self._model_resolved = True
            if self._model == "tip4p":
                energy += self._water_water(pos, widx, L, forces)
            else:
                # three-site models: one DSF pass covers water-water AND the
                # ion terms (the reference family's own summation)
                energy += self._dsf_site_terms(pos, L, forces)
        elif ion_mask.any():
            raise ChaordError(
                "the classical ASE backend realizes water (TIP4P/SPC/E) "
                "and its ions; an ion-only system has no water to solvate")

        if ion_mask.any() and self._model == "tip4p":
            energy += self._ion_terms(pos, water, ion_mask, L, forces,
                                      at.get_chemical_symbols())

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

    def _water_water(self, pos, widx, L, forces) -> float:
        """Water-water TIP4P terms, ase.calculators.tip4p formulas vectorized
        over a KD-tree O-O pair list: adds forces in place, returns energy.

        Every pair is cut on the O-O distance (ASE's scheme, molecule
        geometry whole via the O's lattice shift): LJ on O-O with ASE's
        sigma0/epsilon0, Coulomb between all 4x4 site pairs (including the
        virtual M sites), the cutoff derivative carried by the O-O axis, and
        the M-site force redistributed onto O/H with ASE's formulas."""
        O, H1, H2, M, nhat, gamma = _water_geometry(pos, widx, L)
        Oin = np.mod(O, L)
        pairs = cKDTree(Oin, boxsize=L).query_pairs(
            self.rc, output_type="ndarray")
        if not len(pairs):
            return 0.0  # dialect-exempt: numerical-guard: zero accumulator
        a, b = pairs[:, 0], pairs[:, 1]
        D = Oin[b] - Oin[a]
        D -= L * np.round(D / L)                 # O -> O, min image (ASE shift)
        r = np.linalg.norm(D, axis=1)
        t, dtdd = _cutoff(r, self.rc, self.width)
        Rhat = D / r[:, None]

        # LJ on O-O (ASE TIP4P: epsilon0/sigma0, cutoff on r_OO)
        s6 = (sigma0 / r) ** 6                   # dialect-exempt: exact-geometry (LJ)
        u = 4.0 * epsilon0 * (s6 * s6 - s6)      # dialect-exempt: exact-geometry (LJ)
        dudd = -24.0 * epsilon0 * (2.0 * s6 * s6 - s6) / r  # dialect-exempt: exact-geometry (dLJ/dr)
        energy = float(np.sum(u * t))

        # Coulomb on the 4x4 site pairs; every site of molecule b carries the
        # O-O lattice shift (ASE scheme). The site-site vector is
        # X[p, k, l] = D + dB[l] - dA[k] (site offsets from each O); its
        # length comes from the quadratic expansion and the force sums from
        # reductions over (k, l), never materializing the (P, 4, 4, 3)
        # tensor (the same sums, a fraction of the memory traffic).
        q = np.array([0.0, qH, qH, -2.0 * qH])   # dialect-exempt: TIP4P site charges via ASE's qH
        dA = np.stack([np.zeros_like(D), H1[a] - O[a], H2[a] - O[a],
                       OM_DISTANCE * nhat[a]], axis=1)
        dB = np.stack([np.zeros_like(D), H1[b] - O[b], H2[b] - O[b],
                       OM_DISTANCE * nhat[b]], axis=1)
        DA = (D[:, None, :] * dA).sum(-1)            # (P, 4)  D . dA[k]
        DB = (D[:, None, :] * dB).sum(-1)            # (P, 4)  D . dB[l]
        AB = np.matmul(dA, dB.transpose(0, 2, 1))    # (P, 4, 4)  dA[k] . dB[l]
        d2 = ((r * r)[:, None, None]
              + (dA * dA).sum(-1)[:, :, None]        # dialect-exempt: exact-geometry (site distances)
              + (dB * dB).sum(-1)[:, None, :]
              - 2 * DA[:, :, None] + 2 * DB[:, None, :] - 2 * AB)
        dsite = np.sqrt(d2)
        e = KC * q[None, :, None] * q[None, None, :] / dsite
        energy += float(np.sum(e * t[:, None, None]))

        # site-site pair forces: coefficient ccoef = e t / d^2 along X (ASE's
        # (e/d * t) * D/d), reduced algebraically:
        #   sum_k ccoef X = (sum_k ccoef) (D + dB[l]) - sum_k ccoef dA[k]
        #   sum_l ccoef X = (sum_l ccoef) (D - dA[k]) + sum_l ccoef dB[l]
        ccoef = e / d2 * t[:, None, None]            # dialect-exempt: exact-geometry (Coulomb force)
        cs_k = ccoef.sum(1)                          # (P, 4) sums over k
        cs_l = ccoef.sum(2)                          # (P, 4) sums over l
        fsB = cs_k[:, :, None] * (D[:, None, :] + dB) \
            - np.matmul(ccoef.transpose(0, 2, 1), dA)
        fsA = -(cs_l[:, :, None] * (D[:, None, :] - dA)
                + np.matmul(ccoef, dB))

        # cutoff-derivative force on the O-O axis, once per site k of a
        # (ASE: FOO = -(e_f dtdd) DOO/d, + on b's O, - on a's O)
        axis = -(e.sum(2) * dtdd[:, None])[:, :, None] * Rhat[:, None, :]
        fsB[:, 0] += axis.sum(1)
        fsA[:, 0] -= axis.sum(1)

        # LJ pair force on the O's (ASE formula)
        flj = -(dudd * t + u * dtdd)[:, None] * Rhat  # dialect-exempt: exact-geometry (LJ + cutoff)
        fsB[:, 0] += flj
        fsA[:, 0] -= flj

        # redistribute the M-site row onto O, H1, H2 (ASE formulas)
        Fd = fsA[:, 3].copy()
        rid = OM_DISTANCE * nhat[a]
        proj = (rid * Fd).sum(1) / (rid * rid).sum(1)
        Fd_perp = Fd - proj[:, None] * rid
        fsA[:, 0] += Fd - gamma[a][:, None] * Fd_perp
        fsA[:, 1] += 0.5 * gamma[a][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)
        fsA[:, 2] += 0.5 * gamma[a][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)
        Fd = fsB[:, 3].copy()
        rid = OM_DISTANCE * nhat[b]
        proj = (rid * Fd).sum(1) / (rid * rid).sum(1)
        Fd_perp = Fd - proj[:, None] * rid
        fsB[:, 0] += Fd - gamma[b][:, None] * Fd_perp
        fsB[:, 1] += 0.5 * gamma[b][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)
        fsB[:, 2] += 0.5 * gamma[b][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)

        for s in range(3):   # the M-site rows (s=3) were redistributed
            _scatter_add(forces, widx[a, s], fsA[:, s])
            _scatter_add(forces, widx[b, s], fsB[:, s])
        return energy

    def _dsf_site_terms(self, pos, L, forces) -> float:
        """All site-site terms (water-water, ion-water, ion-ion) of the
        three-site models, as the shipped reference family realizes them:
        one pair list over every atom (KD tree, intramolecular pairs
        excluded), Lennard-Jones between LJ sites (Lorentz-Berthelot,
        energy-shifted at rc, force unshifted) and damped shifted force
        (Wolf/DSF) electrostatics on every charged site pair.

        The scheme and its parameters (alpha = 0.2 1/A, rc = 9 A) are the
        recorded protocol of the nacl_aq reference (Fennell & Gezelter,
        J. Chem. Phys. 124, 234104 (2006)); plain truncated Coulomb measurably
        over-coordinates this reference (continuing one of its own frames
        drifts the mean coordination +0.5), so the realization speaks the
        reference's published truncation, not an approximation of it. rc and
        alpha are read from the dialect's water_models table and rc is capped
        at half the box. Adds forces in place, returns energy."""
        from scipy.special import erfc
        from ..build.molecules import water_model_table
        spec = water_model_table(self.dialect)[self._model]
        alpha = float(spec["dsf_alpha_inv_A"])
        rc = min(float(spec["realization_rc_A"]),
                 0.5 * float(L.min()) - float(_MD_DEFAULTS["rc_margin"]))  # dialect-exempt: exact-geometry (half cell)
        n = len(pos)
        if self._dsf_atoms is None or len(self._dsf_atoms[0]) != n:
            # per-atom sigma/epsilon/charge/molecule ids: composition-only,
            # cached like the masks (they never change mid-run)
            water, ion_mask = self._masks
            widx = np.where(water)[0].reshape(-1, 3)
            iidx = np.where(ion_mask)[0]
            qsite, sig_w, eps_w, _nsites = self._water_params()
            syms = list(self.atoms.get_chemical_symbols())
            sigma = np.zeros(n)
            eps = np.zeros(n)
            qq = np.zeros(n)
            mol = np.full(n, -1, int)
            sigma[widx[:, 0]] = sig_w    # LJ on O only (SPC/E: H carries none)
            eps[widx[:, 0]] = eps_w
            qq[widx] = qsite             # [q_O, q_H, q_H] broadcast over OHH
            mol[widx] = np.arange(len(widx))[:, None]
            sigma[iidx] = [ION_LJ[syms[k]][0] for k in iidx]
            eps[iidx] = [ION_LJ[syms[k]][1] * KCAL for k in iidx]
            qq[iidx] = [ION_CHARGE[syms[k]] for k in iidx]
            mol[iidx] = len(widx) + np.arange(len(iidx))
            self._dsf_atoms = (sigma, eps, qq, mol)

        sigma, eps, qq, mol = self._dsf_atoms
        pos = np.mod(pos, L)
        pos = np.minimum(pos, L * (1 - 1e-9))  # dialect-exempt: strict upper edge for KD trees
        pairs = cKDTree(pos, boxsize=L).query_pairs(rc, output_type="ndarray")
        if not len(pairs):
            return 0.0  # dialect-exempt: numerical-guard: zero accumulator
        i, j = pairs[:, 0], pairs[:, 1]
        keep = mol[i] != mol[j]                # no intramolecular terms
        i, j = i[keep], j[keep]
        # min-image on the pair vectors, in place (250k x 3 arrays: every
        # avoided temporary is measurable at this size)
        D = pos[j]
        D -= pos[i]
        T = D / L
        np.rint(T, out=T)
        T *= L
        D -= T
        r = np.linalg.norm(D, axis=1)

        # LJ, Lorentz-Berthelot, energy-shifted at rc, force unshifted
        # (the recorded reference scheme; the force jump at rc is < 1e-4 eV/A).
        # H carries no LJ site (eps 0), which zeroes every H-involving term.
        # Powers by repeated multiplication: at 250k pairs the pow ufunc is
        # measurably slower than three multiplies.
        s_ij = 0.5 * (sigma[i] + sigma[j])   # dialect-exempt: exact-geometry (Lorentz-Berthelot)
        e_ij = np.sqrt(eps[i] * eps[j])
        q = s_ij / r
        q2 = q * q
        s6 = q2 * q2 * q2                    # (sigma/r)^6
        qc = s_ij / rc
        qc2 = qc * qc
        s6c = qc2 * qc2 * qc2                # (sigma/rc)^6
        u_lj = 4.0 * e_ij * (s6 * s6 - s6 - (s6c * s6c - s6c))          # dialect-exempt: exact-geometry (LJ, rc-shifted)
        f_lj = 24.0 * e_ij * (2.0 * s6 * s6 - s6) / (r * r)             # dialect-exempt: exact-geometry (dLJ/dr)

        # damped shifted force electrostatics (see the docstring)
        qiqj = KC * qq[i] * qq[j]
        ar = alpha * r
        erfc_r = erfc(ar)
        f_r = erfc_r / r
        f_rc = erfc(alpha * rc) / rc
        erfc_coef = 2.0 * alpha / np.sqrt(np.pi)  # dialect-exempt: exact-geometry (DSF erfc derivative)
        g_rc = f_rc / rc + erfc_coef * np.exp(-(alpha * rc) ** 2) / rc
        u_el = qiqj * (f_r - f_rc + (r - rc) * g_rc)   # dialect-exempt: exact-geometry (DSF energy)
        h_r = -erfc_coef * np.exp(-ar * ar) / r - erfc_r / (r * r) + g_rc  # dialect-exempt: exact-geometry (DSF force)

        # both pair terms act along D: one shared coefficient, one product
        coef = (qiqj * h_r / r) - f_lj
        fvec = coef[:, None] * D
        _scatter_add(forces, i, fvec)
        _scatter_sub(forces, j, fvec)
        return float(u_lj.sum() + u_el.sum())

    # ------------------------------------------------------------ ion glue --
    def _ion_terms(self, pos, water, ion_mask, L, forces, sym) -> float:
        """Ion-ion + ion-water LJ/Coulomb: adds forces in place, returns the
        energy. All terms share the ASE TIP4P smooth cutoff on the pair's
        defining distance (O-ion for ion-water, ion-ion otherwise). The
        water side follows the resolved model's site charges and O-LJ
        constants (TIP4P: ASE's own; SPC/E: the dialect's published pair)."""
        widx = np.where(water)[0].reshape(-1, 3)
        iidx = np.where(ion_mask)[0]
        nw = len(widx)

        # whole-molecule water geometry and virtual M sites (ASE construction)
        qsite, sig_w, eps_w, nsites = self._water_params()
        O, H1, H2, M, nhat, gamma = _water_geometry(pos, widx, L)
        nvec = nhat

        sig_i = np.array([ION_LJ[sym[k]][0] for k in iidx])
        eps_i = np.array([ION_LJ[sym[k]][1] * KCAL for k in iidx])
        q_i = np.array([ION_CHARGE[sym[k]] for k in iidx], float)

        ipos = np.mod(pos[iidx], L)
        opos = np.mod(O, L)
        energy = 0.0  # dialect-exempt: numerical-guard: zero accumulator

        # -------------------------------------------------------- ion-ion --
        tree = cKDTree(ipos, boxsize=L)
        pairs = tree.query_pairs(self.rc, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            D = ipos[b] - ipos[a]
            D -= L * np.round(D / L)
            r = np.linalg.norm(D, axis=1)
            t, dtdd = _cutoff(r, self.rc, self.width)
            s6 = (0.5 * (sig_i[a] + sig_i[b]) / r) ** 6  # dialect-exempt: exact-geometry (Lorentz-Berthelot)
            e_ij = np.sqrt(eps_i[a] * eps_i[b])
            u_lj = 4.0 * e_ij * (s6 * s6 - s6)                 # dialect-exempt: exact-geometry (LJ)
            du_lj = -24.0 * e_ij * (2.0 * s6 * s6 - s6) / r    # dialect-exempt: exact-geometry (dLJ/dr)
            u_c = KC * q_i[a] * q_i[b] / r
            energy += float(np.sum((u_lj + u_c) * t))
            g = (t * (du_lj - u_c / r) + dtdd * (u_lj + u_c)) / r  # dialect-exempt: exact-geometry
            np.add.at(forces, iidx[a], g[:, None] * D)
            np.subtract.at(forces, iidx[b], g[:, None] * D)

        # --------------------------------------------------- ion-water --
        centers = np.vstack([opos, ipos])
        ctree = cKDTree(centers, boxsize=L)
        cpairs = ctree.query_pairs(self.rc, output_type="ndarray")
        if len(cpairs):
            lo, hi = cpairs[:, 0], cpairs[:, 1]
            cross_mask = (lo < nw) != (hi < nw)    # exactly one side is water
            wm = np.where(lo < nw, lo, hi)[cross_mask]         # water index
            ki = (np.where(lo < nw, hi, lo) - nw)[cross_mask]  # ion index
            ion = ipos[ki]

            R00 = ion - O[wm]
            shift = L * np.round(R00 / L)
            R = R00 - shift                        # O -> ion, min image
            r = np.linalg.norm(R, axis=1)
            t, dtdd = _cutoff(r, self.rc, self.width)
            Rhat = R / r[:, None]

            # LJ ion-O on the pair axis (Lorentz-Berthelot with the model's
            # water O-LJ constants)
            s6 = (0.5 * (sig_i[ki] + sig_w) / r) ** 6  # dialect-exempt: exact-geometry (Lorentz-Berthelot)
            e_ij = np.sqrt(eps_i[ki] * eps_w)
            u_lj = 4.0 * e_ij * (s6 * s6 - s6)                 # dialect-exempt: exact-geometry (LJ)
            du_lj = -24.0 * e_ij * (2.0 * s6 * s6 - s6) / r    # dialect-exempt: exact-geometry (dLJ/dr)

            # Coulomb ion vs the water model's sites; every site of the
            # molecule carries the O's lattice shift (ASE scheme). TIP4P has
            # four sites (the virtual M included), the three-site models
            # three -- and no M-site row to redistribute.
            site_pos = [O[wm], H1[wm], H2[wm]] + ([M[wm]] if nsites == 4 else [])
            sites = np.stack(site_pos, axis=1) - shift[:, None]
            X = sites - ion[:, None, :]            # ion -> site vectors
            dsite = np.linalg.norm(X, axis=2)
            e = KC * q_i[ki][:, None] * qsite[None, :] / dsite
            energy += float(np.sum(u_lj * t) + np.sum(e * t[:, None]))

            # axis coefficient shared by the LJ term and the cutoff term
            axis = (t * du_lj + dtdd * (u_lj + e.sum(1)))[:, None] * Rhat

            # per-site Coulomb pair force along the ion->site direction
            csite = (t[:, None] * e / dsite ** 2)[:, :, None] * X
            fs = csite.copy()
            fs[:, 0] += axis                       # O carries the axis term
            fion = -csite.sum(1) - axis            # ion: reaction + axis

            # redistribute the M-site force onto O, H1, H2 (ASE formulas;
            # three-site models have no M row)
            if nsites == 4:
                Fd = fs[:, 3]
                rid = OM_DISTANCE * nvec[wm]
                proj = (rid * Fd).sum(1) / (rid * rid).sum(1)
                Fd_perp = Fd - proj[:, None] * rid
                fs[:, 0] += Fd - gamma[wm][:, None] * Fd_perp
                fs[:, 1] += 0.5 * gamma[wm][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)
                fs[:, 2] += 0.5 * gamma[wm][:, None] * Fd_perp  # dialect-exempt: exact-geometry (ASE redistribution)

            for s in range(3):   # the M-site row (s=3, TIP4P) was redistributed
                _scatter_add(forces, widx[wm, s], fs[:, s])
            np.add.at(forces, iidx[ki], fion)
        return energy


class LJFluid(Calculator):
    """Site-wise Lennard-Jones for the dialect's classical molecular species.

    Published parameters (the molecular dialect's
    ``classical_pair_potentials`` table, keyed by element): monatomic
    species carry one LJ site per atom; two-site species carry one LJ site
    per atom with the intramolecular pair excluded (the bond is held rigid
    by the backend's ``RigidBonds`` constraint). Cross terms use
    Lorentz-Berthelot mixing; the cutoff is the same smooth truncation as
    ASE's TIP4P, on the site-site distance. Physical units (eV, A)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, sigma, epsilon, mol_id, rc=None, width=None):
        Calculator.__init__(self)
        self.sigma = np.asarray(sigma, float)
        self.epsilon = np.asarray(epsilon, float)
        self.mol_id = np.asarray(mol_id, int)
        self.rc = float(rc) if rc is not None else float(_MD_DEFAULTS["rc"])
        self.width = (float(width) if width is not None
                      else float(_MD_DEFAULTS["width"]))

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        L = np.asarray(at.cell.lengths(), float)
        pos = np.mod(at.positions, L)
        forces = np.zeros((len(at), 3))
        energy = 0.0  # dialect-exempt: numerical-guard: zero accumulator
        pairs = cKDTree(pos, boxsize=L).query_pairs(
            self.rc, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            keep = self.mol_id[a] != self.mol_id[b]   # no intramolecular LJ
            a, b = a[keep], b[keep]
            if len(a):
                D = pos[b] - pos[a]
                D -= L * np.round(D / L)
                r = np.linalg.norm(D, axis=1)
                t, dtdd = _cutoff(r, self.rc, self.width)
                s6 = (0.5 * (self.sigma[a] + self.sigma[b]) / r) ** 6  # dialect-exempt: exact-geometry (Lorentz-Berthelot)
                e_ij = np.sqrt(self.epsilon[a] * self.epsilon[b])
                u = 4.0 * e_ij * (s6 * s6 - s6)                 # dialect-exempt: exact-geometry (LJ)
                dudd = -24.0 * e_ij * (2.0 * s6 * s6 - s6) / r  # dialect-exempt: exact-geometry (dLJ/dr)
                energy = float(np.sum(u * t))
                g = t * dudd + dtdd * u               # dialect-exempt: exact-geometry (LJ + cutoff)
                fpair = g[:, None] / r[:, None] * D
                _scatter_add(forces, a, fpair)
                _scatter_sub(forces, b, fpair)
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces


class RigidPairs(FixConstraint):
    """FixBondLengths-equivalent SHAKE/RATTLE, vectorized by bond type.

    The same corrections as ase.constraints.FixBondLengths (position:
    x = (d0^2 - d1^2) / (2 d0.d1) with the reduced-mass partition; momentum:
    x = -(dv.d)/d^2), but the Gauss-Seidel sweep runs over bond TYPES as
    vectorized passes over all bonds of a type instead of ASE's Python loop
    over every single bond. ``pairs`` is a list of (a-indices, b-indices)
    arrays, one entry per bond type."""

    maxiter = 500

    def __init__(self, pairs, tolerance=1e-13):  # dialect-exempt: numerical-guard: ASE FixBondLengths default
        self.pairs = [(np.asarray(a, int), np.asarray(b, int))
                      for a, b in pairs]
        self.tolerance = tolerance
        self.bondlengths = None

    def get_indices(self, atoms=None):
        return np.unique(np.concatenate(
            [a for a, _ in self.pairs] + [b for _, b in self.pairs]))

    def _init_lengths(self, atoms):
        pos = atoms.positions
        L = np.asarray(atoms.cell.lengths(), float)
        self.bondlengths = [np.linalg.norm(_mic(pos[b] - pos[a], L), axis=1)
                            for a, b in self.pairs]
        masses = atoms.get_masses()
        wa = np.array([1.0 / masses[a[0]] for a, _ in self.pairs])   # dialect-exempt: exact-geometry (SHAKE)
        wb = np.array([1.0 / masses[b[1]] for _, b in self.pairs])   # dialect-exempt: exact-geometry (SHAKE)
        self._wa, self._wb = wa, wb
        self._wm = 1.0 / (wa + wb)  # dialect-exempt: exact-geometry (reduced mass)
        self._cd2 = [cd ** 2 for cd in self.bondlengths]

    def adjust_positions(self, atoms, new):
        old = atoms.positions
        L = np.asarray(atoms.cell.lengths(), float)
        if self.bondlengths is None:
            self._init_lengths(atoms)
        wa, wb, wm = self._wa, self._wb, self._wm
        # loop invariants: old positions and their minimum images
        r0s = [old[ia] - old[ib] for ia, ib in self.pairs]
        d0s = [_mic(r0, L) for r0 in r0s]
        for _ in range(self.maxiter):
            converged = True
            for t, (ia, ib) in enumerate(self.pairs):
                d1 = new[ia] - new[ib] - r0s[t] + d0s[t]
                d0 = d0s[t]
                d0d1 = (d0 * d1).sum(1)
                d1d1 = (d1 * d1).sum(1)
                x = 0.5 * (self._cd2[t] - d1d1) / d0d1  # dialect-exempt: exact-geometry (SHAKE)
                act = np.abs(x) > self.tolerance
                if act.any():
                    corr = (x * act * wm[t])[:, None] * d0
                    new[ia] += corr * wa[t]
                    new[ib] -= corr * wb[t]
                    converged = False
            if converged:
                return
        raise RuntimeError(f"{type(self).__name__}: positions did not converge")

    def adjust_momenta(self, atoms, p):
        old = atoms.positions
        L = np.asarray(atoms.cell.lengths(), float)
        if self.bondlengths is None:
            self._init_lengths(atoms)
        wm = self._wm
        masses = atoms.get_masses()
        # loop invariant: minimum-image bond vectors of the old positions
        ds = [_mic(old[ia] - old[ib], L) for ia, ib in self.pairs]
        for _ in range(self.maxiter):
            converged = True
            for t, (ia, ib) in enumerate(self.pairs):
                d = ds[t]
                dv = p[ia] / masses[ia][:, None] - p[ib] / masses[ib][:, None]
                dvd = (dv * d).sum(1)
                x = -dvd / self._cd2[t]    # dialect-exempt: exact-geometry (RATTLE)
                act = np.abs(x) > self.tolerance
                if act.any():
                    corr = (x * act * wm[t])[:, None] * d
                    p[ia] += corr
                    p[ib] -= corr
                    converged = False
            if converged:
                return
        raise RuntimeError(f"{type(self).__name__}: momenta did not converge")

    def adjust_forces(self, atoms, forces):
        self.constraint_forces = -forces.copy()
        self.adjust_momenta(atoms, forces)
        self.constraint_forces += forces


class RigidWater(RigidPairs):
    """OHH water triples as three bond types (O-H1, O-H2, H1-H2).

    ``widx`` are the global indices of the OHH triples (the packer emits
    contiguous triples; ions may sit anywhere).

    The three constraints of a water triangle are solved COUPLED (one batched
    3x3 Newton round per call) instead of swept Gauss-Seidel: the sweep that
    ``RigidPairs`` inherits needs tens of passes at SHAKE tolerances because
    each pass only partially untangles the shared-atom couplings. The fixed
    point is the same (bonds exactly restored, momenta orthogonal to them,
    centre of mass untouched); only the iteration path to it differs. A
    singular or unconverged solve falls back to the swept version."""

    def __init__(self, widx, tolerance=1e-13):  # dialect-exempt: numerical-guard: ASE FixBondLengths default
        self.widx = np.asarray(widx).reshape(-1, 3)
        super().__init__([(self.widx[:, 0], self.widx[:, 1]),
                          (self.widx[:, 0], self.widx[:, 2]),
                          (self.widx[:, 1], self.widx[:, 2])], tolerance)

    def get_removed_dof(self, atoms):
        return 3 * len(self.widx)

    def get_indices(self, atoms=None):
        return self.widx.reshape(-1)

    def _geometry(self, atoms):
        """Minimum-image bond vectors (molecules, bond, xyz) for the three
        bond types O-H1, O-H2, H1-H2, and the per-atom inverse masses."""
        if self.bondlengths is None:
            self._init_lengths(atoms)
        pos = atoms.positions
        L = np.asarray(atoms.cell.lengths(), float)
        w = self.widx
        d = np.stack([_mic(pos[w[:, 0]] - pos[w[:, 1]], L),
                      _mic(pos[w[:, 0]] - pos[w[:, 2]], L),
                      _mic(pos[w[:, 1]] - pos[w[:, 2]], L)], axis=1)
        invm = 1.0 / atoms.get_masses()              # dialect-exempt: exact-geometry (SHAKE)
        wi = np.stack([invm[w[:, 0]], invm[w[:, 1]], invm[w[:, 2]]], axis=1)
        return d, wi

    def _matrix(self, d, wi):
        """Coupled-constraint matrix of the water triangle: bonds 0=(O,H1),
        1=(O,H2), 2=(H1,H2); ASE's reduced-mass partition on the diagonal,
        shared-atom couplings off it (the same linearization the swept
        SHAKE/RATTLE passes iterate on)."""
        A = np.empty((len(d), 3, 3))
        g01 = (d[:, 0] * d[:, 1]).sum(1)
        g02 = (d[:, 0] * d[:, 2]).sum(1)
        g12 = (d[:, 1] * d[:, 2]).sum(1)
        A[:, 0, 0] = (wi[:, 0] + wi[:, 1]) * (d[:, 0] * d[:, 0]).sum(1)
        A[:, 1, 1] = (wi[:, 0] + wi[:, 2]) * (d[:, 1] * d[:, 1]).sum(1)
        A[:, 2, 2] = (wi[:, 1] + wi[:, 2]) * (d[:, 2] * d[:, 2]).sum(1)
        A[:, 0, 1] = A[:, 1, 0] = wi[:, 0] * g01
        A[:, 0, 2] = A[:, 2, 0] = -wi[:, 1] * g02
        A[:, 1, 2] = A[:, 2, 1] = wi[:, 2] * g12
        return A

    def _bonds(self, arr):
        """Minimum-image bond vectors of a position-like array."""
        L = np.asarray(self._cell_lengths, float)
        w = self.widx
        return np.stack([_mic(arr[w[:, 0]] - arr[w[:, 1]], L),
                         _mic(arr[w[:, 0]] - arr[w[:, 2]], L),
                         _mic(arr[w[:, 1]] - arr[w[:, 2]], L)], axis=1)

    def adjust_positions(self, atoms, new):
        if self.bondlengths is None:
            self._init_lengths(atoms)
            self._cell_lengths = np.asarray(atoms.cell.lengths(), float)
            self._targets = np.stack(self.bondlengths, axis=1)
        try:
            # SHAKE freezes the correction directions at the start geometry
            # (atoms.positions), exactly as the swept version's `old` does
            d, wi = self._geometry(atoms)
            A = self._matrix(d, wi)
            scale = self.tolerance * float((self._targets ** 2).max())
            for _ in range(self.maxiter):
                dn = self._bonds(new)
                b = 0.5 * (self._targets ** 2 - (dn * dn).sum(2))  # dialect-exempt: exact-geometry (SHAKE)
                if np.abs(b).max() <= scale:
                    return
                mu = np.linalg.solve(A, b[..., None])[..., 0]
                # atom-wise corrections (i of a bond gets +w mu d, j gets -):
                # O is i of bonds 0,1; H1 is j of 0 and i of 2; H2 j of 1,2
                new[self.widx[:, 0]] += wi[:, 0][:, None] * (
                    mu[:, 0][:, None] * d[:, 0] + mu[:, 1][:, None] * d[:, 1])
                new[self.widx[:, 1]] += wi[:, 1][:, None] * (
                    -mu[:, 0][:, None] * d[:, 0] + mu[:, 2][:, None] * d[:, 2])
                new[self.widx[:, 2]] += wi[:, 2][:, None] * (
                    -mu[:, 1][:, None] * d[:, 1] - mu[:, 2][:, None] * d[:, 2])
            raise RuntimeError("RigidWater: positions did not converge")
        except np.linalg.LinAlgError:
            super().adjust_positions(atoms, new)

    def adjust_momenta(self, atoms, p):
        if self.bondlengths is None:
            self._init_lengths(atoms)
            self._cell_lengths = np.asarray(atoms.cell.lengths(), float)
            self._targets = np.stack(self.bondlengths, axis=1)
        try:
            d, wi = self._geometry(atoms)
            invm = 1.0 / atoms.get_masses()          # dialect-exempt: exact-geometry (RATTLE)
            w = self.widx
            scale = self.tolerance * float((self._targets ** 2).max())
            for _ in range(self.maxiter):
                # relative velocities along the three bonds (v = p/m)
                dv = np.stack(
                    [p[w[:, 1]] * invm[w[:, 1]][:, None] - p[w[:, 0]] * invm[w[:, 0]][:, None],
                     p[w[:, 2]] * invm[w[:, 2]][:, None] - p[w[:, 0]] * invm[w[:, 0]][:, None],
                     p[w[:, 2]] * invm[w[:, 2]][:, None] - p[w[:, 1]] * invm[w[:, 1]][:, None]],
                    axis=1)
                b = -(dv * d).sum(2)                 # dialect-exempt: exact-geometry (RATTLE)
                if np.abs(b).max() <= scale:
                    return
                # momentum corrections carry no per-atom mass factor on
                # assignment (dp = nu d); the coupling matrix is the same
                # linearization as for positions, with the opposite residual
                # sign: the projection removes the violation
                nu = np.linalg.solve(self._matrix(d, wi), (-b)[..., None])[..., 0]
                p[w[:, 0]] += nu[:, 0][:, None] * d[:, 0] + nu[:, 1][:, None] * d[:, 1]
                p[w[:, 1]] += -nu[:, 0][:, None] * d[:, 0] + nu[:, 2][:, None] * d[:, 2]
                p[w[:, 2]] += -nu[:, 1][:, None] * d[:, 1] - nu[:, 2][:, None] * d[:, 2]
            raise RuntimeError("RigidWater: momenta did not converge")
        except np.linalg.LinAlgError:
            super().adjust_momenta(atoms, p)


class RigidBonds(RigidPairs):
    """One bond type for rigid diatomics (the two-site classical species,
    e.g. N2 with its template bond length); SHAKE-exact in a single pass."""

    def __init__(self, bond_i, bond_j, tolerance=1e-13):  # dialect-exempt: numerical-guard: ASE FixBondLengths default
        super().__init__([(bond_i, bond_j)], tolerance)

    def get_removed_dof(self, atoms):
        return len(self.pairs[0][0])


class ASEBackend:
    """An ASE calculator behind the LJ-class build(r)/forces(r) interface.

    ``backend`` names the flavor: "classical" (composition-decides TIP4P,
    TIP4P+ions, SPC/E water or the LJ molecular fluid), "tip4p",
    "tip4p-ions", "lj-fluid", "eam" (tabulated), "lj". The dialect may
    override the mapping with an ``ase_backend`` threshold.

    ``model`` fixes the rigid water model of the classical flavors
    ("tip4p", "spce", ...); None (what the builder call sites pass) resolves
    it from the water geometry itself -- the same median-OH classification
    the fluid lift performs -- so the packed template's model is honoured
    end to end. ``relax`` runs the deterministic relaxation protocol (FIRE,
    then seeded Langevin; rigid water and rigid diatomics constrained) and
    returns positions with whole molecules -- wrap them with
    :func:`wrap_molecular` for output frames.
    """

    def __init__(self, cell, symbols, backend="classical", dialect=None,
                 rc=None, width=None, epsilon=None, sigma=None,
                 eam_potential=None, model=None):
        self.L = np.asarray(cell, float)
        self.symbols = list(symbols)
        self.dialect = dialect
        self.md = _md_params(dialect)
        rc = float(rc) if rc is not None else float(self.md["rc"])
        if backend in ("classical", "tip4p", "tip4p-ions", "lj-fluid"):
            # pair kernels take minimum images from KD trees on wrapped
            # coordinates: the cutoff must stay inside half the box
            rc = min(rc, 0.5 * float(self.L.min()) - float(self.md["rc_margin"]))  # dialect-exempt: exact-geometry (half cell)
        self.rc = rc
        width = float(width) if width is not None else float(self.md["width"])
        self._model_arg = model

        self.lj = None
        if backend in ("classical", "tip4p", "tip4p-ions"):
            try:
                water, ion_mask = split_water_ions(self.symbols, dialect)
            except ChaordError as no_water:
                # not a water/ion system: a molecular fluid of species with
                # published LJ parameters still realizes classically
                if backend != "classical":
                    raise
                try:
                    self.lj = split_lj_species(self.symbols, dialect)
                except ChaordError:
                    raise no_water from None
                self.flavor = self._select(backend, None, None)
                self.widx = np.zeros((0, 3), int)
            else:
                self.flavor = self._select(backend, water, ion_mask)
                self.widx = (np.where(water)[0].reshape(-1, 3) if water.any()
                             else np.zeros((0, 3), int))
        else:
            self.flavor = self._select(backend, None, None)
            self.widx = np.zeros((0, 3), int)
            if self.flavor == "lj-fluid":
                self.lj = split_lj_species(self.symbols, dialect)

        if self.flavor in ("tip4p", "tip4p-ions"):
            calc = TIP4PSolution(rc=self.rc, width=width, dialect=dialect,
                                 model=self._model_arg)
        elif self.flavor == "lj-fluid":
            sigma_arr, eps_arr, mol_id, _bonds = self.lj
            calc = LJFluid(sigma_arr, eps_arr, mol_id, rc=self.rc, width=width)
        elif self.flavor == "eam":
            path = Path(eam_potential) if eam_potential else _default_eam_potential()
            if path is None or not Path(path).exists():
                raise ChaordError(
                    "ASE eam backend needs a tabulated potential: pass "
                    "eam_potential= or ship bench/reference/potentials/"
                    "Cu_u3.eam")
            calc = EAM(potential=str(path))
        elif self.flavor == "lj":
            kw = {}
            if epsilon is not None:
                kw["epsilon"] = float(epsilon)
            if sigma is not None:
                kw["sigma"] = float(sigma)
            calc = LennardJones(rc=self.rc, **kw)
        else:
            raise ChaordError(f"unknown ASE backend flavor {self.flavor!r}")

        self.atoms = Atoms(self.symbols,
                           positions=np.zeros((len(self.symbols), 3)),
                           cell=np.diag(self.L), pbc=True)
        self.atoms.calc = calc
        # rigid geometry of the flavor: OHH water triples, diatomic bonds
        # (constraints themselves are built per relax() call: SHAKE caches
        # the reference bond lengths of the configuration it first sees)
        self.bonds = (self.lj[3] if self.lj is not None and len(self.lj[3][0])
                      else None)

    @property
    def model(self) -> str | None:
        """The resolved rigid water model of the classical flavors (explicit
        ``model=`` or measured from the water geometry on the first energy
        call; None for the waterless flavors)."""
        calc = self.atoms.calc
        return calc.model_name if isinstance(calc, TIP4PSolution) else None

    # --------------------------------------------------------- selection --
    def _select(self, backend, water, ion_mask) -> str:
        mapping = None
        if self.dialect is not None:
            try:
                mapping = self.dialect.threshold("ase_backend")
            except ChaordError:
                mapping = None
        if isinstance(mapping, dict) and backend in mapping:
            return str(mapping[backend])
        if backend == "classical":
            if water is not None or ion_mask is not None:
                return "tip4p-ions" if ion_mask.any() else "tip4p"
            return "lj-fluid"       # reached only when the LJ table covered it
        if backend == "tip4p" and ion_mask is not None and ion_mask.any():
            raise ChaordError("tip4p flavor is pure water; "
                              "ions need the tip4p-ions solution flavor")
        return backend

    # ------------------------------------------- LJ-class API (run_md ok) --
    def build(self, r):
        """Interface parity with realize.lj.LJ: prime the calculator."""
        self.atoms.set_positions(np.asarray(r, float))

    def forces(self, r):
        self.build(r)
        return np.array(self.atoms.get_forces())

    def energy(self, r):
        self.build(r)
        return float(self.atoms.get_potential_energy())

    # ---------------------------------------------------------- relaxation --
    def relax(self, r, rng, T_K=None, md_steps=None):
        """FIRE + seeded Langevin; rigid water and rigid diatomics are kept
        rigid.

        Returns positions with molecules whole (no per-atom wrapping)."""
        self.atoms.set_positions(np.asarray(r, float))
        constraints = []
        if len(self.widx):
            constraints.append(RigidWater(self.widx))
        if self.bonds is not None:
            constraints.append(RigidBonds(self.bonds[0], self.bonds[1]))
        if constraints:
            self.atoms.set_constraint(constraints)
        T = float(T_K) if T_K is not None else float(self.md["reference_T_K"])
        try:
            # 1. capped minimization of the packed start
            FIRE(self.atoms, logfile=None).run(
                fmax=float(self.md["fire_fmax"]),
                steps=int(self.md["fire_steps"]))
            # 2. Langevin, seeded from the caller's rng: same program + seed
            #    + backend -> same coordinates (rule 10)
            self._langevin(T, int(self.md["fast_steps"]),
                           float(self.md["fast_dt_fs"]),
                           float(self.md["fast_gamma_per_fs"]), rng)
            steps = int(self.md["relax_steps"] if md_steps is None
                        else md_steps)
            if steps > 0:
                self._langevin(T, steps, float(self.md["relax_dt_fs"]),
                               float(self.md["relax_gamma_per_fs"]), rng)
            return np.array(self.atoms.get_positions())
        finally:
            self.atoms.set_constraint()

    def _langevin(self, T_K, steps, dt_fs, gamma_per_fs, rng):
        if steps <= 0:
            return
        masses = self.atoms.get_masses()
        v = rng.standard_normal((len(self.atoms), 3)) * np.sqrt(
            units.kB * T_K / masses)[:, None]      # dialect-exempt: exact-geometry (Maxwell)
        self.atoms.set_velocities(v)
        Langevin(self.atoms, timestep=dt_fs * units.fs,
                 temperature_K=T_K, friction=gamma_per_fs / units.fs,
                 rng=rng, fixcm=False).run(steps)


def wrap_molecular(pos, symbols, L, dialect=None) -> np.ndarray:
    """Wrap into the box without cutting molecules: each O by itself, its
    H's to the O's image, everything else (ions, LJ molecular species) by
    itself. Per-atom wrapping of the two-site species is observables-safe:
    every consumer measures through the minimum image, so a diatomic cut at
    a boundary is the same microstate (the backend keeps molecules whole
    before this; only the output imaging differs)."""
    pos = np.asarray(pos, float)
    L = np.asarray(L, float)
    try:
        water, _ = split_water_ions(symbols, dialect)
    except ChaordError:
        # a pure LJ molecular fluid (or anything without water): the parse
        # error is the backend's to raise, not the wrapper's
        water = np.zeros(len(symbols), bool)
    out = pos.copy()
    if water.any():
        # water atoms arrive as O,H,H triples in order: group per molecule
        w = out[water].reshape(-1, 3, 3)      # (molecule, atom, xyz)
        O = np.mod(w[:, 0, :], L)
        w[:, 1, :] = O + _mic(w[:, 1, :] - w[:, 0, :], L)
        w[:, 2, :] = O + _mic(w[:, 2, :] - w[:, 0, :], L)
        w[:, 0, :] = O
        out[water] = w.reshape(-1, 3)
    out[~water] = np.mod(out[~water], L)
    return out
