"""Vectorized force kernels for the reference-data MD runs (bench/reference).

Everything here runs on ASE (Atoms, constraints, ase.md integrators) but
replaces ASE's per-pair Python loops — orders of magnitude too slow for the
system sizes of the reference cases — with numpy-vectorized evaluations of
the *same* published potentials:

  * FastEAM   - EAM evaluation with the spline tables parsed by
                ase.calculators.eam.EAM (LAMMPS funcfl/setfl files).
  * FastTIP4P - rigid TIP4P water with the exact formulas, parameters and
                O-O-based smooth pair truncation of ase.calculators.tip4p
                (Jorgensen et al., J. Chem. Phys. 79, 926 (1983), DOI in the
                ASE source), including virtual-site force redistribution.
  * JCSPCEWolf - rigid SPC/E water (Berendsen et al., J. Phys. Chem. 91,
                6269 (1987)) + Joung-Cheatham Na+/Cl- (J. Phys. Chem. B 112,
                9020 (2008)) with damped-shifted-force (Wolf) electrostatics
                (Fennell & Gezelter, J. Chem. Phys. 124, 234104 (2006)).

Each fast calculator carries a validation routine that the generator runs
and records before any dynamics: FastEAM and FastTIP4P are checked against
the corresponding ASE calculators (same numbers to machine precision);
JCSPCEWolf checks that its DSF force is the exact derivative of its DSF
energy and documents its formula.

Nothing in this file may import chaord (circular-validation ban).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import erfc

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.eam import EAM
from ase.calculators.tip4p import (TIP4P, angleHOH, epsilon0, qH, rOH,
                                   sigma0)

kC = units.Hartree * units.Bohr          # 14.3996 eV A for charges in e


def _diag(atoms) -> np.ndarray:
    """Diagonal of the (orthorhombic) cell as a plain array."""
    return np.asarray(atoms.cell.lengths(), float)


# ------------------------------------------------------------------- EAM --

class FastEAM(EAM):
    """EAM with vectorized force evaluation; single-element potentials.

    Uses the spline tables parsed and built by ase.calculators.eam.EAM, so
    the published tabulated potential is evaluated exactly; only the pair
    loops are vectorized."""

    def __init__(self, **kw):
        EAM.__init__(self, **kw)
        if self.Nelements != 1:
            raise ValueError("FastEAM supports single-element potentials")

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        from ase.neighborlist import neighbor_list
        i, j, D, r = neighbor_list('ijDd', at, self.cutoff)
        rho = self.electron_density[0](r)
        phi = self.phi[0, 0](r)
        rho_tot = np.zeros(len(at))
        np.add.at(rho_tot, i, rho)
        energy = float(self.embedded_energy[0](rho_tot).sum()
                       + 0.5 * phi.sum())
        dF = self.d_embedded_energy[0](rho_tot)
        scale = (self.d_phi[0, 0](r)
                 + (dF[i] + dF[j]) * self.d_electron_density[0](r))
        fvec = scale[:, None] * D / r[:, None]
        forces = np.zeros((len(at), 3))
        np.add.at(forces, i, fvec)
        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces

    @staticmethod
    def validate_against_ase(potential_path, element='Cu', a=3.615,
                             seed=0, max_dev=1e-6):
        """Energy+forces must match ase.calculators.eam.EAM."""
        from ase.build import bulk
        ref = bulk(element, 'fcc', a=a, cubic=True).repeat((2, 2, 2))
        ref.rattle(stdev=0.2, seed=seed + 1)
        a1, a2 = ref.copy(), ref.copy()
        a1.calc = EAM(potential=potential_path)
        a2.calc = FastEAM(potential=potential_path)
        de = abs(a1.get_potential_energy() - a2.get_potential_energy())
        df = abs(a1.get_forces() - a2.get_forces()).max()
        if not (de < max_dev and df < max_dev):
            raise AssertionError(f"FastEAM mismatch: dE={de:.2e} dF={df:.2e}")
        return {"dE_eV": float(de), "dF_eV_per_A": float(df)}


# ----------------------------------------------------------------- TIP4P --

class FastTIP4P(Calculator):
    """Rigid TIP4P water: vectorized ase.calculators.tip4p.

    Same parameters (qH, sigma0, epsilon0 from the ASE module, virtual-site
    distance 0.15 A), same O-O-distance smooth pair truncation (rc, width)
    and same virtual-site force redistribution as the ASE calculator;
    molecule pairs are found at once with a KD-tree instead of the Python
    double loop over molecule pairs."""

    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, rc=8.0, width=1.0, **kw):
        Calculator.__init__(self, **kw)
        self.rc = rc
        self.width = width

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        pos = at.positions
        L = _diag(at)
        nmol = len(at) // 3

        # virtual sites O H1 H2 M per molecule (ASE order and construction);
        # built from the *unwrapped* positions so molecules that straddle a
        # periodic boundary keep their geometry
        r_ij = pos[1::3] - pos[0::3]
        r_jk = pos[2::3] - pos[1::3]
        nvec = r_ij + 0.5 * r_jk
        nvec /= np.linalg.norm(nvec, axis=1)[:, None]
        xpos = np.empty((nmol, 4, 3))
        xpos[:, 0] = pos[0::3]
        xpos[:, 1] = pos[1::3]
        xpos[:, 2] = pos[2::3]
        xpos[:, 3] = pos[0::3] + 0.15 * nvec
        q = np.array([0.0, qH, qH, -2 * qH])

        # molecule pairs by O-O distance (ASE scheme: the whole molecule of b
        # is displaced by the same lattice shift as its oxygen)
        tree = cKDTree(np.mod(pos[0::3], L), boxsize=L)
        pairs = tree.query_pairs(self.rc, output_type="ndarray")
        a, b = pairs[:, 0], pairs[:, 1]
        Dw = pos[0::3][b] - pos[0::3][a]
        shift = L * np.round(Dw / L)          # lattice shift applied to b
        DOO = Dw - shift
        d = np.sqrt((DOO * DOO).sum(1))

        x12 = (d > self.rc - self.width) & (d < self.rc)
        y = (d[x12] - self.rc + self.width) / self.width
        t = np.zeros(len(d))
        t[d < self.rc] = 1.0
        t[x12] -= y * y * (3.0 - 2.0 * y)
        dtdd = np.zeros(len(d))
        dtdd[x12] -= 6.0 / self.width * y * (1.0 - y)

        # LJ between O sites, cut on the O-O distance
        e_lj = 4 * epsilon0 * (sigma0 ** 12 / d ** 12 - sigma0 ** 6 / d ** 6)
        coef = (4 * epsilon0 * (12 * sigma0 ** 12 / d ** 13
                                - 6 * sigma0 ** 6 / d ** 7) * t
                - e_lj * dtdd)
        f_lj = coef[:, None] * DOO / d[:, None]          # force on O of b

        # electrostatics: 4x4 site-site pairs per molecule pair, all with the
        # same molecular cutoff factor t(r_OO) (ASE scheme); molecule b uses
        # its oxygen's lattice shift
        Dp = (xpos[b] - shift[:, None, :])[:, None, :, :] - xpos[a][:, :, None, :]
        dp = np.sqrt((Dp * Dp).sum(-1))
        e = kC * q[None, :, None] * q[None, None, :] / dp
        tc = t[:, None, None]
        Fp = (e * tc / dp ** 2)[:, :, :, None] * Dp      # on b's site s_b
        e_f = e.sum(2)                                   # per site of a
        FOO = -(e_f * dtdd[:, None])[:, :, None] * DOO[:, None, :] \
            / d[:, None, None]

        energy = float((e_lj * t).sum() + (e * tc).sum())

        # accumulate forces on the 4 sites of every molecule
        fx = np.zeros((nmol, 4, 3))
        np.add.at(fx, (b[:, None], np.arange(4)[None, :]), Fp.sum(1))
        np.add.at(fx, (a[:, None], np.arange(4)[None, :]), -Fp.sum(2))
        contrib_b = np.zeros((len(b), 4, 3))
        contrib_b[:, 0] = f_lj + FOO.sum(1)
        np.add.at(fx, b, contrib_b)
        contrib_a = np.zeros((len(a), 4, 3))
        contrib_a[:, 0] = -f_lj - FOO.sum(1)
        np.add.at(fx, a, contrib_a)

        # redistribute the M-site force onto O, H1, H2 (ASE formulas)
        Fd = fx[:, 3]
        r_id = 0.15 * nvec
        gamma = 0.15 / np.linalg.norm(r_ij + 0.5 * r_jk, axis=1)
        proj = (r_id * Fd).sum(1) / (r_id * r_id).sum(1)
        Fd_minus_F1 = Fd - proj[:, None] * r_id
        forces = np.empty_like(pos)
        forces[0::3] = fx[:, 0] + Fd - gamma[:, None] * Fd_minus_F1
        forces[1::3] = fx[:, 1] + 0.5 * gamma[:, None] * Fd_minus_F1
        forces[2::3] = fx[:, 2] + 0.5 * gamma[:, None] * Fd_minus_F1

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces

    @staticmethod
    def validate_against_ase(nmol=64, seed=3, rc=None, width=1.0,
                             max_dev=1e-8):
        """Energy+forces must match ase.calculators.tip4p.TIP4P."""
        from ase import Atoms
        rng = np.random.default_rng(seed)
        L = (nmol * 30.0) ** (1 / 3)
        if rc is None:
            rc = min(5.0, L / 2 - 0.05)   # the ASE calculator needs 2*rc <= L
        pos = np.zeros((nmol * 3, 3))
        th = np.radians(angleHOH / 2)
        for m in range(nmol):
            o = rng.uniform(0, L, 3)
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            v = rng.normal(size=3)
            v -= u * (v @ u)
            v /= np.linalg.norm(v)
            pos[3 * m] = o
            pos[3 * m + 1] = o + rOH * (np.cos(th) * u + np.sin(th) * v)
            pos[3 * m + 2] = o + rOH * (np.cos(th) * u - np.sin(th) * v)
        at1 = Atoms('OH2' * nmol, positions=pos, cell=[L, L, L], pbc=True)
        at2 = at1.copy()
        at1.calc = TIP4P(rc=rc, width=width)
        at2.calc = FastTIP4P(rc=rc, width=width)
        de = abs(at1.get_potential_energy() - at2.get_potential_energy())
        df = abs(at1.get_forces() - at2.get_forces()).max()
        if not (de < max_dev and df < max_dev):
            raise AssertionError(
                f"FastTIP4P mismatch: dE={de:.2e} dF={df:.2e}")
        return {"dE_eV": float(de), "dF_eV_per_A": float(df)}


# ------------------------------------------- SPC/E water + Joung-Cheatham --

class JCSPCEWolf(Calculator):
    """Rigid SPC/E water + Joung-Cheatham Na+/Cl- with Wolf/DSF truncation.

    * LJ between O-O, ion-O and ion-ion sites (Lorentz-Berthelot mixing,
      energy-shifted at rc); H carries no LJ.
    * Electrostatics between all charged sites (qO = -0.8476 e,
      qH = +0.4238 e, ions +-1 e) with the damped shifted-force (DSF)
      truncation of the Wolf method, Fennell & Gezelter, J. Chem. Phys.
      124, 234104 (2006):

        V(r) = q_i q_j kC [ erfc(a r)/r - erfc(a rc)/rc
               + (r - rc) ( erfc(a rc)/rc^2
               + (2 a / sqrt(pi)) e^{-a^2 rc^2} / rc ) ]

      continuous in energy and force at r = rc; a = alpha and rc are
      constructor arguments. Intramolecular water pairs are excluded
      (rigid geometry makes their energy constant)."""

    implemented_properties = ['energy', 'free_energy', 'forces']

    # SPC/E: Berendsen, Grigera, Straatsma, J. Phys. Chem. 91, 6269 (1987)
    Q_O, Q_H = -0.8476, 0.4238
    SIGMA_O, EPS_O = 3.166, 0.1553          # Angstrom, kcal/mol
    R_OH, ANGLE_HOH = 1.0, 109.47           # rigid SPC/E geometry
    # Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), Table 3 (SPC/E)
    SIGMA_NA, EPS_NA = 2.159542, 0.3526418  # Angstrom, kcal/mol
    SIGMA_CL, EPS_CL = 4.830486, 0.0300147  # Angstrom, kcal/mol

    def __init__(self, rc=9.0, alpha=0.2, **kw):
        Calculator.__init__(self, **kw)
        self.rc = rc
        self.alpha = alpha
        kcal = units.kcal / units.mol
        self._lj = {'O': (self.SIGMA_O, self.EPS_O * kcal),
                    'Na': (self.SIGMA_NA, self.EPS_NA * kcal),
                    'Cl': (self.SIGMA_CL, self.EPS_CL * kcal)}
        self._q = {'O': self.Q_O, 'H': self.Q_H, 'Na': 1.0, 'Cl': -1.0}

    def _molecule_ids(self, sym):
        n = len(sym)
        mol = np.full(n, -1)
        k = 0
        for i, s in enumerate(sym):
            if s == 'O':
                mol[i:i + 3] = k
                k += 1
            elif s in ('Na', 'Cl'):
                mol[i] = 100000 + i
        return mol

    def calculate(self, atoms=None, properties=('energy', 'forces'),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at = self.atoms
        pos = at.positions
        sym = at.get_chemical_symbols()
        L = _diag(at)
        sig = np.array([self._lj.get(s, (0.0, 0.0))[0] for s in sym])
        eps = np.array([self._lj.get(s, (0.0, 0.0))[1] for s in sym])
        qq = np.array([self._q[s] for s in sym])
        mol = self._molecule_ids(sym)

        tree = cKDTree(np.mod(pos, L), boxsize=L)
        pairs = tree.query_pairs(self.rc, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        keep = mol[i] != mol[j]
        i, j = i[keep], j[keep]
        D = pos[j] - pos[i]
        D -= L * np.round(D / L)
        r = np.sqrt((D * D).sum(1))

        # LJ, Lorentz-Berthelot, energy-shifted at rc (no force shift: the
        # force jump at rc is < 1e-4 eV/A for every pair type here)
        s_ij = 0.5 * (sig[i] + sig[j])
        e_ij = np.sqrt(eps[i] * eps[j])
        has_lj = e_ij > 0
        sr6 = np.where(has_lj, (s_ij / r) ** 6, 0.0)
        sr6c = np.where(has_lj, (s_ij / self.rc) ** 6, 0.0)
        u_lj = 4 * e_ij * (sr6 ** 2 - sr6 - (sr6c ** 2 - sr6c))
        # du/dr / r;  force_i = -du/dr * D/r is folded into the sign below
        f_lj = np.where(has_lj, 24 * e_ij * (2 * sr6 ** 2 - sr6) / r ** 2, 0.0)

        # DSF electrostatics (see class docstring)
        a, rc = self.alpha, self.rc
        qiqj = kC * qq[i] * qq[j]
        erfc_r = erfc(a * r)
        f_r = erfc_r / r
        f_rc = erfc(a * rc) / rc
        g_rc = (f_rc / rc
                + 2 * a / np.sqrt(np.pi) * np.exp(-(a * rc) ** 2) / rc)
        u_el = qiqj * (f_r - f_rc + (r - rc) * g_rc)
        h_r = (-2 * a / np.sqrt(np.pi) * np.exp(-(a * r) ** 2) / r
               - erfc_r / r ** 2 + g_rc)          # d/dr [erfc(a r)/r] + g_rc

        energy = float(u_lj.sum() + u_el.sum())
        fvec = (qiqj * h_r / r)[:, None] * D - f_lj[:, None] * D
        forces = np.zeros((len(at), 3))
        np.add.at(forces, i, fvec)
        np.subtract.at(forces, j, fvec)

        self.results['energy'] = energy
        self.results['free_energy'] = energy
        self.results['forces'] = forces

    def selftest(self):
        """The DSF pair force must be the exact derivative of the DSF pair
        energy (numeric derivative check) — the electrostatics kernel is
        otherwise validated by the physical sanity checks of the case."""
        a, rc = self.alpha, self.rc

        def f(x):
            return erfc(a * x) / x

        g = (f(rc) / rc + 2 * a / np.sqrt(np.pi) * np.exp(-(a * rc) ** 2) / rc)

        def u(x):
            return f(x) - f(rc) + (x - rc) * g

        def du(x):
            return ((-2 * a / np.sqrt(np.pi) * np.exp(-(a * x) ** 2) / x
                     - erfc(a * x) / x ** 2) + g)

        r = np.array([2.4, 3.3, 5.1, 8.7])
        h = 1e-6
        num = (u(r + h) - u(r - h)) / (2 * h)
        if np.abs(num - du(r)).max() > 1e-6:
            raise AssertionError("DSF force is not dV/dr")
        return {"dsf_force_matches_energy_derivative": True}
