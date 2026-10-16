# -*- coding: utf-8 -*-
"""
Build initial LAMMPS data files for LiPF6/EC:EMC electrolyte inside armchair
CNTs (field off/on states differ only in the run input, not the build; the
field-on run uses a different velocity seed).

STANDALONE: imports only stdlib + numpy, never `chaord` (house rule).

Usage:
    python build_systems.py            # builds all four systems
    python build_systems.py cnt10_field0 ...
"""
import json
import math
import os
import sys

import numpy as np

import params as P

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------

def place_from_zmat(pos_a, pos_b, pos_c, r, ang_deg, dih_deg):
    """Position D given A,B,C with |BD|=r, angle(A-B-D)=ang, dihedral(A,B,C,D).
    Standard NeRF/standard z-matrix placement."""
    ang = math.radians(ang_deg)
    dih = math.radians(dih_deg)
    bc = pos_c - pos_b
    bc /= np.linalg.norm(bc)
    ab = pos_b - pos_a
    n = np.cross(ab, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array([-r * math.cos(ang),
                   r * math.sin(ang) * math.cos(dih),
                   r * math.sin(ang) * math.sin(dih)])
    return pos_b + d2[0] * bc + d2[1] * m + d2[2] * n


def hydrogens_on_ch2(carbon, n1, n2, ch_len):
    """Two H's tetrahedral on an sp3 carbon with 2 heavy neighbors n1, n2.
    H-C-H = 109.47 deg, H's symmetric about the heavy-neighbor plane normal."""
    u = (n1 - carbon) / np.linalg.norm(n1 - carbon)
    v = (n2 - carbon) / np.linalg.norm(n2 - carbon)
    bis = u + v
    bis /= np.linalg.norm(bis)
    nrm = np.cross(u, v)
    nrm /= np.linalg.norm(nrm)
    a = math.radians(54.7356)  # gives H-C-H = 109.47 deg
    h1 = carbon - math.cos(a) * bis * ch_len + math.sin(a) * nrm * ch_len
    h2 = carbon - math.cos(a) * bis * ch_len - math.sin(a) * nrm * ch_len
    return h1, h2


def hydrogens_on_ch3(carbon, neighbor, ch_len, ref):
    """Three H's on a methyl carbon (one heavy neighbor)."""
    u = (neighbor - carbon) / np.linalg.norm(neighbor - carbon)
    # orthonormal complement
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, u)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    theta = math.radians(70.5288)  # H-C-heavy = 180 - 109.47
    hs = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0 + 0.35  # deterministic phase
        d = math.cos(theta) * (-u) + math.sin(theta) * (
            math.cos(phi) * e1 + math.sin(phi) * e2)
        hs.append(carbon + d / np.linalg.norm(d) * ch_len)
    return hs


# ----------------------------------------------------------------------------
# molecule templates: list of (label, type_id, q, np.array xyz)
# bonds: list of (i, j) local index pairs (0-based) -> bond type 1
# ----------------------------------------------------------------------------

TYPE_OF = {"C_tube": 1, "CO": 2, "OS": 3, "CT": 4, "OC": 5, "HC": 6,
           "HM": 7, "C_term": 8, "Li": 9, "P": 10, "F": 11}


def build_ec():
    """Ethylene carbonate, rigid, planar 5-ring.

    Bond lengths/angles from [Nutakki2023] Table III (CO-OC 1.229,
    CO-OS 1.327, OS-CT 1.410, CT-CT 1.529, CT-HC 1.09; OC-CO-OS 123.4,
    OS-CO-OS 118.18). NOTE: the three angles at the carbonate carbon in
    the thesis (2 x OC-CO-OS + OS-CO-OS = 364.98 deg) cannot coexist in
    a planar trigonal center; we enforce OC-CO-OS = 123.4 deg exactly on
    both sides, which fixes OS-CO-OS = 360 - 2 x 123.4 = 113.2 deg.
    With the fixed bond lengths and mirror symmetry the remaining ring
    angles follow from closure: C-O-C = 109.32 deg, O-CH2-CH2 =
    104.08 deg (both close to experimental EC ring geometry and better
    than the thesis targets, which do not close). All deviations are
    recorded in the build manifest.
    """
    b = P.GEOM["bonds"]
    l_CO_OS, l_OS_CT, l_CT_CT = b[("CO", "OS")], b[("OS", "CT")], b[("CT", "CT")]
    l_CO_OC, l_CH = b[("CO", "OC")], b[("CT", "HC")]
    beta = math.radians(113.2)          # OS-CO-OS implied by 2x(OC-CO-OS=123.4)
    half = 0.5 * l_CT_CT
    # C1 (CO) on the mirror axis; O2/O5, C3/C4 symmetric pairs
    # O2 = C1 + l_CO_OS * (-sin(beta/2), -cos(beta/2)); |O2 - C3| = l_OS_CT
    # with C3 = (-half, 0):
    dx = -math.sin(beta / 2.0) * l_CO_OS
    dy = -math.cos(beta / 2.0) * l_CO_OS
    t = l_OS_CT ** 2 - (dx + half) ** 2
    assert t > 0
    h1 = math.sqrt(t) - dy               # C1.y such that O2-C3 = l_OS_CT
    C1 = np.array([0.0, h1, 0.0])
    O2 = C1 + np.array([dx, dy, 0.0])
    C3 = np.array([-half, 0.0, 0.0])
    C4 = np.array([half, 0.0, 0.0])
    O5 = C1 + np.array([-dx, dy, 0.0])
    # carbonyl O: in plane, on the mirror axis; verify OC-CO-OS ~ 123.4 deg
    OC = C1 + np.array([0.0, 1.0, 0.0]) * l_CO_OC
    real = math.degrees(math.acos(np.dot(OC - C1, O2 - C1) /
                                  (l_CO_OC * l_CO_OS)))
    assert abs(real - 123.4) < 1.5, f"OC-CO-OS angle {real:.2f}"
    ch = P.GEOM["angles"]  # noqa: F841 (angles carried in params for reference)
    atoms = [
        ("CO", P.EC_CHARGES["CO"], C1),
        ("OC", P.EC_CHARGES["OC"], OC),
        ("OS", P.EC_CHARGES["OS"], O2),
        ("CT", P.EC_CHARGES["CT"], C3),
        ("CT", P.EC_CHARGES["CT"], C4),
        ("OS", P.EC_CHARGES["OS"], O5),
    ]
    # hydrogens on the two CH2
    for C, other, nbr in ((C3, C4, O2), (C4, C3, O5)):
        h1_, h2_ = hydrogens_on_ch2(C, other, nbr, l_CH)
        atoms.append(("HC", P.EC_CHARGES["HC"], h1_))
        atoms.append(("HC", P.EC_CHARGES["HC"], h2_))
    out = [(lab, TYPE_OF[lab], q, xyz) for lab, q, xyz in atoms]
    bonds = [(0, 1), (0, 2), (0, 5), (2, 3), (3, 4), (4, 5),
             (6, 3), (7, 3), (8, 4), (9, 4)]
    # ring angles actually realized (for provenance)
    ang = lambda a, b, c: math.degrees(math.acos(
        np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b))))
    ring_info = dict(
        OS_CO_OS=ang(O2, C1, O5), CO_OC=math.radians(0),
        C_O_C=ang(C1, O2, C3), O_CH2_CH2=ang(O2, C3, C4),
        OC_CO_OS=real,
    )
    return out, bonds, ring_info


def build_emc():
    """Ethyl methyl carbonate, rigid, planar heavy-atom skeleton.

    Internal coordinates from [Nutakki2023] Table III: bonds CO-OC 1.229,
    CO-OS 1.327, OS-CT 1.410, CT-CT 1.529, C-H 1.09; angles OC-CO-OS 123.4,
    CO-OS-CT 116.9, OS-CT-CT 109.5. Dihedrals chosen (documented):
    OC-CO-OS-CT = 0 deg (planar carbonate, ester s-cis/syn conformer, the
    standard ester minimum) on both ester oxygens, and CO-OS-CT-CT = 180 deg
    (anti-periplanar, planar ethyl rotamer).

    REBUILT 2026-10-14: the original z-matrix placement folded the ethoxy
    OS onto the carbonyl OC (0.68 A apart): place_from_zmat enforced
    OC-CO-OS = 143.7 / 30.6 deg instead of the intended 123.4 deg (its
    angle/dihedral semantics did not match the calling convention). This
    explicit planar construction asserts every intended internal
    coordinate. The original H-bond wiring also bonded CT_me to the
    terminal-methyl hydrogens (range(idx-3, idx)) instead of its own
    methyl hydrogens; harmless (bonds are zero-force and
    neigh_modify exclude molecule/intra supersedes 1-2/1-3 exclusions)
    but now wired correctly.
    """
    b = P.GEOM["bonds"]
    l_CO_OC, l_CO_OS, l_OS_CT = b[("CO", "OC")], b[("CO", "OS")], b[("OS", "CT")]
    l_CT_CT, l_CH = b[("CT", "CT")], b[("CT", "HC")]

    def rot2(v, deg):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], 0.0])

    # planar carbonate core in the xy-plane, mirror-symmetric like EC:
    # CO at origin, OC along +y; both OS at 123.4 deg from CO->OC
    # (=> OS-CO-OS = 360 - 2 x 123.4 = 113.2 deg, same closure as in EC).
    CO = np.array([0.0, 0.0, 0.0])
    OC = np.array([0.0, l_CO_OC, 0.0])
    ang = math.radians(123.4)
    u_os = np.array([math.sin(ang), math.cos(ang), 0.0])   # CO -> OS (right)
    OS_et = l_CO_OS * u_os
    OS_me = l_CO_OS * np.array([-u_os[0], u_os[1], 0.0])    # mirrored (left)

    def ct_at(OS, side):
        """CT on an ester O: in-plane, angle CO-OS-CT = 116.9 deg,
        syn-periplanar to OC (OC-CO-OS-CT = 0 deg)."""
        v = (CO - OS) / np.linalg.norm(CO - OS)              # OS -> CO
        for rot in (116.9, -116.9):
            w = rot2(v, rot * side)
            CT = OS + l_OS_CT * w
            axis = OS - CO
            s1 = np.cross(axis, OC - CO)[2]
            s2 = np.cross(axis, CT - CO)[2]
            if s1 * s2 > 0:                                  # same side: syn
                return CT
        raise AssertionError("no syn placement found")

    CT_et = ct_at(OS_et, +1.0)
    CT_me = ct_at(OS_me, -1.0)

    def cterm_at(CT, OS):
        """Terminal CH3: angle OS-CT-CT = 109.5 deg, dihedral
        CO-OS-CT-CT = 180 deg (anti-periplanar)."""
        v = (OS - CT) / np.linalg.norm(OS - CT)              # CT -> OS
        for rot in (109.5, -109.5):
            w = rot2(v, rot)
            CT2 = CT + l_CT_CT * w
            axis = CT - OS
            s1 = np.cross(axis, CO - OS)[2]
            s2 = np.cross(axis, CT2 - OS)[2]
            if s1 * s2 < 0:                                  # opposite: anti
                return CT2
        raise AssertionError("no anti placement found")

    C_term = cterm_at(CT_et, OS_et)

    q = P.EMC_CHARGES
    atoms = [
        ("CO", q["CO"], CO), ("OC", q["OC"], OC),
        ("OS", q["OS_me"], OS_me), ("CT", q["CT_me"], CT_me),
        ("OS", q["OS_et"], OS_et), ("CT", q["CT_et"], CT_et),
        ("C_term", q["C_term"], C_term),
    ]
    for h in hydrogens_on_ch3(CT_me, OS_me, l_CH, None):
        atoms.append(("HC", q["HC_me"], h))                  # idx 7, 8, 9
    for h in hydrogens_on_ch2(CT_et, OS_et, C_term, l_CH):
        atoms.append(("HC", q["HC_et"], h))                  # idx 10, 11
    for h in hydrogens_on_ch3(C_term, CT_et, l_CH, None):
        atoms.append(("HM", q["HM"], h))                     # idx 12, 13, 14
    out = [(lab, TYPE_OF[lab], qq, xyz) for lab, qq, xyz in atoms]
    bonds = [(0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (5, 6),
             (3, 7), (3, 8), (3, 9),                         # methoxy H
             (5, 10), (5, 11),                               # ethoxy CH2 H
             (6, 12), (6, 13), (6, 14)]                      # terminal H

    # ---- verify every intended internal coordinate (honesty by assertion)
    A = np.array([a[2] for a in atoms])
    iCO, iOC, iOSme, iCTme, iOSet, iCTet, iT = range(7)

    def dist(i, j):
        return float(np.linalg.norm(A[i] - A[j]))

    def angle_at(i, j, k):
        u, v = A[i] - A[j], A[k] - A[j]
        return math.degrees(math.acos(
            np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def dih4(i, j, k, l):
        b0 = A[i] - A[j]
        b1 = A[k] - A[j]
        b1 = b1 / np.linalg.norm(b1)
        b2 = A[l] - A[k]
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return math.degrees(-math.atan2(np.dot(np.cross(b1, v), w),
                                        np.dot(v, w)))

    for (i, j), r in [((iCO, iOC), 1.229), ((iCO, iOSme), 1.327),
                      ((iCO, iOSet), 1.327), ((iOSme, iCTme), 1.410),
                      ((iOSet, iCTet), 1.410), ((iCTet, iT), 1.529)]:
        assert abs(dist(i, j) - r) < 1e-6, f"bond {i}-{j}: {dist(i, j)}"
    for iOS_k, iCT_k in ((iOSme, iCTme), (iOSet, iCTet)):
        got = angle_at(iOC, iCO, iOS_k)
        assert abs(got - 123.4) < 1e-3, f"OC-CO-OS {got}"
        got = angle_at(iCO, iOS_k, iCT_k)
        assert abs(got - 116.9) < 1e-3, f"CO-OS-CT {got}"
    got = angle_at(iOSet, iCTet, iT)
    assert abs(got - 109.5) < 1e-3, f"OS-CT-CT {got}"
    got = angle_at(iOSme, iCO, iOSet)
    assert abs(got - 113.2) < 1e-3, f"OS-CO-OS {got}"
    for iOS_k, iCT_k in ((iOSme, iCTme), (iOSet, iCTet)):
        dd = dih4(iOC, iCO, iOS_k, iCT_k)
        assert abs(dd) < 1e-6, f"OC-CO-OS-CT must be syn(0), got {dd}"
    dd = dih4(iCO, iOSet, iCTet, iT)
    assert abs(abs(dd) - 180.0) < 1e-6, f"CO-OS-CT-CT must be anti, got {dd}"
    assert abs(A[:7, 2]).max() < 1e-9, "heavy skeleton not planar"
    # no unintended intra-molecular heavy-atom contacts
    bonded = {(min(x, y), max(x, y)) for x, y in bonds}
    dmat = np.linalg.norm(A[:7][:, None] - A[:7][None], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    for x in range(7):
        for y in range(x + 1, 7):
            if (x, y) not in bonded:
                assert dmat[x, y] > 2.2, (f"non-bonded heavy pair "
                                          f"{x}-{y} at {dmat[x, y]:.3f} A")
    return out, bonds


def build_pf6():
    """PF6-, rigid octahedron. r0(P-F) = 1.606 A, theta(F-P-F) = 90/180 deg
    [Nutakki2023] Table II. Charges: P +1.70, F -0.45 x6 (sum -1.00)."""
    r = P.GEOM["bonds"][("P", "F")]
    dirs = [np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
            np.array([0., 1., 0.]), np.array([0., -1., 0.]),
            np.array([0., 0., 1.]), np.array([0., 0., -1.])]
    atoms = [("P", TYPE_OF["P"], P.PF6_CHARGES["P"], np.zeros(3))]
    for d in dirs:
        atoms.append(("F", TYPE_OF["F"], P.PF6_CHARGES["F"], d * r))
    bonds = [(0, i) for i in range(1, 7)]
    return atoms, bonds


def build_li():
    return [("Li", TYPE_OF["Li"], P.LI_CHARGE, np.zeros(3))], []


# ----------------------------------------------------------------------------
# CNT builder
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# packing
# ----------------------------------------------------------------------------

def sigma_of(t):
    return P.ATOM_TYPES[t][2]


def rand_quaternion(rng):
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = np.array([math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                  math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                  math.sqrt(u1) * math.sin(2 * math.pi * u3),
                  math.sqrt(u1) * math.cos(2 * math.pi * u3)])
    return q


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def min_allowed(t1, t2):
    """Initial-packing minimum distance between atom types t1, t2
    (different molecules / vs tube): 0.85 x LJ sigma_ij (Lorentz-Berthelot),
    floored at 2.0 A (and 2.2 A when H is involved)."""
    s = 0.5 * (sigma_of(t1) + sigma_of(t2))
    e1 = P.ATOM_TYPES[t1][0]
    e2 = P.ATOM_TYPES[t2][0]
    floor = 2.2 if ("H" in (e1, e2)) else 2.0
    return max(floor, 0.85 * s)


# threshold matrix (11x11) for vectorized packing checks
_nt = max(P.ATOM_TYPES) + 1
MIN_ALLOWED = np.zeros((_nt, _nt))
for _i in P.ATOM_TYPES:
    for _j in P.ATOM_TYPES:
        MIN_ALLOWED[_i, _j] = min_allowed(_i, _j)


def rotation_aligning(rng, e, cone_deg):
    """Rotation matrix that maps unit vector e into a random direction
    sampled uniformly in a cone of half-angle cone_deg about +/- z
    (sign chosen at random), composed with a random spin about that
    direction (Shoemake quaternion)."""
    half = math.radians(cone_deg)
    u1, u2 = rng.random(), rng.random()
    theta = half * math.sqrt(u1)          # uniform in the cone
    phi = 2.0 * math.pi * u2
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    d = np.array([sgn * math.sin(theta) * math.cos(phi),
                  sgn * math.sin(theta) * math.sin(phi),
                  sgn * math.cos(theta)])
    # quaternion rotating e -> d
    v = np.cross(e, d)
    c = np.dot(e, d)
    if np.linalg.norm(v) < 1e-12:
        axis_spin = rand_quaternion(rng)
        m = quat_to_mat(axis_spin) if c > 0 else np.eye(3)
        return m
    axis = v / np.linalg.norm(v)
    ang = math.atan2(np.linalg.norm(v), c)
    q_align = np.array([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])
    # random spin about d
    qs = rand_quaternion(rng)
    qsm = quat_to_mat(qs)
    return quat_to_mat(q_align) @ qsm


class Packer:
    """Rejection packer with pack-then-relax completion (2026-10-14).

    Pure rejection (random sequential addition) jams near ~55-60% of the
    bulk-liquid density, so the documented bulk-density compositions cannot
    be reached by rejection alone (they were only reachable for the
    previous agent because the broken EMC template was internally
    collapsed). Following the repo's own W8 'grid + minimise' approach,
    molecules that cannot be placed strictly are placed at the
    best-of-N overlapping candidate (>= 0.70x the contact floor) and the
    whole configuration is then relaxed: every violating inter-body pair
    and wall contact pushes the two rigid bodies apart along the pair
    axis by a fraction of the deficit, until every contact clears the
    strict floor. The final configuration is verified against the same
    floors the strict packer enforces.
    """

    RELAX_RATE = 0.35
    RELAX_SWEEPS_PER_INSERT = 150
    RELAX_SWEEPS_FINAL = 2000
    TOLERANT_RATIO = 0.70

    def __init__(self, R, Lz, rng):
        self.R, self.Lz, self.rng = R, Lz, rng
        self.pos = np.zeros((0, 3))
        self.typ = np.array([], dtype=int)
        self.bodies = []        # (types, local, rot_matrix, com)
        self.n_tolerant = 0

    def _flat_from_bodies(self):
        pos = np.vstack([loc @ m.T + com
                         for _t, loc, m, com in self.bodies])
        typ = np.concatenate([t for t, _l, _m, _c in self.bodies])
        self.pos, self.typ = pos, typ

    def _ok(self, types, xyz, li_sep):
        tt = np.asarray(types)
        # against placed molecules (vectorized, z-PBC)
        if len(self.typ):
            dz = xyz[:, 2][:, None] - self.pos[None, :, 2]
            dz -= self.Lz * np.round(dz / self.Lz)
            d2 = ((xyz[:, 0][:, None] - self.pos[None, :, 0]) ** 2 +
                  (xyz[:, 1][:, None] - self.pos[None, :, 1]) ** 2 + dz ** 2)
            thr = MIN_ALLOWED[tt][:, self.typ] ** 2
            if np.any(d2 < thr):
                return False
            if li_sep and TYPE_OF["P"] in set(self.typ.tolist()):
                pmask = self.typ == TYPE_OF["P"]
                if np.any(d2[:, pmask] < 4.2 ** 2):
                    return False
        # radial gap to the tube wall
        gap = self.R - np.hypot(xyz[:, 0], xyz[:, 1])
        if np.any(gap < MIN_ALLOWED[tt, TYPE_OF["C_tube"]]):
            return False
        return True

    def _candidate(self, template, R_pack, longest_axis):
        types = [t for _, t, _, _ in template]
        local = np.array([xyz for _, _, _, xyz in template])
        tt = np.asarray(types)
        ttube = TYPE_OF["C_tube"]
        best = None            # (worst_ratio, xyz, m)
        for _ in range(20000):
            u, phi = self.rng.random(), 2 * math.pi * self.rng.random()
            r = R_pack * math.sqrt(u)
            z = self.rng.random() * self.Lz
            if self.rng.random() < 0.5 and longest_axis is not None:
                m = rotation_aligning(self.rng, longest_axis, 35.0)
            else:
                m = quat_to_mat(rand_quaternion(self.rng))
            xyz = local @ m.T + np.array([r * math.cos(phi),
                                          r * math.sin(phi), z])
            # worst normalized clearance (1.0 == at the strict floor)
            ratio = 1.0
            if len(self.typ):
                dz = xyz[:, 2][:, None] - self.pos[None, :, 2]
                dz -= self.Lz * np.round(dz / self.Lz)
                d2 = ((xyz[:, 0][:, None] - self.pos[None, :, 0]) ** 2 +
                      (xyz[:, 1][:, None] - self.pos[None, :, 1]) ** 2 +
                      dz ** 2)
                ratios = np.sqrt(d2 / MIN_ALLOWED[tt][:, self.typ] ** 2)
                ratio = min(ratio, float(ratios.min()))
            gap = self.R - np.hypot(xyz[:, 0], xyz[:, 1])
            ratio = min(ratio, float((gap /
                                      MIN_ALLOWED[tt, ttube]).min()))
            if ratio >= 1.0:
                return xyz, m, 1.0          # strict hit
            if best is None or ratio > best[0]:
                best = (ratio, xyz, m)
        return (None, None, best[0]) if best is None else \
            (best[1], best[2], best[0])

    def relax(self, sweeps):
        """Push overlapping rigid bodies (and wall contacts) apart until
        every inter-body pair and wall gap clears the strict floor.
        Translation-only (rigid geometry preserved); z is wrapped."""
        t_tube = TYPE_OF["C_tube"]
        t_li, t_p = TYPE_OF["Li"], TYPE_OF["P"]
        converged_at = None
        for sweep in range(sweeps):
            self._flat_from_bodies()
            pos, typ = self.pos, self.typ
            own = np.concatenate([np.full(len(t), k)
                                  for k, (t, _l, _m, _c)
                                  in enumerate(self.bodies)])
            n_at = len(pos)
            dz = pos[:, 2][:, None] - pos[None, :, 2]
            dz -= self.Lz * np.round(dz / self.Lz)
            d2 = ((pos[:, 0][:, None] - pos[None, :, 0]) ** 2 +
                  (pos[:, 1][:, None] - pos[None, :, 1]) ** 2 + dz ** 2)
            thr = MIN_ALLOWED[typ][:, typ].copy() * 1.002   # aim slightly
            # above the floor so the push-apart settles strictly clear of
            # it (boundary flicker otherwise leaves pairs at ratio 1.0000)
            np.fill_diagonal(thr, np.inf)
            viol = (d2 < thr ** 2) & (own[:, None] != own[None, :])
            ii, jj = np.nonzero(viol)
            ii, jj = ii[ii < jj], jj[ii < jj]
            # wall violations
            gap = self.R - np.hypot(pos[:, 0], pos[:, 1])
            wfloor = MIN_ALLOWED[typ, t_tube] * 1.002
            wv = np.nonzero(gap < wfloor)[0]
            if len(ii) == 0 and len(wv) == 0:
                converged_at = sweep
                break
            push = np.zeros((len(self.bodies), 3))
            rate = self.RELAX_RATE
            for a, b in zip(ii, jj):
                dvec = pos[a] - pos[b]
                dvec[2] -= self.Lz * round(dvec[2] / self.Lz)
                d = math.sqrt((dvec ** 2).sum())
                if d < 1e-9:
                    dvec = np.array([1.0, 0.0, 0.0])
                    d = 1.0
                deficit = thr[a, b] - d
                step = 0.5 * rate * deficit * dvec / d
                push[own[a]] += step
                push[own[b]] -= step
            for a in wv:
                rvec = np.array([pos[a, 0], pos[a, 1], 0.0])
                rn = math.hypot(rvec[0], rvec[1])
                if rn < 1e-9:
                    rvec = np.array([1.0, 0.0, 0.0])
                    rn = 1.0
                deficit = wfloor[a] - gap[a]
                push[own[a]] -= rate * deficit * rvec / rn
            for k, (_t, _l, _m, com) in enumerate(self.bodies):
                com += push[k]
                com[2] -= self.Lz * math.floor(com[2] / self.Lz)
        self._flat_from_bodies()
        return converged_at

    def verify(self):
        """True iff every inter-body pair and wall gap clears the strict
        floor within a 0.1% numerical tolerance (the Li-P 4.2 A separation
        is an insertion heuristic only - contact ion pairs are physical
        and relaxation is not required to undo them)."""
        pos, typ = self.pos, self.typ
        own = np.concatenate([np.full(len(t), k)
                              for k, (t, _l, _m, _c)
                              in enumerate(self.bodies)])
        dz = pos[:, 2][:, None] - pos[None, :, 2]
        dz -= self.Lz * np.round(dz / self.Lz)
        d2 = ((pos[:, 0][:, None] - pos[None, :, 0]) ** 2 +
              (pos[:, 1][:, None] - pos[None, :, 1]) ** 2 + dz ** 2)
        thr = MIN_ALLOWED[typ][:, typ] * 0.999
        np.fill_diagonal(thr, np.inf)
        inter = own[:, None] != own[None, :]
        if np.any((d2 < thr ** 2) & inter):
            return False
        gap = self.R - np.hypot(pos[:, 0], pos[:, 1])
        if np.any(gap < MIN_ALLOWED[typ, TYPE_OF["C_tube"]] * 0.999):
            return False
        return True

    def try_insert(self, template, R_pack, attempts, longest_axis):
        types = [t for _, t, _, _ in template]
        local = np.array([xyz for _, _, _, xyz in template])
        li_sep = template[0][0] == "Li"
        for it in range(attempts):
            u, phi = self.rng.random(), 2 * math.pi * self.rng.random()
            r = R_pack * math.sqrt(u)
            z = self.rng.random() * self.Lz
            if it % 2 == 0 and longest_axis is not None:
                m = rotation_aligning(self.rng, longest_axis, 35.0)
            else:
                m = quat_to_mat(rand_quaternion(self.rng))
            xyz = local @ m.T + np.array([r * math.cos(phi),
                                          r * math.sin(phi), z])
            if self._ok(types, xyz, li_sep):
                com = xyz[0] - (local @ m.T)[0]
                self.bodies.append((np.asarray(types), local, m, com))
                self._flat_from_bodies()
                return xyz
        return None

    def try_insert_relaxed(self, template, R_pack, longest_axis):
        """Strict first; on failure best-of-N tolerant candidate + relax."""
        strict = self.try_insert(template, R_pack, 60000, longest_axis)
        if strict is not None:
            return strict
        xyz, m, ratio = self._candidate(template, R_pack, longest_axis)
        if xyz is None or ratio < self.TOLERANT_RATIO:
            return None
        types = np.asarray([t for _, t, _, _ in template])
        local = np.array([a[3] for a in template])
        com = xyz[0] - (local @ m.T)[0]
        self.bodies.append((types, local, m, com))
        self.n_tolerant += 1
        self.relax(self.RELAX_SWEEPS_PER_INSERT)
        return xyz


# ----------------------------------------------------------------------------
# data-file writing
# ----------------------------------------------------------------------------

def write_data(path, box, tube_xyz, molecules, comment):
    """molecules: list of (mol_id, [(label,type,q,xyz)...], bonds_local)"""
    natoms = len(tube_xyz) + sum(len(m[1]) for m in molecules)
    nbonds = sum(len(m[2]) for m in molecules)
    lines = [f"LAMMPS data file. {comment}", ""]
    lines += [f"{natoms} atoms", f"{nbonds} bonds", "",
              f"{len(P.ATOM_TYPES)} atom types", "1 bond types", ""]
    xlo, xhi, ylo, yhi, zlo, zhi = box
    lines += [f"{xlo:.6f} {xhi:.6f} xlo xhi",
              f"{ylo:.6f} {yhi:.6f} ylo yhi",
              f"{zlo:.6f} {zhi:.6f} zlo zhi", "",
              "Masses", ""]
    for t in sorted(P.ATOM_TYPES):
        lines.append(f"{t} {P.AMU[P.ATOM_TYPES[t][0]]:.4f}")
    lines += ["", "Atoms # full", ""]
    idx = 0
    atom_rows = []
    # tube = molecule 1, frozen, q=0
    for i, xyz in enumerate(tube_xyz):
        idx += 1
        atom_rows.append(f"{idx} 1 {TYPE_OF['C_tube']} 0.0 "
                         f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    bond_rows = []
    offset = idx
    for mol_id, atoms, bonds_local in molecules:
        for lab, t, q, xyz in atoms:
            idx += 1
            atom_rows.append(f"{idx} {mol_id} {t} {q:+.4f} "
                             f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
        for i, j in bonds_local:
            bond_rows.append(f"{len(bond_rows)+1} 1 {offset+i+1} {offset+j+1}")
        offset += len(atoms)
    lines += atom_rows + ["", "Bonds", ""] + bond_rows
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return natoms, nbonds


# ----------------------------------------------------------------------------
# post-build geometric guard (added 2026-10-16, Stage A restart)
# ----------------------------------------------------------------------------

def validate_data_file(path, floor_ratio=0.95, chunk=256):
    """Post-build guard on a written LAMMPS 'full' data file (rows
    'id mol type q x y z'; no image flags). Refuses (RuntimeError) a build
    whose file would make LAMMPS die at setup/step 0:

      1. every INTER-molecular pair distance (minimum image, all 3 dims)
         must be >= floor_ratio x the packing floor MIN_ALLOWED[t1,t2];
      2. every molecule-tube pair distance likewise (the tube is
         molecule 1 / type C_tube; checked explicitly rather than via a
         radial gap so a mis-placed molecule cannot hide behind COM
         sampling radii);
      3. every molecule must be WHOLE in raw (as-written) coordinates:
         its raw diameter must match the minimum-image diameter within
         1 A - a molecule the writer left split across a periodic
         boundary produces huge raw intra-molecular distances;
      4. every written bond must be whole (raw length < 3.0 A; the
         templates' bonds are all 0.99-1.61 A).

    Returns a stats dict for the build manifest. Intra-molecular pairs
    (including the rigid templates' fixed geometry) are exempt from 1-2,
    as in the packer's own verify().
    """
    box = {}
    atoms, bonds = [], []
    sect = None
    with open(path) as f:
        for l in f:
            s = l.strip()
            if not s:
                continue
            low = s.lower()
            hit = False
            for ax in "xyz":
                if low.endswith(f"{ax}lo {ax}hi"):
                    box[ax] = (float(s.split()[0]), float(s.split()[1]))
                    hit = True
            if hit:
                continue
            if s.startswith("Atoms"):
                sect = "atoms"
            elif s.startswith("Bonds"):
                sect = "bonds"
            elif s.startswith(("Masses", "Velocities")):
                sect = None
            elif sect == "atoms":
                p = s.split()
                atoms.append([int(p[0]), int(p[1]), int(p[2]),
                              float(p[4]), float(p[5]), float(p[6])])
            elif sect == "bonds":
                p = s.split()
                bonds.append((int(p[2]), int(p[3])))
    if not box or not atoms:
        raise RuntimeError(f"{path}: could not parse box/atoms for the guard")
    A = np.array(atoms, float)
    order = np.argsort(A[:, 0])
    A = A[order]                       # rows sorted by atom id
    X, mol, typ = A[:, 3:6], A[:, 1].astype(int), A[:, 2].astype(int)
    ids = A[:, 0].astype(int)
    L = np.array([box["x"][1] - box["x"][0], box["y"][1] - box["y"][0],
                  box["z"][1] - box["z"][0]])
    t_tube = TYPE_OF["C_tube"]
    n = len(A)

    # --- 1+2: chunked min-image pair scan (inter-molecule, mol-vs-tube)
    min_inter, min_tube = np.inf, np.inf
    off_inter = off_tube = None
    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        d = X[s0:s1, None, :] - X[None, :, :]
        d -= L * np.round(d / L)
        d2 = (d ** 2).sum(-1)
        same = mol[s0:s1, None] == mol[None, :]
        m = (~same) & (d2 > 0.0)
        d2_m = np.where(m, d2, np.inf)
        tube_pair = m & ((typ[s0:s1, None] == t_tube) |
                         (typ[None, :] == t_tube))
        d2_t = np.where(tube_pair, d2, np.inf)
        k = np.unravel_index(np.argmin(d2_m), d2_m.shape)
        if d2_m[k] < min_inter:
            min_inter, off_inter = d2_m[k], (s0 + k[0], k[1])
        k = np.unravel_index(np.argmin(d2_t), d2_t.shape)
        if d2_t[k] < min_tube:
            min_tube, off_tube = d2_t[k], (s0 + k[0], k[1])
    min_inter, min_tube = math.sqrt(max(min_inter, 0.0)), \
        math.sqrt(max(min_tube, 0.0))

    def _pair_types(off):
        return int(typ[off[0]]), int(typ[off[1]])

    ti, tj = _pair_types(off_inter)
    if min_inter < floor_ratio * MIN_ALLOWED[ti, tj]:
        raise RuntimeError(
            f"{os.path.basename(path)}: BUILD REFUSED - inter-molecular "
            f"contact {min_inter:.3f} A (atom {ids[off_inter[0]]} type {ti} "
            f"- atom {ids[off_inter[1]]} type {tj}) is below "
            f"{floor_ratio:.2f} x the packing floor "
            f"{MIN_ALLOWED[ti, tj]:.3f} A")
    ti, tj = _pair_types(off_tube)
    if min_tube < floor_ratio * MIN_ALLOWED[ti, tj]:
        raise RuntimeError(
            f"{os.path.basename(path)}: BUILD REFUSED - molecule-tube "
            f"contact {min_tube:.3f} A (atom {ids[off_tube[0]]} type {ti} "
            f"- atom {ids[off_tube[1]]} type {tj}) is below "
            f"{floor_ratio:.2f} x the packing floor "
            f"{MIN_ALLOWED[ti, tj]:.3f} A")

    # --- 3: molecule wholeness in raw coordinates. The TUBE (molecule 1)
    # is exempt: it is a single frozen body that intentionally spans the
    # box, so its min-image diameter is legitimately shorter than its raw
    # diameter (any object larger than half the box has no unique min-image
    # diameter). Liquid molecules are all << L/2.
    worst_split = 0.0
    split_mol = None
    for m in np.unique(mol):
        if m == 1:
            continue
        ix = np.flatnonzero(mol == m)
        sub = X[ix]
        if len(ix) == 1:
            continue
        dv = sub[:, None, :] - sub[None, :, :]
        raw_span = float(np.sqrt((dv ** 2).sum(-1).max()))
        dd = dv - L * np.round(dv / L)
        mi_span = float(np.sqrt((dd ** 2).sum(-1).max()))
        if mi_span > 0.5 * L.min():
            continue                    # min-image diameter ill-defined
        if raw_span - mi_span > worst_split:
            worst_split, split_mol = raw_span - mi_span, int(m)
    if worst_split > 1.0:
        raise RuntimeError(
            f"{os.path.basename(path)}: BUILD REFUSED - molecule "
            f"{split_mol} is split across a periodic boundary in the "
            f"written file (raw diameter exceeds min-image diameter by "
            f"{worst_split:.1f} A)")

    # --- 4: bond wholeness
    row_of = {int(k): i for i, k in enumerate(ids)}
    max_bond = 0.0
    bad_bond = None
    for a, b in bonds:
        dv = X[row_of[a]] - X[row_of[b]]
        d = float(np.linalg.norm(dv))
        if d > max_bond:
            max_bond, bad_bond = d, (a, b)
    if max_bond > 3.0:
        raise RuntimeError(
            f"{os.path.basename(path)}: BUILD REFUSED - bond "
            f"{bad_bond[0]}-{bad_bond[1]} raw length {max_bond:.3f} A "
            f"(molecule broken across the boundary in the written file)")
    return dict(min_intermolecular_A=round(min_inter, 3),
                min_mol_tube_A=round(min_tube, 3),
                max_molecule_split_A=round(worst_split, 3),
                max_raw_bond_A=round(max_bond, 3),
                floor_ratio=floor_ratio, checked_pairs="all, min image")


# ----------------------------------------------------------------------------
# per-system build
# ----------------------------------------------------------------------------

def build_system(name):
    spec = P.SYSTEMS[name]
    seed = P.SEEDS[name]["build"]
    rng = np.random.default_rng(seed)
    n = spec["n"]
    ncell = spec["ncell"]

    # --- tube
    a = P.GRAPHENE_A
    R = a * n * math.sqrt(3.0) / (2.0 * math.pi)
    Lz = ncell * a
    tube_xyz = []
    for cell in range(ncell):
        z0 = cell * a
        for k in (0, 1):
            for c in range(n):
                phiA = 2.0 * math.pi * (c + k / 2.0) / n
                phiB = phiA + 2.0 * math.pi / (3.0 * n)
                z = z0 + k * a / 2.0
                tube_xyz.append([R * math.cos(phiA), R * math.sin(phiA), z])
                tube_xyz.append([R * math.cos(phiB), R * math.sin(phiB), z])
    tube_xyz = np.array(tube_xyz)
    d_tube = 2.0 * R

    # --- box: 1 nm vacuum padding beyond the tube in x and y
    pad = 10.0
    xhi = yhi = R + pad
    box = (-xhi, xhi, -yhi, yhi, 0.0, Lz)

    # --- composition
    comp = P.COMPOSITION[n]

    # pore volumes (for reporting/molarity):
    R_vdw = R - 3.2  # sigma_C/2 (1.7) + typical O half-width (1.5)
    V_geom = math.pi * R ** 2 * Lz / 1e3          # nm^3 (geometric cylinder)
    V_vdw = math.pi * R_vdw ** 2 * Lz / 1e3       # nm^3 (vdW-corrected)
    molarity = comp["n_LiPF6"] / (0.6022 * V_vdw)  # mol/L

    # --- pack electrolyte
    packer = Packer(R, Lz, rng)
    R_pack = R - 2.9
    ec_tmpl, ec_bonds, ec_ring = build_ec()
    emc_tmpl, emc_bonds = build_emc()
    pf6_tmpl, pf6_bonds = build_pf6()
    li_tmpl, li_bonds = build_li()

    def longest_axis(tmpl):
        loc = np.array([xyz for _, _, _, xyz in tmpl])
        c = loc - loc.mean(axis=0)
        w, v = np.linalg.eigh(c.T @ c)
        return v[:, np.argmax(w)]

    axis_of = {id(t): longest_axis(t)
               for t in (ec_tmpl, emc_tmpl, pf6_tmpl, li_tmpl)}

    molecules = []       # (mol_id, atoms(list of (label,type,q,xyz)), bonds)
    mol_id = 1           # tube is molecule 1
    # order: PF6, Li (ions separated at start), then EC before EMC.
    # Reordered 2026-10-14: with the corrected extended EMC template the
    # original EMC-first order jams (EMC chains fragment the remaining
    # voids before the more numerous disk-like EC molecules are placed).
    # Insertion order is a sampling detail only - it does not affect the
    # equilibrated physics, just packing success.
    order = ([pf6_tmpl] * comp["n_LiPF6"] + [li_tmpl] * comp["n_LiPF6"] +
             [ec_tmpl] * comp["n_EC"] + [emc_tmpl] * comp["n_EMC"])
    bonds_of = {id(pf6_tmpl): pf6_bonds, id(li_tmpl): li_bonds,
                id(emc_tmpl): emc_bonds, id(ec_tmpl): ec_bonds}
    for tmpl in order:
        strict_only = tmpl[0][0] in ("Li", "P")   # ions: strict insertion
        placed = None
        for rp in (R - 2.9, R - 2.7):   # fallback widens COM sampling only;
            # per-atom tube gaps are still enforced by _ok()
            if strict_only:
                for attempt_budget in (120000, 400000):
                    placed = packer.try_insert(tmpl, R_pack=rp,
                                               attempts=attempt_budget,
                                               longest_axis=axis_of[id(tmpl)])
                    if placed is not None:
                        break
            else:
                placed = packer.try_insert_relaxed(
                    tmpl, R_pack=rp, longest_axis=axis_of[id(tmpl)])
            if placed is not None:
                break
        if placed is None:
            raise RuntimeError(f"{name}: packing failed for a molecule "
                               f"({tmpl[0][0]})")

    # final relaxation + strict-floor verification of the whole packing
    relax_converged_at = packer.relax(packer.RELAX_SWEEPS_FINAL)
    if not packer.verify():
        raise RuntimeError(f"{name}: relaxation failed to clear all "
                           f"contact floors")

    # build the molecule list from the FINAL (post-relaxation) positions
    k = 0
    for tmpl in order:
        mol_id += 1
        atoms = [(lab, t, q, packer.pos[k + i].copy())
                 for i, (lab, t, q, _) in enumerate(tmpl)]
        molecules.append((mol_id, atoms, bonds_of[id(tmpl)]))
        k += len(tmpl)

    # --- charge & count checks
    tot_q = sum(q for _, m, _ in molecules for _, _, q, _ in m)
    assert abs(tot_q) < 1e-6, f"system not neutral: {tot_q}"
    for mid, m, _ in molecules:
        mq = sum(q for _, _, q, _ in m)
        lab0 = m[0][0]
        expect = {"CO": 0.0, "P": -1.0, "Li": 1.0}.get(lab0, None)
        if expect is not None:
            assert abs(mq - expect) < 1e-6, f"mol {mid} charge {mq}"

    # --- write data file
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    data_path = os.path.join(HERE, "data", f"{name}.data")
    natoms, nbonds = write_data(
        data_path, box, tube_xyz, molecules,
        f"LiPF6/EC:EMC in armchair ({n},{n}) CNT; build seed {seed}")

    # --- initial-contact stats (honest packing quality): min distance
    # between atoms of DIFFERENT molecules (intra-body pairs excluded via
    # body ownership; fixed 2026-10-14 - the previous stat was dominated
    # by the 1.09 A C-H bonds inside a molecule), and to the tube wall
    own = np.concatenate([np.full(len(t), k) for k, (t, _l, _m, _c)
                          in enumerate(packer.bodies)])
    n_pl = len(packer.typ)
    best_mm = None
    for i in range(n_pl):
        dz = packer.pos[:, 2] - packer.pos[i, 2]
        dz -= Lz * np.round(dz / Lz)
        d2 = ((packer.pos[:, 0] - packer.pos[i, 0]) ** 2 +
              (packer.pos[:, 1] - packer.pos[i, 1]) ** 2 + dz ** 2)
        d2[i] = np.inf
        cand = d2[own != own[i]]
        if len(cand):
            m = cand.min()
            best_mm = m if best_mm is None else min(best_mm, m)
    wall_min = float(np.min(R - np.hypot(packer.pos[:, 0], packer.pos[:, 1])))
    info = dict(
        name=name, cnt_chirality=[n, n], cnt_diameter_A=round(d_tube, 4),
        cnt_length_A=round(Lz, 4), cnt_atoms=len(tube_xyz),
        R_pack_A=R_pack, box=[round(v, 4) for v in box],
        composition={k: comp[k] for k in comp},
        n_atoms_total=natoms, n_bonds=nbonds,
        V_pore_geom_nm3=round(V_geom, 4), V_pore_vdw_nm3=round(V_vdw, 4),
        molarity_M_vdw_volume=round(molarity, 3),
        ec_ring_angles={k: round(v, 2) for k, v in ec_ring.items()},
        initial_min_atom_gap_A=round(math.sqrt(best_mm), 3),
        initial_min_wall_gap_A=round(wall_min, 3),
        build_seed=seed,
        packer="rejection + pack-then-relax (W8-style push-apart, "
               "2026-10-14): strict floor = max(2.0 A, 2.2 A with H, "
               "0.85 x LB-mixed sigma) inter-body and wall; tolerant "
               "insert >= 0.70 x floor then relaxed back above floor",
        n_tolerant_inserts=packer.n_tolerant,
        final_relax_converged_sweep=relax_converged_at,
    )
    return info, data_path


def main():
    systems = sys.argv[1:] or list(P.SYSTEMS)
    manifest = {}
    for name in systems:
        info, data_path = build_system(name)
        manifest[name] = info
        print(f"[build] {name}: {info['n_atoms_total']} atoms, "
              f"{info['molarity_M_vdw_volume']} M, -> {data_path}")
    with open(os.path.join(HERE, "data", "build_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("[build] wrote data/build_manifest.json")


if __name__ == "__main__":
    main()
