"""RD2 (docs/reviews/research_decisions.md): a nanotube statement with
chirality, its exact build, and the lift-side chirality inference.

Red-first suite: written against the decision record before the code.
Geometry constants follow Saito, Dresselhaus & Dresselhaus, "Physical
Properties of Carbon Nanotubes" (1998), ch. 3: graphene a = 2.46 A,
|Ch| = a sqrt(n^2+nm+m^2), d = |Ch|/pi, d_R = gcd(2m+n, 2n+m),
|T| = sqrt(3)|Ch|/d_R, atoms per translation period = 4(n^2+nm+m^2)/d_R."""
import numpy as np
import pytest

from chaord.lang.errors import ChaordError
from chaord.lang.fmt import format_program
from chaord.lang.ir import ir_equal
from chaord.lang.parser import parse_text

A = 2.46  # graphene lattice constant, A (Saito/Dresselhaus ch. 3)
NL = chr(10)


def _nt_program(chirality="(10,10)", length="39.36", cell="33.564 33.564 39.36",
                conserve="  conserve atoms C 640", geo=("0. 0. 19.68",
                                                        "6.782", "39.36")):
    center, radius, glen = geo
    text = (f"chaord 0.1{NL}dialect core + carbon{NL}{NL}"
            f"system {{{NL}  cell {cell} A{NL}  pbc xyz{NL}  seed 3{NL}"
            f"{conserve}{NL}}}{NL}{NL}"
            f"physics {{{NL}  backend mlp{NL}  model \"mace-mp-0\"{NL}}}{NL}{NL}"
            f"crystal tube : cylinder axis z center {center} "
            f"radius {radius} A length {glen} A {{{NL}"
            f"  nanotube {chirality} length {length} A{NL}}}{NL}{NL}"
            f"residual none{NL}")
    return text


# ------------------------------------------------------------ parse / format --

def test_nanotube_statement_parses_and_round_trips():
    text = _nt_program()
    program = parse_text(text)
    region = next(b for b in program.blocks if b.t == "region")
    stmt = next(s for s in region.statements if s.key == "nanotube")
    assert stmt.kind == "build"
    ch = stmt.values[0]
    assert ch.t == "chir"
    assert (ch.n, ch.m) == (10, 10)
    assert stmt.values[1].text == "length"
    assert stmt.values[2].num == "39.36" and stmt.values[2].unit == "A"
    assert format_program(program) == text
    assert ir_equal(parse_text(format_program(program)), program)


def test_chirality_canonicalisation_on_format():
    """A spaced pair reprints canonically and re-parses to the same IR."""
    text = _nt_program(chirality="(13, 7)", length="49.9666", cell="53.5 53.5 49.9666",
                       conserve="  conserve atoms C 824",
                       geo=("0. 0. 24.9833", "6.8903", "49.9666"))
    program = parse_text(text)
    out = format_program(program)
    assert "nanotube (13,7) length 49.9666 A" in out
    assert ir_equal(parse_text(out), program)


# ------------------------------------------------------------------- build ----

def _d_r(n, m):
    from math import gcd
    return gcd(2 * m + n, 2 * n + m)


def _build(chirality, length, conserve_n=None):
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    n, m = chirality
    dr = _d_r(n, m)
    if dr == 0 or not (n >= m >= 0 and n >= 1):
        # the builder must refuse before any geometry math: a minimal program
        text = (f"chaord 0.1{NL}dialect core + carbon{NL}{NL}"
                f"system {{{NL}  cell 30 30 40 A{NL}}}{NL}{NL}"
                f"crystal tube : all {{{NL}"
                f"  nanotube ({n},{m}) length {length:g} A{NL}}}{NL}")
        program = parse_text(text)
        return (build_program(program, load_dialect(("core", "carbon")),
                              physics=False), 0.0, 0, 0.0)
    N = n * n + n * m + m * m
    ch = A * np.sqrt(N)
    t_len = np.sqrt(3) * ch / _d_r(n, m)
    n_t = max(1, round(length / t_len))
    built = n_t * t_len
    atoms = 4 * N * n_t // _d_r(n, m)
    d = ch / np.pi
    box = d + 2 * 10.0
    # the program must describe the built tube as exactly as the assertions
    # read it (radius about the cell-centre axis to 1e-6 A): the centre is
    # derived from the PRINTED cell so the two agree bit-for-bit, and the
    # axial length is rounded UP so the py = 0 atom never wraps below 0
    box_txt = f"{box:.4f}"
    z_txt = f"{np.ceil(built * 10 ** 10) / 10 ** 10:.10f}"
    cx = float(box_txt) / 2
    cz = float(z_txt) / 2
    conserve = (f"  conserve atoms C {atoms}" if conserve_n is None
                else (f"  conserve atoms C {conserve_n}" if conserve_n else ""))
    text = (f"chaord 0.1{NL}dialect core + carbon{NL}{NL}"
            f"system {{{NL}  cell {box_txt} {box_txt} {z_txt} A{NL}"
            f"  pbc xyz{NL}  seed 3{NL}{conserve}{NL}}}{NL}{NL}"
            f"physics {{{NL}  backend mlp{NL}  model \"mace-mp-0\"{NL}}}{NL}{NL}"
            f"crystal tube : cylinder axis z center {cx:.6f} {cx:.6f} "
            f"{cz:.11f} radius {d / 2:.4f} A length {z_txt} A {{{NL}"
            f"  nanotube ({n},{m}) length {length:g} A{NL}}}{NL}{NL}"
            f"residual none{NL}")
    program = parse_text(text)
    frame = build_program(program, load_dialect(("core", "carbon")),
                          rng=np.random.default_rng(0), physics=False)
    return frame, built, atoms, ch


@pytest.mark.parametrize("chirality,length", [((10, 10), 39.36),
                                              ((16, 0), 38.34),
                                              ((13, 7), 49.9666)])
def test_build_nanotube_is_the_ideal_construction(chirality, length):
    """Atom count, radius and the unrolled graphene mesh are exact (1e-6 A)."""
    n, m = chirality
    frame, built, atoms, ch = _build(chirality, length)
    pos = frame.pos
    assert len(pos) == atoms
    assert set(frame.symbols) == {"C"}
    L = frame.cell_diag
    R = ch / (2 * np.pi)
    # every carbon sits exactly on the tube radius about the cell-centre axis
    r = np.hypot(pos[:, 0] - L[0] / 2, pos[:, 1] - L[1] / 2)
    assert np.allclose(r, R, atol=1e-6)
    # the tube fills the axial cell exactly (commensurate periods)
    assert -1e-9 <= pos[:, 2].min() and pos[:, 2].max() < built + 1e-9
    assert abs(L[2] - built) < 1e-6
    # unroll the built tube and compare against an independent enumeration of
    # the graphene sheet cut by the (Ch, N_T x T) cell (Saito/Dresselhaus ch. 3)
    phi = np.arctan2(pos[:, 1] - L[1] / 2, pos[:, 0] - L[0] / 2)
    x = np.mod(R * phi, ch)
    y = np.mod(pos[:, 2], built)
    a1 = A * np.array([1.0, 0.0])
    a2 = A * np.array([0.5, np.sqrt(3) / 2])
    basis = [np.zeros(2), (a1 + a2) / 3]
    dr = _d_r(n, m)
    chv = n * a1 + m * a2
    chh = chv / ch                              # circumferential unit vector
    tvec = ((2 * m + n) * a1 - (2 * n + m) * a2) / dr
    tlen = np.sqrt(3) * ch / dr
    th = tvec / tlen                            # axial unit vector
    n_t = round(built / tlen)
    t1, t2 = (2 * m + n) // dr, -((2 * n + m) // dr)
    ref = []
    # cover the (Ch, N_T x T) cell's (u, v) parallelogram -- corners (0,0),
    # (n, m), N_T (t1, t2) and their sum -- with a 3-cell margin
    for u in range(-3, n + n_t * t1 + 4):
        for v in range(n_t * t2 - 3, m + 3):
            for b in basis:
                p = u * a1 + v * a2 + b
                px, py = p @ chh, p @ th
                if -1e-9 <= px < ch - 1e-9 * ch and -1e-9 <= py < built - 1e-9 * built:
                    ref.append((px, py))
    assert len(ref) == atoms
    got = sorted(zip(np.round(x, 6), np.round(y, 6)))
    want = sorted((round(px, 6), round(py, 6)) for px, py in ref)
    assert got == want


def test_build_nanotube_refuses_wrong_conserve():
    with pytest.raises(ChaordError):
        _build((10, 10), 39.36, conserve_n=639)


def test_build_nanotube_refuses_noncanonical_chirality():
    with pytest.raises(ChaordError):
        _build((0, 0), 39.36)
    with pytest.raises(ChaordError):
        _build((3, 7), 39.36)  # m > n


def test_build_nanotube_refuses_too_small_cell():
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    text = _nt_program(cell="10.0 10.0 39.36")   # box narrower than the tube
    with pytest.raises(ChaordError):
        build_program(parse_text(text), load_dialect(("core", "carbon")),
                      physics=False)


# -------------------------------------------------------------------- lift ----

def _infer(frame):
    from chaord.lift.nanotube import infer_nanotube_chirality
    return infer_nanotube_chirality(frame.pos, frame.cell)


@pytest.mark.parametrize("chirality,length", [((10, 10), 39.36),
                                              ((16, 0), 38.34),
                                              ((13, 7), 49.9666)])
def test_lift_infers_chirality_back_exactly(chirality, length):
    frame, built, _atoms, _ch = _build(chirality, length)
    got = _infer(frame)
    assert (got.n, got.m) == chirality
    assert abs(got.length - built) < 1e-6


def test_lift_refuses_fcc_carbon():
    """A non-tube frame is honestly refused, never assigned a chirality."""
    a = 3.5
    base = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], float) * a / 2
    reps = (3, 3, 7)
    pos = np.concatenate([base + np.array([i, j, k]) * a
                          for i in range(reps[0]) for j in range(reps[1])
                          for k in range(reps[2])])
    cell = np.diag(np.array(reps, float) * a)
    from chaord.io.frames import Frame
    frame = Frame(pos=pos, cell=cell, symbols=["C"] * len(pos))
    with pytest.raises(ChaordError):
        _infer(frame)


def test_lift_refuses_flat_graphene_sheet():
    """A flat sheet is not a tube either (no single radius about any axis)."""
    a1 = A * np.array([1.0, 0.0, 0.0])
    a2 = A * np.array([0.5, np.sqrt(3) / 2, 0.0])
    basis = [np.zeros(3), (a1 + a2) / 3]
    pos = np.concatenate([u * a1 + v * a2 + b
                          for u in range(20) for v in range(20)
                          for b in basis]).reshape(-1, 3)
    cell = np.diag([20 * A, 20 * A, 20.0])
    from chaord.io.frames import Frame
    frame = Frame(pos=pos, cell=cell, symbols=["C"] * len(pos))
    with pytest.raises(ChaordError):
        _infer(frame)


def test_lift_survives_boundary_rounding():
    """An atom a hair below the box origin wraps to exactly L in floating
    point (np.mod(-1e-16, L) == L); the KD-tree boxsize contract requires
    [0, box) strictly. This bit the CI ubuntu runner (2026-10-14, run
    38564929887: ValueError 'Some input data are greater than the size of
    the periodic box' on chirality1/chirality2) while windows passed -- the
    inference must clamp, not raise."""
    frame, built, _atoms, _ch = _build((10, 10), 39.36)
    pos = frame.pos.copy()
    # rigid translation so the lowest atom sits EXACTLY on the origin (the
    # lattice is untouched; the inference is translation-invariant), then a
    # uniform hair below zero: np.mod wraps that one atom to exactly L
    pos[:, 2] -= pos[:, 2].min()
    pos[:, 2] -= 1e-15
    from chaord.io.frames import Frame
    shifted = Frame(pos=pos, cell=frame.cell, symbols=frame.symbols)
    got = _infer(shifted)
    assert (got.n, got.m) == (10, 10)
    assert abs(got.length - built) < 1e-6
