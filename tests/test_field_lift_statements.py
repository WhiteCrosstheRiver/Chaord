"""RD4 (docs/reviews/research_decisions.md): the field's measurable
consequence, asserted.

Red-first suite: written against the decision record before the code. When
the caller supplies `field=`, the lifted program must carry the statement-level
evidence -- `assert field_dipole_p1 <v> +- <tol> along <axis>` in the system
block next to `state field`, measured with cv.field.dipole_orientation from
the frame's own charges -- or, when the dialect cannot measure it, a
provenance note naming the reason (RD3's binding `state field` statement
stands either way). With `field=` absent: nothing field-related at all.

The expected P1 values are computed test-side from the placed geometry
(Rodrigues rotations of registry templates at known angles), the same
independent-implementation discipline as tests/test_cv_field.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chaord.build.molecules import TEMPLATES               # noqa: E402
from chaord.dialects import load_dialect                   # noqa: E402
from chaord.io.frames import Frame                         # noqa: E402
from chaord.lang.fmt import format_program                 # noqa: E402
from chaord.lang.ir import ir_equal                        # noqa: E402
from chaord.lang.parser import parse_text                  # noqa: E402
from chaord.lift import lift_frame                         # noqa: E402

FIELD = {"component": "E", "value": 0.2, "unit": "V/A", "axis": "z"}

# per-atom site charges of the templates used below, template atom order.
# H2O: SPC/E (Berendsen, Grigera & Straatsma, J. Phys. Chem. 91, 6269
# (1987)), the research dialect's own table. EC: Nutakki 2023 Table III
# (CO +0.99, OS -0.44, OC -0.63, CT +0.22, HC +0.02; the template's atom
# order is C1=CO, O2=OS, C3/C4=CT, O5=OS, O6=OC, H x4=HC) -- the charge
# model the research runs actually used (research/cnt_electrolyte/params.py
# EC_CHARGES). Both sets are neutral, as dipole_orientation requires.
_SITE_CHARGES = {
    "H2O": [-0.8476, 0.4238, 0.4238],
    "EC": [0.99, -0.44, 0.22, 0.22, -0.44, -0.63, 0.02, 0.02, 0.02, 0.02],
}


def _rotation_from_to(u, v):
    """Rotation matrix mapping unit vector u onto unit vector v (Rodrigues)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    c = float(np.dot(u, v))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:                # antiparallel: pi about a perpendicular
        perp = np.cross(u, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-12:
            perp = np.cross(u, [0.0, 1.0, 0.0])
        perp = perp / np.linalg.norm(perp)
        K = np.array([[0.0, -perp[2], perp[1]],
                      [perp[2], 0.0, -perp[0]],
                      [-perp[1], perp[0], 0.0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis = np.cross(u, v)
    axis = axis / np.linalg.norm(axis)
    s = float(np.sqrt(1.0 - c * c))
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _template_frame(name, rotations, centres, L, with_charges=True):
    """A frame of registry-template molecules, one per (rotation, centre).

    `rotations[i]` maps the template's own dipole direction onto the wanted
    orientation; per-atom site charges from `_SITE_CHARGES` ride in
    frame.info["charges"] (the LAMMPS reader's convention) when
    `with_charges` -- the research case's frames always carry them."""
    t = TEMPLATES[name]
    rel = np.asarray(t["rel"], float)
    charge = np.asarray(_SITE_CHARGES[name], float)
    pos = np.vstack([np.asarray(c, float) + rel @ R.T
                     for R, c in zip(rotations, centres)])
    symbols = list(t["symbols"]) * len(rotations)
    info = {"charges": np.tile(charge, len(rotations))} if with_charges else {}
    return Frame(pos=pos, cell=np.diag(np.asarray(L, float)),
                 symbols=symbols, pbc=(True, True, True), info=info)


def _template_unit_dipole(name):
    """The template's own dipole direction under its site-charge model."""
    t = TEMPLATES[name]
    rel = np.asarray(t["rel"], float)
    q = np.asarray(_SITE_CHARGES[name], float)
    mu = (q[:, None] * rel).sum(axis=0)
    return mu / np.linalg.norm(mu)


def _species_view(dl, species):
    """The same dialect with `field_dipole_species` named by the caller (the
    measurement view the first-result pipeline passes; RD4: the element-keyed
    site-charge table cannot express EC's per-site charges, so the carbonate
    solvents join as a view over the dialect, never as its default)."""
    from chaord.dialects import Dialect
    thresholds = dict(dl._data["thresholds"])
    thresholds["field_dipole_species"] = list(species)
    return Dialect(dl.names, {**dl._data, "thresholds": thresholds},
                   dl.versions)

def _system_statements(program):
    return next(b for b in program.blocks if b.t == "system").statements


def _provenance_notes(program):
    prov = next((b for b in program.blocks if b.t == "provenance"), None)
    if prov is None:
        return []
    return [s.values[0].text for s in prov.statements
            if s.key == "note" and s.values and s.values[0].t == "s"]


def _p1_statement(program):
    return next((s for s in _system_statements(program)
                 if s.kind == "assert" and s.key == "field_dipole_p1"), None)


# ------------------------------------------------------------- the assert --

@pytest.mark.parametrize("name,mode", [("H2O", "fluid"), ("EC", "auto")])
def test_lift_with_field_asserts_measured_p1(name, mode):
    """A field program carries the P1 assert with the measured value.

    Three template molecules are placed with their dipoles along +z, at
    60 degrees to +z, and in the xy plane: the expected P1 is the plain mean
    of (1, cos 60, 0) -- test-side arithmetic, no library CV involved."""
    u = _template_unit_dipole(name)
    thetas = (0.0, np.pi / 3, np.pi / 2)
    rots = [_rotation_from_to(u, [np.sin(t), 0.0, np.cos(t)]) for t in thetas]
    centres = [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]]
    frame = _template_frame(name, rots, centres, [12.0, 12.0, 18.0])
    expected = float(np.mean([1.0, np.cos(np.pi / 3), 0.0]))
    dl = load_dialect(("core", "molecular", "research"))
    if name == "EC":      # the carbonate solvent joins as a species view (RD4)
        dl = _species_view(dl, ["C3H4O3"])
    program = lift_frame(frame, dl, T=300.0, mode=mode, field=FIELD)
    stmt = _p1_statement(program)
    assert stmt is not None, (f"a {name} program lifted with field= carries no "
                              "field_dipole_p1 assert (RD4)")
    value, tol = stmt.values[0], stmt.values[1]
    assert value.t == "q" and value.num == f"{expected:.3f}"
    assert tol.t == "tol" and tol.value.num == f"{1.0 / np.sqrt(3):.3f}"
    assert stmt.values[2].text == "along" and stmt.values[3].text == "z"


def test_p1_assert_sits_next_to_state_field_in_the_system_block():
    u = _template_unit_dipole("H2O")
    frame = _template_frame("H2O",
                            [np.eye(3), np.eye(3), np.eye(3)],
                            [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]],
                            [12.0, 12.0, 18.0])
    dl = load_dialect(("core", "molecular", "research"))
    program = lift_frame(frame, dl, T=300.0, mode="fluid", field=FIELD)
    stmts = _system_statements(program)
    keys = [(s.kind, s.key) for s in stmts]
    assert ("state", "field") in keys and ("assert", "field_dipole_p1") in keys
    assert keys.index(("state", "field")) + 1 == keys.index(
        ("assert", "field_dipole_p1"))


def test_p1_assert_value_is_measured_not_the_field_value():
    """The assert carries the physics (the orientation average), never the
    caller's field strength: 0.2 V/A must not appear as the P1."""
    u = _template_unit_dipole("H2O")
    frame = _template_frame("H2O",
                            [_rotation_from_to(u, [0.0, 0.0, 1.0])] * 3,
                            [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]],
                            [12.0, 12.0, 18.0])
    dl = load_dialect(("core", "molecular", "research"))
    program = lift_frame(frame, dl, T=300.0, mode="fluid", field=FIELD)
    stmt = _p1_statement(program)
    assert stmt is not None
    assert stmt.values[0].num == "1.000"    # three perfect cosines
    assert "0.2" not in stmt.values[0].num


def test_p1_assert_provenance_records_the_measurement_and_the_bound():
    u = _template_unit_dipole("H2O")
    frame = _template_frame("H2O",
                            [_rotation_from_to(u, [0.0, 0.0, 1.0])] * 3,
                            [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]],
                            [12.0, 12.0, 18.0])
    dl = load_dialect(("core", "molecular", "research"))
    program = lift_frame(frame, dl, T=300.0, mode="fluid", field=FIELD)
    notes = _provenance_notes(program)
    assert any("field P1 measured" in n and "frame's own" in n for n in notes), notes
    assert any("a-priori standard-error bound" in n and "not an empirical floor"
               in n for n in notes), notes


def test_p1_assert_parses_and_round_trips():
    u = _template_unit_dipole("H2O")
    frame = _template_frame("H2O",
                            [_rotation_from_to(u, [0.0, 0.0, 1.0])] * 3,
                            [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]],
                            [12.0, 12.0, 18.0])
    dl = load_dialect(("core", "molecular", "research"))
    program = lift_frame(frame, dl, T=300.0, mode="fluid", field=FIELD)
    text = format_program(program)
    assert "assert field_dipole_p1 1.000 +- 0.577 along z" in text
    units = sorted(dl.units | {"V/A", "K"})
    again = parse_text(text, units=units)
    assert ir_equal(again, program)


def test_field_assert_key_declared_in_research_dialect():
    assert load_dialect(("core", "research")).key_allowed("system",
                                                          "field_dipole_p1")


# --------------------------------------------------------------- absence --

def test_no_field_means_no_field_asserts():
    u = _template_unit_dipole("H2O")
    frame = _template_frame("H2O",
                            [_rotation_from_to(u, [0.0, 0.0, 1.0])] * 3,
                            [[6.0, 6.0, 3.0], [6.0, 6.0, 9.0], [6.0, 6.0, 15.0]],
                            [12.0, 12.0, 18.0])
    dl = load_dialect(("core", "molecular", "research"))
    program = lift_frame(frame, dl, T=300.0, mode="fluid")
    text = format_program(program)
    assert "field" not in text and "field_dipole_p1" not in text


# ------------------------------------------------- degrade, never silence --

def _neon_frame(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 25.0, size=(32, 3))
    return Frame(pos=pos, cell=np.diag([25.0, 25.0, 25.0]),
                 symbols=["Ne"] * 32, pbc=(True, True, True))


def test_dialect_without_the_observable_degrades_with_a_note():
    """RD3's neon lift (core + lj) keeps its binding statement; the missing
    assert is recorded in provenance, never silent (RD4)."""
    dl = load_dialect(("core", "lj"))
    program = lift_frame(_neon_frame(), dl, mode="fluid", field=FIELD)
    text = format_program(program)
    assert "state field E 0.2 V/A along z" in text
    assert "field_dipole_p1" not in text
    notes = _provenance_notes(program)
    assert any("field observables not asserted" in n for n in notes), notes


def test_uncharged_carbonate_frame_degrades_with_the_reason():
    """No frame charges and no site-charge model for C3H4O3: the reason is
    named (fail-closed CV), the field statement still prints."""
    u = _template_unit_dipole("EC")
    frame = _template_frame("EC",
                            [_rotation_from_to(u, [0.0, 0.0, 1.0])] * 4,
                            [[8.0, 6.0, 4.0], [8.0, 6.0, 12.0],
                             [8.0, 14.0, 4.0], [8.0, 14.0, 12.0]],
                            [16.0, 20.0, 16.0], with_charges=False)
    dl = _species_view(load_dialect(("core", "molecular", "research")),
                       ["C3H4O3", "C4H8O3"])
    program = lift_frame(frame, dl, T=300.0, mode="auto", field=FIELD)
    assert _p1_statement(program) is None
    notes = _provenance_notes(program)
    assert any("field observables not asserted" in n
               and "site-charge" in n for n in notes), notes
