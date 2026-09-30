"""Defences that HELD under attack (green regression assets).  Each of these
inputs is a plausible way to fool a criterion that the current code already
handles correctly; they are kept so a regression cannot silently reopen the
hole.  Numbers in each docstring were measured on 2026-09-30."""
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chaord.build import build_program
from chaord.build.crystal import build_conventional
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lang.parser import parse_text
from chaord.lift import lift_frame
from tools import acceptance as acc

from helpers_frames import divacancy_fcc_cu, mixed_l12_defects, thermal

ROOT = Path(__file__).resolve().parents[2]
METAL = ("core", "metal")


def _detected(text):
    return {t: int(c) for t, c in re.findall(r"defect (\S+) count (\d+)", text)}


def test_green_a4_adjacent_vacancies_reported_as_two_vacancies():
    """The suite always plants separated defects; the physically common
    ADJACENT pair (divacancy) is handled: planted 2 adjacent V_Cu -> detected
    {'V_Cu': 2} at the 0.8-Tm-equivalent amplitude."""
    metal = load_dialect(METAL)
    perfect = build_conventional("fcc", {"a": 3.615}, ("Cu",), (3, 3, 3))
    d = acc.median_nn_distance(perfect.pos, perfect.cell_diag)
    hot = thermal(divacancy_fcc_cu(),
                  float(metal.threshold("thermal_test_amplitude")) * d, 7)
    det = _detected(acc.format_program_text(lift_frame(hot, metal, mode="defects")))
    assert det == {"V_Cu": 2}, (
        f"adjacent vacancy pair must lift as 2 V_Cu (or an explicit "
        f"divacancy token), got {det}"
    )


def test_green_a4_mixed_defect_cell_detected_exactly():
    """Mixed types in ONE cell (V_Ni 3 + Al_Ni 4 + Ni_i 3 at the 0.8-Tm
    amplitude): the lift detects exactly the planted multiset (P=R=1.000) --
    the suite never plants mixed cells (the per-cell metric cannot score
    them; see test_power_mutations) but the lift itself is correct."""
    metal = load_dialect(METAL)
    frame, edge = mixed_l12_defects()
    d = acc.median_nn_distance(frame.pos, frame.cell_diag)
    hot = thermal(frame, float(metal.threshold("thermal_test_amplitude")) * d, 11)
    det = _detected(acc.format_program_text(lift_frame(hot, metal, mode="defects")))
    assert det == {"V_Ni": 3, "Al_Ni": 4, "Ni_i": 3}, (
        f"mixed cell must be detected exactly, got {det}"
    )


@pytest.mark.slow
def test_green_a4_hot_perfect_crystal_reports_no_false_defects():
    """A defect-free fcc-Cu frame swept to the 1.0-Tm-equivalent amplitude
    (the promise covers 0.9 Tm): 0 false defects at every step
    (0.80/0.85/0.90/0.95/1.00 Tm measured)."""
    metal = load_dialect(METAL)
    perfect = build_conventional("fcc", {"a": 3.615}, ("Cu",), (3, 3, 3))
    d = acc.median_nn_distance(perfect.pos, perfect.cell_diag)
    anchor = float(metal.threshold("thermal_test_amplitude"))
    for tm in (0.80, 0.90, 1.00):
        hot = thermal(perfect, anchor * (tm / 0.8) ** 0.5 * d, int(tm * 1000))
        det = _detected(acc.format_program_text(
            lift_frame(hot, metal, mode="defects")))
        assert not det, f"{tm:.2f} Tm-equivalent perfect crystal: false {det}"


def test_green_a10_lift_text_stable_across_hash_seeds():
    """PYTHONHASHSEED 0/1/42 in separate processes: identical sha256 over the
    lifted text of 5 diverse cases (defect crystal, interface, solution,
    reactive)."""
    prog = r"""
import sys, hashlib
sys.path.insert(0, {root!r}/src)
sys.path.insert(0, {root!r})
from chaord.dialects import load_dialect
from chaord.io.frames import read_frame
from chaord.lift import lift_frame
from tools import acceptance as acc
h = hashlib.sha256()
for cid, names in (("defects/l12_ni3al_vac_antisite", ("core", "metal")),
                   ("interface/lj_solid_liquid", ("core", "lj")),
                   ("interfaces/cu_water", ("core", "metal")),
                   ("solutions/nacl_aq", ("core", "molecular")),
                   ("reactive/water_oh_h_box20", ("core", "molecular"))):
    case = acc.case_by_id(cid)
    frame = read_frame(case["frames"][0])
    h.update(acc.format_program_text(
        lift_frame(frame, load_dialect(names))).encode())
print(h.hexdigest())
""".format(root=str(ROOT))
    import os
    digests = set()
    for seed in ("0", "1", "42"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", prog],
                           capture_output=True, text=True, timeout=300,
                           cwd=str(ROOT), env=env)
        digests.add(r.stdout.strip())
    assert len(digests) == 1, f"lift text differs across hash seeds: {digests}"


def test_green_a3_l12_thermal_frame_round_trips_byte_identical():
    """The one stored thermal crystal frame whose lift builds: byte-identical
    round-trip text (measured 1/18 overall; keep the survivor green)."""
    dl = load_dialect(METAL)
    case = acc.case_by_id("crystals/l12_ni3al")
    frame = read_frame(case["frames"][0])
    t1 = acc.format_program_text(lift_frame(frame, dl))
    rebuilt = build_program(parse_text(t1), dl, rng=np.random.default_rng(5))
    t2 = acc.format_program_text(lift_frame(rebuilt, dl))
    assert t2 == t1
