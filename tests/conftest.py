"""Top-level tests conftest.

O12a (Reviews 4-7) registry: strict xfail is allowed only for REGISTERED
open findings awaiting the owner's decision.  The registry lives HERE
(top level) so its collection hook sees every test in the tree -- a
subdirectory conftest's hook only sees its own subtree, which is why
this moved out of tests/adversarial/conftest.py on 2026-10-09.
Registration classes: red-team findings (a criterion currently fooled,
reports/redteam_findings.md) and recorded open items from a review work
order awaiting the reviewer's sign-off (currently the W5 strained-box
identity gap; tests/test_strained_boxes.py _GAP is the readable mirror,
kept in sync by test_gap_set_matches_registry).  Strictness is the
enforcement: the moment a registered test passes, the XPASS fails the
suite and the id must come out -- a fix cannot silently bypass its own
registered failure.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# Strained-box identity gap (registered 2026-10-09): the 94 matrix configs
# of tests/test_strained_boxes.py whose first lift (defect arm, scan-fitted
# constant) and re-lift (spglib arm, idealised constant) disagree on the
# a/c line -- the W5 open item awaiting the reviewer's sign-off on the
# lift-side snap.  Registered here per O12a: strict xfail only for
# registered open findings; the moment the snap lands these XPASS, the
# strict marker turns CI red, and the ids come out of this registry.
FOILED = {
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe-+0.005|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe--0.002|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/bcc_fe--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.015|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.015|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/diamond_si--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_crconi--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/fcc_cu--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/hcp_mg-+0.002|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/hcp_mg-+0.002|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/hcp_mg--0.002|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/hcp_mg--0.002|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/l12_ni3al--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.015|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.015|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/perovskite_srtio3--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.002|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.015|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.002|iso]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[crystals/rocksalt_nacl--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.005|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random-+0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random--0.005|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random--0.005|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random--0.015|x]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random--0.015|y]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",
    "tests/test_strained_boxes.py::test_strain_matrix_lift_build_lift_text_identical[solutions/cuau_random--0.015|z]":
        "W5 recorded open item (docs/reviews/open_items_v2.md + module docstring): defect-arm scan-fit vs spglib idealisation differ on the a/c line under strain (0.001-0.012 A); lift-side snap awaits reviewer sign-off -- remove this id when it lands",}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid in FOILED:
            reason = FOILED[item.nodeid]
            if not isinstance(reason, str):
                reason = ("red team: criterion currently fooled "
                          "(reports/redteam_findings.md); remove this id "
                          "when the fix lands")
            item.add_marker(pytest.mark.xfail(strict=True, reason=reason))
