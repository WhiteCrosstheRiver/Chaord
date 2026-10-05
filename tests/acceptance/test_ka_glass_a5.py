"""W7 steps 6-7: A5 evidence for the Kob-Andersen binary glass (D9).

The A5 glass reference case is ka_glass (Kob-Andersen 80:20 at rho* = 1.2);
these tests exercise the criterion machinery on it exactly as
tools/acceptance.check_a5 does (lift with the provenance T, three rebuild
draws, frame-averaged observables, 1.5x max(mean, P90) cross-quench floor),
plus the power mutation A5 needs (physics-off rebuild must FAIL) and the
step-7 quench-rate / anneal-truncation resolution measurements.

MAINLINE PATCH APPLIED IN-PROCESS (documented in the W7 report; the file is
outside this stream's ownership): the legacy cascade's DEFECTS arm
(lift_crystal_defects -> defects.fit_crystal) raises a bare ValueError on
placeholder species ('A'/'B' are not ASE elements), which the W1 fail-closed
ladder correctly refuses to swallow.  fit_crystal must state that verdict as
its designed no-fit ChaordError instead; until it does, the guard below
converts it for non-element frames so the cascade reaches the amorphous arm
(no physics changes: the crystal arms' verdict on a binary glass is
'not this phase' either way).

Run:  python -m pytest tests/acceptance/test_ka_glass_a5.py -m slow
"""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from tools import acceptance as acc

KA = acc.ROOT / "bench" / "reference" / "ka_glass"
DIALECT = ("core", "glass", "lj_mixtures")
SEEDS = (7, 13, 29)


@pytest.fixture(scope="module")
def ka_program_text():
    from conftest import apply_ka_defects_arm_guard
    apply_ka_defects_arm_guard()
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lift import lift_frame
    dl = load_dialect(DIALECT)
    floors = json.loads((acc.ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))
    t_provenance = acc._provenance_tstar(
        json.loads((KA / "provenance.json").read_text(encoding="utf-8")))
    frame = read_frame(str(KA / "frame_0.npz"))
    program = lift_frame(frame, dl, T=t_provenance)
    return acc.format_program_text(program)


def test_ka_glass_lifts_as_binary_amorphous(ka_program_text):
    """The A5 program text: multi-species conserve/composition, the physics
    model (inferred, stated as assumed), the history protocol."""
    text = ka_program_text
    assert "conserve atoms A 1600 B 400" in text
    assert "composition A 1600 B 400" in text
    assert "model kob_andersen" in text
    assert "amorphous" in text
    assert re.search(r"history melt [\d.]+ for \d+ -> quench to", text)
    assert "model kob_andersen assumed" in text


@pytest.mark.slow
def test_ka_glass_a5_averaged_round_trip(ka_program_text):
    """The criterion verdict, A5's own protocol: reference = mean
    observables over the floor record's ref_frames; rebuild = mean over 3
    draws; gate at 1.5x max(mean, P90) of the recorded averaged floor."""
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.parser import parse_text
    from chaord.cv.noise import observables, distance
    dl = load_dialect(DIALECT)
    floors = json.loads((acc.ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))["ka_glass"]
    avg = floors["avg"]
    o_ref = acc._mean_observables(
        [acc._a5_obs(read_frame(str(KA / f"frame_{i}.npz")), dl)
         for i in avg["ref_frames"]])
    draws = []
    for seed in SEEDS:
        prog = parse_text(ka_program_text)
        frame = build_program(prog, dl, rng=np.random.default_rng(seed))
        draws.append(acc._a5_obs(frame, dl))
    o_rb = acc._mean_observables(draws)
    dist = distance(o_ref, o_rb, dialect=dl)
    floor_eff = acc._a5_effective_floor(avg)
    ratios = {k: dist[k] / max(floor_eff[k], 1e-6) for k in dist}
    for k, r in sorted(ratios.items()):
        print(f"    {k}: {dist[k]:.4f} vs floor {floor_eff[k]:.4f} (x{r:.2f})")
    assert all(r <= 1.5 for r in ratios.values()), (dist, floor_eff, ratios)


@pytest.mark.slow
def test_ka_glass_physics_off_mutation_fails(ka_program_text):
    """Power mutation (AGENTS, post Review 3): the packing prior without the
    MD must FAIL the floor gate, or the criterion has no physics evidence."""
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.parser import parse_text
    from chaord.cv.noise import distance
    dl = load_dialect(DIALECT)
    floors = json.loads((acc.ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))["ka_glass"]
    avg = floors["avg"]
    o_ref = acc._mean_observables(
        [acc._a5_obs(read_frame(str(KA / f"frame_{i}.npz")), dl)
         for i in avg["ref_frames"]])
    prog = parse_text(ka_program_text)
    frame = build_program(prog, dl, rng=np.random.default_rng(7),
                          physics=False)
    dist = distance(o_ref, acc._a5_obs(frame, dl), dialect=dl)
    floor_eff = acc._a5_effective_floor(avg)
    ratios = {k: dist[k] / max(floor_eff[k], 1e-6) for k in dist}
    assert all(r > 1.5 for r in ratios.values()), \
        f"physics-off rebuild passed the gate: {ratios}"


# ------------------------------------------- W7 step 7: protocol resolution --

def _ka_energy_per_atom(frame) -> float:
    """Cut-and-shifted KA potential energy per atom (per-pair 2.5 sigma)."""
    from chaord.realize.lj import LJMixture
    dl_entry = load_dialect_entry()
    mix = LJMixture(frame.cell_diag, frame.symbols, dl_entry["epsilon"],
                    dl_entry["sigma"], float(dl_entry["rc_factor"]), skin=0.3)
    return mix.energy(np.mod(frame.pos, frame.cell_diag)) / len(frame.pos)


def load_dialect_entry():
    from chaord.dialects import load_dialect
    return load_dialect(DIALECT).threshold("lj_mixtures")["kob_andersen"]


@pytest.mark.slow
def test_ka_glass_quench_rate_resolution(ka_program_text):
    """W7 step 7: default protocol vs 10x faster quench vs 100-step anneal.

    Reports (and asserts only the criterion-defining property): whether the
    structural observables and the potential energy per atom separate a 10x
    quench-rate change and a truncated anneal beyond the cross-quench floor.
    If structure does not separate 10x but energy does, energy per atom
    becomes a glass observable with its own floor (see the W7 report)."""
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.parser import parse_text
    from chaord.cv.noise import observables, distance
    dl = load_dialect(DIALECT)
    floors = json.loads((acc.ROOT / "reports" / "noise_floors.json")
                        .read_text(encoding="utf-8"))["ka_glass"]
    avg = floors["avg"]
    ref_frames = [read_frame(str(KA / f"frame_{i}.npz"))
                  for i in avg["ref_frames"]]
    o_ref = acc._mean_observables([acc._a5_obs(f, dl) for f in ref_frames])
    floor_eff = acc._a5_effective_floor(avg)

    hist_re = re.compile(r"(?m)^(\s*history melt [\d.]+ for )(\d+)"
                         r"( -> quench to [\d.]+ at )([\d.eE+-]+)"
                         r"( -> anneal [\d.]+ for )(\d+)\s*$")

    def variants():
        m = hist_re.search(ka_program_text)
        assert m, "no history line in the lifted program"
        melt, rate, anneal = m.group(2), m.group(4), m.group(6)
        t_hi = float(re.search(r"history melt ([\d.]+)", ka_program_text).group(1))
        t_lo = float(re.search(r"quench to ([\d.]+)", ka_program_text).group(1))
        n_q = round((t_hi - t_lo) / float(rate))
        yield "default", ka_program_text
        # 10x faster quench: rate x10
        yield "quench10x", hist_re.sub(
            lambda mm: mm.group(1) + mm.group(2) + mm.group(3)
            + f"{float(mm.group(4)) * 10:g}" + mm.group(5) + mm.group(6),
            ka_program_text)
        # anneal truncated to 100 steps
        yield "anneal100", hist_re.sub(
            lambda mm: mm.group(1) + mm.group(2) + mm.group(3) + mm.group(4)
            + mm.group(5) + "100", ka_program_text)

    report = {}
    means = {}
    for name, text in variants():
        obs_list, energies = [], []
        for seed in SEEDS:
            frame = build_program(parse_text(text), dl,
                                  rng=np.random.default_rng(seed))
            obs_list.append(acc._a5_obs(frame, dl))
            energies.append(_ka_energy_per_atom(frame))
        o = acc._mean_observables(obs_list)
        means[name] = o
        d = distance(o_ref, o, dialect=dl)
        report[name] = {
            "structural": {k: d[k] / max(floor_eff[k], 1e-6)
                           for k in sorted(d)},
            "energy_per_atom_mean": float(np.mean(energies)),
            "energy_per_atom_spread": float(np.ptp(energies)),
        }
    # protocol-vs-protocol distances: can the observables tell the DEFAULT
    # rebuild apart from a 10x faster quench / a truncated anneal?  (the
    # W7 step-7 question is asked BETWEEN protocols, not only vs reference)
    for name in list(report):
        if name == "default":
            continue
        dp = distance(means["default"], means[name], dialect=dl)
        report[name]["vs_default"] = {k: round(v / max(floor_eff[k], 1e-6), 3)
                                      for k, v in dp.items()}
        de = (report[name]["energy_per_atom_mean"]
              - report["default"]["energy_per_atom_mean"])
        report[name]["energy_vs_default"] = round(de, 4)
    e_ref = float(np.mean([_ka_energy_per_atom(f) for f in ref_frames]))
    e_ref_spread = float(np.ptp([_ka_energy_per_atom(f) for f in ref_frames]))
    for name, rep in report.items():
        print(f"    {name}: structure "
              + ", ".join(f"{k} x{v:.2f}"
                          for k, v in rep["structural"].items())
              + f"; E/atom {rep['energy_per_atom_mean']:.4f}"
              f" (ref {e_ref:.4f}, ref spread {e_ref_spread:.4f})")
    # the default protocol must pass; the mutated ones are REPORTED, their
    # pass/fail is the step-7 finding (asserted in the report, not here --
    # whichever way they land is a measurement, not a regression)
    assert all(r <= 1.5 for r in report["default"]["structural"].values())
    with open(acc.ROOT / "reports" / "ka_glass_quench_resolution.json",
              "w", encoding="utf-8") as fh:
        json.dump({"reference_energy_per_atom": e_ref,
                   "reference_energy_spread": e_ref_spread,
                   "floor_effective": floor_eff,
                   "variants": report}, fh, indent=2)
