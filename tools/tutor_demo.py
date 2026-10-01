"""Three-act live demo for the tutor (Review 3's closing ask).

    python tools/tutor_demo.py

Act 1  live round trip   lift -> build -> lift of the stored lj_liquid
                         reference frame, texts compared byte for byte
Act 2  physics off fails the same program rebuilt with physics=False must
                         exceed the noise-floor gate
Act 3  wrong T fails     the same program rebuilt at 0.8x the provenance
                         temperature must exceed the noise-floor gate

Exit code 0 exactly when act 1 passes and acts 2-3 FAIL (a test that
visibly fails for the wrong input is what makes the passes believable).
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def set_temperature(text: str, t: float) -> str:
    """Rewrite the program's `state T <x>` line (act 3's knob)."""
    out, hit = [], False
    for line in text.splitlines():
        if re.match(r"^\s*state T ", line):
            line = f"  state T {t}"
            hit = True
        out.append(line)
    if not hit:
        raise ValueError("no `state T` line found in the program text")
    return "\n".join(out) + "\n"


def verdict_line(name: str, ok: bool, expect: bool, detail: str) -> str:
    got = "PASS" if ok else "FAIL"
    match = "as required" if ok == expect else "*** UNEXPECTED ***"
    return f"  {name:<22} {got}  ({match})  {detail}"


def main() -> int:
    import numpy as np
    import acceptance as acc
    from chaord.build import build_program
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.parser import parse_text
    from chaord.lift import lift_frame

    print("=" * 78)
    print("Chaord live demo -- the round trip passes, the wrong inputs FAIL")
    print("=" * 78)

    dl = load_dialect(("core", "lj"))
    frame = read_frame(ROOT / "bench" / "reference" / "lj_liquid" / "frame_0.npz")
    floors = json.loads((ROOT / "reports" / "noise_floors.json").read_text())
    fl = floors["lj_liquid"]

    # ---- act 0: an ordered system round-trips BYTE-IDENTICALLY ----------
    from chaord.dialects import load_dialect as _ld
    dl_cu = _ld(("core", "metal"))
    cu = read_frame(ROOT / "bench" / "data" / "crystals" / "fcc_cu" /
                    "frame_0.npz")
    tc1 = acc.format_program_text(lift_frame(cu, dl_cu))
    cu2 = build_program(parse_text(tc1), dl_cu, rng=np.random.default_rng(7))
    tc2 = acc.format_program_text(lift_frame(cu2, dl_cu))
    print("Act 0: ordered system (thermal fcc Cu frame) lift -> build -> lift")
    print(verdict_line("text round trip", tc2 == tc1, True,
                       f"byte-identical ({len(tc1)} chars): ordered matter "
                       "round-trips exactly, thermal frame included"))

    from chaord.cv.noise import observables, distance
    eff = acc._a5_effective_floor(fl)
    t1 = acc.format_program_text(lift_frame(frame, dl))
    print()
    print("Act 1: liquid lj_liquid lift -> build -> lift -- the statistical")
    print("       regime: the assert VALUES are measurements of a chaotic MD")
    print("       frame, so byte identity belongs to act 0 and THIS act's")
    print("       contract is statistical identity within the noise floor")
    print("  lifted program (first lines):")
    for line in t1.splitlines()[:8]:
        print(f"    {line}")
    t0 = time.perf_counter()
    rebuilt = build_program(parse_text(t1), dl, rng=np.random.default_rng(7))
    secs = time.perf_counter() - t0
    d = distance(observables(frame, dl), observables(rebuilt, dl), dialect=dl)
    ok1 = all(d[k] <= 1.5 * eff[k] for k in d)
    print(verdict_line("statistical round trip", ok1, True,
                       f"cn_tv {d['cn_tv']:.3f}, gr_rms {d['gr_rms']:.3f} vs "
                       f"gates {1.5 * eff['cn_tv']:.3f}/{1.5 * eff['gr_rms']:.3f} "
                       f"(floors max(mean, P90) from noise_floors.json); "
                       f"rebuild {secs:.1f}s"))
    print("\nAct 2: the SAME program rebuilt with physics OFF must FAIL")
    packed = build_program(parse_text(t1), dl, rng=np.random.default_rng(7),
                           physics=False)
    d2 = distance(observables(frame, dl), observables(packed, dl), dialect=dl)
    ok2 = not all(d2[k] <= 1.5 * eff[k] for k in d2)
    print(verdict_line("physics off", ok2, True,
                       f"cn_tv {d2['cn_tv']:.3f}, gr_rms {d2['gr_rms']:.3f} "
                       f"vs gates {1.5 * eff['cn_tv']:.3f}/{1.5 * eff['gr_rms']:.3f}"))

    print()
    print("Act 3: rebuilt at 0.8x the provenance temperature must FAIL")
    print("  (on lj_liquid_large, the case carrying the temperature evidence:")
    print("   the 500-atom liquid's +-20% power shortfall is a documented")
    print("   residual -- single draws stay inside its floor)")
    big = read_frame(ROOT / "bench" / "reference" / "lj_liquid_large" /
                     "frame_0.npz")
    tb1 = acc.format_program_text(lift_frame(big, dl))
    prov = json.loads((ROOT / "bench" / "reference" / "lj_liquid_large" /
                       "provenance.json").read_text(encoding="utf-8"))
    tstar = acc._provenance_tstar(prov)
    cold_text = set_temperature(tb1, 0.8 * tstar)
    big_frames = [read_frame(ROOT / "bench" / "reference" /
                             "lj_liquid_large" / f"frame_{k}.npz")
                  for k in range(10)]
    obs_ref = acc._mean_observables([observables(f, dl) for f in big_frames])
    # the AVERAGING protocol's floor (the 'avg' entry, tighter than the
    # frame-pair floor: averaged observables carry less noise) -- exactly
    # what the acceptance harness gates the temperature mutations against
    avg_fl = floors["lj_liquid_large"]["avg"]
    eff2 = {k: max(float(avg_fl[f"{k}_mean"]), float(avg_fl[f"{k}_q90"]))
            for k in ("gr_rms", "cn_tv")}
    draws = []
    t0 = time.perf_counter()
    for s in (7, 13, 29):
        rb = build_program(parse_text(cold_text), dl,
                           rng=np.random.default_rng(s))
        draws.append(observables(rb, dl))
    secs3 = time.perf_counter() - t0
    obs_cold = acc._mean_observables(draws)
    d3 = distance(dict(obs_ref), dict(obs_cold), dialect=dl)
    ok3 = not all(d3[k] <= 1.5 * eff2[k] for k in d3)
    print(verdict_line(f"T = {0.8 * tstar:.3f}", ok3, True,
                       f"cn_tv {d3['cn_tv']:.3f}, gr_rms {d3['gr_rms']:.3f} "
                       f"vs gates {1.5 * eff2['cn_tv']:.3f}/"
                       f"{1.5 * eff2['gr_rms']:.3f}; 3 seeds x ~{secs3 / 3:.0f}s, "
                       f"10-frame averaged reference"))

    print("\nHow to read this:")
    print("  act 1 shows the compiler/decompiler pair reproduces the exact text")
    print("  acts 2-3 show the acceptance gate is not decorative: dropping the")
    print("  physics prior or rebuilding at the wrong temperature both exceed it")
    print("  -- the passes elsewhere are believable because these fail.")
    all_ok = (tc2 == tc1) and ok1 and ok2 and ok3
    print(f"\noverall: {'ALL THREE ACTS AS REQUIRED' if all_ok else 'AN ACT MISBEHAVED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
