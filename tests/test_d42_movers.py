"""Z2 (Review 11): the D42 bar is implemented and, on the current
full-ladder data, yields NO secondary movers — pinning that state (a
physical regeneration that creates a genuine mover flips this test red
and forces the report language to change)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D42 = ROOT / "research" / "cnt_electrolyte" / "analysis" / "d42_movers.py"


def test_d42_current_data_yield_no_movers():
    """D42 (a)+(b)+(c) applied to the full-ladder results: 'movers' is empty.

    The interesting near-miss is fluid.ion_pairs.AGG at (16,16): Holm-
    significant at 0.2 V/A and monotone over 3 fields, but D42c rejects it
    (6 Li+ < 10 ions per kind). The test also asserts that rejection so the
    ion-count bar cannot be silently dropped."""
    r = subprocess.run([sys.executable, str(D42)],
                       capture_output=True, text=True,
                       cwd=str(ROOT), timeout=1200)
    assert r.returncode == 0, r.stderr[-500:]
    import json
    out = json.loads(r.stdout)
    assert out["movers"] == [], (
        f"D42 movers appeared -- update the reports language: {out['movers']}")
    # the ion-count bar actually fired on the AGG near-miss
    rej = out.get("ion_bar_rejections", {})
    assert rej.get("16|fluid.ion_pairs.AGG") == "6 Li+ < 10", (
        f"the D42c ion-count bar did not fire on the AGG near-miss: {rej}")
