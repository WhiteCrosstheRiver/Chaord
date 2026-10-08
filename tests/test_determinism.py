"""W10 determinism: same input + dialect + version -> same text, regardless
of what the process lifted before (AGENTS rule 10).

The reported defect: `surfaces/si001_2x1` lifted to `site far` after other
lifts in the same process while a fresh process lifted the same frame to the
correct site.  The red test for it lifts the case in a fresh subprocess and
again at the END of a process that lifted every other bench frame first, in
several orders (forward, reverse, seeded shuffles); the texts must be
byte-identical.  Running the two arms in separate subprocesses also makes
the comparison sensitive to PYTHONHASHSEED-dependent output (each child gets
a fresh random seed), covering the set/dict-order class of hidden state.

This module doubles as the child entry point: ``python tests/test_determinism.py
--child ...`` prints the si001_2x1 texts of one arm between BEGIN/END markers
(UTF-8) for the parent to compare.
"""
import difflib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
BENCH = ROOT / "bench" / "data"
TARGET_CASE = "surfaces/si001_2x1"

# lift_frame's own mode vocabulary (mirrors tools/acceptance.py): a ground
# truth 'lift_mode' that names a bench category (not a mode) lifts auto
_LIFT_FRAME_MODES = {"pipeline", "crystal", "defects", "amorphous", "surface",
                     "fluid", "slab"}


def _bench_cases() -> list[dict]:
    """[{id, frames [Path], dialect tuple, mode str}] like tools/acceptance.py."""
    manifest = json.loads((BENCH / "index.json").read_text(encoding="utf-8"))
    cases = []
    for entry in manifest["cases"]:
        gt = json.loads((BENCH / entry["ground_truth"]).read_text(encoding="utf-8"))
        cases.append(dict(
            id=entry["id"], frames=[BENCH / f for f in entry["frames"]],
            dialect=tuple(gt.get("lift_dialect", ["core"])),
            mode=gt.get("lift_mode", "auto")))
    return cases


# ------------------------------------------------------------------- child --

def _lift_one(frame, dialect, mode):
    """Lift with the recorded mode; auto on refusal, like the acceptance loop."""
    from chaord.lang.errors import ChaordError
    from chaord.lift import lift_frame
    if mode in _LIFT_FRAME_MODES:
        try:
            return lift_frame(frame, dialect, mode=mode)
        except ChaordError:
            return lift_frame(frame, dialect)
    return lift_frame(frame, dialect)


def _child(arm: str, seed: int) -> int:
    """Run one arm in this (fresh) process; print the target case's texts.

    arm 'fresh': lift only the target case (the ground-truth baseline).
    arm 'orders': for every order (forward, reverse, seeded shuffles of the
    flattened frame list), lift EVERY other bench frame first -- dialects
    shared per stack, exactly like tools/acceptance.py's cached loop -- then
    the target case, and print its texts after each order."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.fmt import format_program

    sys.stdout.reconfigure(encoding="utf-8")

    def emit(tag, k, text):
        print(f"BEGIN {tag} {k}")
        print(text)
        print(f"END {tag} {k}")

    def target_texts():
        case = next(c for c in _bench_cases() if c["id"] == TARGET_CASE)
        dl = load_dialect(case["dialect"])
        return [format_program(_lift_one(read_frame(p), dl, case["mode"]))
                for p in case["frames"]]

    if arm == "fresh":
        for k, text in enumerate(target_texts()):
            emit("fresh", k, text)
        return 0

    cases = _bench_cases()
    others = [(c, p) for c in cases if c["id"] != TARGET_CASE for p in c["frames"]]
    target = next(c for c in cases if c["id"] == TARGET_CASE)
    orders = {"forward": list(others),
              "reverse": list(reversed(others))}
    for s in (seed, seed + 1):
        shuffled = list(others)
        random.Random(s).shuffle(shuffled)
        orders[f"shuffle:{s}"] = shuffled
    for tag, seq in orders.items():
        dialects: dict = {}
        for case, path in seq:
            dl = dialects.setdefault(case["dialect"], load_dialect(case["dialect"]))
            _lift_one(read_frame(path), dl, case["mode"])
        dl = dialects.setdefault(target["dialect"], load_dialect(target["dialect"]))
        for k, path in enumerate(target["frames"]):
            emit(tag, k, format_program(_lift_one(read_frame(path), dl, target["mode"])))
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        raise SystemExit(_child(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0))
    raise SystemExit("usage: test_determinism.py --child fresh|orders [seed]")


# ------------------------------------------------------------------ parent --

def _parse_blocks(stdout: str) -> dict[str, list[str]]:
    """{'fresh': [text, ...], 'forward': [...], ...} from the child markers."""
    out: dict[str, list[str]] = {}
    tag = None
    k = -1
    lines: list[str] = []
    for line in stdout.splitlines():
        if line.startswith("BEGIN "):
            _, tag, k = line.split()
            lines = []
        elif line.startswith("END "):
            assert line.split() == ["END", tag, k], line
            block = out.setdefault(tag, [])
            assert len(block) == int(k), f"{tag}: frame {k} out of order"
            block.append("\n".join(lines))
            tag = None
        elif tag is not None:
            lines.append(line)
    assert out and all(len(v) == 5 for v in out.values()), \
        f"child output incomplete: {sorted(out)}"
    return out


def _run_child(args: list[str]) -> dict[str, list[str]]:
    r = subprocess.run([sys.executable, str(Path(__file__)), "--child", *args],
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=ROOT, timeout=3600)
    assert r.returncode == 0, f"child {args} failed:\n{r.stdout}\n{r.stderr}"
    return _parse_blocks(r.stdout)


def _first_diff(a: str, b: str) -> str:
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(),
                                     "fresh", "after-others", lineterm=""))
    return "\n".join(diff[:12]) if diff else "(byte-level difference)"


@pytest.mark.slow
def test_si001_text_invariant_under_process_history_and_order():
    """si001_2x1 lifted fresh == lifted after every other bench frame.

    Forward, reverse and two seeded shuffle orders of the full bench set
    (125 frames) must all reproduce the fresh-process text byte-for-byte.
    A mismatch is printed as a unified diff of the first differing frame."""
    baseline = _run_child(["fresh"])["fresh"]
    after = _run_child(["orders", "20261002"])
    for tag, texts in sorted(after.items()):
        for k, (want, got) in enumerate(zip(baseline, texts)):
            assert got == want, (
                f"{TARGET_CASE} frame {k}: text after order '{tag}' differs "
                f"from the fresh-process lift:\n{_first_diff(want, got)}")


def test_si001_text_invariant_within_one_process():
    """Fast in-process guard: the surface-family lifts around si001_2x1.

    The reported trigger was 'after other lifts in the same process'; this
    interleaves the other surface/slab bench cases (the family the surface
    lifter shares code with) before and between the si001_2x1 frames.  It is
    no substitute for the full-bench test above (which stays the regression
    guard) but it fails in seconds, not minutes."""
    from chaord.dialects import load_dialect
    from chaord.io.frames import read_frame
    from chaord.lang.fmt import format_program

    target = next(c for c in _bench_cases() if c["id"] == TARGET_CASE)
    neighbours = [c for c in _bench_cases()
                  if c["id"] != TARGET_CASE
                  and c["id"].split("/")[0] in ("surface", "surfaces", "interfaces")]
    dialects: dict = {}

    def text_of(case, path):
        dl = dialects.setdefault(case["dialect"], load_dialect(case["dialect"]))
        return format_program(_lift_one(read_frame(path), dl, case["mode"]))

    first = [text_of(target, p) for p in target["frames"]]
    for case in neighbours:                     # warm the process with the family
        for path in case["frames"]:
            text_of(case, path)
    again = [text_of(target, p) for p in target["frames"]]
    for k, (want, got) in enumerate(zip(first, again)):
        assert got == want, (
            f"{TARGET_CASE} frame {k}: text changed after lifting the "
            f"surface/interface bench family:\n{_first_diff(want, got)}")


# ------------------------------------------------------ quantized sort keys --

def test_quantized_keys_ignore_subnano_noise():
    """W10 binding: canonical sort keys on float coordinates are quantized
    to 1e-6 A, so coordinates differing by <1e-9 A (BLAS reduction-order
    noise; the hcp_mg regeneration flipped a lexsort on exactly that) sort
    identically, while the 1e-6 grid still separates real differences."""
    from chaord.lift.crystal import quantize_coord
    x = 2.718281828459045
    noisy = [x, x + 4.9e-10, x - 4.9e-10, x + 8.1e-10, x - 8.1e-10]
    keys = [quantize_coord(v) for v in noisy]
    assert len(set(keys)) == 1, f"sub-nano noise changed the key: {keys}"
    assert quantize_coord(x) != quantize_coord(x + 1.5e-6), \
        "the 1e-6 A quantum no longer separates coordinates"


def test_quantized_lexsort_picks_the_same_representative_under_noise():
    """The hcp_mg class at sort level: a lexsort over coordinates whose
    values wobble by <1e-9 A keeps one canonical winner when (and only
    when) the sort keys are quantized."""
    import numpy as np
    from chaord.lift.crystal import quantize_coords
    offs = np.array([
        [1.9999999996, 0.5, 0.25],
        [2.0000000003, 0.5, 0.25],   # same cluster, other side of 2.0
        [3.0, 0.0625, 0.03125],      # later in every order
    ])
    # sub-nano noise that swaps the first two rows across x = 2.0
    eps = np.array([[4.9e-10, 0.0, 0.0], [-4.9e-10, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def winner(arr, quantized):
        keys = quantize_coords(arr) if quantized else arr
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        return int(order[0])

    assert winner(offs, True) == winner(offs + eps, True), \
        "quantized lexsort changed winner under sub-nano noise"
    # the raw lexsort IS flippable by that noise (first two rows straddle
    # 2.0): the quantized one must not inherit the fragility
    raw_flips = winner(offs, False) != winner(offs + eps, False)
    assert raw_flips, "noise fixture no longer straddles a raw-sort boundary"
