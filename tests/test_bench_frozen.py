"""bench/data is frozen: synthetic inputs must not move with code changes.

Post Review 3 (2026-09-30, problem 12): a packing change in the generator
once silently rewrote the frames of five synthetic cases and the NaCl ground
truth. The manifest bench/data/checksums.json pins every shipped frame and
ground-truth file; regenerating bench/data is its own reviewed change that
updates the manifest in the same commit, never a side effect.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "bench" / "data"
MANIFEST = DATA / "checksums.json"


def _content_hash(p: Path) -> str:
    """Line-ending-agnostic content hash: git normalises CRLF to LF on
    commit, so a Windows working tree and a Linux checkout must both verify
    against the same manifest (only .json text is normalised; frames are
    binary npz and hash raw)."""
    data = p.read_bytes()
    if p.suffix == ".json":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def test_bench_data_matches_frozen_checksums():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    shipped = sorted(str(p.relative_to(DATA)).replace("\\", "/")
                     for p in DATA.rglob("*")
                     if p.is_file() and p.name != MANIFEST.name)
    assert shipped == sorted(manifest), (
        "bench/data changed without updating checksums.json -- regenerate in "
        "its own reviewed commit if the change is intentional; the set of "
        f"files differs: {set(shipped) ^ set(manifest)}")
    for rel, want in manifest.items():
        got = _content_hash(DATA / rel)
        assert got == want, (
            f"{rel} differs from the frozen manifest -- a code change must "
            "not silently rewrite synthetic bench inputs")
