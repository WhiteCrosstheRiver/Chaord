"""CLI end-to-end: fmt, diff, check, lift, build, roundtrip exit codes."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def run(*args, **kw):
    return subprocess.run([sys.executable, "-m", "chaord.cli", *args],
                          capture_output=True, text=True, cwd=ROOT, **kw)


def test_fmt_check_canonical(tmp_path):
    p = tmp_path / "a.chaord"
    p.write_text((ROOT / "spec" / "examples" / "06_gas.chaord").read_text())
    r = run("fmt", str(p), "--check")
    assert r.returncode == 0, r.stdout + r.stderr


def test_fmt_check_noncanonical(tmp_path):
    p = tmp_path / "a.chaord"
    p.write_text("chaord 0.1\nsystem{seed 7}\n")
    r = run("fmt", str(p), "--check")
    assert r.returncode == 1


def test_diff_ignores_provenance(tmp_path):
    a = tmp_path / "a.chaord"
    b = tmp_path / "b.chaord"
    a.write_text('chaord 0.1\n\nsystem {\n  seed 7\n}\n')
    b.write_text('chaord 0.1\n\nsystem {\n  seed 7\n}\n\nprovenance {\n  note "x"\n}\n')
    r = run("diff", str(a), str(b))
    assert r.returncode == 0, r.stdout


def test_diff_detects_difference(tmp_path):
    a = tmp_path / "a.chaord"
    b = tmp_path / "b.chaord"
    a.write_text('chaord 0.1\n\nsystem {\n  seed 7\n}\n')
    b.write_text('chaord 0.1\n\nsystem {\n  seed 8\n}\n')
    r = run("diff", str(a), str(b))
    assert r.returncode == 1


def test_lift_writes_program(tmp_path):
    out = tmp_path / "lifted.chaord"
    r = run("lift", str(ROOT / "prototype" / "snap.npz"), "-o", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    text = out.read_text()
    assert text.startswith("chaord 0.1")
    assert "conserve atoms X 2301" in text


def test_build_and_check(tmp_path):
    prog = tmp_path / "p.chaord"
    run("lift", str(ROOT / "prototype" / "snap.npz"), "-o", str(prog))
    coords = tmp_path / "rebuilt.npz"
    r = run("build", str(prog), "-o", str(coords), "--seed", "3", "--no-physics")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2301" in r.stdout
    r = run("check", str(prog), str(coords))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout


def test_roundtrip_no_physics(tmp_path):
    r = run("roundtrip", str(ROOT / "prototype" / "snap.npz"), "--no-physics")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "atoms" in r.stdout
