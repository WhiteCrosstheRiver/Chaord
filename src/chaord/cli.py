"""The chaord command line: build, lift, check, fmt, diff, roundtrip."""
from __future__ import annotations

import argparse
import difflib
import sys

import numpy as np


def _cmd_fmt(args) -> int:
    from .lang.api import from_json, load, save, to_json
    from .lang.fmt import format_program
    from .lang.parser import parse_text
    from pathlib import Path
    text = Path(args.infile).read_text(encoding="utf-8")
    out = format_program(parse_text(text))
    if args.check:
        if out != text:
            import difflib
            for line in difflib.unified_diff(
                    text.splitlines(), out.splitlines(),
                    fromfile=args.infile, tofile="canonical", lineterm="", n=1):
                print(line)
            return 1
        print(f"{args.infile}: already canonical")
        return 0
    if args.outfile:
        Path(args.outfile).write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)
    if args.json:
        print(to_json(parse_text(out)))
    return 0


def _cmd_lift(args) -> int:
    from .check.statics import run_checks
    from .dialects import load_dialect
    from .io.frames import read_frame
    from .lang.fmt import format_program
    from .lift import lift_frame
    dialect = load_dialect(args.dialect.split("+"))
    frame = read_frame(args.infile, index=args.frame)
    program = lift_frame(frame, dialect, T=args.T, mode=args.mode)
    text = format_program(program)
    checks = run_checks(program, frame, dialect)
    for c in checks:
        print(f"[{ 'PASS' if c.passed else 'FAIL' }] {c.name}: {c.detail}", file=sys.stderr)
    if args.outfile:
        from pathlib import Path
        Path(args.outfile).write_text(text, encoding="utf-8")
        print(f"wrote {args.outfile}")
    else:
        sys.stdout.write(text)
    return 0 if all(c.passed for c in checks) else 2


def _cmd_build(args) -> int:
    from .build import build_program
    from .dialects import dialect_from_program
    from .io.frames import write_frame
    from .lang.api import load
    program = load(args.infile)
    dialect = dialect_from_program(program)
    rng = np.random.default_rng(args.seed)
    frame = build_program(program, dialect, rng, physics=not args.no_physics,
                          md_steps=args.md_steps)
    write_frame(args.outfile, frame)
    print(f"built {len(frame)} atoms -> {args.outfile}")
    return 0


def _cmd_check(args) -> int:
    from .check.statics import run_checks
    from .dialects import dialect_from_program
    from .io.frames import read_frame
    from .lang.api import load
    program = load(args.program)
    dialect = dialect_from_program(program)
    frame = read_frame(args.structure)
    checks = run_checks(program, frame, dialect)
    failed = False
    for c in checks:
        print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        failed |= not c.passed
    return 1 if failed else 0


def _cmd_diff(args) -> int:
    from .lang.api import load
    from .lang.fmt import format_program

    def canonical(path):
        p = load(path)
        p.blocks = [b for b in p.blocks if b.t != "provenance"]  # provenance never counts
        return format_program(p).splitlines()

    a, b = canonical(args.a), canonical(args.b)
    if a == b:
        print("programs are equivalent (provenance ignored)")
        return 0
    for line in difflib.unified_diff(a, b, fromfile=args.a, tofile=args.b, lineterm=""):
        print(line)
    return 1


def _cmd_roundtrip(args) -> int:
    from .build import build_program
    from .cv.noise import observables, distance
    from .dialects import load_dialect
    from .io.frames import read_frame
    from .lang.api import load, save
    from .lift import lift_frame
    from .lift.slab import decompile
    import tempfile, pathlib
    dialect = load_dialect(args.dialect.split("+"))
    frame = read_frame(args.infile)
    n = max(1, int(args.samples))
    prog0 = lift_frame(frame, dialect, T=args.T, mode=args.mode)
    with tempfile.TemporaryDirectory() as td:
        prog_path = pathlib.Path(td) / "p.chaord"
        save(prog0, prog_path)
        rebuilt = load(prog_path)
        frames = [build_program(rebuilt, dialect, np.random.default_rng(args.seed + i),
                                physics=not args.no_physics)
                  for i in range(n)]
        if n == 1:
            p0 = decompile(frame.pos, frame.cell_diag, args.T, dialect)
            p1 = decompile(frames[0].pos, frames[0].cell_diag, args.T, dialect)

            def row(name, f):
                return f"{name:<32}" + "".join(f"{f(x):>14}" for x in (p0, p1))
            print(row("atoms", lambda Q: f"{Q['N']}"))
            print(row("net vacancies", lambda Q: f"{sum(c['net'] for c in Q['defects'])}"))
            print(row("lattice a", lambda Q: f"{Q['a_x']:.3f}"))
            print(row("strain zz %", lambda Q: f"{100*(Q['a_z']/Q['a_x']-1):+.1f}"))
            print(row("liquid density", lambda Q: f"{Q['rho']:.3f}"))
            m = lambda Q: (Q["rm"] > 0.8)
            print(row("g(r) RMS dist", lambda Q: f"{np.sqrt(np.mean((Q['gr'][m(Q)]-p0['gr'][m(p0)])**2)):.3f}"))
            print(row("angle distr. TV", lambda Q: f"{0.5*np.abs(Q['bad']-p0['bad']).sum():.3f}"))
            return 0
        # several independent rebuilds (one seed each): per-sample observable
        # distance, then mean +- std against the rebuild-to-rebuild noise floor
        o0 = observables(frame, dialect)
        obs = [observables(f, dialect) for f in frames]
        ds = [distance(o0, o) for o in obs]
        floors = [distance(obs[i], obs[j])
                  for i in range(n) for j in range(i + 1, n)]
        print(f"roundtrip: {n} samples (seeds {args.seed}..{args.seed + n - 1})")
        for i, (f, d) in enumerate(zip(frames, ds)):
            print(f"sample {i + 1}: seed {args.seed + i}  atoms {len(f)}  "
                  f"gr_rms {d['gr_rms']:.3f}  cn_tv {d['cn_tv']:.3f}")
        for k in ds[0]:
            vals = np.array([d[k] for d in ds])
            fl = float(np.mean([m[k] for m in floors]))
            print(f"mean {k}: {vals.mean():.3f} +- {vals.std():.3f} over {n} samples"
                  f"  |  noise floor {fl:.3f}"
                  f"  |  ratio x{vals.mean() / max(fl, 1e-9):.1f}")  # dialect-exempt: degenerate floor
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chaord",
                                 description="describe the macrostate, sample the microstate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fmt", help="print the canonical form of a program")
    p.add_argument("infile")
    p.add_argument("-o", "--outfile")
    p.add_argument("--check", action="store_true", help="exit 1 if not already canonical")
    p.add_argument("--json", action="store_true", help="also print the JSON IR")
    p.set_defaults(fn=_cmd_fmt)

    p = sub.add_parser("lift", help="coordinates -> program")
    p.add_argument("infile")
    p.add_argument("-o", "--outfile")
    p.add_argument("--frame", type=int, default=-1)
    p.add_argument("--dialect", default="core+lj")
    p.add_argument("--T", type=float, default=0.65,
                   help="temperature is metadata: stated in the program, not measured")
    p.add_argument("--mode", default="auto", choices=["auto", "crystal", "slab"])
    p.set_defaults(fn=_cmd_lift)

    p = sub.add_parser("build", help="program -> coordinates")
    p.add_argument("infile")
    p.add_argument("-o", "--outfile", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-physics", action="store_true")
    p.add_argument("--md-steps", type=int, default=None)
    p.set_defaults(fn=_cmd_build)

    p = sub.add_parser("check", help="run static checks on a program + structure")
    p.add_argument("program")
    p.add_argument("structure")
    p.set_defaults(fn=_cmd_check)

    p = sub.add_parser("diff", help="compare two programs canonically")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=_cmd_diff)

    p = sub.add_parser("roundtrip", help="lift -> build -> lift and compare")
    p.add_argument("infile")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--samples", type=int, default=5,
                   help="independent rebuilds (seed, seed+1, ...); "
                        "1 prints the classic two-column table")
    p.add_argument("--dialect", default="core+lj")
    p.add_argument("--T", type=float, default=0.65)
    p.add_argument("--mode", default="auto", choices=["auto", "crystal", "slab"])
    p.add_argument("--no-physics", action="store_true")
    p.set_defaults(fn=_cmd_roundtrip)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
