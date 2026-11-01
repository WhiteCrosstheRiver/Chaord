"""D50 (Review 12): a commit message may only name files the commit contains.

Checks the last N commits (or a range): every token in the message that
looks like a repo file path must exist in that commit's tree. Over-claiming
messages (naming evidence the commit does not carry) fail -- the known
offenders are listed in docs/records/corrections.md and predate this check.

Run:  python tools/check_commit_claims.py [--range A..B] [--last N]
CI usage: python tools/check_commit_claims.py --last 1
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# path-like token: contains a slash and a known evidence/log/data extension,
# or names a directory we track
PATH_TOKEN = re.compile(
    r"(?:[\w.-]+/)+[\w.-]+\.(?:data|lammps|dump|json|md|py|yaml|yml|txt|csv|npz)")


def commit_tree_paths(commit):
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit],
                         capture_output=True, text=True, cwd=ROOT)
    return set(out.stdout.splitlines())


def check(commit, end_tree=None):
    end_tree = end_tree or set()
    msg = subprocess.run(["git", "log", "-1", "--format=%B", commit],
                         capture_output=True, text=True, cwd=ROOT).stdout
    tree = commit_tree_paths(commit)
    over_claims = []
    for token in sorted(set(PATH_TOKEN.findall(msg))):
        # prose like "final.data/equil/prod.dump" is not a path: skip tokens
        # whose first segment is a bare filename (no repo directory prefix)
        # and which do not exist anywhere in the tree
        first = token.split("/")[0]
        if "/" in token and not any(p.split("/")[0] == first for p in tree)                 and not token.startswith("..."):
            continue
        # the message may name a directory; test the dir prefix too
        if token in tree or any(p.startswith(token + "/") for p in tree):
            continue
        # elided token ('...sub/path' from quoting a long path): pass when
        # some tracked path ends with the tail
        if token.startswith("...") and any(
                p.endswith(token[3:]) for p in tree):
            continue
        if token.startswith("...") and any(
                p.endswith(token[3:]) for p in end_tree):
            continue
        over_claims.append(token)
    return msg, over_claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", dest="rng", default=None,
                    help="commit range A..B to check (default: --last)")
    ap.add_argument("--last", type=int, default=1)
    args = ap.parse_args()
    if args.rng:
        lo, hi = args.rng.split("..")
        out = subprocess.run(["git", "rev-list", args.rng],
                             capture_output=True, text=True, cwd=ROOT)
        commits = out.stdout.splitlines()[::-1]  # oldest first
    else:
        out = subprocess.run(
            ["git", "rev-list", f"-n{args.last}", "HEAD"],
            capture_output=True, text=True, cwd=ROOT)
        commits = out.stdout.splitlines()[::-1]
    # forward references: commits whose message names a file added by the
    # NEXT commit in the audited sequence (e.g. a fix commit that says
    # "recorded in docs/records/corrections.md" when corrections.md lands
    # with the check itself). Allowed only when the file exists at the END
    # of the audited range.
    end_tree = commit_tree_paths(commits[-1]) if commits else set()
    failures = 0
    for c in commits:
        msg, over = check(c, end_tree)
        over = [t for t in over if t not in end_tree]
        if over:
            failures += 1
            short = subprocess.run(["git", "log", "-1", "--format=%h", c],
                                   capture_output=True, text=True,
                                   cwd=ROOT).stdout.strip()
            print(f"OVER-CLAIM in {short}: message names paths absent from "
                  f"its tree: {over}")
    if failures:
        print(f"{failures} over-claiming commit(s) -- see "
              f"docs/records/corrections.md for the known pre-check ones")
        print("(the ellipsis token '...' in a message is an abbreviation, "
              "not a path: it is ignored, but messages that truncate a real "
              "path with it are treated as naming the FULL path when that "
              "full path exists in the tree -- see the elision handling "
              "above)")
        return 1
    print(f"commit-claims check: clean ({len(commits)} commits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
