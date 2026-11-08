"""CI-equivalent gate on a gerui compute node: sketch check, magic-number gate,
fast test suite and the acceptance harness, the last two in parallel.
Each step's output goes to out/<step>.log; out/gate.json records the exit codes."""
import glob, json, shutil, subprocess, sys, time
from pathlib import Path

py = sys.executable
steps = {
    "sketch": [py, "tools/sketch_check.py"] + sorted(glob.glob("spec/examples/*.chaord")),
    "magic": [py, "tools/check_magic_numbers.py"],
}
parallel = {
    "pytest": [py, "-m", "pytest", "-p", "no:cacheprovider", "tests", "-m", "not slow",
               "-q", "--tb=short", "-rfE"],
    "acceptance": [py, "tools/acceptance.py", "-o", "out/acceptance.json",
                   "--details", "out/acceptance_details.md"],
}
out = Path("out")
res = {}
for name, cmd in steps.items():
    t = time.time()
    with open(out / f"{name}.log", "w") as f:
        res[name] = dict(rc=subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT),
                         seconds=round(time.time() - t))
procs = {}
for name, cmd in parallel.items():
    f = open(out / f"{name}.log", "w")
    procs[name] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f, time.time())
for name, (p, f, t) in procs.items():
    res[name] = dict(rc=p.wait(), seconds=round(time.time() - t))
    f.close()
(out / "gate.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
sys.exit(0 if all(r["rc"] == 0 for r in res.values()) else 1)
