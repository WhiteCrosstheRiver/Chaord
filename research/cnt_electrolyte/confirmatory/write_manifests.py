# -*- coding: utf-8 -*-
"""Write the pre-flight sha256 manifest per confirmatory job (18 files under
confirmatory/manifests/): inputs, sbatch, seeds, protocol, smoke result.
The postflight section is filled by update_manifests_postflight.py after the
cluster results are downloaded (Z3 'Done when': all 18 with manifests).

STANDALONE: does not import chaord.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    with open(os.path.join(HERE, "jobs_index.json"), encoding="utf-8") as f:
        index = json.load(f)
    os.makedirs(os.path.join(HERE, "manifests"), exist_ok=True)
    for j in index:
        name = j["name"]
        d = os.path.join(HERE, "jobs", name)
        files = []
        for role, fn in (("in.lammps", "in.lammps"),
                         ("system.data", "system.data"),
                         ("sbatch.sh", "sbatch.sh")):
            p = os.path.join(d, fn)
            files.append(dict(role=role, path=os.path.relpath(p, HERE),
                              bytes=os.path.getsize(p), sha256=sha256(p)))
        # smoke record
        smoke_d = os.path.join(HERE, "smoke", name)
        exit_f = os.path.join(smoke_d, "exit_code")
        smoke = None
        if os.path.isfile(exit_f):
            code = open(exit_f).read().strip()
            log = os.path.join(smoke_d, "log.lammps")
            smoke = dict(ok=(code == "0" and os.path.isfile(log)
                             and "Total wall time" in open(
                                 log, encoding="utf-8", errors="ignore"
                             ).read()),
                         exit_code=int(code),
                         log=os.path.relpath(log, HERE))
        man = dict(
            job=name, kind="confirmatory",
            work_order="Z3 (open_items_v5.md); D41; PLAN.md pre-registration",
            tube=16, tube_len_A=j["tube_len_A"], build=j["build"],
            build_seed=j["build_seed"], field=j["field"],
            field_mv=j["field_mv"], atoms=j["atoms"], dof=j["dof"],
            protocol=dict(ramp_ps=j["ramp_ps"], equil_ps=j["equil_ps"],
                          settle_ps=j["settle_ps"], block_ps=j["block_ps"],
                          n_blocks=j["n_blocks"],
                          steps_total=j["steps_total"],
                          start_from=j["source"],
                          filling="pilot pack-into-pore (PLAN.md section 4)"),
            seeds=j["seeds"],
            lmp=dict(production="cluster module lammps/3Mar20-icc.2022 "
                                "(LAMMPS 3 Mar 2020), 8-rank MPI",
                     smoke="C:/ProgramData/anaconda3/Scripts/lmp.exe "
                           "(22 Jul 2025 Update 4), validation only"),
            smoke=smoke,
            files=files,
            postflight=None,
        )
        out = os.path.join(HERE, "manifests", name + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2)
        s = "smoke ok" if (smoke and smoke["ok"]) else \
            ("SMOKE FAILED" if smoke else "smoke pending")
        print(f"[manifest] {name}: {len(files)} files hashed, {s}")


if __name__ == "__main__":
    main()
