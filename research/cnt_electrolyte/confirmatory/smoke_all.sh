#!/bin/bash
# Smoke every confirmatory input on the LOCAL LAMMPS (22 Jul 2025 Update 4):
# lmp -var smoke 1 -in in.lammps (500-step versions of every stage).
# Runs in isolated sandbox dirs smoke/<name>/ so the real job dirs stay
# pristine for upload; records <name>.exit for write_manifests.py.
# Usage: ./smoke_all.sh [N-CONCURRENT]   (default 9; ~16 min each, 4116 atoms)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LMP="C:/ProgramData/anaconda3/Scripts/lmp.exe"
CONC="${1:-9}"
mkdir -p "$HERE/smoke"

run_one() {
  local name="$1" d
  d="$HERE/smoke/$name"
  rm -rf "$d"; mkdir -p "$d"
  cp "$HERE/jobs/$name/in.lammps" "$HERE/jobs/$name/system.data" "$d/"
  ( cd "$d" && "$LMP" -var smoke 1 -in in.lammps > stdout.log 2>&1 )
  echo "$?" > "$d/exit_code"
  if [ "$(cat "$d/exit_code")" = "0" ] && grep -q "Total wall time" "$d/log.lammps" 2>/dev/null; then
    echo "[smoke] $name OK"
  else
    echo "[smoke] $name FAILED (exit $(cat "$d/exit_code" 2>/dev/null || echo '?'))"
  fi
}

i=0
for name in sb16L_e0_b1 sb16L_e0_b2 sb16L_e0_b3 sb16L_e0_b4 sb16L_e0_b5 sb16L_e0_b6 \
            sb16L_e10_b1 sb16L_e10_b2 sb16L_e10_b3 sb16L_e10_b4 sb16L_e10_b5 sb16L_e10_b6 \
            sb16L_e20_b1 sb16L_e20_b2 sb16L_e20_b3 sb16L_e20_b4 sb16L_e20_b5 sb16L_e20_b6; do
  run_one "$name" &
  i=$((i+1))
  if [ $((i % CONC)) -eq 0 ]; then wait; fi
done
wait
echo "[smoke] all done: $(grep -l '^0$' "$HERE"/smoke/*/exit_code 2>/dev/null | wc -l) / 18 OK"
