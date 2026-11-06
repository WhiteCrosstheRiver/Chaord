# Write the real production input for one job into smoke/<job>/ for -var smoke 1.
import os, shutil, sys
sys.path.insert(0, os.getcwd())
import r13_inputs as I
d = os.path.join(I.HERE, "smoke", "r13_e10_b1")
os.makedirs(d, exist_ok=True)
with open(os.path.join(d, "in.lammps"), "w", newline="\n") as f:
    f.write(I.r13_input("r13_e10_b1", 0.01, 1))
shutil.copyfile(os.path.join(I.DATA, "r13_b1.data"), os.path.join(d, "system.data"))
print("wrote", d)
