"""LAMMPS data-file and dump readers.

Provenance: after-gate research target (docs/reviews/open_items_v2.md,
section 5, item 1): "A LAMMPS data/dump reader (type -> element from the
data file's masses), triclinic and NPT boxes".

Entry points
------------
* ``read_lammps_data(path) -> (Frame, metadata)``
* ``read_lammps_dump(path, frame_indices=None) -> list[Frame]``
* ``LammpsTypeMap`` -- the type-id -> element mapping derived from a data
  file's Masses section, shareable with dumps (dumps carry only integer
  type ids; ``LammpsTypeMap.apply(frames)`` writes the symbols onto
  dump-read frames).

Frame triclinic decision
------------------------
``Frame.cell`` (src/chaord/io/frames.py) is a full (3, 3) row-vector
matrix, so triclinic cells ARE representable and are fully supported:
LAMMPS tilt factors (xy, xz, yz) populate the lower-triangular cell

    [[xhi-xlo, 0,      0      ],
     [xy,      yhi-ylo, 0     ],
     [xz,      yz,      zhi-zlo]]

which is ASE's convention.  Downstream lift arms may still refuse
non-orthogonal cells (e.g. src/chaord/lift/defect_program.py raises for
them); that is their documented behaviour, not data loss here.

Units
-----
LAMMPS files do not state their unit system.  No conversion is
performed: lengths are taken as angstrom, charges as multiples of e and
masses as amu (the real/metal convention the owner's runs use).  A file
produced under another unit system must be converted by the caller.

Parse decisions (data files)
----------------------------
1. The first line is the title and is always skipped (LAMMPS read_data
   treats it as a comment).  Blank lines and lines starting with ``#``
   are skipped everywhere.
2. Header lines match, in any order and until the first section
   keyword: ``N atoms|bonds|angles|dihedrals|impropers``,
   ``M atom|bond|... types``, ``lo hi xlo xhi`` (and y/z), and the
   optional triclinic line ``xy xz yz``.  Any other header line is a
   ChaordError (fail closed) with its line number.
3. A section starts at a line whose first token is not numeric; the
   section name is the text up to an optional ``#`` comment.  Masses,
   Atoms, Velocities, Bonds and Angles are parsed; every other section
   (Dihedrals, Impropers, *Coeffs, ...) is counted into
   ``metadata["skipped_sections"]`` as ``(name, n_rows)`` -- recorded,
   never silently dropped.
4. ``Atoms`` style: the ``# style`` suffix on the section line is
   authoritative when present (LAMMPS write_data emits it).  Otherwise
   the style is inferred from the token count of each row:
   5 = atomic, 6 = charge, 7 = full, 9 = charge+images, 10 = full+images,
   11 = molecular+images, 8 = atomic+images when the last three tokens
   are integer-formatted else molecular.  Counts that fit no pattern
   (or rows disagreeing with the declared style) raise a ChaordError
   naming the line.  Unsupported exotic styles (e.g. ``hybrid``) are
   refused, not guessed.
5. Atom ids must be unique and contiguous 1..N (a LAMMPS read_data
   requirement); the row count must equal the header count.  Atoms are
   stored sorted by id.  Molecule ids, image flags, velocities, bonds
   and angles (raw 1-based ids, as written) are carried in metadata.
6. Positions are shifted by -(xlo, ylo, zlo) (Frame has no origin) and
   must already lie inside the box (within 1e-6 fractional slack);
   a data file with an atom outside the box is refused by name -- it
   would be refused by LAMMPS itself.
7. Type -> element: nearest ``ase.data.atomic_masses`` entry within
   0.05 amu (an exact table match is distance 0 and always wins).
   Types with no element inside the tolerance map to symbol "X" and are
   listed in ``metadata["unmapped_types"]`` -- reported, never a crash.
8. Boundary conditions are not stored in data files; frames are built
   with pbc = (True, True, True) and ``metadata["boundary"]`` records
   that this is an assumption.

Parse decisions (dump files)
----------------------------
1. Snapshots follow the fixed LAMMPS order ITEM: TIMESTEP,
   NUMBER OF ATOMS, BOX BOUNDS, ATOMS.  An unknown ITEM section (e.g.
   ``ITEM: TIME``) is a ChaordError naming it -- honest refusal beats
   silently skipping a block of an evolving file.
2. ``BOX BOUNDS`` labels ending in ``xy xz yz`` mark a triclinic box;
   each of the three following lines then carries ``lo hi tilt``.
   Otherwise each line is ``lo hi`` (orthogonal).  Every snapshot gets
   its own cell, which is what makes NPT dumps (bounds changing per
   snapshot) work.
3. ``ITEM: ATOMS`` column names are read from the header.  Required:
   ``id`` and ``type``.  Coordinates: either ``x y z`` (cartesian,
   shifted by -(lo) and validated inside the box as above) or
   ``xs ys zs`` (scaled; wrapped into [0, 1) by a fractional modulo,
   then ``pos = frac @ cell`` which applies the tilt for triclinic
   boxes).  Unwrapped-only columns (``xu yu zu`` / ``xsu ysu zsu``)
   raise a ChaordError: Frame keeps positions wrapped into the cell,
   and unwrapped coordinates would silently violate that invariant.
   Optional: ``q`` (charges into ``frame.info``) and ``ix iy iz``
   (image flags recorded, NOT applied).
4. ``q`` is one column per atom; charges live in ``frame.info["charges"]``
   because Frame carries symbols only.
5. pbc per axis is True only for a ``pp`` label; fixed/shrink-wrapped
   labels give False, and the raw labels are kept in
   ``frame.info["box_boundary_labels"]`` because a bool per axis cannot
   express a mixed label such as ``mp`` (recorded, not dropped).
6. Rows must match NUMBER OF ATOMS; ids must be unique (any values --
   dumps of LAMMPS groups are not contiguous); atoms are stored sorted
   by id.  Symbols default to "X" until a LammpsTypeMap is applied;
   the integer type ids ride along in ``frame.info["type_ids"]``.
7. ``frame_indices`` selects snapshots after parsing: None -> all, an
   int -> that one, a list/tuple of ints, or a slice.  Out-of-range
   indices raise a ChaordError naming the available count.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, chemical_symbols

from ..lang.errors import ChaordError
from .frames import Frame

# Work-order tolerance (open_items_v2.md section 5 item 1): a LAMMPS
# Masses value maps to an element when an ase.data table entry lies
# within 0.05 amu; nearest wins, exact table matches are distance 0.
_MASS_TOLERANCE_AMU = 0.05

# Fractional slack when validating that positions lie inside the box:
# guards the last-digit fuzz of text float round trips without letting
# genuinely outside-the-box coordinates through.
_BOX_SLACK = 1e-6


# --------------------------------------------------------------- elements

def element_from_mass(mass: float,
                      tolerance: float = _MASS_TOLERANCE_AMU) -> str | None:
    """Nearest element symbol by atomic mass, or None outside tolerance.

    Distance is |mass - ase.data.atomic_masses[Z]| for Z = 1..118; the
    minimum over Z is returned when it lies within ``tolerance`` amu
    (an exact table match has distance 0 and therefore always wins).
    """
    mass = float(mass)
    best_symbol, best_d = None, None
    for z in range(1, len(chemical_symbols)):
        d = abs(float(atomic_masses[z]) - mass)
        if best_d is None or d < best_d:
            best_symbol, best_d = chemical_symbols[z], d
    if best_d is not None and best_d <= tolerance:
        return best_symbol
    return None


class LammpsTypeMap:
    """LAMMPS integer atom type -> element symbol, from a Masses table.

    Built directly from ``{type_id: mass}`` or obtained from
    ``read_lammps_data`` via ``metadata["type_map"]``; share it with
    dump-read frames through ``apply`` (dumps carry only type ids).
    """

    def __init__(self, masses: dict, tolerance: float = _MASS_TOLERANCE_AMU):
        self.masses = {int(t): float(m) for t, m in masses.items()}
        self.tolerance = float(tolerance)
        self._symbols = {}
        for t, m in self.masses.items():
            s = element_from_mass(m, self.tolerance)
            self._symbols[t] = s if s is not None else "X"
        # types with no element inside the tolerance: reported, kept as "X"
        self.unmapped = sorted(t for t, s in self._symbols.items() if s == "X")

    def symbol(self, type_id: int) -> str:
        """Element for a type id declared in the Masses table; "X" if the
        type is unmapped or absent from the table."""
        return self._symbols.get(int(type_id), "X")

    def apply(self, frames, strict: bool = False) -> list:
        """Write element symbols onto dump-read frames (in place).

        Uses each frame's ``info["type_ids"]``.  Returns the sorted list
        of type ids encountered that have no mapping (absent from the
        Masses table or unmapped there); those atoms get symbol "X".
        With ``strict=True`` any such id raises instead -- use it when
        an "X" in the output would be worse than a refusal.
        """
        unknown = set()
        for fr in frames:
            tids = fr.info.get("type_ids")
            if tids is None:
                raise ChaordError(
                    "frame carries no type_ids in info; LammpsTypeMap.apply "
                    "works on frames read by read_lammps_dump")
            syms = []
            for t in tids:
                t = int(t)
                if t not in self._symbols or self._symbols[t] == "X":
                    unknown.add(t)
                    syms.append("X")
                else:
                    syms.append(self._symbols[t])
            fr.symbols = syms
        if strict and unknown:
            raise ChaordError(
                "unmapped LAMMPS type id(s) "
                f"{sorted(unknown)} in dump; extend the Masses section or "
                "orthogonalise before relying on elements")
        return sorted(unknown)


# ------------------------------------------------------------------ cells

def _cell_from_bounds(lo, hi, tilt=None) -> np.ndarray:
    """Lower-triangular row-vector cell from box bounds (+ tilt)."""
    if tilt is None:
        return np.diag(np.asarray(hi, float) - np.asarray(lo, float))
    xy, xz, yz = (float(t) for t in tilt)
    lx, ly, lz = (float(hi[k]) - float(lo[k]) for k in range(3))
    return np.array([[lx, 0.0, 0.0],
                     [xy, ly, 0.0],
                     [xz, yz, lz]])


def _validate_inside_box(pos, cell, where: str) -> None:
    """Refuse cartesian positions outside the cell (fail closed): LAMMPS
    itself rejects such files, so a Chaord here names our own bug or a
    hand-edited file instead of wrapping silently."""
    frac = np.asarray(pos, float) @ np.linalg.inv(np.asarray(cell, float))
    bad = (frac < -_BOX_SLACK) | (frac > 1.0 + _BOX_SLACK)
    if bad.any():
        n = int(bad.any(axis=1).sum())
        raise ChaordError(
            f"{where}: {n} atom position(s) outside the simulation box; "
            "LAMMPS data/dump coordinates must already be wrapped")


# ------------------------------------------------------------- data files

_COUNT_RE = re.compile(r"^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)$")
_TYPES_RE = re.compile(r"^(\d+)\s+(atom|bond|angle|dihedral|improper)\s+types$")
_BOX_RE = re.compile(r"^(\S+)\s+(\S+)\s+([xyz])lo\s+\3hi$")
_TILT_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+xy\s+xz\s+yz$")

_PARSED_SECTIONS = {"Masses", "Atoms", "Velocities", "Bonds", "Angles"}

# style -> (token counts without/with image flags)
_STYLE_TOKENS = {
    "atomic": (5, 8),
    "charge": (6, 9),
    "full": (7, 10),
    "molecular": (8, 11),
}


def _is_int_token(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


def _parse_atoms_row(tokens, style, lineno, path):
    """Split one Atoms row per the style table; returns a dict."""
    n = len(tokens)
    has_img = False
    if style is None:
        # no "# style" hint: infer from token count (see module docstring 4)
        if n == 5:
            style = "atomic"
        elif n == 6:
            style = "charge"
        elif n == 7:
            style = "full"
        elif n == 8:
            style = "molecular"
            if all(_is_int_token(t) for t in tokens[5:8]):
                style = "atomic"
                has_img = True
        elif n == 9:
            style, has_img = "charge", True
        elif n == 10:
            style, has_img = "full", True
        elif n == 11:
            style, has_img = "molecular", True
        else:
            raise ChaordError(
                f"{path.name}: line {lineno}: unrecognised Atoms row with "
                f"{n} tokens: {' '.join(tokens)}")
    else:
        plain, with_img = _STYLE_TOKENS.get(style, (None, None))
        if plain is None:
            raise ChaordError(
                f"{path.name}: line {lineno}: unsupported Atoms style "
                f"'{style}' (supported: {sorted(_STYLE_TOKENS)})")
        if n == with_img:
            has_img = True
        elif n != plain:
            raise ChaordError(
                f"{path.name}: line {lineno}: Atoms row has {n} tokens, "
                f"style '{style}' needs {plain} (or {with_img} with image "
                f"flags): {' '.join(tokens)}")

    try:
        aid = int(tokens[0])
        if style == "atomic":
            if has_img:
                type_i, q, xyz, img = 1, None, slice(2, 5), slice(5, 8)
            else:
                type_i, q, xyz, img = 1, None, slice(2, 5), None
        elif style == "charge":
            if has_img:
                type_i, q, xyz, img = 1, 2, slice(3, 6), slice(6, 9)
            else:
                type_i, q, xyz, img = 1, 2, slice(3, 6), None
        elif style == "molecular":
            if has_img:
                type_i, q, xyz, img = 2, None, slice(3, 6), slice(6, 9)
            else:
                type_i, q, xyz, img = 2, None, slice(3, 6), None
        else:  # full
            if has_img:
                type_i, q, xyz, img = 2, 3, slice(4, 7), slice(7, 10)
            else:
                type_i, q, xyz, img = 2, 3, slice(4, 7), None
        tid = int(tokens[type_i])
        charge = float(tokens[q]) if q is not None else None
        pos = [float(t) for t in tokens[xyz]]
        image = [int(t) for t in tokens[img]] if img is not None else None
        mol = int(tokens[1]) if style in ("full", "molecular") else None
    except (ValueError, IndexError):
        raise ChaordError(
            f"{path.name}: line {lineno}: cannot parse Atoms row "
            f"(style '{style}'): {' '.join(tokens)}") from None
    return {"id": aid, "type": tid, "q": charge, "pos": pos,
            "img": image, "mol": mol}


def read_lammps_data(path) -> tuple[Frame, dict]:
    """Read a LAMMPS data file -> (Frame, metadata).

    See the module docstring for the full list of parse decisions.  The
    frame's symbols come from the Masses table (nearest element within
    0.05 amu; "X" when unmapped, with ``metadata["unmapped_types"]``
    reporting each such type).  Charges, molecule ids, image flags,
    velocities, bonds, angles and the raw lower box corner live in
    ``metadata`` (None when the file carries none).
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines:
        raise ChaordError(f"{path.name}: empty file")

    counts, type_counts = {}, {}
    box, tilt = {}, None
    sections = []            # [(name, style_hint|None, [(lineno, tokens)])]
    current = None           # index into sections while scanning

    for lineno, raw in enumerate(lines[1:], start=2):   # line 1 = title
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        first = s.split()[0]
        if current is None and not first[0].isalpha():
            # ---- header line (numbers first)
            m = _COUNT_RE.match(s)
            if m:
                counts[m.group(2)] = int(m.group(1))
                continue
            m = _TYPES_RE.match(s)
            if m:
                type_counts[m.group(2)] = int(m.group(1))
                continue
            m = _BOX_RE.match(s)
            if m:
                axis = m.group(3)
                try:
                    box[axis] = (float(m.group(1)), float(m.group(2)))
                except ValueError:
                    raise ChaordError(
                        f"{path.name}: line {lineno}: bad box bounds: {s}"
                        ) from None
                continue
            m = _TILT_RE.match(s)
            if m:
                try:
                    tilt = (float(m.group(1)), float(m.group(2)),
                            float(m.group(3)))
                except ValueError:
                    raise ChaordError(
                        f"{path.name}: line {lineno}: bad tilt line: {s}"
                        ) from None
                continue
            raise ChaordError(
                f"{path.name}: line {lineno}: unrecognised header line: {s}")
        # ---- section keyword or section data row
        if first[0].isalpha():
            name = s.split("#", 1)[0].strip()
            style = None
            if "#" in s and name == "Atoms":
                style = s.split("#", 1)[1].strip() or None
            current = len(sections)
            sections.append((name, style, []))
        else:
            if current is None:
                raise ChaordError(
                    f"{path.name}: line {lineno}: data before any section: {s}")
            sections[current][2].append((lineno, s.split()))

    if not ({"x", "y", "z"} <= box.keys()):
        raise ChaordError(f"{path.name}: header is missing box bounds")
    if "atoms" not in counts:
        raise ChaordError(f"{path.name}: header is missing the atom count")

    skipped = []
    parsed = {}
    for name, style, rows in sections:
        if name not in _PARSED_SECTIONS:
            skipped.append((name, len(rows)))
        else:
            parsed[name] = (style, rows)
    if "Atoms" not in parsed:
        raise ChaordError(f"{path.name}: no Atoms section")
    if "Masses" not in parsed:
        raise ChaordError(f"{path.name}: no Masses section (element mapping "
                          "comes from the masses; refusing to guess)")

    masses = {}
    for lineno, tok in parsed["Masses"][1]:
        if len(tok) < 2:
            raise ChaordError(
                f"{path.name}: line {lineno}: bad Masses row: {' '.join(tok)}")
        try:
            masses[int(tok[0])] = float(tok[1])
        except ValueError:
            raise ChaordError(
                f"{path.name}: line {lineno}: bad Masses row: {' '.join(tok)}"
            ) from None

    style_hint, atom_rows = parsed["Atoms"]
    atoms = [_parse_atoms_row(tok, style_hint, lineno, path)
             for lineno, tok in atom_rows]
    atoms.sort(key=lambda a: a["id"])
    ids = [a["id"] for a in atoms]
    if ids != list(range(1, len(atoms) + 1)):
        raise ChaordError(
            f"{path.name}: atom ids must be unique and contiguous 1..N "
            f"(got {len(ids)} rows, first gap or duplicate near "
            f"{sorted(set(range(1, len(ids) + 1)) - set(ids))[:3]})")
    if counts["atoms"] != len(atoms):
        raise ChaordError(
            f"{path.name}: header declares {counts['atoms']} atoms but the "
            f"Atoms section has {len(atoms)} rows")

    n_types = type_counts.get("atom")
    for a in atoms:
        if n_types is not None and not 1 <= a["type"] <= n_types:
            raise ChaordError(
                f"{path.name}: atom {a['id']} has type {a['type']} outside "
                f"the declared 1..{n_types} atom types")
        if a["type"] not in masses:
            raise ChaordError(
                f"{path.name}: atom {a['id']} has type {a['type']} with no "
                "Masses entry")

    lo = np.array([box["x"][0], box["y"][0], box["z"][0]])
    hi = np.array([box["x"][1], box["y"][1], box["z"][1]])
    cell = _cell_from_bounds(lo, hi, tilt)
    pos = np.array([a["pos"] for a in atoms], float) - lo
    _validate_inside_box(pos, cell, f"{path.name} Atoms section")

    tmap = LammpsTypeMap(masses)
    symbols = [tmap.symbol(a["type"]) for a in atoms]
    charges = ([a["q"] for a in atoms]
               if all(a["q"] is not None for a in atoms) else None)
    molecule_ids = ([a["mol"] for a in atoms]
                    if all(a["mol"] is not None for a in atoms) else None)
    image_flags = (np.array([a["img"] for a in atoms], int)
                   if all(a["img"] is not None for a in atoms) else None)

    velocities = None
    if "Velocities" in parsed:
        velocities = np.zeros((len(atoms), 3))
        seen = set()
        for lineno, tok in parsed["Velocities"][1]:
            if len(tok) != 4:
                raise ChaordError(
                    f"{path.name}: line {lineno}: bad Velocities row: "
                    f"{' '.join(tok)}")
            try:
                aid = int(tok[0])
                velocities[aid - 1] = [float(t) for t in tok[1:4]]
            except ValueError:
                raise ChaordError(
                    f"{path.name}: line {lineno}: bad Velocities row: "
                    f"{' '.join(tok)}") from None
            seen.add(aid)
        if seen != set(ids):
            raise ChaordError(
                f"{path.name}: Velocities ids do not cover atoms 1..{len(ids)}")

    def _topology(section, width, label):
        if section not in parsed:
            return None
        out = []
        for lineno, tok in parsed[section][1]:
            if len(tok) != width:
                raise ChaordError(
                    f"{path.name}: line {lineno}: bad {label} row: "
                    f"{' '.join(tok)}")
            try:
                out.append(tuple(int(t) for t in tok))
            except ValueError:
                raise ChaordError(
                    f"{path.name}: line {lineno}: bad {label} row: "
                    f"{' '.join(tok)}") from None
        return out

    bonds = _topology("Bonds", 4, "Bonds")
    angles = _topology("Angles", 5, "Angles")

    frame = Frame(pos=pos, cell=cell, symbols=symbols,
                  pbc=(True, True, True))
    style_name = style_hint
    if style_name is None:
        # name the interpretation actually used (see docstring decision 4)
        style_name = ("full" if charges is not None and molecule_ids is not None
                      else "charge" if charges is not None else "atomic")
    metadata = {
        "type_map": tmap,
        "unmapped_types": list(tmap.unmapped),
        "charges": charges,
        "molecule_ids": molecule_ids,
        "image_flags": image_flags,
        "velocities": velocities,
        "bonds": bonds,
        "angles": angles,
        "atom_style": style_name,
        "n_atom_types": n_types if n_types is not None else len(masses),
        "triclinic": tilt is not None,
        "box_lo": tuple(float(v) for v in lo),
        "counts": counts,
        "skipped_sections": skipped,
        "boundary": "assumed periodic (data files do not store boundaries)",
    }
    return frame, metadata


# ------------------------------------------------------------ dump files

def read_lammps_dump(path, frame_indices=None) -> list:
    """Read a LAMMPS text dump -> list of Frames (one per snapshot).

    NPT dumps work because each snapshot carries its own BOX BOUNDS and
    therefore its own cell; triclinic dumps populate the tilt factors
    (Frame stores a full 3x3 cell).  ``frame_indices`` selects snapshots
    (None = all; int; list/tuple of ints; slice).  See the module
    docstring for every parse decision.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    frames = []
    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if not s.startswith("ITEM:"):
            raise ChaordError(
                f"{path.name}: line {i + 1}: expected an ITEM: header, "
                f"got {s!r}")
        head = s[len("ITEM:"):].strip()
        if head != "TIMESTEP":
            raise ChaordError(
                f"{path.name}: line {i + 1}: expected ITEM: TIMESTEP to "
                f"start a snapshot, got ITEM: {head}")
        if i + 1 >= n:
            raise ChaordError(f"{path.name}: file ends after TIMESTEP header")
        try:
            timestep = int(lines[i + 1].strip())
        except ValueError:
            raise ChaordError(
                f"{path.name}: line {i + 2}: bad timestep "
                f"{lines[i + 1]!r}") from None
        i += 2

        # NUMBER OF ATOMS
        _expect(path, lines, i, "NUMBER OF ATOMS")
        i += 1
        try:
            natoms = int(lines[i].strip())
        except ValueError:
            raise ChaordError(
                f"{path.name}: line {i + 1}: bad atom count "
                f"{lines[i]!r}") from None
        i += 1

        # BOX BOUNDS
        labels = _expect(path, lines, i, "BOX BOUNDS")
        i += 1
        tilt_labels = labels[-3:] == ["xy", "xz", "yz"]
        axis_labels = labels[:-3] if tilt_labels else labels
        if len(axis_labels) != 3:
            raise ChaordError(
                f"{path.name}: line {i + 1}: cannot read BOX BOUNDS labels "
                f"{labels}")
        bounds = []
        for k in range(3):
            try:
                vals = [float(t) for t in lines[i + k].split()]
            except ValueError:
                raise ChaordError(
                    f"{path.name}: line {i + k + 1}: bad box bounds line "
                    f"{lines[i + k]!r}") from None
            want = 3 if tilt_labels else 2
            if len(vals) != want:
                raise ChaordError(
                    f"{path.name}: line {i + k + 1}: box line needs {want} "
                    f"numbers, got {len(vals)}")
            bounds.append(vals)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        tilt = (bounds[0][2], bounds[1][2], bounds[2][2]) if tilt_labels \
            else None
        cell = _cell_from_bounds(lo, hi, tilt)
        i += 3

        # ATOMS
        columns = _expect(path, lines, i, "ATOMS")
        i += 1
        rows = []
        for k in range(natoms):
            if i + k >= n:
                raise ChaordError(
                    f"{path.name}: file ends before the declared "
                    f"{natoms} atom rows")
            if lines[i + k].strip().startswith("ITEM:"):
                raise ChaordError(
                    f"{path.name}: snapshot declares {natoms} atoms but "
                    f"only {k} row(s) appear before the next ITEM header")
            tok = lines[i + k].split()
            if len(tok) != len(columns):
                raise ChaordError(
                    f"{path.name}: line {i + k + 1}: row has {len(tok)} "
                    f"tokens but the header declares {len(columns)} columns")
            rows.append(tok)
        i += natoms

        frames.append(_dump_snapshot(path, timestep, natoms, columns, rows,
                                     lo, cell, tilt, axis_labels))

    total = len(frames)
    if frame_indices is None:
        selected = frames
    elif isinstance(frame_indices, slice):
        selected = frames[frame_indices]
    elif isinstance(frame_indices, int):
        selected = [_select_one(path, frames, frame_indices, total)]
    else:
        selected = [_select_one(path, frames, j, total)
                    for j in frame_indices]
    return list(selected)


def _select_one(path, frames, index, total):
    if not -total <= index < total:
        raise ChaordError(
            f"{path.name}: frame index {index} out of range; the dump has "
            f"{total} snapshot(s)")
    return frames[index]


def _expect(path, lines, i, keyword):
    s = lines[i].strip() if i < len(lines) else ""
    if not s.startswith("ITEM:"):
        raise ChaordError(
            f"{path.name}: line {i + 1}: expected ITEM: {keyword}, got {s!r}")
    head = s[len("ITEM:"):].strip()
    if not head.startswith(keyword):
        raise ChaordError(
            f"{path.name}: line {i + 1}: expected ITEM: {keyword}, "
            f"got ITEM: {head}")
    # labels after the (possibly multi-word) keyword, e.g. "pp pp pp"
    return head.split()[len(keyword.split()):]


def _dump_snapshot(path, timestep, natoms, columns, rows, lo, cell,
                   tilt, axis_labels) -> Frame:
    def col(name):
        return columns.index(name)

    for need in ("id", "type"):
        if need not in columns:
            raise ChaordError(
                f"{path.name} (timestep {timestep}): ATOMS header lacks the "
                f"'{need}' column: {' '.join(columns)}")
    cart = all(c in columns for c in ("x", "y", "z"))
    scaled = all(c in columns for c in ("xs", "ys", "zs"))
    unwrapped = (all(c in columns for c in ("xu", "yu", "zu"))
                 or all(c in columns for c in ("xsu", "ysu", "zsu")))
    if cart:
        mode = "cart"
    elif scaled:
        mode = "scaled"
    elif unwrapped:
        raise ChaordError(
            f"{path.name} (timestep {timestep}): only unwrapped coordinate "
            "columns (xu/yu/zu or xsu/ysu/zsu) are present; Frame keeps "
            "positions wrapped into the cell, so re-dump with x y z (or "
            "xs ys zs) instead of silently storing unwrapped coordinates")
    else:
        raise ChaordError(
            f"{path.name} (timestep {timestep}): no usable coordinate "
            f"columns in ATOMS header: {' '.join(columns)}")

    ids = np.array([int(r[col("id")]) for r in rows])
    tids = np.array([int(r[col("type")]) for r in rows])
    if len(set(ids.tolist())) != len(ids):
        raise ChaordError(
            f"{path.name} (timestep {timestep}): duplicate atom ids in the "
            "ATOMS block")
    order = np.argsort(ids)

    if mode == "cart":
        pos = np.array([[float(r[col(c)]) for c in ("x", "y", "z")]
                        for r in rows], float)[order] - lo
        _validate_inside_box(
            pos, cell, f"{path.name} (timestep {timestep})")
    else:
        frac = np.array([[float(r[col(c)]) for c in ("xs", "ys", "zs")]
                         for r in rows], float)[order]
        # wrap scaled coordinates into [0, 1) (module docstring decision 3)
        frac = np.mod(frac, 1.0)
        pos = frac @ cell

    charges = None
    if "q" in columns:
        charges = np.array([float(r[col("q")]) for r in rows], float)[order]
    image_flags = None
    if all(c in columns for c in ("ix", "iy", "iz")):
        image_flags = np.array([[int(r[col(c)]) for c in ("ix", "iy", "iz")]
                                for r in rows], int)[order]

    pbc = tuple(
        len(lab) == 2 and lab[0] == "p" and lab[1] == "p"
        for lab in axis_labels)
    frame = Frame(pos=pos, cell=cell, symbols=["X"] * natoms, pbc=pbc,
                  info={
                      "timestep": timestep,
                      "type_ids": tids[order],
                      "charges": charges,
                      "image_flags": image_flags,
                      "triclinic": tilt is not None,
                      "box_lo": tuple(float(v) for v in lo),
                      "box_boundary_labels": list(axis_labels),
                  })
    return frame
