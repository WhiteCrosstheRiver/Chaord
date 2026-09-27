# Third-party licence audit (M0 deliverable, completed post-hoc)

Chaord is MIT (see LICENSE). Core dependencies ship with the package;
heavy tools are optional extras. Audit date: 2026-09-28.

## Core dependencies (required)

| Package | Licence | Redistribution with MIT Chaord |
| --- | --- | --- |
| numpy | BSD-3-Clause | OK (permissive) |
| scipy | BSD-3-Clause | OK |
| ASE | LGPL-2.1+ (dynamic linking; pip dependency) | OK as a dependency — Chaord does not embed ASE code; LGPL applies to ASE itself, not to importers |
| Lark | MIT | OK |
| Pydantic | MIT | OK |
| PyYAML | MIT | OK |

## Test dependencies

| Package | Licence | Note |
| --- | --- | --- |
| pytest | MIT | OK |
| Hypothesis | MPL-2.0 | file-level copyleft; test-only, not distributed with Chaord |
| spglib | BSD-3-Clause | used in tests and the crystal lifter; OK |

## Optional extras (never imported by core)

| Tool | Licence | Note for packaging as an extra |
| --- | --- | --- |
| OVITO (Python module) | Free for non-commercial ONLY | **Do not bundle.** Document that commercial use needs an OVITO licence; core ships its own passes (Wigner-Seitz, q6, ring statistics) precisely for this reason |
| LAMMPS | GPL-2.0 | invoke as a separate binary, never link; GPL then does not extend to Chaord |
| PLUMED | LGPL-3.0 | separate binary/binding OK; core has protocol/restraint fallbacks |
| MACE | **needs human confirmation** (community licence with commercial restrictions at audit time) | document the terms in the extra's README before depending on it |
| icet | **needs human confirmation** (MPL-style; verify version) | core ships SQS-lite fallback |
| RDKit | BSD-3-Clause | OK |
| pymatgen | MIT | OK |

## Conclusion

The MIT core depends only on permissive/LGPL-as-dependency packages, so
distribution is unencumbered. Two optional extras (MACE, icet) need their
licence terms re-verified at the exact versions before an extras wheel is
published; OVITO must remain opt-in with its non-commercial restriction
called out.
