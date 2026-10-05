"""Versioned dialects: thresholds, units and key vocabulary live here and nowhere else.

A dialect is a YAML file next to this module. `dialect core + lj` loads core first
and overlays lj (later dialects win on conflicts). Pass and builder code reads
every number through `dialect.threshold(name)`; the CI check
`tools/check_magic_numbers.py` enforces that no bare float literal appears in
`src/chaord/lift` or `src/chaord/build`.
"""
from __future__ import annotations

from importlib.resources import files as _files
from typing import Iterable, Sequence

import yaml

from ..lang.errors import ChaordError

DIALECT_NAMES = ("core", "metal", "ionic", "molecular", "surface", "glass",
                 "carbon", "lj", "lj_mixtures")


class Dialect:
    def __init__(self, names: Sequence[str], data: dict, versions: dict[str, str]):
        self.names = list(names)
        self._data = data
        self.versions = dict(versions)
        self.units = frozenset(data.get("units", ()))
        self.keys = data.get("keys", {})

    @property
    def version_string(self) -> str:
        return " + ".join(f"{n} {self.versions[n]}" for n in self.names)

    def threshold(self, name: str):
        try:
            return self._data["thresholds"][name]
        except KeyError:
            raise ChaordError(
                f"dialect '{' + '.join(self.names)}' defines no threshold {name!r}"
            ) from None

    def key_allowed(self, scope: str, key: str) -> bool:
        return key in self.keys.get(scope, ())


def _load_yaml(name: str) -> dict:
    text = (_files("chaord.dialects") / f"{name}.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_dialect(names: Iterable[str] = ("core",)) -> Dialect:
    names = list(names)
    if not names:
        raise ChaordError("at least one dialect is required (core is always implied)")
    if names[0] != "core":
        names = ["core"] + names
    merged: dict = {"units": [], "keys": {}, "thresholds": {}}
    versions: dict[str, str] = {}
    for name in names:
        if name not in DIALECT_NAMES:
            raise ChaordError(f"unknown dialect {name!r}; known: {', '.join(DIALECT_NAMES)}")
        data = _load_yaml(name)
        versions[name] = str(data.get("version", "0"))
        merged["units"] = list(dict.fromkeys(merged["units"] + data.get("units", ())))
        for scope, keys in data.get("keys", {}).items():
            merged["keys"].setdefault(scope, [])
            merged["keys"][scope] = list(dict.fromkeys(merged["keys"][scope] + list(keys)))
        merged["thresholds"].update(data.get("thresholds", {}))
    return Dialect(names, merged, versions)


def dialect_from_program(program) -> Dialect:
    return load_dialect(program.dialects or ("core",))
