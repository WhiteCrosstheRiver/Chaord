"""CV registry: names in, floats out; one home per collective variable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..lang.errors import ChaordError


@dataclass
class CVDef:
    name: str
    unit: Optional[str]
    description: str
    thresholds: tuple[str, ...] = ()      # dialect threshold names the measure reads
    measure: Optional[Callable[..., float]] = None
    restrain: Optional[Callable] = None   # build-side (M5)
    dialect: str = "core"

    def is_measurable(self) -> bool:
        return self.measure is not None


CVS: dict[str, CVDef] = {}


def register(name: str, unit: Optional[str], description: str,
             thresholds: tuple[str, ...] = (), dialect: str = "core",
             measure: Optional[Callable[..., float]] = None,
             restrain: Optional[Callable] = None) -> CVDef:
    if name in CVS:
        raise ChaordError(f"CV {name!r} already registered (one definition per quantity)")
    cv = CVDef(name=name, unit=unit, description=description, thresholds=thresholds,
               dialect=dialect, measure=measure, restrain=restrain)
    CVS[name] = cv
    return cv


def missing_measure(name: str, cv: CVDef):
    def _raise(*args, **kwargs):
        raise ChaordError(
            f"CV {name!r} has no measure yet (planned for a later milestone)")
    return _raise


def measure(name: str, frame, dialect, **kwargs) -> float:
    try:
        cv = CVS[name]
    except KeyError:
        raise ChaordError(f"unknown CV {name!r}; known: {', '.join(sorted(CVS))}") from None
    if cv.measure is None:
        raise ChaordError(f"CV {name!r} has no measure yet")
    return cv.measure(frame, dialect, **kwargs)
