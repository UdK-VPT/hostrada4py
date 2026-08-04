"""hostrada4py 0.42.0 with provider-separated DWD and CERRA access.

Public convenience objects are imported lazily.  This keeps independent
submodules such as :mod:`hostrada4py.hostradaArea` usable even when optional or
separately patched point functionality is temporarily unavailable.
"""
from __future__ import annotations

from typing import Any

__version__ = "0.42.0"
__upstream_version__ = "0.42.0"
__provider_overlay_version__ = "1.0.1-lazy-package-imports"

__all__ = [
    "HostradaDiffuse",
    "extract_diffuse_radiation_for_point",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Load public convenience exports only when they are requested."""
    if name == "HostradaDiffuse":
        from .hostradaDiffuse import HostradaDiffuse

        return HostradaDiffuse
    if name == "extract_diffuse_radiation_for_point":
        try:
            from .hostradaPoint import extract_diffuse_radiation_for_point
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.hostradaPoint":
                raise ModuleNotFoundError(
                    "The module 'hostrada4py.hostradaPoint' is missing from the "
                    "installation. Restore hostrada4py/hostradaPoint.py from the "
                    "complete HOSTRADA package."
                ) from exc
            raise
        return extract_diffuse_radiation_for_point
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
