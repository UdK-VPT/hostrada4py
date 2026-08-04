from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import pandas as pd

TimeoutValue = float | int | Tuple[float, float]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Machine-readable description of one weather-data backend."""

    name: str
    variables: frozenset[str]
    temporal_resolution: str
    spatial_resolution_m: float
    crs: str = "EPSG:3034"
    start: Optional[str] = None
    end: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)


class WeatherProvider(ABC):
    """Provider contract used by the backwards-compatible hostrada facade.

    Every provider returns a local NetCDF file with the public HOSTRADA schema:
    one canonical variable, a ``time`` coordinate and rectilinear ``X``/``Y``
    coordinates in EPSG:3034.  This keeps the existing point, polygon, route and
    weather-file code independent of the original data source.
    """

    name: str

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def supports(self, var: str) -> bool:
        return var in self.capabilities.variables

    def require_variable(self, var: str) -> None:
        if not self.supports(var):
            available = ", ".join(sorted(self.capabilities.variables))
            raise NotImplementedError(
                f"Provider '{self.name}' does not provide '{var}'. "
                f"Available canonical variables: {available}."
            )

    @abstractmethod
    def filename(self, var: str, year: int, month: int) -> str:
        raise NotImplementedError

    def url(self, var: str, year: int, month: int) -> str:
        raise NotImplementedError(
            f"Provider '{self.name}' is API-backed and has no static monthly URL."
        )

    @abstractmethod
    def ensure_month_file(
        self,
        var: str,
        year: int,
        month: int,
        cache_dir: Path,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        selector: Optional[Mapping[str, Any]] = None,
        subset_mode: Optional[str] = None,
        subset_margin_cells: Optional[int] = None,
        timeout: Optional[TimeoutValue] = None,
        retries: Optional[int] = None,
        backoff: Optional[float] = None,
        verbose: bool = True,
    ) -> Path:
        """Return a provider-normalised monthly NetCDF file."""
        raise NotImplementedError

    def required_month_files(
        self,
        variables: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        cache_dir: Path,
    ) -> list[Path]:
        from .common import month_range

        seen: set[str] = set()
        result: list[Path] = []
        for var in variables:
            if var in seen:
                continue
            seen.add(var)
            self.require_variable(var)
            for year, month in month_range(start, end):
                result.append(Path(cache_dir) / self.name / self.filename(var, year, month))
        return result
