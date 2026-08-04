#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backwards-compatible data-access facade for hostrada4py.

The original public functions remain available. DWD-specific URL/file/cache
logic lives in :mod:`hostrada4py.providers.dwd_hostrada`; CERRA access lives in
:mod:`hostrada4py.providers.cerra`. Existing notebooks continue to use DWD by
default and can switch providers through ``HOSTRADA_PROVIDER=cerra`` or the
``use_provider`` context manager.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import xarray as xr

from hostrada4py.providers.base import TimeoutValue, WeatherProvider
from hostrada4py.providers.cerra import CERRAProvider
from hostrada4py.providers.common import (
    DOWNLOAD_BACKOFF,
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_CONNECT_TIMEOUT,
    DOWNLOAD_LOCK_TIMEOUT,
    DOWNLOAD_MIN_BYTES,
    DOWNLOAD_READ_TIMEOUT,
    DOWNLOAD_RETRIES,
    SUBSET_MARGIN_CELLS_DEFAULT,
    download_file,
    find_variable,
    infer_cell_size,
    is_cached_file,
    month_range,
    point_to_epsg3034,
)
from hostrada4py.providers.dwd_hostrada import (
    BASE_URLS,
    FILE_PREFIXES,
    FILE_PREFIXS,
    DWDHostradaProvider,
)

SUBSET_MODE_DEFAULT = os.getenv("HOSTRADA_NETCDF_SUBSET_MODE", "full")
SUBSET_HTTP_RANGE_BLOCK_SIZE = int(
    os.getenv("HOSTRADA_HTTP_RANGE_BLOCK_SIZE", str(2 * 1024 * 1024))
)
SUBSET_FALLBACK_TO_FULL = os.getenv("HOSTRADA_NETCDF_SUBSET_FALLBACK", "1").lower() not in {
    "0", "false", "no", "off"
}
SUBSET_DROP_FULL_AFTER_CREATE = os.getenv("HOSTRADA_DROP_FULL_AFTER_SUBSET", "0").lower() in {
    "1", "true", "yes", "on"
}

_PROVIDERS: dict[str, WeatherProvider] = {}
_PROVIDER_CONTEXT: ContextVar[Optional[str]] = ContextVar(
    "hostrada4py_provider", default=None
)


def register_provider(provider: WeatherProvider, *aliases: str, replace: bool = False) -> None:
    """Register a provider and optional aliases."""
    names = (provider.name, *aliases)
    for name in names:
        key = name.strip().lower()
        if key in _PROVIDERS and not replace and _PROVIDERS[key] is not provider:
            raise KeyError(f"Provider alias '{key}' is already registered.")
        _PROVIDERS[key] = provider


register_provider(DWDHostradaProvider(), "hostrada", "dwd_hostrada")
register_provider(CERRAProvider(), "copernicus_cerra")


def available_providers() -> tuple[str, ...]:
    return tuple(sorted({provider.name for provider in _PROVIDERS.values()}))


def get_provider(provider: str | WeatherProvider | None = None) -> WeatherProvider:
    if isinstance(provider, WeatherProvider):
        return provider
    selected = provider or _PROVIDER_CONTEXT.get() or os.getenv("HOSTRADA_PROVIDER", "dwd")
    key = str(selected).strip().lower()
    try:
        return _PROVIDERS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown weather provider '{selected}'. Available: {available_providers()}"
        ) from exc


def get_provider_name(provider: str | WeatherProvider | None = None) -> str:
    """Return the active provider name without changing notebook APIs."""
    return get_provider(provider).name


def supported_variables(provider: str | WeatherProvider | None = None) -> tuple[str, ...]:
    """Return canonical variables supported by the active provider."""
    return tuple(sorted(get_provider(provider).capabilities.variables))


def set_default_provider(provider: str) -> None:
    """Set the process-wide default used by existing notebooks."""
    get_provider(provider)  # validation
    os.environ["HOSTRADA_PROVIDER"] = provider


@contextmanager
def use_provider(provider: str | WeatherProvider):
    """Temporarily route all existing extractor calls to one provider."""
    selected = get_provider(provider)
    token = _PROVIDER_CONTEXT.set(selected.name)
    try:
        yield selected
    finally:
        _PROVIDER_CONTEXT.reset(token)


def provider_capabilities(provider: str | WeatherProvider | None = None):
    return get_provider(provider).capabilities


def hostrada_filename(
    var: str,
    year: int,
    month: int,
    provider: str | WeatherProvider | None = None,
) -> str:
    return get_provider(provider).filename(var, year, month)


def hostrada_url(
    var: str,
    year: int,
    month: int,
    provider: str | WeatherProvider | None = None,
) -> str:
    return get_provider(provider).url(var, year, month)


def ensure_month_file(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    verbose: bool = True,
    provider: str | WeatherProvider | None = None,
) -> Path:
    return get_provider(provider).ensure_month_file(
        var,
        year,
        month,
        Path(cache_dir),
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        verbose=verbose,
    )


def ensure_month_file_for_point(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    lon: float,
    lat: float,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    subset_mode: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    verbose: bool = True,
    provider: str | WeatherProvider | None = None,
) -> Path:
    x, y = point_to_epsg3034(lon, lat)
    selector: Mapping[str, object] = {
        "type": "point_epsg3034",
        "x": x,
        "y": y,
    }
    return get_provider(provider).ensure_month_file(
        var,
        year,
        month,
        Path(cache_dir),
        start=start,
        end=end,
        selector=selector,
        subset_mode=subset_mode,
        subset_margin_cells=subset_margin_cells,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        verbose=verbose,
    )


def ensure_month_file_for_bbox(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    bbox_epsg3034: Tuple[float, float, float, float],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    subset_mode: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    verbose: bool = True,
    provider: str | WeatherProvider | None = None,
) -> Path:
    selector: Mapping[str, object] = {
        "type": "bbox_epsg3034",
        "bbox": tuple(map(float, bbox_epsg3034)),
    }
    return get_provider(provider).ensure_month_file(
        var,
        year,
        month,
        Path(cache_dir),
        start=start,
        end=end,
        selector=selector,
        subset_mode=subset_mode,
        subset_margin_cells=subset_margin_cells,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
        verbose=verbose,
    )


def required_month_files(
    vars: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
    provider: str | WeatherProvider | None = None,
) -> List[Path]:
    return get_provider(provider).required_month_files(vars, start, end, Path(cache_dir))


def read_month_file(path: Path) -> xr.Dataset:
    """Open DWD or provider-normalised NetCDF with a portable engine fallback."""
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except (ImportError, ModuleNotFoundError, ValueError):
        return xr.open_dataset(path)


# Small compatibility helpers retained for callers that used private functions.
def _normalise_subset_mode(mode: Optional[str] = None) -> str:
    return DWDHostradaProvider._normalise_mode(mode)


def _normalise_margin_cells(margin_cells: Optional[int]) -> int:
    if margin_cells is None:
        return SUBSET_MARGIN_CELLS_DEFAULT
    return max(0, int(margin_cells))


def _point_to_epsg3034(lon: float, lat: float) -> Tuple[float, float]:
    return point_to_epsg3034(lon, lat)


__all__ = [
    "BASE_URLS",
    "FILE_PREFIXES",
    "FILE_PREFIXS",
    "available_providers",
    "register_provider",
    "get_provider",
    "set_default_provider",
    "use_provider",
    "provider_capabilities",
    "hostrada_filename",
    "hostrada_url",
    "is_cached_file",
    "download_file",
    "ensure_month_file",
    "ensure_month_file_for_point",
    "ensure_month_file_for_bbox",
    "required_month_files",
    "read_month_file",
    "month_range",
    "find_variable",
    "infer_cell_size",
]
