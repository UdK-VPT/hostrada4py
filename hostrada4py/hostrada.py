#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Required installations:
  
  pip install netcdf4

"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import random
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
import requests
import xarray as xr
import pandas as pd
import numpy as np
from typing import Iterable, List, Tuple, Sequence, Optional, Union

BASE_URLS = {"tas":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/air_temperature_mean",
             "uhi":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/urban_heat_island_intensity",
             "sfcWind":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/wind_speed",
             "sfcWind_direction":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/wind_direction",
             "rsds":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/radiation_downwelling",
             "clt":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/cloud_cover",
             "hurs":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/humidity_relative",
             "mixr":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/humidity_mixing_ratio",
             "ps":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/pressure_surface",
             "psl":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/pressure_sealevel",
             "tdew":"https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/dew_point"}

FILE_PREFIXS = {"tas":"tas_1hr_HOSTRADA-v1-0_BE_gn",
                "uhi":"uhi_1hr_HOSTRADA-v1-0_BE_gn",
                "sfcWind":"sfcWind_1hr_HOSTRADA-v1-0_BE_gn",
                "sfcWind_direction":"sfcWind_direction_1hr_HOSTRADA-v1-0_BE_gn",
                "rsds":"rsds_1hr_HOSTRADA-v1-0_BE_gn",
                "clt":"clt_1hr_HOSTRADA-v1-0_BE_gn",
                "hurs":"hurs_1hr_HOSTRADA-v1-0_BE_gn",
                "mixr":"mixr_1hr_HOSTRADA-v1-0_BE_gn",
                "ps":"ps_1hr_HOSTRADA-v1-0_BE_gn",
                "psl":"psl_1hr_HOSTRADA-v1-0_BE_gn",
                "tdew":"tdew_1hr_HOSTRADA-v1-0_BE_gn"}

# Robust download defaults. They can be overridden per call or via environment
# variables, which is useful for long annual weather-file exports on unstable
# network connections.
DOWNLOAD_CONNECT_TIMEOUT = float(os.getenv("HOSTRADA_CONNECT_TIMEOUT", "20"))
DOWNLOAD_READ_TIMEOUT = float(os.getenv("HOSTRADA_READ_TIMEOUT", "180"))
DOWNLOAD_RETRIES = int(os.getenv("HOSTRADA_DOWNLOAD_RETRIES", "6"))
DOWNLOAD_BACKOFF = float(os.getenv("HOSTRADA_DOWNLOAD_BACKOFF", "2.0"))
DOWNLOAD_CHUNK_SIZE = int(os.getenv("HOSTRADA_DOWNLOAD_CHUNK_SIZE", str(1024 * 1024)))
DOWNLOAD_LOCK_TIMEOUT = float(os.getenv("HOSTRADA_DOWNLOAD_LOCK_TIMEOUT", "900"))
DOWNLOAD_MIN_BYTES = int(os.getenv("HOSTRADA_DOWNLOAD_MIN_BYTES", "1024"))

# Optional NetCDF subsetting. The default remains the original full monthly-file
# cache. Set HOSTRADA_NETCDF_SUBSET_MODE to one of:
#   full       -> original behaviour (default)
#   subset     -> download full file if needed, write small local subset cache
#   http_range -> try HTTP range/chunked remote reads first, fall back if enabled
#   auto       -> try http_range, then local subset fallback
# The HTTP range path is best-effort and requires optional packages
# fsspec + h5netcdf and server support for byte ranges.
SUBSET_MODE_DEFAULT = os.getenv("HOSTRADA_NETCDF_SUBSET_MODE", "full")
SUBSET_MARGIN_CELLS_DEFAULT = int(os.getenv("HOSTRADA_NETCDF_SUBSET_MARGIN_CELLS", "1"))
SUBSET_HTTP_RANGE_BLOCK_SIZE = int(os.getenv("HOSTRADA_HTTP_RANGE_BLOCK_SIZE", str(2 * 1024 * 1024)))
SUBSET_FALLBACK_TO_FULL = os.getenv("HOSTRADA_NETCDF_SUBSET_FALLBACK", "1").strip().lower() not in {"0", "false", "no", "off"}
SUBSET_DROP_FULL_AFTER_CREATE = os.getenv("HOSTRADA_DROP_FULL_AFTER_SUBSET", "0").strip().lower() in {"1", "true", "yes", "on"}

TimeoutValue = Union[float, int, Tuple[float, float]]


def hostrada_filename(var: str, year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{FILE_PREFIXS[var]}_{year:04d}{month:02d}0100-{year:04d}{month:02d}{last_day:02d}23.nc"


def hostrada_url(var: str, year: int, month: int) -> str:
    return f"{BASE_URLS[var]}/{hostrada_filename(var, year, month)}"


def _normalise_timeout(timeout: Optional[TimeoutValue]) -> Tuple[float, float]:
    if timeout is None:
        return (DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT)
    if isinstance(timeout, tuple):
        return (float(timeout[0]), float(timeout[1]))
    return (float(timeout), float(timeout))


def is_cached_file(path: Path, min_bytes: int = DOWNLOAD_MIN_BYTES) -> bool:
    """Return True if a cached HOSTRADA file is present and plausibly complete."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


@contextmanager
def _file_lock(lock_path: Path, timeout: float = DOWNLOAD_LOCK_TIMEOUT, poll_interval: float = 0.25):
    """Small dependency-free lock to prevent concurrent downloads of one file."""
    start = time.monotonic()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        except FileExistsError:
            if time.monotonic() - start > timeout:
                # The previous process may have crashed. Remove the stale lock
                # only after the generous timeout has elapsed.
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            time.sleep(poll_interval)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _content_length_from_response(response: requests.Response) -> Optional[int]:
    value = response.headers.get("Content-Length")
    if value is None:
        return None
    try:
        length = int(value)
    except ValueError:
        return None
    return length if length >= 0 else None


def _expected_final_size(response: requests.Response, resume_from: int) -> Optional[int]:
    """Return expected final target size for 200/206 responses if known."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total != "*":
            try:
                return int(total)
            except ValueError:
                pass

    length = _content_length_from_response(response)
    if length is None:
        return None
    if response.status_code == 206:
        return resume_from + length
    return length


def _sleep_before_retry(attempt: int, backoff: float) -> None:
    # Exponential backoff plus a small jitter prevents many parallel exports from
    # retrying at the exact same moment.
    delay = backoff * (2 ** max(0, attempt - 1))
    delay += random.uniform(0.0, min(1.0, backoff))
    time.sleep(delay)


def _download_file_once(
    url: str,
    tmp_target: Path,
    timeout: Optional[TimeoutValue],
    chunk_size: int,
    allow_resume: bool,
) -> int:
    resume_from = tmp_target.stat().st_size if allow_resume and tmp_target.exists() else 0
    headers = {"User-Agent": "hostrada4py/robust-downloader"}
    mode = "wb"
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    with requests.get(url, stream=True, timeout=_normalise_timeout(timeout), headers=headers) as r:
        if r.status_code == 416:
            # The partial file is inconsistent with the server object. Let the
            # retry loop restart from byte 0.
            raise IOError("HTTP 416: cached partial download is inconsistent with remote file")
        r.raise_for_status()

        # Some servers ignore Range and answer with 200. In that case overwrite
        # the partial file rather than appending a duplicate byte sequence.
        if resume_from > 0 and r.status_code != 206:
            resume_from = 0
            mode = "wb"

        expected_size = _expected_final_size(r, resume_from)
        bytes_written = resume_from
        with open(tmp_target, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)

    if expected_size is not None and bytes_written != expected_size:
        raise IOError(
            f"Incomplete download for {url}: got {bytes_written} bytes, "
            f"expected {expected_size} bytes"
        )
    if bytes_written < DOWNLOAD_MIN_BYTES:
        raise IOError(f"Downloaded file is unexpectedly small: {bytes_written} bytes")
    return bytes_written


def download_file(
    url: str,
    target: Path,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    allow_resume: bool = True,
) -> Path:
    """Download *url* to *target* only if the target is not cached yet.

    The implementation is robust for large annual weather exports that require
    many monthly NetCDF files:

    - configurable connect/read timeouts,
    - retry with exponential backoff for transient network/server errors,
    - atomic ``*.part`` downloads followed by ``replace()``,
    - optional resume of interrupted ``*.part`` files via HTTP Range,
    - final byte-count checks when the server provides a length,
    - a small lock file so concurrent processes do not corrupt the same cache.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_cached_file(target):
        return target

    lock_path = target.with_name(target.name + ".lock")
    tmp_target = target.with_name(target.name + ".part")

    with _file_lock(lock_path):
        # Another process may have completed the file while we waited.
        if is_cached_file(target):
            return target
        if target.exists() and not is_cached_file(target):
            target.unlink()

        last_error: Optional[BaseException] = None
        for attempt in range(1, retries + 1):
            try:
                _download_file_once(
                    url=url,
                    tmp_target=tmp_target,
                    timeout=timeout,
                    chunk_size=chunk_size,
                    allow_resume=allow_resume,
                )
                tmp_target.replace(target)
                return target
            except Exception as exc:  # noqa: BLE001 - re-raised with context below
                last_error = exc
                if isinstance(exc, IOError) and "HTTP 416" in str(exc):
                    try:
                        tmp_target.unlink()
                    except FileNotFoundError:
                        pass
                if attempt >= retries:
                    break
                _sleep_before_retry(attempt, backoff)

        if last_error is not None:
            raise RuntimeError(
                f"Download failed after {retries} attempts: {url} -> {target}. "
                f"The partial file is kept as {tmp_target} and will be resumed on the next run. "
                f"Last error: {last_error}"
            ) from last_error

    return target


def ensure_month_file(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    verbose: bool = True,
) -> Path:
    """Return the local monthly HOSTRADA file, downloading it only if needed.

    All higher-level extractors should go through this function so the DWD
    server is contacted only for files that are both required by the requested
    variable/date range and not already available in the local cache. Interrupted
    downloads are retried and resumable ``*.part`` files are reused.
    """
    filename = hostrada_filename(var, year, month)
    target = Path(cache_dir) / filename

    if is_cached_file(target):
        if verbose:
            print(f"Cache: {target}")
        return target

    url = hostrada_url(var, year, month)
    if verbose:
        print(f"Download: {url}")
    return download_file(url, target, timeout=timeout, retries=retries, backoff=backoff)


def _normalise_subset_mode(mode: Optional[str] = None) -> str:
    value = (mode or os.getenv("HOSTRADA_NETCDF_SUBSET_MODE", SUBSET_MODE_DEFAULT) or "full").strip().lower()
    aliases = {
        "0": "full",
        "false": "full",
        "no": "full",
        "off": "full",
        "1": "subset",
        "true": "subset",
        "yes": "subset",
        "on": "subset",
        "range": "http_range",
        "remote": "http_range",
        "remote_subset": "http_range",
        "http-range": "http_range",
    }
    value = aliases.get(value, value)
    valid = {"full", "subset", "http_range", "auto"}
    if value not in valid:
        raise ValueError(
            "Unknown HOSTRADA NetCDF subset mode "
            f"'{value}'. Use one of: {sorted(valid)}."
        )
    return value


def _normalise_margin_cells(margin_cells: Optional[int]) -> int:
    if margin_cells is None:
        margin_cells = int(os.getenv("HOSTRADA_NETCDF_SUBSET_MARGIN_CELLS", str(SUBSET_MARGIN_CELLS_DEFAULT)))
    return max(0, int(margin_cells))


def _subset_cache_path(cache_dir: Path, filename: str, kind: str, params: dict) -> Path:
    serialisable = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha1(serialisable.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / "subsets" / kind / f"{Path(filename).stem}.{digest}.nc"


def _find_xy_dim_names_for_var(ds: xr.Dataset, var: str) -> Tuple[str, str]:
    var_name = find_variable(var, ds)
    spatial_dims = [d for d in ds[var_name].dims if d.lower() != "time"]
    if len(spatial_dims) != 2:
        raise KeyError(f"Expected exactly two spatial dimensions for {var_name}, found {ds[var_name].dims}")
    x_candidates = [d for d in spatial_dims if d.lower() == "x"]
    y_candidates = [d for d in spatial_dims if d.lower() == "y"]
    if len(x_candidates) == 1 and len(y_candidates) == 1:
        return x_candidates[0], y_candidates[0]
    return spatial_dims[1], spatial_dims[0]


def _axis_values(ds: xr.Dataset, x_dim: str, y_dim: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_dim not in ds.coords or y_dim not in ds.coords:
        raise KeyError(f"Spatial coordinates are missing: x_dim={x_dim}, y_dim={y_dim}")
    x_vals = np.asarray(ds.coords[x_dim].values)
    y_vals = np.asarray(ds.coords[y_dim].values)
    if x_vals.ndim != 1 or y_vals.ndim != 1:
        raise ValueError(f"Expected 1D x/y coordinates, got {x_vals.ndim}D/{y_vals.ndim}D")
    return x_vals, y_vals


def _point_to_epsg3034(lon: float, lat: float) -> Tuple[float, float]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    return float(x), float(y)


def _index_slice_around_value(values: np.ndarray, value: float, margin_cells: int) -> slice:
    idx = int(np.abs(values - value).argmin())
    start = max(0, idx - margin_cells)
    stop = min(len(values), idx + margin_cells + 1)
    return slice(start, stop)


def _index_slice_for_bounds(values: np.ndarray, low: float, high: float, margin_cells: int) -> slice:
    lo, hi = sorted((float(low), float(high)))
    idx = np.where((values >= lo) & (values <= hi))[0]
    if idx.size == 0:
        centre = 0.5 * (lo + hi)
        nearest = int(np.abs(values - centre).argmin())
        start = max(0, nearest - margin_cells)
        stop = min(len(values), nearest + margin_cells + 1)
        return slice(start, stop)
    start = max(0, int(idx.min()) - margin_cells)
    stop = min(len(values), int(idx.max()) + margin_cells + 1)
    return slice(start, stop)


def _time_slice(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> slice:
    if start is None and end is None:
        return slice(None)
    start_naive = pd.Timestamp(start).tz_localize(None) if start is not None else None
    end_naive = pd.Timestamp(end).tz_localize(None) if end is not None else None
    return slice(start_naive, end_naive)


def _write_subset_from_dataset(
    ds: xr.Dataset,
    target: Path,
    var: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    x_slice: slice,
    y_slice: slice,
) -> Path:
    var_name = find_variable(var, ds)
    x_dim, y_dim = _find_xy_dim_names_for_var(ds, var)
    indexers = {x_dim: x_slice, y_dim: y_slice}
    subset = ds.isel(indexers)
    if "time" in subset.coords or "time" in subset.dims:
        subset = subset.sel(time=_time_slice(start, end))
    # Keep only the requested HOSTRADA variable and coordinates. This avoids
    # accidentally writing unrelated auxiliary variables when future files add
    # additional data variables.
    keep_vars = [var_name]
    subset = subset[keep_vars]
    subset = subset.load()

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(target.name + ".part")
    if tmp_target.exists():
        tmp_target.unlink()
    subset.to_netcdf(tmp_target, engine="netcdf4")
    tmp_target.replace(target)
    return target


def _subset_local_file(
    source: Path,
    target: Path,
    var: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    selector: dict,
    margin_cells: int,
) -> Path:
    with xr.open_dataset(source, engine="netcdf4") as ds:
        var_name = find_variable(var, ds)
        x_dim, y_dim = _find_xy_dim_names_for_var(ds, var)
        x_vals, y_vals = _axis_values(ds, x_dim, y_dim)

        if selector["type"] == "point_epsg3034":
            x_slice = _index_slice_around_value(x_vals, selector["x"], margin_cells)
            y_slice = _index_slice_around_value(y_vals, selector["y"], margin_cells)
        elif selector["type"] == "bbox_epsg3034":
            minx, miny, maxx, maxy = selector["bbox"]
            x_slice = _index_slice_for_bounds(x_vals, minx, maxx, margin_cells)
            y_slice = _index_slice_for_bounds(y_vals, miny, maxy, margin_cells)
        else:
            raise ValueError(f"Unknown subset selector: {selector}")

        return _write_subset_from_dataset(ds, target, var, start, end, x_slice, y_slice)


def _subset_remote_http_range(
    url: str,
    target: Path,
    var: str,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    selector: dict,
    margin_cells: int,
) -> Path:
    """Best-effort remote NetCDF subsetting via HTTP byte ranges.

    This path can reduce transferred bytes only if the server supports HTTP
    Range requests and the optional packages fsspec + h5netcdf are installed.
    It is intentionally optional because the DWD OpenData HTTP endpoint is a
    static-file service rather than a guaranteed OPeNDAP/NCSS subset service.
    """
    import fsspec  # optional dependency

    engine = os.getenv("HOSTRADA_HTTP_RANGE_ENGINE", "h5netcdf")
    block_size = int(os.getenv("HOSTRADA_HTTP_RANGE_BLOCK_SIZE", str(SUBSET_HTTP_RANGE_BLOCK_SIZE)))
    headers = {"User-Agent": "hostrada4py/http-range-subsetter"}

    opened = fsspec.open(url, mode="rb", block_size=block_size, cache_type="bytes", headers=headers)
    fobj = opened.open()
    ds = None
    try:
        ds = xr.open_dataset(fobj, engine=engine)
        x_dim, y_dim = _find_xy_dim_names_for_var(ds, var)
        x_vals, y_vals = _axis_values(ds, x_dim, y_dim)

        if selector["type"] == "point_epsg3034":
            x_slice = _index_slice_around_value(x_vals, selector["x"], margin_cells)
            y_slice = _index_slice_around_value(y_vals, selector["y"], margin_cells)
        elif selector["type"] == "bbox_epsg3034":
            minx, miny, maxx, maxy = selector["bbox"]
            x_slice = _index_slice_for_bounds(x_vals, minx, maxx, margin_cells)
            y_slice = _index_slice_for_bounds(y_vals, miny, maxy, margin_cells)
        else:
            raise ValueError(f"Unknown subset selector: {selector}")

        return _write_subset_from_dataset(ds, target, var, start, end, x_slice, y_slice)
    finally:
        if ds is not None:
            ds.close()
        fobj.close()


def _ensure_month_subset_file(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    subset_target: Path,
    selector: dict,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    subset_mode: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    verbose: bool = True,
) -> Path:
    mode = _normalise_subset_mode(subset_mode)
    if mode == "full":
        return ensure_month_file(var, year, month, cache_dir, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)

    subset_target = Path(subset_target)
    if is_cached_file(subset_target):
        if verbose:
            print(f"Subset cache: {subset_target}")
        return subset_target

    margin_cells = _normalise_margin_cells(subset_margin_cells)
    url = hostrada_url(var, year, month)
    errors = []

    if mode in {"http_range", "auto"}:
        try:
            if verbose:
                print(f"HTTP-range subset: {url}")
            return _subset_remote_http_range(
                url=url,
                target=subset_target,
                var=var,
                start=start,
                end=end,
                selector=selector,
                margin_cells=margin_cells,
            )
        except Exception as exc:  # noqa: BLE001 - best-effort path with fallback
            errors.append(exc)
            if not SUBSET_FALLBACK_TO_FULL and mode == "http_range":
                raise RuntimeError(f"HTTP-range NetCDF subsetting failed for {url}: {exc}") from exc
            warnings.warn(
                "HTTP-range NetCDF subsetting failed; falling back to full monthly "
                f"download followed by local subsetting. Last error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    full_file = ensure_month_file(
        var, year, month, cache_dir, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose
    )
    if verbose:
        print(f"Create subset cache: {subset_target}")
    result = _subset_local_file(
        source=full_file,
        target=subset_target,
        var=var,
        start=start,
        end=end,
        selector=selector,
        margin_cells=margin_cells,
    )

    drop_full = os.getenv("HOSTRADA_DROP_FULL_AFTER_SUBSET", "1" if SUBSET_DROP_FULL_AFTER_CREATE else "0").strip().lower() in {"1", "true", "yes", "on"}
    if drop_full and Path(full_file).exists():
        try:
            Path(full_file).unlink()
        except OSError:
            pass
    return result


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
) -> Path:
    """Return a monthly HOSTRADA file for one point.

    By default this is exactly the original full monthly cache file. If
    ``subset_mode`` or ``HOSTRADA_NETCDF_SUBSET_MODE`` is set to ``subset``,
    ``http_range`` or ``auto``, a small NetCDF file containing only the relevant
    time window and a small spatial neighbourhood around the point is returned.
    """
    mode = _normalise_subset_mode(subset_mode)
    if mode == "full":
        return ensure_month_file(var, year, month, cache_dir, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)

    x, y = _point_to_epsg3034(lon, lat)
    margin_cells = _normalise_margin_cells(subset_margin_cells)
    filename = hostrada_filename(var, year, month)
    params = {
        "version": 1,
        "kind": "point",
        "var": var,
        "year": year,
        "month": month,
        "lon": round(float(lon), 8),
        "lat": round(float(lat), 8),
        "start": str(pd.Timestamp(start).tz_localize(None) if start is not None else ""),
        "end": str(pd.Timestamp(end).tz_localize(None) if end is not None else ""),
        "margin_cells": margin_cells,
    }
    target = _subset_cache_path(cache_dir, filename, "point", params)
    selector = {"type": "point_epsg3034", "x": x, "y": y}
    return _ensure_month_subset_file(
        var=var,
        year=year,
        month=month,
        cache_dir=cache_dir,
        subset_target=target,
        selector=selector,
        start=start,
        end=end,
        subset_mode=mode,
        subset_margin_cells=margin_cells,
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
) -> Path:
    """Return a monthly HOSTRADA file restricted to an EPSG:3034 bounding box.

    This is useful for polygon/area queries. The default is the unchanged full
    monthly download; set ``subset_mode`` or ``HOSTRADA_NETCDF_SUBSET_MODE`` for
    a smaller cache file.
    """
    mode = _normalise_subset_mode(subset_mode)
    if mode == "full":
        return ensure_month_file(var, year, month, cache_dir, timeout=timeout, retries=retries, backoff=backoff, verbose=verbose)

    minx, miny, maxx, maxy = [float(v) for v in bbox_epsg3034]
    margin_cells = _normalise_margin_cells(subset_margin_cells)
    filename = hostrada_filename(var, year, month)
    params = {
        "version": 1,
        "kind": "bbox",
        "var": var,
        "year": year,
        "month": month,
        "bbox_epsg3034": [round(minx, 3), round(miny, 3), round(maxx, 3), round(maxy, 3)],
        "start": str(pd.Timestamp(start).tz_localize(None) if start is not None else ""),
        "end": str(pd.Timestamp(end).tz_localize(None) if end is not None else ""),
        "margin_cells": margin_cells,
    }
    target = _subset_cache_path(cache_dir, filename, "bbox", params)
    selector = {"type": "bbox_epsg3034", "bbox": (minx, miny, maxx, maxy)}
    return _ensure_month_subset_file(
        var=var,
        year=year,
        month=month,
        cache_dir=cache_dir,
        subset_target=target,
        selector=selector,
        start=start,
        end=end,
        subset_mode=mode,
        subset_margin_cells=margin_cells,
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
) -> List[Path]:
    """List the minimum monthly files needed for variables and time range.

    Duplicate variables are ignored while preserving the first occurrence. This
    helper does not download anything; it only exposes the exact download plan
    used by callers and tests.
    """
    seen_vars = set()
    unique_vars = []
    for var in vars:
        if var not in seen_vars:
            unique_vars.append(var)
            seen_vars.add(var)

    files: List[Path] = []
    for var in unique_vars:
        for year, month in month_range(start, end):
            files.append(Path(cache_dir) / hostrada_filename(var, year, month))
    return files


def read_month_file(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Tuple[int, int]]:
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")

    while current <= last:
        yield current.year, current.month
        if current.month == 12:
            current = pd.Timestamp(year=current.year + 1, month=1, day=1, tz="UTC")
        else:
            current = pd.Timestamp(year=current.year, month=current.month + 1, day=1, tz="UTC")


def find_variable(var: str, ds: xr.Dataset) -> str:
    if var in ds.data_vars:
        return var

    candidates = []
    for var_name, da in ds.data_vars.items():
        dims_lower = {d.lower() for d in da.dims}
        if "time" in dims_lower and len(da.dims) >= 3:
            candidates.append(var_name)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"No suitable variable found. Available: {list(ds.data_vars)}")

    raise KeyError(f"Multiple meaning variables found: {candidates}")
