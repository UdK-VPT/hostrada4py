from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer

from .base import TimeoutValue

DOWNLOAD_CONNECT_TIMEOUT = float(os.getenv("HOSTRADA_CONNECT_TIMEOUT", "20"))
DOWNLOAD_READ_TIMEOUT = float(os.getenv("HOSTRADA_READ_TIMEOUT", "180"))
DOWNLOAD_RETRIES = int(os.getenv("HOSTRADA_DOWNLOAD_RETRIES", "6"))
DOWNLOAD_BACKOFF = float(os.getenv("HOSTRADA_DOWNLOAD_BACKOFF", "2.0"))
DOWNLOAD_CHUNK_SIZE = int(os.getenv("HOSTRADA_DOWNLOAD_CHUNK_SIZE", str(1024 * 1024)))
DOWNLOAD_LOCK_TIMEOUT = float(os.getenv("HOSTRADA_DOWNLOAD_LOCK_TIMEOUT", "900"))
DOWNLOAD_MIN_BYTES = int(os.getenv("HOSTRADA_DOWNLOAD_MIN_BYTES", "1024"))
SUBSET_MARGIN_CELLS_DEFAULT = int(os.getenv("HOSTRADA_NETCDF_SUBSET_MARGIN_CELLS", "1"))


def normalise_timeout(timeout: Optional[TimeoutValue]) -> Tuple[float, float]:
    if timeout is None:
        return DOWNLOAD_CONNECT_TIMEOUT, DOWNLOAD_READ_TIMEOUT
    if isinstance(timeout, tuple):
        return float(timeout[0]), float(timeout[1])
    return float(timeout), float(timeout)


def is_cached_file(path: Path, min_bytes: int = DOWNLOAD_MIN_BYTES) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


@contextmanager
def file_lock(lock_path: Path, timeout: float = DOWNLOAD_LOCK_TIMEOUT, poll_interval: float = 0.25):
    start = time.monotonic()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}\n".encode())
        except FileExistsError:
            if time.monotonic() - start > timeout:
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


def download_file(
    url: str,
    target: Path,
    timeout: Optional[TimeoutValue] = None,
    retries: int = DOWNLOAD_RETRIES,
    backoff: float = DOWNLOAD_BACKOFF,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
    allow_resume: bool = True,
) -> Path:
    """Resumable, atomic download shared by static-file providers."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_cached_file(target):
        return target
    part = target.with_name(target.name + ".part")
    with file_lock(target.with_name(target.name + ".lock")):
        if is_cached_file(target):
            return target
        last_error: BaseException | None = None
        for attempt in range(1, retries + 1):
            try:
                resume_from = part.stat().st_size if allow_resume and part.exists() else 0
                headers = {"User-Agent": "hostrada4py/provider-refactor"}
                mode = "wb"
                if resume_from:
                    headers["Range"] = f"bytes={resume_from}-"
                    mode = "ab"
                with requests.get(
                    url,
                    stream=True,
                    timeout=normalise_timeout(timeout),
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    if resume_from and response.status_code != 206:
                        resume_from = 0
                        mode = "wb"
                    with part.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                stream.write(chunk)
                if part.stat().st_size < DOWNLOAD_MIN_BYTES:
                    raise IOError(f"Downloaded file is unexpectedly small: {part}")
                part.replace(target)
                return target
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == retries:
                    break
                delay = backoff * 2 ** (attempt - 1) + random.uniform(0, min(backoff, 1.0))
                time.sleep(delay)
        raise RuntimeError(
            f"Download failed after {retries} attempts: {url} -> {target}. "
            f"The partial file is kept for resume. Last error: {last_error}"
        ) from last_error


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[int, int]]:
    start = as_utc_timestamp(start)
    end = as_utc_timestamp(end)
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    last = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= last:
        yield current.year, current.month
        current = current + pd.offsets.MonthBegin(1)


def as_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def naive_utc(value: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return as_utc_timestamp(value).tz_localize(None)


def point_to_epsg3034(lon: float, lat: float) -> tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    x, y = transformer.transform(float(lon), float(lat))
    return float(x), float(y)


def bbox_3034_to_lonlat(bbox: tuple[float, float, float, float], pad_deg: float = 0.15) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = map(float, bbox)
    transformer = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
    xs = np.array([minx, minx, maxx, maxx, (minx + maxx) / 2])
    ys = np.array([miny, maxy, miny, maxy, (miny + maxy) / 2])
    lon, lat = transformer.transform(xs, ys)
    return (
        float(np.nanmin(lon) - pad_deg),
        float(np.nanmin(lat) - pad_deg),
        float(np.nanmax(lon) + pad_deg),
        float(np.nanmax(lat) + pad_deg),
    )


def cache_key(params: Mapping[str, object], length: int = 16) -> str:
    payload = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def find_variable(var: str, ds: xr.Dataset) -> str:
    if var in ds.data_vars:
        return var
    candidates = []
    for name, da in ds.data_vars.items():
        dims = {d.lower() for d in da.dims}
        if "time" in dims and len(da.dims) >= 3:
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise KeyError(f"No suitable variable found. Available: {list(ds.data_vars)}")
    raise KeyError(f"Multiple meaning variables found: {candidates}")


def find_xy_dimensions(da: xr.DataArray) -> tuple[str, str]:
    spatial = [d for d in da.dims if d.lower() != "time"]
    if len(spatial) != 2:
        raise KeyError(f"Expected two spatial dimensions, found {da.dims}")
    x = [d for d in spatial if d.lower() == "x"]
    y = [d for d in spatial if d.lower() == "y"]
    return (x[0], y[0]) if len(x) == len(y) == 1 else (spatial[1], spatial[0])


def infer_cell_size(ds: xr.Dataset, var: str | None = None, default: float = 1000.0) -> float:
    attr = ds.attrs.get("hostrada4py_cell_size")
    if attr is not None:
        try:
            return float(attr)
        except (TypeError, ValueError):
            pass
    name = var if var in ds.data_vars else next(iter(ds.data_vars), None)
    if name is None:
        return default
    x_dim, y_dim = find_xy_dimensions(ds[name])
    candidates: list[float] = []
    for dim in (x_dim, y_dim):
        if dim in ds.coords and ds[dim].size > 1:
            diff = np.diff(np.asarray(ds[dim].values, dtype=float))
            diff = np.abs(diff[np.isfinite(diff) & (diff != 0)])
            if diff.size:
                candidates.append(float(np.median(diff)))
    return float(np.median(candidates)) if candidates else default


def write_netcdf_atomic(ds: xr.Dataset, target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    engine = "netcdf4"
    try:
        ds.to_netcdf(part, engine=engine)
    except (ImportError, ModuleNotFoundError, ValueError):
        ds.to_netcdf(part)
    part.replace(target)
    return target


def subset_rectilinear_dataset(
    ds: xr.Dataset,
    var: str,
    selector: Optional[Mapping[str, object]],
    margin_cells: int = SUBSET_MARGIN_CELLS_DEFAULT,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> xr.Dataset:
    name = find_variable(var, ds)
    da = ds[name]
    x_dim, y_dim = find_xy_dimensions(da)
    result = ds[[name]]
    if selector:
        xvals = np.asarray(result[x_dim].values, dtype=float)
        yvals = np.asarray(result[y_dim].values, dtype=float)
        if selector["type"] == "point_epsg3034":
            ix = int(np.abs(xvals - float(selector["x"])).argmin())
            iy = int(np.abs(yvals - float(selector["y"])).argmin())
            result = result.isel(
                {
                    x_dim: slice(max(0, ix - margin_cells), min(len(xvals), ix + margin_cells + 1)),
                    y_dim: slice(max(0, iy - margin_cells), min(len(yvals), iy + margin_cells + 1)),
                }
            )
        elif selector["type"] == "bbox_epsg3034":
            minx, miny, maxx, maxy = map(float, selector["bbox"])
            xidx = np.where((xvals >= minx) & (xvals <= maxx))[0]
            yidx = np.where((yvals >= miny) & (yvals <= maxy))[0]
            if not xidx.size:
                xidx = np.array([int(np.abs(xvals - (minx + maxx) / 2).argmin())])
            if not yidx.size:
                yidx = np.array([int(np.abs(yvals - (miny + maxy) / 2).argmin())])
            result = result.isel(
                {
                    x_dim: slice(max(0, int(xidx.min()) - margin_cells), min(len(xvals), int(xidx.max()) + margin_cells + 1)),
                    y_dim: slice(max(0, int(yidx.min()) - margin_cells), min(len(yvals), int(yidx.max()) + margin_cells + 1)),
                }
            )
        else:
            raise ValueError(f"Unknown selector: {selector}")
    if "time" in result.coords:
        result = result.sel(time=slice(naive_utc(start), naive_utc(end)))
    return result.load()
