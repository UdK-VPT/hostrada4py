#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Required installations:
  
  pip install netcdf4

"""

from __future__ import annotations

import calendar
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
import requests
import xarray as xr
import pandas as pd
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
