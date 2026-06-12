#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Required installations:
  
  pip install netcdf4

"""

import calendar
import requests
from pathlib import Path
import xarray as xr
import pandas as pd
from typing import Iterable, List, Tuple, Sequence

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

def hostrada_filename(var: str, year: int, month: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    return f"{FILE_PREFIXS[var]}_{year:04d}{month:02d}0100-{year:04d}{month:02d}{last_day:02d}23.nc"

def hostrada_url(var: str, year: int, month: int) -> str:
    return f"{BASE_URLS[var]}/{hostrada_filename(var, year, month)}"

def is_cached_file(path: Path) -> bool:
    """Return True if a cached HOSTRADA file is present and non-empty."""
    return path.exists() and path.is_file() and path.stat().st_size > 0


def download_file(url: str, target: Path, timeout: int = 120) -> Path:
    """Download *url* to *target* only if the target is not cached yet.

    The file is first written to ``*.part`` and then moved into place atomically.
    This avoids treating interrupted downloads as valid cache entries later on.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_cached_file(target):
        return target

    tmp_target = target.with_name(target.name + ".part")
    if tmp_target.exists():
        tmp_target.unlink()

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        tmp_target.replace(target)
    except Exception:
        if tmp_target.exists():
            tmp_target.unlink()
        raise

    return target


def ensure_month_file(
    var: str,
    year: int,
    month: int,
    cache_dir: Path,
    timeout: int = 120,
    verbose: bool = True,
) -> Path:
    """Return the local monthly HOSTRADA file, downloading it only if needed.

    All higher-level extractors should go through this function so the DWD
    server is contacted only for files that are both required by the requested
    variable/date range and not already available in the local cache.
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
    return download_file(url, target, timeout=timeout)


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