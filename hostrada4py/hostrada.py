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
from typing import Iterable, List, Tuple

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

def download_file(url: str, target: Path, timeout: int = 120) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and target.stat().st_size > 0:
        return target

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return target    

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