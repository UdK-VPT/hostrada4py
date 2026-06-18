#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hostradaPoint.py reads hourly HOSTRADA values for a specific 1 km x 1 km grid.

Features:
- Enter a location as longitude/latitude (WGS84, EPSG:4326)
- Transform to HOSTRADA projection (EPSG:3034)
- Download the required monthly NetCDF files from the DWD
- Select the nearest 1-km grid point
- Export hourly HOSTRADA values as a CSV file

Required installations:
    pip install numpy pandas requests xarray pyproj  
"""

from __future__ import annotations
import calendar
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer
import hostrada4py.hostrada as hs
from hostrada4py.hostradaDiffuse import HostradaDiffuse, combine_point_variables

CACHE_DIR = Path("hostrada_cache")

def normalize_time_index(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError("No ‘time’ coordinate found in the NetCDF file.")
    return ds

def find_xy_dim_names(ds: xr.Dataset, var_name: str) -> Tuple[str, str]:
    """
    Returns the actual spatial dimensions of the data variable.
    Typically expects (‘time’, ‘Y’, ‘X’) or (‘time’, ‘y’, ‘x’).
    Returns: (x_dim, y_dim)
    """
    dims = ds[var_name].dims

    spatial_dims = [d for d in dims if d.lower() != "time"]

    if len(spatial_dims) != 2:
        raise KeyError(
            f"Expected exactly 2 spatial dimensions in addition to ‘time’, found: {dims}"
        )

    y_candidates = [d for d in spatial_dims if d.lower() == "y"]
    x_candidates = [d for d in spatial_dims if d.lower() == "x"]

    if len(x_candidates) == 1 and len(y_candidates) == 1:
        return x_candidates[0], y_candidates[0]

    # Fallback: erste Raumdimension = y, zweite = x
    y_dim, x_dim = spatial_dims[0], spatial_dims[1]
    return x_dim, y_dim

def get_xy_axis_values(ds: xr.Dataset, x_dim: str, y_dim: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the 1D axis values for the room dimensions.
    """
    if x_dim not in ds.coords or y_dim not in ds.coords:
        raise KeyError(
            f"Spatial dimensions are not available as 1D coordinates.. "
            f"x_dim={x_dim}, y_dim={y_dim}, coords={list(ds.coords)}"
        )

    x_vals = ds.coords[x_dim].values
    y_vals = ds.coords[y_dim].values

    if x_vals.ndim != 1 or y_vals.ndim != 1:
        raise ValueError(
            f"Expect 1D axis coordinates. "
            f"{x_dim}.ndim={x_vals.ndim}, {y_dim}.ndim={y_vals.ndim}"
        )

    return x_vals, y_vals

def transform_lonlat_to_epsg3034(lon: float, lat: float) -> Tuple[float, float]:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)

def extract_from_dataset(
    var: str,
    ds: xr.Dataset,
    lon: float,
    lat: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    ds = normalize_time_index(ds)

    if var in ["tas", "uhi", "sfcWind", "sfcWind_direction", "rsds", "clt", "hurs", "mixr", "ps", "psl", "tdew"]:
        var_name = hs.find_variable(var, ds)
    else:
        print("unknown variable")
        
    x_dim, y_dim = find_xy_dim_names(ds, var_name)
    x_vals, y_vals = get_xy_axis_values(ds, x_dim, y_dim)

    x3034, y3034 = transform_lonlat_to_epsg3034(lon, lat)

    # Determine the nearest grid index
    ix = int(np.abs(x_vals - x3034).argmin())
    iy = int(np.abs(y_vals - y3034).argmin())

    # Selection based on actual room dimensions
    da = ds[var_name].isel({x_dim: ix, y_dim: iy})

    # Crop the time window
    start_naive = pd.Timestamp(start).tz_localize(None)
    end_naive = pd.Timestamp(end).tz_localize(None)
    da = da.sel(time=slice(start_naive, end_naive))

    selected_x = float(x_vals[ix])
    selected_y = float(y_vals[iy])

    df = da.to_dataframe(name=var).reset_index()
    df["input_lon"] = lon
    df["input_lat"] = lat
    df["grid_x_epsg3034"] = selected_x
    df["grid_y_epsg3034"] = selected_y

    # If available: Include 2D longitude and latitude at the selected grid point
    if "lon" in ds.coords and set(ds["lon"].dims) == {y_dim, x_dim}:
        df["grid_lon"] = float(ds["lon"].isel({y_dim: iy, x_dim: ix}).values)

    if "lat" in ds.coords and set(ds["lat"].dims) == {y_dim, x_dim}:
        df["grid_lat"] = float(ds["lat"].isel({y_dim: iy, x_dim: ix}).values)

    unit = da.attrs.get("units", "")
    if unit:
        df["unit"] = unit

    long_name = da.attrs.get("long_name", "")
    if long_name:
        df["variable_description"] = long_name

    return df

def extract_values_for_point(
    var: str,
    lon: float,
    lat: float,
    start: str,
    end: str,
    cache_dir: Path = CACHE_DIR,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    if end_ts < start_ts:
        raise ValueError("'end' muss >= 'start' sein.")

    monthly_frames: List[pd.DataFrame] = []

    for year, month in hs.month_range(start_ts, end_ts):
        target = hs.ensure_month_file_for_point(
            var,
            year,
            month,
            cache_dir,
            lon=lon,
            lat=lat,
            start=start_ts,
            end=end_ts,
            subset_mode=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )

        print(f"Read: {target}")
        with hs.read_month_file(target) as ds:
            df = extract_from_dataset(var, ds, lon, lat, start_ts, end_ts)
            monthly_frames.append(df)

    if not monthly_frames:
        return pd.DataFrame()

    result = pd.concat(monthly_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return result


def extract_multiple_values_for_point(
    vars: Iterable[str],
    lon: float,
    lat: float,
    start: str,
    end: str,
    cache_dir: Path = CACHE_DIR,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract multiple HOSTRADA variables for one point and return them as a single
    wide dataframe indexed by time.

    Duplicate variable names are evaluated only once so no duplicate monthly
    files are requested from the DWD server.
    """
    frames: List[pd.DataFrame] = []
    seen_vars = set()
    for var in vars:
        if var in seen_vars:
            continue
        seen_vars.add(var)
        frames.append(
            extract_values_for_point(
                var,
                lon,
                lat,
                start,
                end,
                cache_dir=cache_dir,
                cache_strategy=cache_strategy,
                subset_margin_cells=subset_margin_cells,
            )
        )
    return combine_point_variables(frames)


def _diffuse_required_vars(apply_weather_correction: bool) -> List[str]:
    """Return the smallest HOSTRADA variable set needed by the DHI helper.

    ``rsds`` is the only strictly required input for the Erbs-Driesse model.
    ``tas`` and ``ps`` are kept in the default path because they are used by
    pvlib's solar-position calculation when present, matching the previous
    numerical behaviour without downloading unrelated weather fields.
    """
    required = ["rsds", "tas", "ps"]

    if apply_weather_correction:
        required.extend([
            "clt",
            "hurs",
            "tdew",
            "mixr",
            "sfcWind",
            "uhi",
            "psl",
        ])

    return required


def extract_diffuse_radiation_for_point(
    lon: float,
    lat: float,
    start: str,
    end: str,
    altitude: float | None = None,
    tz: str = "UTC",
    cache_dir: Path = CACHE_DIR,
    apply_weather_correction: bool = False,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
) -> pd.DataFrame:
    """
    One-line helper for point-based HOSTRADA diffuse radiation calculation.

    Downloads the required HOSTRADA variables, combines them into one dataframe,
    and calculates DHI/DNI/kd using HostradaDiffuse with erbs_driesse as the
    robust base method.
    """
    needed_vars = _diffuse_required_vars(apply_weather_correction)

    data = extract_multiple_values_for_point(
        vars=needed_vars,
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
    )

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        tz=tz,
    ) 

    df = model.estimate(data, apply_weather_correction=apply_weather_correction)

    # Index -> Create an explicit time column
    df_out = df.copy()

    # Ensure that the index is of type Datetime
    if not isinstance(df_out.index, pd.DatetimeIndex):
        raise ValueError("The index must be a datetime index")
    
    # Add a time column
    df_out.insert(0, "time", df_out.index)
    
    # Optional: Reset index (recommended for export / CSV / user-friendliness)
    df_out = df_out.reset_index(drop=True)
    
    return df_out
