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

Performance notes:
- Reuses the CRS transformation and spatial grid selection across months/variables
- Uses binary nearest-neighbour lookup for monotonic HOSTRADA axes
- Builds point dataframes directly instead of materialising xarray MultiIndexes
- Combines multiple variables by aligned index concatenation instead of repeated merges

Required installations:
    pip install numpy pandas xarray pyproj
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import hostrada4py.hostrada as hs
from hostrada4py.hostradaDiffuse import HostradaDiffuse, combine_point_variables

CACHE_DIR = Path("hostrada_cache")

_SUPPORTED_VARIABLES = frozenset(
    {
        "tas",
        "uhi",
        "sfcWind",
        "sfcWind_direction",
        "rsds",
        "clt",
        "hurs",
        "mixr",
        "ps",
        "psl",
        "tdew",
    }
)

_METADATA_COLUMNS = (
    "input_lon",
    "input_lat",
    "grid_x_epsg3034",
    "grid_y_epsg3034",
    "grid_lon",
    "grid_lat",
    "X",
    "Y",
    "x",
    "y",
    "lon",
    "lat",
)

_NON_VALUE_COLUMNS = frozenset(
    {
        "time",
        "unit",
        "variable_description",
        *_METADATA_COLUMNS,
    }
)

_LONLAT_TO_EPSG3034 = Transformer.from_crs(
    "EPSG:4326", "EPSG:3034", always_xy=True
)


def normalize_time_index(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError("No ‘time’ coordinate found in the NetCDF file.")
    return ds


def find_xy_dim_names(ds: xr.Dataset, var_name: str) -> Tuple[str, str]:
    """
    Return the actual spatial dimensions of the data variable.

    Typically expects ("time", "Y", "X") or ("time", "y", "x").
    Returns ``(x_dim, y_dim)``.
    """
    dims = ds[var_name].dims
    spatial_dims = [dim for dim in dims if dim.lower() != "time"]

    if len(spatial_dims) != 2:
        raise KeyError(
            "Expected exactly 2 spatial dimensions in addition to ‘time’, "
            f"found: {dims}"
        )

    y_candidates = [dim for dim in spatial_dims if dim.lower() == "y"]
    x_candidates = [dim for dim in spatial_dims if dim.lower() == "x"]

    if len(x_candidates) == 1 and len(y_candidates) == 1:
        return x_candidates[0], y_candidates[0]

    # Fallback: first spatial dimension = y, second = x.
    y_dim, x_dim = spatial_dims
    return x_dim, y_dim


def get_xy_axis_values(
    ds: xr.Dataset, x_dim: str, y_dim: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the 1D axis values for the spatial dimensions."""
    if x_dim not in ds.coords or y_dim not in ds.coords:
        raise KeyError(
            "Spatial dimensions are not available as 1D coordinates. "
            f"x_dim={x_dim}, y_dim={y_dim}, coords={list(ds.coords)}"
        )

    x_vals = np.asarray(ds.coords[x_dim].values)
    y_vals = np.asarray(ds.coords[y_dim].values)

    if x_vals.ndim != 1 or y_vals.ndim != 1:
        raise ValueError(
            "Expected 1D axis coordinates. "
            f"{x_dim}.ndim={x_vals.ndim}, {y_dim}.ndim={y_vals.ndim}"
        )
    if x_vals.size == 0 or y_vals.size == 0:
        raise ValueError("Spatial axis coordinates must not be empty.")

    return x_vals, y_vals


@lru_cache(maxsize=256)
def transform_lonlat_to_epsg3034(lon: float, lat: float) -> Tuple[float, float]:
    """Transform a WGS84 point to EPSG:3034 and cache repeated locations."""
    x, y = _LONLAT_TO_EPSG3034.transform(float(lon), float(lat))
    return float(x), float(y)


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _to_naive_utc(value: object) -> pd.Timestamp:
    return _as_utc_timestamp(value).tz_localize(None)


def _axis_signature(values: np.ndarray) -> Tuple[object, ...]:
    """Return a cheap, stable signature for regular HOSTRADA coordinate axes."""
    values = np.asarray(values)
    positions = sorted({0, 1, values.size // 2, values.size - 2, values.size - 1})
    samples = tuple(float(values[pos]) for pos in positions if 0 <= pos < values.size)
    return values.size, values.dtype.str, samples


def _nearest_axis_index(values: np.ndarray, target: float) -> int:
    """Find the nearest axis value, using O(log n) lookup for monotonic axes.

    Ties are resolved in favour of the lower original index, matching
    ``np.abs(values - target).argmin()`` from the previous implementation.
    """
    values = np.asarray(values)
    size = values.size
    if size == 1:
        return 0

    first = values[0]
    last = values[-1]
    ascending = bool(last >= first)
    ordered = values if ascending else values[::-1]

    # HOSTRADA axes are monotonic. Keep a safe fallback for custom NetCDF files.
    if np.any(ordered[1:] < ordered[:-1]):
        return int(np.abs(values - target).argmin())

    insertion = int(np.searchsorted(ordered, target, side="left"))
    ordered_candidates = {
        max(0, min(size - 1, insertion - 1)),
        max(0, min(size - 1, insertion)),
    }
    original_candidates = [
        candidate if ascending else size - 1 - candidate
        for candidate in ordered_candidates
    ]
    return min(
        original_candidates,
        key=lambda index: (abs(float(values[index]) - target), index),
    )


@dataclass(slots=True)
class _PointExtractionContext:
    lon: float
    lat: float
    x3034: float = field(init=False)
    y3034: float = field(init=False)
    grid_index_cache: Dict[Tuple[object, ...], Tuple[int, int]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.lon = float(self.lon)
        self.lat = float(self.lat)
        self.x3034, self.y3034 = transform_lonlat_to_epsg3034(self.lon, self.lat)

    def grid_indices(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        x_dim: str,
        y_dim: str,
    ) -> Tuple[int, int]:
        key = (
            x_dim,
            y_dim,
            _axis_signature(x_vals),
            _axis_signature(y_vals),
        )
        cached = self.grid_index_cache.get(key)
        if cached is not None:
            return cached

        indices = (
            _nearest_axis_index(x_vals, self.x3034),
            _nearest_axis_index(y_vals, self.y3034),
        )
        self.grid_index_cache[key] = indices
        return indices


def _variable_name(var: str, ds: xr.Dataset) -> str:
    if var not in _SUPPORTED_VARIABLES:
        raise ValueError(
            f"Unknown HOSTRADA variable {var!r}. "
            f"Supported variables: {sorted(_SUPPORTED_VARIABLES)}"
        )
    return hs.find_variable(var, ds)


def _point_dataarray_to_dataframe(da: xr.DataArray, output_name: str) -> pd.DataFrame:
    """Convert a one-dimensional point DataArray without xarray's MultiIndex path."""
    if "time" not in da.dims:
        raise KeyError("Selected data array has no ‘time’ dimension.")

    values = np.asarray(da.values)
    if values.ndim != 1:
        values = values.reshape(-1)

    data: Dict[str, object] = {"time": np.asarray(da.coords["time"].values)}

    # Preserve scalar and time-dependent coordinates emitted by the old
    # ``to_dataframe(...).reset_index()`` path (e.g. X, Y, lon and lat).
    for coord_name, coord in da.coords.items():
        if coord_name == "time" or coord_name == output_name:
            continue
        coord_values = np.asarray(coord.values)
        if coord_values.ndim == 0:
            data[coord_name] = np.repeat(coord_values.reshape(1), values.size)
        elif coord.dims == ("time",):
            data[coord_name] = coord_values

    data[output_name] = values
    return pd.DataFrame(data, copy=False)


def _extract_from_dataset_with_context(
    var: str,
    ds: xr.Dataset,
    context: _PointExtractionContext,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    ds = normalize_time_index(ds)
    var_name = _variable_name(var, ds)

    x_dim, y_dim = find_xy_dim_names(ds, var_name)
    x_vals, y_vals = get_xy_axis_values(ds, x_dim, y_dim)
    ix, iy = context.grid_indices(x_vals, y_vals, x_dim, y_dim)

    # Select the single grid cell before loading values from NetCDF.
    da = ds[var_name].isel({x_dim: ix, y_dim: iy})
    da = da.sel(time=slice(_to_naive_utc(start), _to_naive_utc(end)))

    selected_x = float(x_vals[ix])
    selected_y = float(y_vals[iy])
    df = _point_dataarray_to_dataframe(da, var)

    df["input_lon"] = context.lon
    df["input_lat"] = context.lat
    df["grid_x_epsg3034"] = selected_x
    df["grid_y_epsg3034"] = selected_y

    lon_coord = ds.coords.get("lon")
    if lon_coord is not None and set(lon_coord.dims) == {y_dim, x_dim}:
        df["grid_lon"] = float(lon_coord.isel({y_dim: iy, x_dim: ix}).values)

    lat_coord = ds.coords.get("lat")
    if lat_coord is not None and set(lat_coord.dims) == {y_dim, x_dim}:
        df["grid_lat"] = float(lat_coord.isel({y_dim: iy, x_dim: ix}).values)

    unit = da.attrs.get("units", "")
    if unit:
        df["unit"] = unit

    long_name = da.attrs.get("long_name", "")
    if long_name:
        df["variable_description"] = long_name

    return df


def extract_from_dataset(
    var: str,
    ds: xr.Dataset,
    lon: float,
    lat: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Extract one variable for the nearest HOSTRADA grid cell."""
    context = _PointExtractionContext(lon=lon, lat=lat)
    return _extract_from_dataset_with_context(
        var=var,
        ds=ds,
        context=context,
        start=_as_utc_timestamp(start),
        end=_as_utc_timestamp(end),
    )


def _extract_values_for_point_timestamps(
    var: str,
    context: _PointExtractionContext,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    months: Tuple[Tuple[int, int], ...],
    cache_dir: Path,
    cache_strategy: Optional[str],
    subset_margin_cells: Optional[int],
) -> pd.DataFrame:
    monthly_frames: List[pd.DataFrame] = []

    for year, month in months:
        target = hs.ensure_month_file_for_point(
            var,
            year,
            month,
            cache_dir,
            lon=context.lon,
            lat=context.lat,
            start=start_ts,
            end=end_ts,
            subset_mode=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )

        print(f"Read: {target}")
        with hs.read_month_file(target) as ds:
            monthly_frames.append(
                _extract_from_dataset_with_context(
                    var, ds, context, start_ts, end_ts
                )
            )

    if not monthly_frames:
        return pd.DataFrame()

    if len(monthly_frames) == 1:
        result = monthly_frames[0]
    else:
        result = pd.concat(monthly_frames, ignore_index=True, copy=False)

    # Overlap can occur in custom subset files; retain the original behaviour.
    if result["time"].duplicated().any():
        result = result.drop_duplicates(subset=["time"])
    if not result["time"].is_monotonic_increasing:
        result = result.sort_values("time")
    return result.reset_index(drop=True)


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
    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)

    if end_ts < start_ts:
        raise ValueError("'end' muss >= 'start' sein.")

    context = _PointExtractionContext(lon=lon, lat=lat)
    months = tuple(hs.month_range(start_ts, end_ts))
    return _extract_values_for_point_timestamps(
        var=var,
        context=context,
        start_ts=start_ts,
        end_ts=end_ts,
        months=months,
        cache_dir=Path(cache_dir),
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
    )


def _value_column(frame: pd.DataFrame) -> str:
    candidates = [column for column in frame.columns if column not in _NON_VALUE_COLUMNS]
    known = [column for column in candidates if column in _SUPPORTED_VARIABLES]
    if not known and len(candidates) == 1:
        known = candidates
    if len(known) != 1:
        raise ValueError(
            "Expected exactly one HOSTRADA value column per frame after excluding "
            f"metadata/coordinates. Found candidates: {candidates}, selected: {known}"
        )
    return known[0]


def _combine_point_variables_fast(
    frames: Iterable[pd.DataFrame], time_col: str = "time"
) -> pd.DataFrame:
    """Combine point frames with a zero-merge fast path for aligned hourly data.

    HOSTRADA variables normally share exactly the same hourly timestamps. In that
    common case the result is built directly from the underlying arrays. Custom
    inputs with differing timelines fall back to the established general helper.
    """
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()

    first = non_empty[0]
    if time_col not in first.columns:
        raise KeyError(f"Missing time column '{time_col}' in one input frame.")

    base_time = first[time_col].to_numpy(copy=False)
    aligned = not pd.Index(base_time).has_duplicates
    value_columns: List[Tuple[str, np.ndarray]] = []

    for frame in non_empty:
        if time_col not in frame.columns:
            raise KeyError(f"Missing time column '{time_col}' in one input frame.")
        frame_time = frame[time_col].to_numpy(copy=False)
        if (
            frame_time.shape != base_time.shape
            or not np.array_equal(frame_time, base_time, equal_nan=True)
        ):
            aligned = False
            break
        value_col = _value_column(frame)
        value_columns.append((value_col, frame[value_col].to_numpy(copy=False)))

    if not aligned:
        return combine_point_variables(non_empty, time_col=time_col)

    index = pd.DatetimeIndex(pd.to_datetime(base_time, utc=True), name=time_col)
    data = {name: values for name, values in value_columns}

    # Keep the previous column order: values first, then metadata from frame one.
    for column in _METADATA_COLUMNS:
        if column in first.columns:
            data[column] = first[column].to_numpy(copy=False)

    return pd.DataFrame(data, index=index, copy=False)


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
    start_ts = _as_utc_timestamp(start)
    end_ts = _as_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("'end' muss >= 'start' sein.")

    # Preserve first occurrence while removing duplicates.
    unique_vars = tuple(dict.fromkeys(vars))
    if not unique_vars:
        return pd.DataFrame()

    context = _PointExtractionContext(lon=lon, lat=lat)
    months = tuple(hs.month_range(start_ts, end_ts))
    frames = [
        _extract_values_for_point_timestamps(
            var=var,
            context=context,
            start_ts=start_ts,
            end_ts=end_ts,
            months=months,
            cache_dir=Path(cache_dir),
            cache_strategy=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )
        for var in unique_vars
    ]
    return _combine_point_variables_fast(frames)


def _diffuse_required_vars(apply_weather_correction: bool) -> List[str]:
    """Return the smallest HOSTRADA variable set needed by the DHI helper.

    ``rsds`` is the only strictly required input for the Erbs-Driesse model.
    ``tas`` and ``ps`` remain in the default path because pvlib uses them for
    solar-position calculation when present, matching previous numerical output.
    """
    required = ["rsds", "tas", "ps"]

    if apply_weather_correction:
        required.extend(
            [
                "clt",
                "hurs",
                "tdew",
                "mixr",
                "sfcWind",
                "uhi",
                "psl",
            ]
        )

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
    result = model.estimate(
        data, apply_weather_correction=apply_weather_correction
    )

    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("The index must be a datetime index")

    # ``rename_axis().reset_index()`` avoids the previous copy + insert + reset.
    return result.rename_axis("time").reset_index()
