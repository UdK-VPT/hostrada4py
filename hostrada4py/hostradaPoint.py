#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Point extraction independent of the original weather-data provider.

Data access is routed through
:mod:`hostrada4py.hostrada`; DWD is the default and CERRA can be selected with
``HOSTRADA_PROVIDER=cerra`` or ``hostrada.use_provider('cerra')``.
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
_SUPPORTED_VARIABLES = frozenset({
    "tas", "uhi", "sfcWind", "sfcWind_direction", "rsds", "clt",
    "hurs", "mixr", "ps", "psl", "tdew",
})
_METADATA_COLUMNS = (
    "input_lon", "input_lat", "grid_x_epsg3034", "grid_y_epsg3034",
    "grid_lon", "grid_lat", "X", "Y", "x", "y", "lon", "lat",
)
_NON_VALUE_COLUMNS = frozenset({"time", "unit", "variable_description", *_METADATA_COLUMNS})
_LONLAT_TO_EPSG3034 = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)


def normalize_time_index(ds: xr.Dataset) -> xr.Dataset:
    if "time" not in ds.coords and "time" not in ds.dims:
        raise KeyError("No 'time' coordinate found in the NetCDF file.")
    return ds


def find_xy_dim_names(ds: xr.Dataset, var_name: str) -> Tuple[str, str]:
    dims = ds[var_name].dims
    spatial = [d for d in dims if d.lower() != "time"]
    if len(spatial) != 2:
        raise KeyError(f"Expected two spatial dimensions in addition to time, found {dims}.")
    xs = [d for d in spatial if d.lower() == "x"]
    ys = [d for d in spatial if d.lower() == "y"]
    if len(xs) == len(ys) == 1:
        return xs[0], ys[0]
    y_dim, x_dim = spatial
    return x_dim, y_dim


def get_xy_axis_values(ds: xr.Dataset, x_dim: str, y_dim: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_dim not in ds.coords or y_dim not in ds.coords:
        raise KeyError(f"Missing 1-D spatial coordinates {x_dim!r}/{y_dim!r}.")
    x = np.asarray(ds[x_dim].values)
    y = np.asarray(ds[y_dim].values)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or y.size == 0:
        raise ValueError("Spatial axes must be non-empty and one-dimensional.")
    return x, y


@lru_cache(maxsize=256)
def transform_lonlat_to_epsg3034(lon: float, lat: float) -> Tuple[float, float]:
    x, y = _LONLAT_TO_EPSG3034.transform(float(lon), float(lat))
    return float(x), float(y)


def _as_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _to_naive_utc(value: object) -> pd.Timestamp:
    return _as_utc_timestamp(value).tz_localize(None)


def _axis_signature(values: np.ndarray) -> Tuple[object, ...]:
    values = np.asarray(values)
    positions = sorted({0, 1, values.size // 2, values.size - 2, values.size - 1})
    samples = tuple(float(values[p]) for p in positions if 0 <= p < values.size)
    return values.size, values.dtype.str, samples


def _nearest_axis_index(values: np.ndarray, target: float) -> int:
    values = np.asarray(values)
    if values.size == 1:
        return 0
    ascending = bool(values[-1] >= values[0])
    ordered = values if ascending else values[::-1]
    if np.any(ordered[1:] < ordered[:-1]):
        return int(np.abs(values - target).argmin())
    insertion = int(np.searchsorted(ordered, target, side="left"))
    candidates = {max(0, min(values.size - 1, insertion - 1)), max(0, min(values.size - 1, insertion))}
    original = [c if ascending else values.size - 1 - c for c in candidates]
    return min(original, key=lambda i: (abs(float(values[i]) - target), i))


@dataclass(slots=True)
class _PointExtractionContext:
    lon: float
    lat: float
    x3034: float = field(init=False)
    y3034: float = field(init=False)
    grid_index_cache: Dict[Tuple[object, ...], Tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.lon, self.lat = float(self.lon), float(self.lat)
        self.x3034, self.y3034 = transform_lonlat_to_epsg3034(self.lon, self.lat)

    def grid_indices(self, x_vals, y_vals, x_dim, y_dim):
        key = (x_dim, y_dim, _axis_signature(x_vals), _axis_signature(y_vals))
        if key not in self.grid_index_cache:
            self.grid_index_cache[key] = (
                _nearest_axis_index(x_vals, self.x3034),
                _nearest_axis_index(y_vals, self.y3034),
            )
        return self.grid_index_cache[key]


def _variable_name(var: str, ds: xr.Dataset) -> str:
    if var not in _SUPPORTED_VARIABLES:
        raise ValueError(f"Unknown weather variable {var!r}. Supported: {sorted(_SUPPORTED_VARIABLES)}")
    return hs.find_variable(var, ds)


def _point_dataarray_to_dataframe(da: xr.DataArray, output_name: str) -> pd.DataFrame:
    if "time" not in da.dims:
        raise KeyError("Selected data array has no time dimension.")
    values = np.asarray(da.values).reshape(-1)
    data: Dict[str, object] = {"time": np.asarray(da["time"].values)}
    for name, coord in da.coords.items():
        if name in {"time", output_name}:
            continue
        a = np.asarray(coord.values)
        if a.ndim == 0:
            data[name] = np.repeat(a.reshape(1), values.size)
        elif coord.dims == ("time",):
            data[name] = a
    data[output_name] = values
    return pd.DataFrame(data, copy=False)


def _extract_from_dataset_with_context(var, ds, context, start, end):
    ds = normalize_time_index(ds)
    name = _variable_name(var, ds)
    x_dim, y_dim = find_xy_dim_names(ds, name)
    x_vals, y_vals = get_xy_axis_values(ds, x_dim, y_dim)
    ix, iy = context.grid_indices(x_vals, y_vals, x_dim, y_dim)
    da = ds[name].isel({x_dim: ix, y_dim: iy}).sel(time=slice(_to_naive_utc(start), _to_naive_utc(end)))
    df = _point_dataarray_to_dataframe(da, var)
    selected_x, selected_y = float(x_vals[ix]), float(y_vals[iy])
    df["input_lon"], df["input_lat"] = context.lon, context.lat
    df["grid_x_epsg3034"], df["grid_y_epsg3034"] = selected_x, selected_y
    lon_coord, lat_coord = ds.coords.get("lon"), ds.coords.get("lat")
    if lon_coord is not None and set(lon_coord.dims) == {y_dim, x_dim}:
        df["grid_lon"] = float(lon_coord.isel({y_dim: iy, x_dim: ix}).values)
    if lat_coord is not None and set(lat_coord.dims) == {y_dim, x_dim}:
        df["grid_lat"] = float(lat_coord.isel({y_dim: iy, x_dim: ix}).values)
    if da.attrs.get("units"):
        df["unit"] = da.attrs["units"]
    if da.attrs.get("long_name"):
        df["variable_description"] = da.attrs["long_name"]
    return df


def extract_from_dataset(var, ds, lon, lat, start, end):
    return _extract_from_dataset_with_context(var, ds, _PointExtractionContext(lon, lat), _as_utc_timestamp(start), _as_utc_timestamp(end))


def extract_values_for_point(
    var: str,
    lon: float,
    lat: float,
    start: object,
    end: object,
    cache_dir: Path | str = CACHE_DIR,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    provider=None,
    verbose: bool = True,
) -> pd.DataFrame:
    start_ts, end_ts = _as_utc_timestamp(start), _as_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must not be earlier than start")
    context = _PointExtractionContext(lon, lat)
    frames: List[pd.DataFrame] = []
    with hs.use_provider(provider) if provider is not None else _nullcontext():
        for year, month in hs.month_range(start_ts, end_ts):
            path = hs.ensure_month_file_for_point(
                var, year, month, Path(cache_dir), lon=context.lon, lat=context.lat,
                start=start_ts, end=end_ts, subset_mode=cache_strategy,
                subset_margin_cells=subset_margin_cells, verbose=verbose,
            )
            with hs.read_month_file(path) as ds:
                frames.append(_extract_from_dataset_with_context(var, ds, context, start_ts, end_ts))
    if not frames:
        return pd.DataFrame(columns=["time", var])
    result = pd.concat(frames, ignore_index=True)
    result["time"] = pd.to_datetime(result["time"])
    result = result.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    mask = (result["time"] >= _to_naive_utc(start_ts)) & (result["time"] <= _to_naive_utc(end_ts))
    return result.loc[mask].reset_index(drop=True)


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def extract_multiple_values_for_point(
    variables: Iterable[str], lon: float, lat: float, start: object, end: object,
    cache_dir: Path | str = CACHE_DIR, cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None, provider=None, verbose: bool = True,
) -> pd.DataFrame:
    unique = list(dict.fromkeys(variables))
    frames = [extract_values_for_point(v, lon, lat, start, end, cache_dir, cache_strategy,
                                       subset_margin_cells, provider, verbose) for v in unique]
    return combine_point_variables(frames)


def _diffuse_required_vars(apply_weather_correction: bool = False) -> List[str]:
    required = ["rsds"]
    if apply_weather_correction:
        required.extend(["clt", "hurs", "uhi"])
    available = hs.provider_capabilities().variables
    return [variable for variable in required if variable in available]


def extract_diffuse_radiation_for_point(
    lon: float, lat: float, start: object, end: object, tz: str = "UTC",
    apply_weather_correction: bool = False, cache_dir: Path | str = CACHE_DIR,
    cache_strategy: Optional[str] = None, subset_margin_cells: Optional[int] = None,
    provider=None, verbose: bool = True, **kwargs,
) -> pd.DataFrame:
    variables = _diffuse_required_vars(apply_weather_correction)
    data = extract_multiple_values_for_point(
        variables, lon, lat, start, end, cache_dir, cache_strategy,
        subset_margin_cells, provider, verbose,
    )
    model = HostradaDiffuse(latitude=lat, longitude=lon, tz=tz)
    return model.estimate(data, apply_weather_correction=apply_weather_correction)

# Historic aliases used by external notebooks/scripts.
extract_values = extract_values_for_point
extract_multiple_variables_for_point = extract_multiple_values_for_point
