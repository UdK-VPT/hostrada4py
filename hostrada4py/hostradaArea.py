#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hostradaArea.py includes functions which read hourly HOSTRADA values for a large number
of 1 km x 1 km grids, which are defined by a polygon with at least three points (lat/lon).

Input:
- Polygon as a list of (lon, lat) points in EPSG:4326, at least 3 points
- Timestamp as a UTC timestamp, e.g., “2024-01-02T12:00:00”

Output:
- GeoDataFrame containing all 1-km cells that lie entirely within the polygon
- Per cell: HOSTRADA value, cell center in EPSG:3034, geometry
- Optional: Export to GeoJSON and CSV
- Optional: Interactive Leaflet/OpenStreetMap map as HTML

Note:
- The polygon points are specified in WGS84 (Lon/Lat).
- Internally, the polygon is transformed to EPSG:3034 because HOSTRADA is available in this CRS.
- By default, only cells that lie entirely within the polygon are returned.
  If all touching cells are desired instead, `selection_mode=“intersects”` can be set.

Required installations:
  
  pip install folium numpy geopandas pandas xarray branca pyproj shapely

"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple, List
import re
import folium
from folium.features import DivIcon
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from branca.colormap import LinearColormap, linear
from pyproj import Transformer
import shapely
from shapely.geometry import Polygon, box as geometry_box

import hostrada4py.hostrada as hs

CACHE_DIR = Path("hostrada_cache")

def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    """Accepts #RGB, #RGBA, #RRGGBB, and #RRGGBBAA; alpha is ignored."""
    if color is None:
        raise ValueError("Farbe darf nicht None sein.")

    color = str(color).strip()

    # Optional accepts rgba(...)
    rgba_match = re.fullmatch(
        r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)",
        color,
        flags=re.IGNORECASE,
    )
    if rgba_match:
        r, g, b = (int(rgba_match.group(i)) for i in (1, 2, 3))
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError(f"Ungültige RGB-Farbe: {color}")
        return r, g, b

    if color.startswith("#"):
        color = color[1:]

    # #RGB or #RGBA
    if len(color) in (3, 4):
        color = "".join(ch * 2 for ch in color)

    # #RRGGBBAA -> cut Alpha
    if len(color) == 8:
        color = color[:6]

    if len(color) != 6:
        raise ValueError(f"Unvalid hex color: {color}")

    try:
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Unvalid hex color: {color}") from exc


def _relative_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)

    def channel(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r_l, g_l, b_l = channel(r), channel(g), channel(b)
    return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l


def _auto_contrast_text_color(background_color: str) -> str:
    return "#111111" if _relative_luminance(background_color) > 0.45 else "#ffffff"


def _label_location_latlon(row: pd.Series) -> Tuple[float, float]:
    if "grid_lat" in row and "grid_lon" in row and pd.notna(row["grid_lat"]) and pd.notna(row["grid_lon"]):
        return float(row["grid_lat"]), float(row["grid_lon"])

    representative_point = row.geometry.representative_point()
    return float(representative_point.y), float(representative_point.x)


def normalize_xy_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "X" in df.columns:
        rename_map["X"] = "grid_x_epsg3034"
    if "Y" in df.columns:
        rename_map["Y"] = "grid_y_epsg3034"
    if "x" in df.columns:
        rename_map["x"] = "grid_x_epsg3034"
    if "y" in df.columns:
        rename_map["y"] = "grid_y_epsg3034"
    return df.rename(columns=rename_map)


@lru_cache(maxsize=8)
def _get_transformer(source_crs: str, target_crs: str) -> Transformer:
    """Cache CRS transformers because they are reused for every extraction/map."""
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def _find_xy_dimensions(da: xr.DataArray) -> Tuple[str, str]:
    """Return x/y dimension names for a rectilinear HOSTRADA DataArray."""
    spatial_dims = [dim for dim in da.dims if dim.lower() != "time"]
    if len(spatial_dims) != 2:
        raise KeyError(
            f"Expected exactly two spatial dimensions, found {tuple(da.dims)}."
        )

    x_candidates = [dim for dim in spatial_dims if dim.lower() == "x"]
    y_candidates = [dim for dim in spatial_dims if dim.lower() == "y"]
    if len(x_candidates) == 1 and len(y_candidates) == 1:
        return x_candidates[0], y_candidates[0]

    # HOSTRADA variables conventionally use (..., Y, X). Keep the former
    # fallback behaviour for files whose dimensions have non-standard names.
    return spatial_dims[1], spatial_dims[0]


def _axis_values(da: xr.DataArray, dim: str) -> np.ndarray:
    if dim not in da.coords:
        raise KeyError(f"Spatial coordinate '{dim}' is missing.")
    values = np.asarray(da.coords[dim].values)
    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional coordinate for '{dim}', got {values.ndim}D."
        )
    return values


def _make_square_geometries(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cell_size: float = 1000.0,
) -> np.ndarray:
    """Create square geometries vectorially with a compatibility fallback."""
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)
    half = float(cell_size) / 2.0

    vectorized_box = getattr(shapely, "box", None)
    if vectorized_box is not None:
        return np.asarray(
            vectorized_box(
                x_centers - half,
                y_centers - half,
                x_centers + half,
                y_centers + half,
            ),
            dtype=object,
        )

    # Shapely < 2 compatibility. The candidate set has already been reduced to
    # the polygon bounding box, so this fallback remains substantially faster
    # than constructing every cell in the complete Germany grid.
    return np.asarray(
        [
            geometry_box(x - half, y - half, x + half, y + half)
            for x, y in zip(x_centers, y_centers)
        ],
        dtype=object,
    )


def make_square_polygon(x_center: float, y_center: float, cell_size: float = 1000.0) -> Polygon:
    half = cell_size / 2.0
    return Polygon([
        (x_center - half, y_center - half),
        (x_center + half, y_center - half),
        (x_center + half, y_center + half),
        (x_center - half, y_center + half),
        (x_center - half, y_center - half),
    ])


def polygon_lonlat_to_epsg3034(points_lonlat: Sequence[Tuple[float, float]]) -> Polygon:
    if len(points_lonlat) < 3:
        raise ValueError("The polygon has to have a minimum of three vertices.")

    points = np.asarray(points_lonlat, dtype=float)
    transformer = _get_transformer("EPSG:4326", "EPSG:3034")
    x, y = transformer.transform(points[:, 0], points[:, 1])
    poly = Polygon(np.column_stack((x, y)))

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or not poly.is_valid:
        raise ValueError("The input polygon is not valid.")

    return poly


def transform_centers_to_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    """Transform all grid centres in one vectorized pyproj operation."""
    result = df.copy()
    if result.empty:
        result["grid_lon"] = pd.Series(dtype=float)
        result["grid_lat"] = pd.Series(dtype=float)
        return result

    transformer = _get_transformer("EPSG:3034", "EPSG:4326")
    lon, lat = transformer.transform(
        result["grid_x_epsg3034"].to_numpy(dtype=float, copy=False),
        result["grid_y_epsg3034"].to_numpy(dtype=float, copy=False),
    )
    result["grid_lon"] = np.asarray(lon, dtype=float)
    result["grid_lat"] = np.asarray(lat, dtype=float)
    return result


def prepare_static_grid_mask(
    var: str,
    ds: xr.Dataset,
    polygon_lonlat: Sequence[Tuple[float, float]],
    selection_mode: str = "within",
) -> gpd.GeoDataFrame:
    """
    Set up the grid once and determine the cells which belong to the polygon.

    Only coordinate vectors inside the polygon bounding box are expanded into
    grid cells. This avoids converting the complete HOSTRADA raster to a
    DataFrame and avoids row-wise polygon construction.
    """
    var_name = hs.find_variable(var, ds)
    da0 = ds[var_name].isel(time=0)
    x_dim, y_dim = _find_xy_dimensions(da0)
    x_values = _axis_values(da0, x_dim)
    y_values = _axis_values(da0, y_dim)

    polygon_3034 = polygon_lonlat_to_epsg3034(polygon_lonlat)
    minx, miny, maxx, maxy = polygon_3034.bounds

    # The geometry-based .cx selection used previously also admitted cells that
    # touched the bbox. Expanding by half a 1-km cell reproduces that candidate
    # set before the exact topological predicate is applied.
    half_cell = 500.0
    x_candidates = x_values[
        (x_values >= minx - half_cell) & (x_values <= maxx + half_cell)
    ]
    y_candidates = y_values[
        (y_values >= miny - half_cell) & (y_values <= maxy + half_cell)
    ]

    if x_candidates.size == 0 or y_candidates.size == 0:
        return gpd.GeoDataFrame(
            columns=["grid_x_epsg3034", "grid_y_epsg3034", "geometry"],
            geometry="geometry",
            crs="EPSG:3034",
        )

    x_grid, y_grid = np.meshgrid(x_candidates, y_candidates)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    geometries = _make_square_geometries(x_flat, y_flat)

    grid_gdf = gpd.GeoDataFrame(
        {
            "grid_x_epsg3034": x_flat,
            "grid_y_epsg3034": y_flat,
        },
        geometry=gpd.GeoSeries(geometries, crs="EPSG:3034"),
        crs="EPSG:3034",
    )

    if selection_mode == "within":
        mask = grid_gdf.geometry.within(polygon_3034)
    elif selection_mode == "intersects":
        mask = grid_gdf.geometry.intersects(polygon_3034)
    elif selection_mode == "centroid":
        # The square centres are exactly the raster coordinates. Creating point
        # objects is faster than calculating centroids of every square.
        vectorized_points = getattr(shapely, "points", None)
        if vectorized_points is not None:
            centres = gpd.GeoSeries(
                vectorized_points(x_flat, y_flat), crs="EPSG:3034"
            )
        else:
            centres = grid_gdf.geometry.centroid
        mask = centres.within(polygon_3034)
    else:
        raise ValueError("selection_mode muss 'within', 'intersects' oder 'centroid' sein.")

    selected_grid = grid_gdf.loc[mask].copy()
    selected_grid = selected_grid.sort_values(
        ["grid_y_epsg3034", "grid_x_epsg3034"], kind="stable"
    ).reset_index(drop=True)
    return selected_grid


def _coordinate_positions(
    axis_values: np.ndarray,
    selected_values: np.ndarray,
    axis_name: str,
) -> np.ndarray:
    """Map selected coordinate values to positional xarray indices."""
    positions = pd.Index(axis_values).get_indexer(selected_values)
    if np.any(positions < 0):
        missing = selected_values[positions < 0][:5]
        raise KeyError(
            f"Selected {axis_name} coordinates are missing in a monthly file: "
            f"{missing.tolist()}"
        )
    return positions.astype(np.intp, copy=False)


def _extract_selected_matrix(
    da: xr.DataArray,
    selected_grid: gpd.GeoDataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Load only the selected polygon cells as a time-by-cell NumPy matrix."""
    da = da.sel(
        time=slice(start_ts.tz_localize(None), end_ts.tz_localize(None))
    )
    if da.sizes.get("time", 0) == 0:
        return pd.DatetimeIndex([]), np.empty((0, len(selected_grid)), dtype=float)

    x_dim, y_dim = _find_xy_dimensions(da)
    x_values = _axis_values(da, x_dim)
    y_values = _axis_values(da, y_dim)

    selected_x = selected_grid["grid_x_epsg3034"].to_numpy(copy=False)
    selected_y = selected_grid["grid_y_epsg3034"].to_numpy(copy=False)
    x_positions = _coordinate_positions(x_values, selected_x, x_dim)
    y_positions = _coordinate_positions(y_values, selected_y, y_dim)

    cell_dim = "__hostrada_cell"
    selected = da.isel(
        {
            x_dim: xr.DataArray(x_positions, dims=cell_dim),
            y_dim: xr.DataArray(y_positions, dims=cell_dim),
        }
    ).transpose("time", cell_dim)

    values = selected.to_numpy()
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values)

    times = pd.DatetimeIndex(pd.to_datetime(selected["time"].to_numpy()))
    if not times.is_monotonic_increasing:
        order = np.argsort(times.asi8, kind="stable")
        times = times.take(order)
        values = values[order]

    return times, values


def _matrix_to_long_frame(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    selected_grid: gpd.GeoDataFrame,
    value_column: str,
) -> pd.DataFrame:
    n_times, n_cells = values.shape
    if n_times == 0 or n_cells == 0:
        return pd.DataFrame(
            columns=[
                "time",
                "grid_y_epsg3034",
                "grid_x_epsg3034",
                value_column,
                "__hostrada_cell",
            ]
        )

    return pd.DataFrame(
        {
            "time": np.repeat(times.to_numpy(), n_cells),
            "grid_y_epsg3034": np.tile(
                selected_grid["grid_y_epsg3034"].to_numpy(copy=False), n_times
            ),
            "grid_x_epsg3034": np.tile(
                selected_grid["grid_x_epsg3034"].to_numpy(copy=False), n_times
            ),
            value_column: values.reshape(-1),
            "__hostrada_cell": np.tile(np.arange(n_cells, dtype=np.intp), n_times),
        }
    )

def _empty_polygon_values_result(
    value_column: Optional[str],
    selection_mode: str,
    return_geodataframe: bool,
) -> gpd.GeoDataFrame | pd.DataFrame:
    columns = [
        "time",
        "grid_x_epsg3034",
        "grid_y_epsg3034",
        value_column or "value",
        "selection_mode",
        "geometry",
    ]
    if return_geodataframe:
        return gpd.GeoDataFrame(
            columns=columns,
            geometry="geometry",
            crs="EPSG:3034",
        )
    return pd.DataFrame(columns=[col for col in columns if col != "geometry"])


def extract_values_for_polygon(
    var: str,
    polygon_lonlat: Sequence[Tuple[float, float]],
    start_utc: str,
    end_utc: str,
    cache_dir: Path = CACHE_DIR,
    selection_mode: str = "within",
    return_geodataframe: bool = True,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Extract hourly values while loading only cells selected by the polygon."""
    start_ts = pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc, tz="UTC")

    if end_ts < start_ts:
        raise ValueError("'end_utc' must >= 'start_utc'.")

    selected_grid: Optional[gpd.GeoDataFrame] = None
    frames: List[pd.DataFrame] = []
    var_name: Optional[str] = None
    polygon_3034_bbox = polygon_lonlat_to_epsg3034(polygon_lonlat).bounds

    for year, month in hs.month_range(start_ts, end_ts):
        target = hs.ensure_month_file_for_bbox(
            var,
            year,
            month,
            cache_dir,
            bbox_epsg3034=polygon_3034_bbox,
            start=start_ts,
            end=end_ts,
            subset_mode=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )

        print(f"Lese Datei: {target}")
        with hs.read_month_file(target) as ds:
            var_name = hs.find_variable(var, ds)

            if selected_grid is None:
                selected_grid = prepare_static_grid_mask(
                    var=var,
                    ds=ds,
                    polygon_lonlat=polygon_lonlat,
                    selection_mode=selection_mode,
                )
                if selected_grid.empty:
                    return _empty_polygon_values_result(
                        var_name, selection_mode, return_geodataframe
                    )

            times, matrix = _extract_selected_matrix(
                ds[var_name], selected_grid, start_ts, end_ts
            )
            if len(times) == 0:
                continue
            frames.append(
                _matrix_to_long_frame(times, matrix, selected_grid, var_name)
            )

    if not frames or selected_grid is None:
        return _empty_polygon_values_result(
            var_name or var, selection_mode, return_geodataframe
        )

    result = pd.concat(frames, ignore_index=True, copy=False)

    # Monthly HOSTRADA files normally contain disjoint, ordered timestamps. The
    # conditional check preserves the former duplicate-removal semantics without
    # always performing an expensive full sort/drop operation.
    duplicate_keys = ["time", "grid_x_epsg3034", "grid_y_epsg3034"]
    duplicate_mask = result.duplicated(subset=duplicate_keys, keep="first")
    if duplicate_mask.any():
        result = result.loc[~duplicate_mask].copy()

    cell_indices = result["__hostrada_cell"].to_numpy(dtype=np.intp, copy=False)
    transformed_centres = transform_centers_to_lonlat(
        selected_grid[["grid_x_epsg3034", "grid_y_epsg3034"]]
    )
    result["grid_lon"] = transformed_centres["grid_lon"].to_numpy()[cell_indices]
    result["grid_lat"] = transformed_centres["grid_lat"].to_numpy()[cell_indices]

    if return_geodataframe:
        result["geometry"] = selected_grid.geometry.to_numpy()[cell_indices]

    result = result.drop(columns="__hostrada_cell")
    result["selection_mode"] = selection_mode

    # Retain the public column order of the previous implementation.
    ordered_columns = [
        "time",
        "grid_y_epsg3034",
        "grid_x_epsg3034",
        var_name or var,
    ]
    if return_geodataframe:
        ordered_columns.append("geometry")
    ordered_columns.extend(["selection_mode", "grid_lon", "grid_lat"])
    remaining_columns = [
        col for col in result.columns if col not in ordered_columns
    ]
    result = result[ordered_columns + remaining_columns].reset_index(drop=True)

    if return_geodataframe:
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")
    return pd.DataFrame(result)


def _numeric_matrix(values: np.ndarray) -> np.ndarray:
    """Convert an extracted matrix to floating point while retaining NaNs."""
    if np.issubdtype(values.dtype, np.number):
        return values.astype(float, copy=False)
    converted = pd.to_numeric(
        pd.Series(values.reshape(-1)), errors="coerce"
    ).to_numpy(dtype=float)
    return converted.reshape(values.shape)


def _empty_mean_result(
    var: str,
    mean_column: Optional[str],
    include_statistics: bool,
    return_geodataframe: bool,
) -> gpd.GeoDataFrame | pd.DataFrame:
    value_col = mean_column or var
    columns = [
        "time",
        "grid_x_epsg3034",
        "grid_y_epsg3034",
        value_col,
        "time_start",
        "time_end",
        "time_count",
        "selection_mode",
        "geometry",
    ]
    if include_statistics:
        columns[4:4] = [
            f"{var}_min",
            f"{var}_max",
            f"{var}_std",
        ]
    empty = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:3034")
    if return_geodataframe:
        return empty
    return pd.DataFrame(empty.drop(columns="geometry"))


def extract_mean_values_for_polygon(
    var: str,
    polygon_lonlat: Sequence[Tuple[float, float]],
    start_utc: str,
    end_utc: str,
    cache_dir: Path = CACHE_DIR,
    selection_mode: str = "within",
    return_geodataframe: bool = True,
    mean_column: Optional[str] = None,
    include_statistics: bool = True,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Calculates spatially resolved temporal mean values for all 1-km cells inside
    a polygon over a given UTC time period.

    The result contains one row per selected grid cell. By default, the temporal
    mean is written back into the HOSTRADA value column. If a separate column
    name is desired, set mean_column, e.g. mean_column="tas_mean".

    Statistics are accumulated directly from monthly time-by-cell arrays. This
    avoids a potentially very large intermediate GeoDataFrame containing one
    repeated geometry per hour and grid cell.

    Parameters:
    - var: HOSTRADA variable or alias, e.g. "tas"
    - polygon_lonlat: Polygon vertices as (lon, lat) points in EPSG:4326
    - start_utc / end_utc: UTC period boundaries, inclusive
    - cache_dir: Directory for downloaded HOSTRADA files
    - selection_mode: "within", "intersects", or "centroid"
    - return_geodataframe: If True, returns a GeoDataFrame with cell geometry
    - mean_column: Optional output column name for the temporal mean
    - include_statistics: Adds min/max/std and non-null time_count per cell

    Output:
    - One row per 1-km cell
    - Per cell: temporal mean over the period, cell center coordinates,
      selection mode, period start/end, and geometry when return_geodataframe=True
    """
    start_ts = pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc, tz="UTC")
    if end_ts < start_ts:
        raise ValueError("'end_utc' must >= 'start_utc'.")

    polygon_3034_bbox = polygon_lonlat_to_epsg3034(polygon_lonlat).bounds
    selected_grid: Optional[gpd.GeoDataFrame] = None
    var_name: Optional[str] = None

    total_count: Optional[np.ndarray] = None
    running_mean: Optional[np.ndarray] = None
    running_m2: Optional[np.ndarray] = None
    running_min: Optional[np.ndarray] = None
    running_max: Optional[np.ndarray] = None
    period_start: Optional[pd.Timestamp] = None
    period_end: Optional[pd.Timestamp] = None
    source_dtype: Optional[np.dtype] = None

    for year, month in hs.month_range(start_ts, end_ts):
        target = hs.ensure_month_file_for_bbox(
            var,
            year,
            month,
            cache_dir,
            bbox_epsg3034=polygon_3034_bbox,
            start=start_ts,
            end=end_ts,
            subset_mode=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )

        print(f"Lese Datei: {target}")
        with hs.read_month_file(target) as ds:
            var_name = hs.find_variable(var, ds)
            if selected_grid is None:
                selected_grid = prepare_static_grid_mask(
                    var=var,
                    ds=ds,
                    polygon_lonlat=polygon_lonlat,
                    selection_mode=selection_mode,
                )
                if selected_grid.empty:
                    return _empty_mean_result(
                        var, mean_column, include_statistics, return_geodataframe
                    )

                n_cells = len(selected_grid)
                total_count = np.zeros(n_cells, dtype=np.int64)
                running_mean = np.zeros(n_cells, dtype=float)
                running_m2 = np.zeros(n_cells, dtype=float)
                running_min = np.full(n_cells, np.nan, dtype=float)
                running_max = np.full(n_cells, np.nan, dtype=float)

            times, raw_matrix = _extract_selected_matrix(
                ds[var_name], selected_grid, start_ts, end_ts
            )
            if len(times) == 0:
                continue

            current_start = pd.Timestamp(times[0])
            current_end = pd.Timestamp(times[-1])
            period_start = current_start if period_start is None else min(period_start, current_start)
            period_end = current_end if period_end is None else max(period_end, current_end)

            if source_dtype is None:
                source_dtype = np.asarray(raw_matrix).dtype
            matrix = _numeric_matrix(raw_matrix)
            valid = ~np.isnan(matrix)
            month_count = valid.sum(axis=0, dtype=np.int64)
            has_values = month_count > 0
            if not has_values.any():
                continue

            safe_matrix = np.where(valid, matrix, 0.0)
            month_sum = safe_matrix.sum(axis=0, dtype=float)
            month_mean = np.zeros_like(month_sum)
            np.divide(month_sum, month_count, out=month_mean, where=has_values)

            deviations = np.where(valid, matrix - month_mean, 0.0)
            month_m2 = np.square(deviations).sum(axis=0, dtype=float)

            month_min = np.min(np.where(valid, matrix, np.inf), axis=0)
            month_max = np.max(np.where(valid, matrix, -np.inf), axis=0)
            month_min[~has_values] = np.nan
            month_max[~has_values] = np.nan

            old_count = total_count.copy()
            new_count = old_count + month_count
            combine = has_values
            delta = month_mean - running_mean

            running_mean[combine] += (
                delta[combine]
                * month_count[combine]
                / new_count[combine]
            )
            running_m2[combine] += (
                month_m2[combine]
                + np.square(delta[combine])
                * old_count[combine]
                * month_count[combine]
                / new_count[combine]
            )
            total_count = new_count
            running_min = np.fmin(running_min, month_min)
            running_max = np.fmax(running_max, month_max)

    if (
        selected_grid is None
        or total_count is None
        or running_mean is None
        or period_start is None
        or period_end is None
    ):
        return _empty_mean_result(
            var, mean_column, include_statistics, return_geodataframe
        )

    value_col = var_name or var
    output_mean_col = mean_column or value_col
    means = running_mean.copy()
    means[total_count == 0] = np.nan

    result = pd.DataFrame(
        {
            "grid_x_epsg3034": selected_grid["grid_x_epsg3034"].to_numpy(copy=False),
            "grid_y_epsg3034": selected_grid["grid_y_epsg3034"].to_numpy(copy=False),
            output_mean_col: means,
            "time": period_end,
            "time_start": period_start,
            "time_end": period_end,
            "time_count": total_count,
            "selection_mode": selection_mode,
        }
    )

    if include_statistics:
        std = np.full(len(result), np.nan, dtype=float)
        enough_values = total_count > 1
        std[enough_values] = np.sqrt(
            np.maximum(running_m2[enough_values], 0.0)
            / (total_count[enough_values] - 1)
        )
        result[f"{value_col}_min"] = running_min
        result[f"{value_col}_max"] = running_max
        result[f"{value_col}_std"] = std

    # pandas preserves float32 for grouped float32 climate values. Retain that
    # public dtype while using float64 internally for numerically stable running
    # statistics.
    if source_dtype is not None and np.issubdtype(source_dtype, np.floating):
        result[output_mean_col] = result[output_mean_col].astype(source_dtype)
        if include_statistics:
            for statistic in ("min", "max", "std"):
                result[f"{value_col}_{statistic}"] = result[
                    f"{value_col}_{statistic}"
                ].astype(source_dtype)

    result = transform_centers_to_lonlat(result)

    if return_geodataframe:
        result["geometry"] = selected_grid.geometry.to_numpy()
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")

    return pd.DataFrame(result)

def summarize_values_period(gdf_or_df: gpd.GeoDataFrame | pd.DataFrame, var: str) -> pd.DataFrame:
    """Calculate the hourly area statistics without copying geometry columns."""
    if "time" not in gdf_or_df.columns:
        raise KeyError("The 'time' column is missing.")
    if var not in gdf_or_df.columns:
        raise KeyError(f"The '{var}' column is missing.")

    # Selecting only the two required columns avoids copying the often very
    # large and expensive Shapely geometry array into an intermediate frame.
    df = pd.DataFrame(gdf_or_df.loc[:, ["time", var]])
    summary = (
        df.groupby("time", as_index=False, sort=True, observed=True)
        .agg(
            value_mean=(var, "mean"),
            value_min=(var, "min"),
            value_max=(var, "max"),
            cell_count=(var, "count"),
        )
        .reset_index(drop=True)
    )
    return summary


def _prepare_leaflet_frame(
    gdf_or_df: gpd.GeoDataFrame | pd.DataFrame,
    var: str,
    time_utc: Optional[str] = None,
) -> gpd.GeoDataFrame:
    if isinstance(gdf_or_df, gpd.GeoDataFrame):
        gdf = gdf_or_df.copy()
    else:
        if "geometry" not in gdf_or_df.columns:
            raise ValueError("A geometry column is required for the Leaflet map.")
        gdf = gpd.GeoDataFrame(gdf_or_df.copy(), geometry="geometry", crs="EPSG:3034")

    if var not in gdf.columns:
        raise KeyError(f"Value column '{var}' not found. Available: {list(gdf.columns)}")

    if time_utc is not None:
        ts = pd.Timestamp(time_utc, tz="UTC").tz_localize(None)
        gdf = gdf.loc[pd.to_datetime(gdf["time"]) == ts].copy()
    elif "time" in gdf.columns and gdf["time"].nunique() > 1:
        raise ValueError(
            "The GeoDataFrame contains multiple time points. Please set 'time_utc'."
        )

    if gdf.empty:
        raise ValueError("No data available for the Leaflet map.")

    if gdf.crs is None:
        raise ValueError("The GeoDataFrame has no CRS.")

    return gdf.to_crs("EPSG:4326")


def make_leaflet_map_timepoint(
    gdf_or_df,
    var,
    time_utc,
    show_cell_values=False,
    decimals=1,
    fill_opacity=0.5,
    line_opacity=0.4,
    line_weight=1,
    tiles="OpenStreetMap",
    map_zoom_start=None,
    value_label_font_size=10,
    value_label_color="auto",
    tooltip=True,
    save_html=None,
    title=None,
    subtitle=None,
    reverse_colormap=False,
    vmin=None,
    vmax=None,
):
    """
    Creates an interactive Leaflet map with an OpenStreetMap background and a semi-transparent
    temperature display for the HOSTRADA squares.

     Parameters:
     - gdf_or_df: Result from extract_values_for_polygon(..., return_geodataframe=True)
     - var: Name of the value column, e.g., ‘tas’
     - time_utc: Optional UTC timestamp, if multiple timestamps are present in the DataFrame
     - show_cell_values: Displays the value of each cell as text centered within the square
     - decimals: Number of decimal places for value labels and tooltips
     - fill_opacity: Opacity of the areas, default 0.5
     - value_label_color: Text color or ‘auto’ for automatic black/white selection
     - save_html: Optional file path for saving the HTML map
    """

    def _hex_to_rgb(color: str):
        """Accepts #RGB, #RGBA, #RRGGBB, #RRGGBBAA and rgb(...) / rgba(...)."""
        if color is None:
            raise ValueError("The color cannot be “None”.")

        color = str(color).strip()

        rgba_match = re.fullmatch(
            r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)",
            color,
            flags=re.IGNORECASE,
        )
        if rgba_match:
            r, g, b = (int(rgba_match.group(i)) for i in (1, 2, 3))
            if not all(0 <= v <= 255 for v in (r, g, b)):
                raise ValueError(f"Unvalid RGB color: {color}")
            return r, g, b

        if color.startswith("#"):
            color = color[1:]

        if len(color) in (3, 4):
            color = "".join(ch * 2 for ch in color)

        if len(color) == 8:
            color = color[:6]

        if len(color) != 6:
            raise ValueError(f"Unvalid hex color: {color}")

        try:
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError as exc:
            raise ValueError(f"UUnvalid hex color: {color}") from exc

    def _relative_luminance(color: str) -> float:
        r, g, b = _hex_to_rgb(color)

        def channel(v: int) -> float:
            x = v / 255.0
            return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

        r_l, g_l, b_l = channel(r), channel(g), channel(b)
        return 0.2126 * r_l + 0.7152 * g_l + 0.0722 * b_l

    def _auto_contrast_text_color(background_color: str) -> str:
        return "#111111" if _relative_luminance(background_color) > 0.45 else "#ffffff"
    
    def _clip_value(value, lo, hi):
        return max(lo, min(hi, float(value)))

    if gdf_or_df is None or len(gdf_or_df) == 0:
        raise ValueError("gdf_or_df is empty.")

    df = pd.DataFrame(gdf_or_df).copy()

    if "time" not in df.columns:
        raise KeyError("The ‘time’ column is missing.")
    if var not in df.columns:
        raise KeyError(f"The ‘{var}’ column is missing.")
    if "geometry" not in gdf_or_df.columns:
        raise KeyError("The ‘geometry’ column is missing.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    target_time = pd.to_datetime(time_utc, utc=True)

    df = df.loc[df["time"] == target_time].copy()
    if df.empty:
        raise ValueError(f"No data for time_utc={time_utc} found.")

    if isinstance(gdf_or_df, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_or_df.crs)
    else:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3034")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3034")

    gdf_wgs84 = gdf.to_crs(epsg=4326).copy()

    vals = pd.to_numeric(gdf_wgs84[var], errors="coerce")
    valid_vals = vals[np.isfinite(vals)]
    if valid_vals.empty:
        raise ValueError(f"No numerical values in column '{var}' found.")

    data_min = float(valid_vals.min())
    data_max = float(valid_vals.max())

    min_val = data_min if vmin is None else float(vmin)
    max_val = data_max if vmax is None else float(vmax)

    if min_val >= max_val:
        raise ValueError(
            f"Unvalid values for vmin/vmax: vmin={min_val}, vmax={max_val}. "
            "The following must apply: vmin < vmax."
        )

    if reverse_colormap:
        colors = ["#ff0000", "#ff7f00", "#ffff00", "#00ffff", "#0000ff"]
    else:
        colors = ["#0000ff", "#00ffff", "#ffff00", "#ff7f00", "#ff0000"]

    colormap = LinearColormap(
        colors=colors,
        vmin=min_val,
        vmax=max_val,
    )
    colormap.caption = f"{var}"

    if "grid_lon" in gdf_wgs84.columns and "grid_lat" in gdf_wgs84.columns:
        center_lat = float(pd.to_numeric(gdf_wgs84["grid_lat"], errors="coerce").mean())
        center_lon = float(pd.to_numeric(gdf_wgs84["grid_lon"], errors="coerce").mean())
    else:
        centroids = gdf_wgs84.geometry.centroid
        center_lat = float(centroids.y.mean())
        center_lon = float(centroids.x.mean())

    if map_zoom_start is None:
        m = folium.Map(location=[center_lat, center_lon], tiles=tiles, zoomSnap=0.25)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=map_zoom_start, tiles=tiles, zoomSnap=0.25)

    if subtitle is None and time_utc is not None:
        try:
            subtitle = pd.to_datetime(time_utc).strftime("%d.%m.%Y %H:%M UTC")
        except Exception:
            subtitle = f"{time_utc} UTC"

    if title or subtitle:
        header_html = f"""
        <div style="
            position: fixed;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(255, 255, 255, 0.92);
            padding: 10px 18px;
            border: 1px solid #666;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
            max-width: 80vw;
        ">
            <div style="
                font-size: 18px;
                font-weight: 600;
                line-height: 1.2;
                margin-bottom: 2px;
                white-space: normal;
            ">
                {title or ""}
            </div>
            <div style="
                font-size: 13px;
                color: #444;
                line-height: 1.2;
                white-space: normal;
            ">
                {subtitle or ""}
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(header_html))

    def _style_function(feature):
        value = feature["properties"].get(var)
        if value is None or pd.isna(value):
            fill_color = "#808080"
        else:
            fill_color = colormap(_clip_value(value, min_val, max_val))
        return {
            "fillColor": fill_color,
            "color": "#333333",
            "weight": line_weight,
            "opacity": line_opacity,
            "fillOpacity": fill_opacity,
        }

    # Generate a JSON-serializable copy for Folium
    gdf_json = gdf_wgs84.copy()

    if "time" in gdf_json.columns:
        gdf_json["time_str"] = pd.to_datetime(
            gdf_json["time"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M UTC")
        gdf_json = gdf_json.drop(columns=["time"])

    for col in gdf_json.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf_json[col]):
            gdf_json[col] = pd.to_datetime(
                gdf_json[col], utc=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M UTC")

    if tooltip:
        tooltip_fields = [var]
        tooltip_aliases = [f"{var}: "]

        if "time_str" in gdf_json.columns:
            tooltip_fields.append("time_str")
            tooltip_aliases.append("Zeit: ")

        folium.GeoJson(
            gdf_json,
            style_function=_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False,
                labels=True,
            ),
        ).add_to(m)
    else:
        folium.GeoJson(
            gdf_json,
            style_function=_style_function,
        ).add_to(m)

    if show_cell_values:
        for _, row in gdf_wgs84.iterrows():
            if pd.isna(row[var]):
                continue

            value = float(row[var])
            fill_color = colormap(value)

            if isinstance(value_label_color, str) and value_label_color.lower() == "auto":
                label_color = _auto_contrast_text_color(fill_color)
            else:
                label_color = value_label_color

            if "grid_lon" in row and "grid_lat" in row and pd.notna(row["grid_lon"]) and pd.notna(row["grid_lat"]):
                lon = float(row["grid_lon"])
                lat = float(row["grid_lat"])
            else:
                rp = row.geometry.representative_point()
                lon = float(rp.x)
                lat = float(rp.y)

            label_html = f"""
            <div style="
                font-size: {value_label_font_size}px;
                color: {label_color};
                font-weight: 600;
                text-align: center;
                white-space: nowrap;
                text-shadow:
                    -1px -1px 0 rgba(255,255,255,0.35),
                     1px -1px 0 rgba(255,255,255,0.35),
                    -1px  1px 0 rgba(255,255,255,0.35),
                     1px  1px 0 rgba(255,255,255,0.35);
            ">
                {value:.{decimals}f}
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 18),
                    html=label_html,
                ),
            ).add_to(m)

    colormap.add_to(m)

    try:
        bounds = gdf_wgs84.total_bounds  # minx, miny, maxx, maxy
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        pass

    if save_html:
        save_path = Path(save_html)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(save_path)

    return m



def make_leaflet_map_timeperiod(
    gdf_or_df,
    var,
    show_cell_values=False,
    decimals=1,
    fill_opacity=0.5,
    line_opacity=0.4,
    line_weight=1,
    tiles="OpenStreetMap",
    map_zoom_start=None,
    value_label_font_size=10,
    value_label_color="auto",
    tooltip=True,
    save_html=None,
    title=None,
    subtitle=None,
    reverse_colormap=False,
    vmin=None,
    vmax=None,
    value_column: Optional[str] = None,
):
    """
    Creates an interactive Leaflet map for spatially resolved temporal mean values
    calculated by extract_mean_values_for_polygon(...).

    In contrast to make_leaflet_map_timepoint(...), this function does not filter
    by a single timestamp. It expects one row per grid cell, as returned by
    extract_mean_values_for_polygon(..., return_geodataframe=True).

    Parameters:
    - gdf_or_df: Result from extract_mean_values_for_polygon(..., return_geodataframe=True)
    - var: HOSTRADA variable name, e.g. "tas". By default this is also the value column.
    - value_column: Optional explicit mean-value column, e.g. "tas_mean" if
      extract_mean_values_for_polygon(..., mean_column="tas_mean") was used.
    - show_cell_values: Displays the mean value of each cell as centered text
    - decimals: Number of decimal places for value labels and tooltips
    - fill_opacity: Opacity of the areas, default 0.5
    - value_label_color: Text color or "auto" for automatic black/white selection
    - save_html: Optional file path for saving the HTML map
    """

    def _clip_value(value, lo, hi):
        return max(lo, min(hi, float(value)))

    def _format_utc(value, fallback=""):
        if value is None or pd.isna(value):
            return fallback
        try:
            return pd.to_datetime(value, utc=True).strftime("%d.%m.%Y %H:%M UTC")
        except Exception:
            return str(value)

    if gdf_or_df is None or len(gdf_or_df) == 0:
        raise ValueError("gdf_or_df is empty.")

    df = pd.DataFrame(gdf_or_df).copy()
    map_value_col = value_column or var

    if map_value_col not in df.columns:
        possible_mean_col = f"{var}_mean"
        if value_column is None and possible_mean_col in df.columns:
            map_value_col = possible_mean_col
        else:
            raise KeyError(
                f"The value column '{map_value_col}' is missing. "
                f"Available columns: {list(df.columns)}. "
                "Set value_column if a custom mean_column was used."
            )

    if "geometry" not in gdf_or_df.columns:
        raise KeyError("The ‘geometry’ column is missing.")

    if isinstance(gdf_or_df, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=gdf_or_df.crs)
    else:
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3034")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3034")

    gdf_wgs84 = gdf.to_crs(epsg=4326).copy()

    vals = pd.to_numeric(gdf_wgs84[map_value_col], errors="coerce")
    valid_vals = vals[np.isfinite(vals)]
    if valid_vals.empty:
        raise ValueError(f"No numerical values in column '{map_value_col}' found.")

    data_min = float(valid_vals.min())
    data_max = float(valid_vals.max())

    min_val = data_min if vmin is None else float(vmin)
    max_val = data_max if vmax is None else float(vmax)

    if min_val >= max_val:
        raise ValueError(
            f"Unvalid values for vmin/vmax: vmin={min_val}, vmax={max_val}. "
            "The following must apply: vmin < vmax."
        )

    if reverse_colormap:
        colors = ["#ff0000", "#ff7f00", "#ffff00", "#00ffff", "#0000ff"]
    else:
        colors = ["#0000ff", "#00ffff", "#ffff00", "#ff7f00", "#ff0000"]

    colormap = LinearColormap(
        colors=colors,
        vmin=min_val,
        vmax=max_val,
    )
    colormap.caption = f"{map_value_col} period mean"

    if "grid_lon" in gdf_wgs84.columns and "grid_lat" in gdf_wgs84.columns:
        center_lat = float(pd.to_numeric(gdf_wgs84["grid_lat"], errors="coerce").mean())
        center_lon = float(pd.to_numeric(gdf_wgs84["grid_lon"], errors="coerce").mean())
    else:
        centroids = gdf_wgs84.geometry.centroid
        center_lat = float(centroids.y.mean())
        center_lon = float(centroids.x.mean())

    if map_zoom_start is None:
        m = folium.Map(location=[center_lat, center_lon], tiles=tiles, zoomSnap=0.25)
    else:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=map_zoom_start, tiles=tiles, zoomSnap=0.25)

    if subtitle is None:
        if "time_start" in gdf_wgs84.columns and "time_end" in gdf_wgs84.columns:
            period_start = pd.to_datetime(gdf_wgs84["time_start"], utc=True, errors="coerce").min()
            period_end = pd.to_datetime(gdf_wgs84["time_end"], utc=True, errors="coerce").max()
            if pd.notna(period_start) and pd.notna(period_end):
                subtitle = f"{_format_utc(period_start)} – {_format_utc(period_end)}"
        elif "time" in gdf_wgs84.columns:
            period_time = pd.to_datetime(gdf_wgs84["time"], utc=True, errors="coerce").max()
            if pd.notna(period_time):
                subtitle = f"Zeitraummittel bis {_format_utc(period_time)}"

    if title or subtitle:
        header_html = f"""
        <div style="
            position: fixed;
            top: 12px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background: rgba(255, 255, 255, 0.92);
            padding: 10px 18px;
            border: 1px solid #666;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18);
            max-width: 80vw;
        ">
            <div style="
                font-size: 18px;
                font-weight: 600;
                line-height: 1.2;
                margin-bottom: 2px;
                white-space: normal;
            ">
                {title or ""}
            </div>
            <div style="
                font-size: 13px;
                color: #444;
                line-height: 1.2;
                white-space: normal;
            ">
                {subtitle or ""}
            </div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(header_html))

    def _style_function(feature):
        value = feature["properties"].get(map_value_col)
        if value is None or pd.isna(value):
            fill_color = "#808080"
        else:
            fill_color = colormap(_clip_value(value, min_val, max_val))
        return {
            "fillColor": fill_color,
            "color": "#333333",
            "weight": line_weight,
            "opacity": line_opacity,
            "fillOpacity": fill_opacity,
        }

    # Generate a JSON-serializable copy for Folium.
    gdf_json = gdf_wgs84.copy()

    if "time_start" in gdf_json.columns:
        gdf_json["time_start_str"] = pd.to_datetime(
            gdf_json["time_start"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M UTC")
        gdf_json = gdf_json.drop(columns=["time_start"])

    if "time_end" in gdf_json.columns:
        gdf_json["time_end_str"] = pd.to_datetime(
            gdf_json["time_end"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M UTC")
        gdf_json = gdf_json.drop(columns=["time_end"])

    if "time" in gdf_json.columns:
        gdf_json["time_str"] = pd.to_datetime(
            gdf_json["time"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M UTC")
        gdf_json = gdf_json.drop(columns=["time"])

    for col in gdf_json.columns:
        if col == "geometry":
            continue
        if pd.api.types.is_datetime64_any_dtype(gdf_json[col]):
            gdf_json[col] = pd.to_datetime(
                gdf_json[col], utc=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d %H:%M UTC")

    if tooltip:
        tooltip_fields = [map_value_col]
        tooltip_aliases = [f"{map_value_col}: "]

        for field, alias in [
            ("time_start_str", "Start: "),
            ("time_end_str", "Ende: "),
            ("time_count", "Zeitschritte: "),
            (f"{var}_min", f"{var} min: "),
            (f"{var}_max", f"{var} max: "),
            (f"{var}_std", f"{var} std: "),
        ]:
            if field in gdf_json.columns:
                tooltip_fields.append(field)
                tooltip_aliases.append(alias)

        folium.GeoJson(
            gdf_json,
            style_function=_style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                sticky=False,
                labels=True,
            ),
        ).add_to(m)
    else:
        folium.GeoJson(
            gdf_json,
            style_function=_style_function,
        ).add_to(m)

    if show_cell_values:
        for _, row in gdf_wgs84.iterrows():
            if pd.isna(row[map_value_col]):
                continue

            value = float(row[map_value_col])
            fill_color = colormap(_clip_value(value, min_val, max_val))

            if isinstance(value_label_color, str) and value_label_color.lower() == "auto":
                label_color = _auto_contrast_text_color(fill_color)
            else:
                label_color = value_label_color

            if "grid_lon" in row and "grid_lat" in row and pd.notna(row["grid_lon"]) and pd.notna(row["grid_lat"]):
                lon = float(row["grid_lon"])
                lat = float(row["grid_lat"])
            else:
                rp = row.geometry.representative_point()
                lon = float(rp.x)
                lat = float(rp.y)

            label_html = f"""
            <div style="
                font-size: {value_label_font_size}px;
                color: {label_color};
                font-weight: 600;
                text-align: center;
                white-space: nowrap;
                text-shadow:
                    -1px -1px 0 rgba(255,255,255,0.35),
                     1px -1px 0 rgba(255,255,255,0.35),
                    -1px  1px 0 rgba(255,255,255,0.35),
                     1px  1px 0 rgba(255,255,255,0.35);
            ">
                {value:.{decimals}f}
            </div>
            """

            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 18),
                    html=label_html,
                ),
            ).add_to(m)

    colormap.add_to(m)

    try:
        bounds = gdf_wgs84.total_bounds  # minx, miny, maxx, maxy
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        pass

    if save_html:
        save_path = Path(save_html)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(save_path)

    return m
