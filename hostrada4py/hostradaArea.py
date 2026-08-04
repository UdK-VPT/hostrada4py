"""Polygon and area evaluation for HOSTRADA and provider-normalised grids.

This module only contains grid selection, statistics and visualisation logic.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Sequence, Tuple
import re

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Polygon, box as geometry_box
from shapely.ops import transform as shapely_transform

from hostrada4py import hostrada as hs

CACHE_DIR = Path("hostrada_cache")
_TO_3034 = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
_TO_4326 = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)


def _as_utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def polygon_lonlat_to_epsg3034(points_lonlat) -> Polygon:
    if hasattr(points_lonlat, "geom_type"):
        poly = points_lonlat
    else:
        points = list(points_lonlat)
        if len(points) < 3:
            raise ValueError("The polygon has to have a minimum of three vertices.")
        poly = Polygon(points)
    poly = shapely_transform(_TO_3034.transform, poly)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        raise ValueError("The input polygon is not valid.")
    return poly


def transform_centers_to_lonlat(data, y=None):
    """Transform grid centres; supports both the legacy DataFrame and arrays."""
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        xcol = "grid_x_epsg3034" if "grid_x_epsg3034" in result else "X"
        ycol = "grid_y_epsg3034" if "grid_y_epsg3034" in result else "Y"
        lon, lat = _TO_4326.transform(result[xcol].to_numpy(), result[ycol].to_numpy())
        result["grid_lon"] = lon
        result["grid_lat"] = lat
        return result
    return _TO_4326.transform(np.asarray(data), np.asarray(y))


def make_square_polygon(x_center: float, y_center: float, cell_size: float = 1000.0) -> Polygon:
    half = float(cell_size) / 2.0
    return geometry_box(x_center-half, y_center-half, x_center+half, y_center+half)


def _xy_dims(da: xr.DataArray) -> tuple[str, str]:
    spatial = [d for d in da.dims if d.lower() != "time"]
    xs = [d for d in spatial if d.lower() == "x"]
    ys = [d for d in spatial if d.lower() == "y"]
    if len(xs) == len(ys) == 1:
        return xs[0], ys[0]
    if len(spatial) != 2:
        raise ValueError(f"Expected two spatial dimensions, got {da.dims}")
    return spatial[1], spatial[0]


def prepare_static_grid_mask(
    var: str | xr.Dataset,
    ds: xr.Dataset | str,
    polygon_lonlat,
    selection_mode: str = "within",
):
    """Return the selected static cell grid.

    The 0.42.0 implementation was used in two positional forms in downstream
    material: ``prepare_static_grid_mask(var, ds, polygon)`` and
    ``prepare_static_grid_mask(ds, var, polygon)``.  The first form returns the
    GeoDataFrame used by the current area extraction path.  The dataset-first
    form retains the historical five-item tuple
    ``(x_dim, y_dim, x_indices, y_indices, geometries)``.
    """
    import geopandas as gpd

    legacy_tuple = isinstance(var, xr.Dataset)
    if legacy_tuple:
        var, ds = ds, var
    if not isinstance(var, str) or not isinstance(ds, xr.Dataset):
        raise TypeError("Expected (variable, dataset, polygon) or (dataset, variable, polygon).")

    var_name = hs.find_variable(var, ds)
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    x_dim, y_dim = _xy_dims(da)
    x_values = np.asarray(ds[x_dim].values, dtype=float)
    y_values = np.asarray(ds[y_dim].values, dtype=float)
    cell_size = float(hs.infer_cell_size(ds, var_name))
    # The dataset-first compatibility form historically accepted a polygon in
    # the dataset CRS (EPSG:3034).  The normal variable-first public API keeps
    # accepting longitude/latitude polygons.
    if legacy_tuple and hasattr(polygon_lonlat, "geom_type"):
        poly = polygon_lonlat
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or not poly.is_valid:
            raise ValueError("The input polygon is not valid.")
    else:
        poly = polygon_lonlat_to_epsg3034(polygon_lonlat)
    minx, miny, maxx, maxy = poly.bounds
    half = cell_size / 2.0
    xs = x_values[(x_values >= minx-half) & (x_values <= maxx+half)]
    ys = y_values[(y_values >= miny-half) & (y_values <= maxy+half)]
    if xs.size == 0 or ys.size == 0:
        if legacy_tuple:
            return x_dim, y_dim, np.array([], dtype=int), np.array([], dtype=int), []
        return gpd.GeoDataFrame(
            columns=["grid_x_epsg3034", "grid_y_epsg3034", "geometry"],
            geometry="geometry", crs="EPSG:3034"
        )
    xx, yy = np.meshgrid(xs, ys)
    x_flat, y_flat = xx.ravel(), yy.ravel()
    geoms = [make_square_polygon(x, y, cell_size) for x, y in zip(x_flat, y_flat)]
    effective_mode = "centroid" if legacy_tuple and selection_mode == "within" else selection_mode
    if effective_mode == "within":
        keep = [g.within(poly) for g in geoms]
    elif effective_mode == "intersects":
        keep = [g.intersects(poly) for g in geoms]
    elif effective_mode in {"centroid", "center", "contains_center"}:
        keep = [poly.covers(g.centroid) for g in geoms]
    else:
        raise ValueError("selection_mode must be 'within', 'intersects' or 'centroid'.")
    keep_array = np.asarray(keep, dtype=bool)
    selected_geometries = [g for g, flag in zip(geoms, keep) if flag]
    if legacy_tuple:
        selected_x = x_flat[keep_array]
        selected_y = y_flat[keep_array]
        x_lookup = {float(value): index for index, value in enumerate(x_values)}
        y_lookup = {float(value): index for index, value in enumerate(y_values)}
        x_indices = np.asarray([x_lookup[float(value)] for value in selected_x], dtype=int)
        y_indices = np.asarray([y_lookup[float(value)] for value in selected_y], dtype=int)
        return x_dim, y_dim, x_indices, y_indices, selected_geometries
    selected = gpd.GeoDataFrame(
        {"grid_x_epsg3034": x_flat[keep_array],
         "grid_y_epsg3034": y_flat[keep_array]},
        geometry=selected_geometries, crs="EPSG:3034"
    )
    return selected.sort_values(["grid_y_epsg3034", "grid_x_epsg3034"]).reset_index(drop=True)


def _resolve_legacy_args(
    polygon_lonlat=None, start_utc=None, end_utc=None, selection_mode="within",
    return_geodataframe=True, *, polygon_points=None, start=None, end=None,
    predicate=None, as_geodataframe=None,
):
    polygon = polygon_lonlat if polygon_lonlat is not None else polygon_points
    start_value = start_utc if start_utc is not None else start
    end_value = end_utc if end_utc is not None else end
    mode = predicate if predicate is not None else selection_mode
    return_gdf = return_geodataframe if as_geodataframe is None else as_geodataframe
    if polygon is None or start_value is None or end_value is None:
        raise TypeError("polygon_lonlat, start_utc and end_utc are required")
    return polygon, start_value, end_value, mode, return_gdf


def extract_values_for_polygon(
    var: str,
    polygon_lonlat=None,
    start_utc=None,
    end_utc=None,
    cache_dir: Path | str = CACHE_DIR,
    selection_mode: str = "within",
    return_geodataframe: bool = True,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    provider=None,
    *,
    polygon_points=None,
    start=None,
    end=None,
    predicate=None,
    as_geodataframe=None,
    verbose: bool = True,
):
    """Extract hourly values for all selected cells.

    The first arguments and column order are compatible with version 0.42.0.
    The keyword-only aliases make the function usable by newer provider code.
    """
    import geopandas as gpd

    polygon, start_value, end_value, mode, return_gdf = _resolve_legacy_args(
        polygon_lonlat, start_utc, end_utc, selection_mode, return_geodataframe,
        polygon_points=polygon_points, start=start, end=end,
        predicate=predicate, as_geodataframe=as_geodataframe,
    )
    start_ts, end_ts = _as_utc(start_value), _as_utc(end_value)
    if end_ts < start_ts:
        raise ValueError("'end_utc' must >= 'start_utc'.")
    poly_bbox = polygon_lonlat_to_epsg3034(polygon).bounds
    frames = []
    selected_grid = None
    value_name = var
    context = hs.use_provider(provider) if provider is not None else nullcontext()
    with context:
        for year, month in hs.month_range(start_ts, end_ts):
            target = hs.ensure_month_file_for_bbox(
                var, year, month, Path(cache_dir), bbox_epsg3034=poly_bbox,
                start=start_ts, end=end_ts, subset_mode=cache_strategy,
                subset_margin_cells=subset_margin_cells, verbose=verbose,
            )
            if verbose:
                print(f"Lese Datei: {target}")
            with hs.read_month_file(target) as ds:
                value_name = hs.find_variable(var, ds)
                if selected_grid is None:
                    selected_grid = prepare_static_grid_mask(var, ds, polygon, mode)
                    if selected_grid.empty:
                        break
                da = ds[value_name]
                x_dim, y_dim = _xy_dims(da)
                x_values = np.asarray(ds[x_dim].values)
                y_values = np.asarray(ds[y_dim].values)
                xi = pd.Index(x_values).get_indexer(selected_grid["grid_x_epsg3034"])
                yi = pd.Index(y_values).get_indexer(selected_grid["grid_y_epsg3034"])
                cell_dim = "__hostrada_cell"
                selected = da.isel({
                    x_dim: xr.DataArray(xi, dims=cell_dim),
                    y_dim: xr.DataArray(yi, dims=cell_dim),
                })
                if "time" in selected.dims:
                    selected = selected.sel(time=slice(start_ts.tz_localize(None), end_ts.tz_localize(None))).transpose("time", cell_dim)
                    times = pd.to_datetime(selected["time"].values)
                    values = np.asarray(selected.values)
                else:
                    times = pd.DatetimeIndex([start_ts.tz_localize(None)])
                    values = np.asarray(selected.values)[None, :]
                if values.size == 0:
                    continue
                ntime, ncell = values.shape
                frames.append(pd.DataFrame({
                    "time": np.repeat(times, ncell),
                    "grid_y_epsg3034": np.tile(selected_grid["grid_y_epsg3034"].to_numpy(), ntime),
                    "grid_x_epsg3034": np.tile(selected_grid["grid_x_epsg3034"].to_numpy(), ntime),
                    value_name: values.reshape(-1),
                    "__hostrada_cell": np.tile(np.arange(ncell), ntime),
                }))
    columns = ["time", "grid_y_epsg3034", "grid_x_epsg3034", value_name,
               "geometry", "selection_mode", "grid_lon", "grid_lat"]
    if not frames or selected_grid is None or selected_grid.empty:
        empty = gpd.GeoDataFrame(columns=columns, geometry="geometry", crs="EPSG:3034")
        return empty if return_gdf else pd.DataFrame(empty.drop(columns="geometry"))
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["time", "grid_x_epsg3034", "grid_y_epsg3034"])
    idx = result["__hostrada_cell"].to_numpy(dtype=int)
    lon, lat = _TO_4326.transform(
        selected_grid["grid_x_epsg3034"].to_numpy(),
        selected_grid["grid_y_epsg3034"].to_numpy(),
    )
    result["selection_mode"] = mode
    result["grid_lon"] = np.asarray(lon)[idx]
    result["grid_lat"] = np.asarray(lat)[idx]
    if return_gdf:
        result["geometry"] = selected_grid.geometry.to_numpy()[idx]
    result = result.drop(columns="__hostrada_cell")
    ordered = ["time", "grid_y_epsg3034", "grid_x_epsg3034", value_name]
    if return_gdf:
        ordered.append("geometry")
    ordered += ["selection_mode", "grid_lon", "grid_lat"]
    result = result[ordered].reset_index(drop=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034") if return_gdf else pd.DataFrame(result)


def extract_mean_values_for_polygon(
    var: str,
    polygon_lonlat=None,
    start_utc=None,
    end_utc=None,
    cache_dir: Path | str = CACHE_DIR,
    selection_mode: str = "within",
    return_geodataframe: bool = True,
    mean_column: Optional[str] = None,
    include_statistics: bool = True,
    cache_strategy: Optional[str] = None,
    subset_margin_cells: Optional[int] = None,
    provider=None,
    **aliases,
):
    """Return one row per grid cell with the temporal mean over the period."""
    import geopandas as gpd

    data = extract_values_for_polygon(
        var, polygon_lonlat, start_utc, end_utc, cache_dir, selection_mode,
        True, cache_strategy, subset_margin_cells, provider, **aliases,
    )
    value_col = var if var in data.columns else hs.find_variable(var, xr.Dataset({var: xr.DataArray([])})) if False else var
    if value_col not in data.columns:
        candidates = [c for c in data.columns if c not in {"time","grid_x_epsg3034","grid_y_epsg3034","geometry","selection_mode","grid_lon","grid_lat"}]
        value_col = candidates[0] if candidates else var
    output_col = mean_column or value_col
    if data.empty:
        cols = ["time", "grid_x_epsg3034", "grid_y_epsg3034", output_col,
                "time_start", "time_end", "time_count", "selection_mode", "geometry",
                "grid_lon", "grid_lat"]
        empty = gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:3034")
        return empty if return_geodataframe else pd.DataFrame(empty.drop(columns="geometry"))
    agg = {output_col: (value_col, "mean"), "time_count": (value_col, "count")}
    if include_statistics:
        agg.update({f"{value_col}_min": (value_col, "min"),
                    f"{value_col}_max": (value_col, "max"),
                    f"{value_col}_std": (value_col, "std")})
    result = data.groupby(["grid_x_epsg3034", "grid_y_epsg3034"], as_index=False).agg(**agg)
    period_start = pd.to_datetime(data["time"]).min()
    period_end = pd.to_datetime(data["time"]).max()
    result.insert(0, "time", period_start)
    result["time_start"] = period_start
    result["time_end"] = period_end
    result["selection_mode"] = selection_mode
    static = data.drop_duplicates(["grid_x_epsg3034", "grid_y_epsg3034"])[
        ["grid_x_epsg3034", "grid_y_epsg3034", "grid_lon", "grid_lat", "geometry"]
    ]
    result = result.merge(static, on=["grid_x_epsg3034", "grid_y_epsg3034"], how="left")
    if return_geodataframe:
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")
    return pd.DataFrame(result.drop(columns="geometry"))


def summarize_values_period(gdf_or_df, var: str) -> pd.DataFrame:
    if "time" not in gdf_or_df.columns:
        raise KeyError("The 'time' column is missing.")
    if var not in gdf_or_df.columns:
        candidates = [c for c in gdf_or_df.columns if c not in {"time","grid_x_epsg3034","grid_y_epsg3034","geometry","selection_mode","grid_lon","grid_lat","time_start","time_end","time_count"} and pd.api.types.is_numeric_dtype(gdf_or_df[c])]
        if len(candidates) == 1:
            var = candidates[0]
        else:
            raise KeyError(f"The '{var}' column is missing.")
    return (
        pd.DataFrame(gdf_or_df[["time", var]])
        .groupby("time", as_index=False, sort=True)
        .agg(value_mean=(var, "mean"), value_min=(var, "min"),
             value_max=(var, "max"), cell_count=(var, "count"))
    )


def _prepare_leaflet_frame(gdf_or_df, var: str, time_utc=None):
    import geopandas as gpd
    if isinstance(gdf_or_df, gpd.GeoDataFrame):
        gdf = gdf_or_df.copy()
    else:
        if "geometry" not in gdf_or_df.columns:
            raise ValueError("A geometry column is required for the Leaflet map.")
        gdf = gpd.GeoDataFrame(gdf_or_df.copy(), geometry="geometry", crs="EPSG:3034")
    if time_utc is not None and "time" in gdf:
        target = pd.Timestamp(time_utc)
        if target.tzinfo is not None:
            target = target.tz_convert("UTC").tz_localize(None)
        gdf = gdf[pd.to_datetime(gdf["time"]) == target]
    if gdf.empty:
        raise ValueError("No data available for the Leaflet map.")
    if var not in gdf.columns:
        raise KeyError(f"Value column '{var}' not found. Available: {list(gdf.columns)}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3034")
    return gdf.to_crs("EPSG:4326")


def _make_leaflet_map(
    gdf, value_col, *, show_cell_values=False, decimals=1, fill_opacity=.5,
    line_opacity=.4, line_weight=1, tiles="OpenStreetMap", map_zoom_start=None,
    value_label_font_size=10, value_label_color="auto", tooltip=True,
    save_html=None, title=None, subtitle=None, reverse_colormap=False,
    vmin=None, vmax=None,
):
    import folium
    from branca.colormap import LinearColormap
    from folium.features import DivIcon

    values = pd.to_numeric(gdf[value_col], errors="coerce")
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if vmin is None and len(finite) else float(vmin or 0)
    hi = float(finite.max()) if vmax is None and len(finite) else float(vmax or 1)
    if hi <= lo:
        hi = lo + 1e-9
    colors = ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]
    if reverse_colormap:
        colors.reverse()
    cmap = LinearColormap(colors, vmin=lo, vmax=hi, caption=title or value_col)
    bounds = gdf.total_bounds
    center = [(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2]
    m = folium.Map(location=center, zoom_start=map_zoom_start or 8, tiles=tiles, control_scale=True)
    if title or subtitle:
        heading = f"<h3 style='margin:0'>{title or ''}</h3><div>{subtitle or ''}</div>"
        m.get_root().html.add_child(folium.Element(f"<div style='position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:8px'>{heading}</div>"))
    for _, row in gdf.iterrows():
        value = float(row[value_col]) if pd.notna(row[value_col]) else np.nan
        color = cmap(value) if np.isfinite(value) else "#999999"
        tt = f"{value_col}: {value:.{decimals}f}" if np.isfinite(value) else f"{value_col}: NaN"
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _feature, color=color: {
                "fillColor": color, "color": color, "weight": line_weight,
                "opacity": line_opacity, "fillOpacity": fill_opacity,
            },
            tooltip=tt if tooltip else None,
        ).add_to(m)
        if show_cell_values and np.isfinite(value):
            p = row.geometry.representative_point()
            label_color = "#111111" if value_label_color == "auto" else value_label_color
            folium.Marker(
                [p.y, p.x],
                icon=DivIcon(html=f"<div style='font-size:{value_label_font_size}px;color:{label_color};font-weight:600'>{value:.{decimals}f}</div>"),
            ).add_to(m)
    cmap.add_to(m)
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    if save_html:
        path = Path(save_html); path.parent.mkdir(parents=True, exist_ok=True); m.save(str(path))
    return m


def make_leaflet_map_timepoint(
    gdf_or_df, var, time_utc, show_cell_values=False, decimals=1,
    fill_opacity=.5, line_opacity=.4, line_weight=1, tiles="OpenStreetMap",
    map_zoom_start=None, value_label_font_size=10, value_label_color="auto",
    tooltip=True, save_html=None, title=None, subtitle=None,
    reverse_colormap=False, vmin=None, vmax=None,
):
    gdf = _prepare_leaflet_frame(gdf_or_df, var, time_utc)
    return _make_leaflet_map(gdf, var, show_cell_values=show_cell_values,
        decimals=decimals, fill_opacity=fill_opacity, line_opacity=line_opacity,
        line_weight=line_weight, tiles=tiles, map_zoom_start=map_zoom_start,
        value_label_font_size=value_label_font_size, value_label_color=value_label_color,
        tooltip=tooltip, save_html=save_html, title=title, subtitle=subtitle,
        reverse_colormap=reverse_colormap, vmin=vmin, vmax=vmax)


def make_leaflet_map_timeperiod(
    gdf_or_df, var, show_cell_values=False, decimals=1, fill_opacity=.5,
    line_opacity=.4, line_weight=1, tiles="OpenStreetMap", map_zoom_start=None,
    value_label_font_size=10, value_label_color="auto", tooltip=True,
    save_html=None, title=None, subtitle=None, reverse_colormap=False,
    vmin=None, vmax=None, value_column: Optional[str] = None,
):
    value_col = value_column or (var if var in gdf_or_df.columns else f"{var}_mean")
    gdf = _prepare_leaflet_frame(gdf_or_df, value_col, None)
    return _make_leaflet_map(gdf, value_col, show_cell_values=show_cell_values,
        decimals=decimals, fill_opacity=fill_opacity, line_opacity=line_opacity,
        line_weight=line_weight, tiles=tiles, map_zoom_start=map_zoom_start,
        value_label_font_size=value_label_font_size, value_label_color=value_label_color,
        tooltip=tooltip, save_html=save_html, title=title, subtitle=subtitle,
        reverse_colormap=reverse_colormap, vmin=vmin, vmax=vmax)


def create_folium_map(data, value_column, time=None, zoom_start=6):
    if time is None and "time" in data.columns and data["time"].nunique() > 1:
        time = pd.to_datetime(data["time"]).iloc[0]
    return make_leaflet_map_timepoint(data, value_column, time, map_zoom_start=zoom_start)


extract_area_values = extract_values_for_polygon
extract_mean_area_values = extract_mean_values_for_polygon
