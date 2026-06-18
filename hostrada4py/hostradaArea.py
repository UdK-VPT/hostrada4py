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
from shapely.geometry import Polygon

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

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    points_3034 = [transformer.transform(lon, lat) for lon, lat in points_lonlat]

    poly = Polygon(points_3034)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or not poly.is_valid:
        raise ValueError("The input polygon is not valid.")

    return poly


def transform_centers_to_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
    coords = df.apply(
        lambda row: transformer.transform(row["grid_x_epsg3034"], row["grid_y_epsg3034"]),
        axis=1,
    )
    df = df.copy()
    df["grid_lon"] = [c[0] for c in coords]
    df["grid_lat"] = [c[1] for c in coords]
    return df


def prepare_static_grid_mask(
    var: str,
    ds: xr.Dataset,
    polygon_lonlat: Sequence[Tuple[float, float]],
    selection_mode: str = "within",
) -> gpd.GeoDataFrame:
    """
    Set up the grid once and estimates the cells which belong to the polygon.
    """
    var_name = hs.find_variable(var, ds)
    da0 = ds[var_name].isel(time=0)

    grid_df = da0.to_dataframe(name=var_name).reset_index()
    grid_df = normalize_xy_columns(grid_df)

    needed = {"grid_x_epsg3034", "grid_y_epsg3034"}
    missing = needed - set(grid_df.columns)
    if missing:
        raise KeyError(f"Fehlende Rasterspalten: {missing}")

    grid_df = grid_df[list(needed)].drop_duplicates().copy()
    grid_df["geometry"] = grid_df.apply(
        lambda row: make_square_polygon(row["grid_x_epsg3034"], row["grid_y_epsg3034"]),
        axis=1,
    )

    grid_gdf = gpd.GeoDataFrame(grid_df, geometry="geometry", crs="EPSG:3034")

    polygon_3034 = polygon_lonlat_to_epsg3034(polygon_lonlat)

    minx, miny, maxx, maxy = polygon_3034.bounds
    grid_gdf = grid_gdf.cx[minx:maxx, miny:maxy].copy()

    if selection_mode == "within":
        mask = grid_gdf.geometry.within(polygon_3034)
    elif selection_mode == "intersects":
        mask = grid_gdf.geometry.intersects(polygon_3034)
    elif selection_mode == "centroid":
        mask = grid_gdf.geometry.centroid.within(polygon_3034)
    else:
        raise ValueError("selection_mode muss 'within', 'intersects' oder 'centroid' sein.")

    selected_grid = grid_gdf.loc[mask].copy()
    return selected_grid


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
    start_ts = pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc, tz="UTC")

    if end_ts < start_ts:
        raise ValueError("'end_utc' must >= 'start_utc'.")

    selected_grid = None
    frames: List[gpd.GeoDataFrame] = []
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
                    if return_geodataframe:
                        return gpd.GeoDataFrame(
                            columns=[
                                "time",
                                "grid_x_epsg3034",
                                "grid_y_epsg3034",
                                var_name,
                                "selection_mode",
                                "geometry",
                            ],
                            geometry="geometry",
                            crs="EPSG:3034",
                        )
                    return pd.DataFrame()

            da = ds[var_name].sel(
                time=slice(start_ts.tz_localize(None), end_ts.tz_localize(None))
            )

            values_df = da.to_dataframe(name=var_name).reset_index()
            values_df = normalize_xy_columns(values_df)

            merged = values_df.merge(
                selected_grid[["grid_x_epsg3034", "grid_y_epsg3034", "geometry"]],
                on=["grid_x_epsg3034", "grid_y_epsg3034"],
                how="inner",
            )

            month_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:3034")
            month_gdf["selection_mode"] = selection_mode
            frames.append(month_gdf)

    if not frames:
        if return_geodataframe:
            return gpd.GeoDataFrame(
                columns=[
                    "time",
                    "grid_x_epsg3034",
                    "grid_y_epsg3034",
                    var_name,
                    "selection_mode",
                    "geometry",
                ],
                geometry="geometry",
                crs="EPSG:3034",
            )
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        subset=["time", "grid_x_epsg3034", "grid_y_epsg3034"]
    ).sort_values(["time", "grid_y_epsg3034", "grid_x_epsg3034"]).reset_index(drop=True)

    result = transform_centers_to_lonlat(result)

    if return_geodataframe:
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")

    return pd.DataFrame(result.drop(columns="geometry"))


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
    values = extract_values_for_polygon(
        var=var,
        polygon_lonlat=polygon_lonlat,
        start_utc=start_utc,
        end_utc=end_utc,
        cache_dir=cache_dir,
        selection_mode=selection_mode,
        return_geodataframe=True,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
    )

    if values.empty:
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

    values = values.copy()

    if var in values.columns:
        value_col = var
    else:
        metadata_cols = {
            "time",
            "grid_x_epsg3034",
            "grid_y_epsg3034",
            "grid_lon",
            "grid_lat",
            "selection_mode",
            "geometry",
        }
        value_candidates = [
            col for col in values.columns
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(values[col])
        ]
        if len(value_candidates) != 1:
            raise KeyError(
                f"Value column for var='{var}' could not be determined. "
                f"Candidates: {value_candidates}"
            )
        value_col = value_candidates[0]

    output_mean_col = mean_column or value_col
    values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
    values["time"] = pd.to_datetime(values["time"], utc=True, errors="coerce").dt.tz_localize(None)

    group_cols = ["grid_x_epsg3034", "grid_y_epsg3034"]
    agg_map = {
        output_mean_col: (value_col, "mean"),
        "time": ("time", "max"),
        "time_start": ("time", "min"),
        "time_end": ("time", "max"),
        "time_count": (value_col, "count"),
        "selection_mode": ("selection_mode", "first"),
    }

    if include_statistics:
        agg_map.update({
            f"{value_col}_min": (value_col, "min"),
            f"{value_col}_max": (value_col, "max"),
            f"{value_col}_std": (value_col, "std"),
        })

    if "grid_lon" in values.columns:
        agg_map["grid_lon"] = ("grid_lon", "first")
    if "grid_lat" in values.columns:
        agg_map["grid_lat"] = ("grid_lat", "first")
    if "geometry" in values.columns:
        agg_map["geometry"] = ("geometry", "first")

    result = (
        values.groupby(group_cols, as_index=False)
        .agg(**agg_map)
        .sort_values(["grid_y_epsg3034", "grid_x_epsg3034"])
        .reset_index(drop=True)
    )

    if return_geodataframe:
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")

    if "geometry" in result.columns:
        result = result.drop(columns="geometry")
    return pd.DataFrame(result)


def summarize_values_period(gdf_or_df: gpd.GeoDataFrame | pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Calculates an hourly time series of area mean values of the polygon.
    """
    df = pd.DataFrame(gdf_or_df).copy()

    summary = (
        df.groupby("time", as_index=False)
        .agg(
            value_mean=(var, "mean"),
            value_min=(var, "min"),
            value_max=(var, "max"),
            cell_count=(var, "count"),
        )
        .sort_values("time")
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
