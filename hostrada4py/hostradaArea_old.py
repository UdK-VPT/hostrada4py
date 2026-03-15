#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hostradaArea.py includes functions which read hourly HOSTRADA values for a large number of 1 km x 1 km grids, which are defined by a polygon with at least three points (lat/lon).

Eingabe:
- Polygon als Liste von (lon, lat)-Punkten in EPSG:4326, mindestens 3 Punkte
- Zeitpunkt als UTC-Zeitstempel, z. B. "2024-01-02T12:00:00"

Ausgabe:
- GeoDataFrame mit allen 1-km-Zellen, die vollständig innerhalb des Polygons liegen
- je Zelle: NOSTRADA-Wert, Zellzentrum in EPSG:3034, Geometrie
- optional: Export nach GeoJSON und CSV

Hinweis:
- Die Polygonpunkte werden in WGS84 (Lon/Lat) angegeben.
- Intern wird das Polygon nach EPSG:3034 transformiert, weil HOSTRADA in diesem CRS vorliegt.
- Standardmäßig werden nur Zellen zurückgegeben, die vollständig im Polygon liegen.
  Wenn stattdessen alle berührten Zellen gewünscht sind, kann `selection_mode="intersects"` gesetzt werden.

Benötigte Pakete:
    pip install calendar pathlib typing geopandas pandas requests xarray pyproj shapely   
"""

from __future__ import annotations

import calendar
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List
import geopandas as gpd
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Polygon
import hostrada4py.hostrada as hs

CACHE_DIR = Path("hostrada_cache")

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
        raise ValueError("The polygon has to have a mininum of three vertices.")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
    points_3034 = [transformer.transform(lon, lat) for lon, lat in points_lonlat]

    poly = Polygon(points_3034)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or not poly.is_valid:
        raise ValueError("The input polygon ist not valid.")

    return poly

def transform_centers_to_lonlat(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
    coords = df.apply(
        lambda row: transformer.transform(row["grid_x_epsg3034"], row["grid_y_epsg3034"]),
        axis=1
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
        axis=1
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
) -> gpd.GeoDataFrame | pd.DataFrame:
    start_ts = pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc, tz="UTC")

    if end_ts < start_ts:
        raise ValueError("'end_utc' must >= 'start_utc'.")

    selected_grid = None
    frames: List[gpd.GeoDataFrame] = []
    
    for year, month in hs.month_range(start_ts, end_ts):
        filename = hs.hostrada_filename(var, year, month)
        url = hs.hostrada_url(var, year, month)
        target = cache_dir / filename

        print(f"Lade Datei: {url}")
        hs.download_file(url, target)

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
                        return gpd.GeoDataFrame(columns=[
                            "time", "grid_x_epsg3034", "grid_y_epsg3034",
                            var_name, "selection_mode", "geometry"
                        ], geometry="geometry", crs="EPSG:3034")
                    return pd.DataFrame()

            da = ds[var_name].sel(
                time=slice(start_ts.tz_localize(None), end_ts.tz_localize(None))
            )

            values_df = da.to_dataframe(name=var_name).reset_index()
            values_df = normalize_xy_columns(values_df)

            merged = values_df.merge(
                selected_grid[["grid_x_epsg3034", "grid_y_epsg3034", "geometry"]],
                on=["grid_x_epsg3034", "grid_y_epsg3034"],
                how="inner"
            )

            month_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:3034")
            month_gdf["selection_mode"] = selection_mode
            frames.append(month_gdf)

    if not frames:
        if return_geodataframe:
            return gpd.GeoDataFrame(columns=[
                "time", "grid_x_epsg3034", "grid_y_epsg3034",
                var_name, "selection_mode", "geometry"
            ], geometry="geometry", crs="EPSG:3034")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        subset=["time", "grid_x_epsg3034", "grid_y_epsg3034"]
    ).sort_values(["time", "grid_y_epsg3034", "grid_x_epsg3034"]).reset_index(drop=True)

    result = transform_centers_to_lonlat(result)

    if return_geodataframe:
        return gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:3034")

    return pd.DataFrame(result.drop(columns="geometry"))


def summarize_values_period(gdf_or_df: gpd.GeoDataFrame | pd.DataFrame, var = str) -> pd.DataFrame:
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