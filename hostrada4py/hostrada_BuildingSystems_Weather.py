#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_BuildingSystems_Weather.py

Create point-based BuildingSystems ASCII/CSV weather tables from HOSTRADA data.

BuildingSystems also provides an ASCII weather-data base class
``BuildingSystems.Climate.WeatherData.BaseClasses.WeatherDataFileASCII``. It
reads a numeric table with ``Modelica.Blocks.Tables.CombiTable1Ds`` and maps
seven user-selected table columns to the weather outputs. This module writes a
comma-separated numeric table that can be used with this reader.

The default table layout is chosen so that the Modelica weather block can use::

    final tabNam   = "tab1"
    final timeFac  = 1.0/3600.0
    final deltaTime = 1800.0
    final columns  = {5, 6, 3, 8, 9, 4, 7}
    final scaleFac = {1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 1.0}

The first selected column is global horizontal radiation and the second selected
column is diffuse horizontal radiation. The BuildingSystems ASCII base class
calculates direct horizontal radiation internally as ``global - diffuse``.

The implementation reuses the EnergyPlus preparation code, so the HOSTRADA
request set remains minimal for the selected options. By default only ``rsds``,
``tas``, ``ps``, ``hurs``, ``sfcWind`` and ``sfcWind_direction`` are requested.
Optional sky-cover and weather-correction options request only their additionally
needed variables.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from hostrada4py.hostrada_EnergyPlus_Weather import (
    create_energyplus_weather_dataframe,
    _cloud_cover_tenths,
    _format_number,
    _time_zone_offset_hours,
)
from hostrada4py.hostradaPoint import CACHE_DIR

BUILDINGSYSTEMS_CSV_COLUMNS = [
    "Time_h",
    "Time_s",
    "TAirRef_degC",
    "RelHum_pct",
    "GlobalHorizontalRadiation_W_m2",
    "DiffuseHorizontalRadiation_W_m2",
    "CloudCover_okta",
    "WindSpeed_m_s",
    "WindDirection_deg",
]

# One-based column mapping for BuildingSystems.Climate.WeatherData.BaseClasses.
# WeatherDataFileASCII. The base class computes IrrDirHor = col[1] - col[2] and
# IrrDifHor = col[2].
BUILDINGSYSTEMS_CSV_ASCII_COLUMNS = [5, 6, 3, 8, 9, 4, 7]
BUILDINGSYSTEMS_CSV_ASCII_SCALE_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 1.0]
BUILDINGSYSTEMS_CSV_TIME_FACTOR = 1.0 / 3600.0
BUILDINGSYSTEMS_CSV_DELTA_TIME = 1800.0


def _seconds_from_year_start(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) == 0:
        return np.array([], dtype=int)
    localized = index if index.tz is not None else index.tz_localize("UTC")
    year_start = pd.DatetimeIndex(
        [pd.Timestamp(year=int(year), month=1, day=1, tz=localized.tz) for year in localized.year]
    )
    return ((localized - year_start).total_seconds()).astype(int).to_numpy()


def _format_csv_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "0"
    except TypeError:
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _cloud_cover_okta(epw: pd.DataFrame) -> pd.Series:
    """Return cloud cover in oktas for the BuildingSystems ASCII reader."""
    sky_cover = epw.get("Total Sky Cover")
    if sky_cover is None:
        return pd.Series(0.0, index=epw.index, dtype=float)

    sky_cover = sky_cover.astype(float)
    # EPW missing sky cover marker is 99. The ASCII reader expects 0..8 oktas.
    sky_cover = sky_cover.where(sky_cover < 90.0, 0.0)
    # ``create_energyplus_weather_dataframe`` writes sky cover in tenths. Convert
    # to oktas because BuildingSystems.WeatherDataReader uses cloudCover/8.
    return (sky_cover / 10.0 * 8.0).round().clip(lower=0.0, upper=8.0)


def create_buildingsystems_csv_weather_dataframe(
    lon: float,
    lat: float,
    start: str,
    end: str,
    *,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    cache_dir: str | Path = CACHE_DIR,
    apply_weather_correction: bool = False,
    include_sky_cover: bool = False,
) -> pd.DataFrame:
    """Create a BuildingSystems ASCII/CSV weather dataframe for one point.

    The returned dataframe is numeric and uses the column order expected by the
    helper ``buildingsystems_csv_modelica_block``. ``Time_h`` is seconds since
    local January 1 midnight divided by 3600. The recommended Modelica reader
    settings are ``timeFac=1/3600`` and ``deltaTime=1800`` so that hourly values
    are sampled at the middle of each hour.
    """
    epw = create_energyplus_weather_dataframe(
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        altitude=altitude,
        tz=tz,
        cache_dir=cache_dir,
        apply_weather_correction=apply_weather_correction,
        include_sky_cover=include_sky_cover,
    )

    seconds = _seconds_from_year_start(epw.index)
    out = pd.DataFrame(index=epw.index)
    out["Time_h"] = seconds / 3600.0
    out["Time_s"] = seconds
    out["TAirRef_degC"] = epw["Dry Bulb Temperature"].astype(float)
    out["RelHum_pct"] = epw["Relative Humidity"].astype(float).clip(lower=0.0, upper=100.0)
    out["GlobalHorizontalRadiation_W_m2"] = epw["Global Horizontal Radiation"].astype(float).clip(lower=0.0)
    out["DiffuseHorizontalRadiation_W_m2"] = epw["Diffuse Horizontal Radiation"].astype(float).clip(lower=0.0)
    out["CloudCover_okta"] = _cloud_cover_okta(epw)
    out["WindSpeed_m_s"] = epw["Wind Speed"].astype(float).clip(lower=0.0)
    out["WindDirection_deg"] = epw["Wind Direction"].astype(float) % 360.0
    return out[BUILDINGSYSTEMS_CSV_COLUMNS]


def _metadata_lines(
    df: pd.DataFrame,
    *,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    location_name: str,
    longitude_0: float | None,
    time_zone: float | None,
    table_name: str,
) -> list[str]:
    elevation = 0.0 if altitude is None or math.isnan(float(altitude)) else float(altitude)
    offset = _time_zone_offset_hours(tz, df.index) if time_zone is None else float(time_zone)
    lon0 = offset * 15.0 if longitude_0 is None else float(longitude_0)

    return [
        f"# BuildingSystems ASCII/CSV weather table generated by hostrada4py",
        f"# LOCATION,{location_name},{_format_number(lat, 6)},{_format_number(lon, 6)},{_format_number(lon0, 6)},{_format_number(elevation, 1)}",
        "# Modelica reader: BuildingSystems.Climate.WeatherData.BaseClasses.WeatherDataFileASCII",
        f"# tabNam={table_name}",
        f"# timeFac={BUILDINGSYSTEMS_CSV_TIME_FACTOR:.12g}",
        f"# deltaTime={BUILDINGSYSTEMS_CSV_DELTA_TIME:.12g}",
        "# columns={" + ",".join(str(v) for v in BUILDINGSYSTEMS_CSV_ASCII_COLUMNS) + "}",
        "# scaleFac={" + ",".join(_format_csv_value(v) for v in BUILDINGSYSTEMS_CSV_ASCII_SCALE_FACTORS) + "}",
        "# columnNames=" + ",".join(BUILDINGSYSTEMS_CSV_COLUMNS),
    ]


def write_buildingsystems_csv(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    lon: float,
    lat: float,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    location_name: str = "HOSTRADA",
    longitude_0: float | None = None,
    time_zone: float | None = None,
    table_name: str = "tab1",
    delimiter: str = ",",
    include_modelica_table_header: bool = True,
    include_metadata: bool = True,
    include_column_header: bool = False,
) -> Path:
    """Write a BuildingSystems ASCII/CSV weather table.

    By default the file starts with ``#1 double tab1(n,9)`` and is therefore
    directly readable by Modelica's ``CombiTable1Ds``. The numeric rows are
    comma-separated. Set ``include_column_header=True`` only for inspection or
    post-processing; a textual header line is not suitable for direct Modelica
    table reading.
    """
    missing = [col for col in BUILDINGSYSTEMS_CSV_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing BuildingSystems CSV output columns: {missing}")

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        if include_modelica_table_header:
            f.write(f"#1 double {table_name}({len(df)},{len(BUILDINGSYSTEMS_CSV_COLUMNS)})\n")
        if include_metadata:
            for line in _metadata_lines(
                df=df,
                lon=lon,
                lat=lat,
                altitude=altitude,
                tz=tz,
                location_name=location_name,
                longitude_0=longitude_0,
                time_zone=time_zone,
                table_name=table_name,
            ):
                f.write(line + "\n")
        if include_column_header:
            f.write(delimiter.join(BUILDINGSYSTEMS_CSV_COLUMNS) + "\n")
        for _, row in df[BUILDINGSYSTEMS_CSV_COLUMNS].iterrows():
            f.write(delimiter.join(_format_csv_value(row[col]) for col in BUILDINGSYSTEMS_CSV_COLUMNS) + "\n")

    return path


def buildingsystems_csv_modelica_block(
    block_name: str,
    file_name: str | Path,
    *,
    lon: float,
    lat: float,
    time_zone: float = 1.0,
    longitude_0: float | None = None,
    info: str = "Generated by hostrada4py from HOSTRADA data",
    table_name: str = "tab1",
) -> str:
    """Return a small Modelica block for the generated CSV weather table.

    The returned block extends ``WeatherDataFileASCII`` and uses the column
    mapping matching ``BUILDINGSYSTEMS_CSV_COLUMNS``.
    """
    lon0 = time_zone * 15.0 if longitude_0 is None else float(longitude_0)
    file_name = str(file_name).replace('\\', '/')
    return f'''block {block_name}
  "HOSTRADA weather data for BuildingSystems ASCII/CSV reader"
  extends BuildingSystems.Climate.WeatherData.BaseClasses.WeatherDataFileASCII(
    info="{info}",
    filNam=Modelica.Utilities.Files.loadResource("{file_name}"),
    final tabNam="{table_name}",
    final timeFac={BUILDINGSYSTEMS_CSV_TIME_FACTOR:.12g},
    final deltaTime={BUILDINGSYSTEMS_CSV_DELTA_TIME:.12g},
    final columns={{{','.join(str(v) for v in BUILDINGSYSTEMS_CSV_ASCII_COLUMNS)}}},
    final scaleFac={{{','.join(_format_csv_value(v) for v in BUILDINGSYSTEMS_CSV_ASCII_SCALE_FACTORS)}}},
    final latitudeDeg={float(lat):.8g},
    final longitudeDeg={float(lon):.8g},
    final longitudeDeg_0={lon0:.8g});
end {block_name};'''


def write_buildingsystems_csv_modelica_block(
    output_file: str | Path,
    block_name: str,
    weather_file_name: str | Path,
    *,
    lon: float,
    lat: float,
    time_zone: float = 1.0,
    longitude_0: float | None = None,
    info: str = "Generated by hostrada4py from HOSTRADA data",
    table_name: str = "tab1",
) -> Path:
    """Write a companion Modelica block for the generated CSV weather file."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        buildingsystems_csv_modelica_block(
            block_name=block_name,
            file_name=weather_file_name,
            lon=lon,
            lat=lat,
            time_zone=time_zone,
            longitude_0=longitude_0,
            info=info,
            table_name=table_name,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def create_buildingsystems_csv_weather_file(
    lon: float,
    lat: float,
    start: str,
    end: str,
    output_file: str | Path,
    *,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    cache_dir: str | Path = CACHE_DIR,
    apply_weather_correction: bool = False,
    include_sky_cover: bool = False,
    location_name: str = "HOSTRADA",
    longitude_0: float | None = None,
    time_zone: float | None = None,
    table_name: str = "tab1",
    delimiter: str = ",",
    include_modelica_table_header: bool = True,
    include_metadata: bool = True,
    include_column_header: bool = False,
    modelica_block_file: str | Path | None = None,
    modelica_block_name: str = "Climate_HOSTRADA",
) -> Path:
    """Create a BuildingSystems ASCII/CSV weather file.

    The generated numeric CSV table can be read by
    ``BuildingSystems.Climate.WeatherData.BaseClasses.WeatherDataFileASCII``. If
    ``modelica_block_file`` is provided, a companion ``.mo`` block with the
    correct column mapping, scaling factors and location parameters is written as
    well.
    """
    df = create_buildingsystems_csv_weather_dataframe(
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        altitude=altitude,
        tz=tz,
        cache_dir=cache_dir,
        apply_weather_correction=apply_weather_correction,
        include_sky_cover=include_sky_cover,
    )

    path = write_buildingsystems_csv(
        df,
        output_file,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        location_name=location_name,
        longitude_0=longitude_0,
        time_zone=time_zone,
        table_name=table_name,
        delimiter=delimiter,
        include_modelica_table_header=include_modelica_table_header,
        include_metadata=include_metadata,
        include_column_header=include_column_header,
    )

    if modelica_block_file is not None:
        offset = _time_zone_offset_hours(tz, df.index) if time_zone is None else float(time_zone)
        write_buildingsystems_csv_modelica_block(
            output_file=modelica_block_file,
            block_name=modelica_block_name,
            weather_file_name=path.name,
            lon=lon,
            lat=lat,
            time_zone=offset,
            longitude_0=longitude_0,
            table_name=table_name,
        )

    return path


# Short aliases for users who search for ASCII or CSV wording directly.
create_buildingsystems_ascii_weather_dataframe = create_buildingsystems_csv_weather_dataframe
create_buildingsystems_ascii_weather_file = create_buildingsystems_csv_weather_file
write_buildingsystems_ascii = write_buildingsystems_csv
create_buildingsystems_csv_dataframe = create_buildingsystems_csv_weather_dataframe
create_buildingsystems_csv_file = create_buildingsystems_csv_weather_file
write_buildingsystems_csv_file = write_buildingsystems_csv
