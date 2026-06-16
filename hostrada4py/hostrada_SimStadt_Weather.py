#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_SimStadt_Weather.py

Create point-based SimStadt weather files from HOSTRADA data.

SimStadt can use local Meteonorm/TMY3 weather files. For the standard
SimStadt weather processor the relevant columns are the solar irradiances
(GHI, DNI, DHI) and the dry-bulb ambient temperature; the remaining TMY3
columns are kept in place so that the required column ids are not shifted.

The implementation reuses the EnergyPlus HOSTRADA preparation pipeline and
therefore downloads only the minimum HOSTRADA variables required for the
selected options.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from hostrada4py.hostradaPoint import CACHE_DIR
from hostrada4py.hostrada_EnergyPlus_Weather import (
    EPW_MISSING,
    _cloud_cover_tenths,
    _dew_point_from_t_rh,
    _format_number,
    _parse_utc_timestamp,
    _time_zone_offset_hours,
    create_energyplus_weather_dataframe,
)

TMY3_DATA_COLUMNS = [
    "Date (MM/DD/YYYY)",
    "Time (HH:MM)",
    "ETR (W/m^2)",
    "ETRN (W/m^2)",
    "GHI (W/m^2)",
    "GHI source",
    "GHI uncert (%)",
    "DNI (W/m^2)",
    "DNI source",
    "DNI uncert (%)",
    "DHI (W/m^2)",
    "DHI source",
    "DHI uncert (%)",
    "GH illum (lx)",
    "GH illum source",
    "Global illum uncert (%)",
    "DN illum (lx)",
    "DN illum source",
    "DN illum uncert (%)",
    "DH illum (lx)",
    "DH illum source",
    "DH illum uncert (%)",
    "Zenith lum (cd/m^2)",
    "Zenith lum source",
    "Zenith lum uncert (%)",
    "TotCld (tenths)",
    "TotCld source",
    "TotCld uncert (code)",
    "OpqCld (tenths)",
    "OpqCld source",
    "OpqCld uncert (code)",
    "Dry-bulb (C)",
    "Dry-bulb source",
    "Dry-bulb uncert (code)",
    "Dew-point (C)",
    "Dew-point source",
    "Dew-point uncert (code)",
    "RHum (%)",
    "RHum source",
    "RHum uncert (code)",
    "Pressure (mbar)",
    "Pressure source",
    "Pressure uncert (code)",
    "Wdir (degrees)",
    "Wdir source",
    "Wdir uncert (code)",
    "Wspd (m/s)",
    "Wspd source",
    "Wspd uncert (code)",
    "Hvis (m)",
    "Hvis source",
    "Hvis uncert (code)",
    "CeilHgt (m)",
    "CeilHgt source",
    "CeilHgt uncert (code)",
    "Pwat (cm)",
    "Pwat source",
    "Pwat uncert (code)",
    "AOD (unitless)",
    "AOD source",
    "AOD uncert (code)",
    "Alb (unitless)",
    "Alb source",
    "Alb uncert (code)",
    "Lprecip depth (mm)",
    "Lprecip quantity (hr)",
    "Lprecip source",
    "Lprecip uncert (code)",
    "PresWth (METAR code)",
    "PresWth source",
    "PresWth uncert (code)",
]

# Meteonorm/TMY3 source/uncertainty placeholders. The SimStadt documentation
# explicitly allows non-required columns to be left as-is or set to NaN, as long
# as they are not deleted. Numeric placeholders keep the file readable by tools
# that expect numeric TMY3 cells.
_SOURCE_FLAG = "H"
_UNCERT_FLAG = 0
_NAN_TEXT = "NaN"


def _empty_tmy3_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a TMY3 dataframe with all required columns in the right order."""
    df = pd.DataFrame(index=index)
    for col in TMY3_DATA_COLUMNS:
        df[col] = _NAN_TEXT

    # Stable defaults for numeric fields that are optional for SimStadt but are
    # commonly read by general TMY3 tooling.
    numeric_defaults = {
        "ETR (W/m^2)": 9999,
        "ETRN (W/m^2)": 9999,
        "GH illum (lx)": 999999,
        "DN illum (lx)": 999999,
        "DH illum (lx)": 999999,
        "Zenith lum (cd/m^2)": 9999,
        "TotCld (tenths)": 99,
        "OpqCld (tenths)": 99,
        "Hvis (m)": 7777,
        "CeilHgt (m)": 7777,
        "Pwat (cm)": 99,
        "AOD (unitless)": 0.999,
        "Alb (unitless)": 0.2,
        "Lprecip depth (mm)": 0,
        "Lprecip quantity (hr)": 0,
        "PresWth (METAR code)": 999999999,
    }
    for col, value in numeric_defaults.items():
        df[col] = value

    for col in TMY3_DATA_COLUMNS:
        if col.endswith(" source"):
            df[col] = _SOURCE_FLAG
        elif "uncert" in col:
            df[col] = _UNCERT_FLAG

    return df


def _to_tmy3_hour_and_date(local_index: pd.DatetimeIndex) -> tuple[list[str], list[str]]:
    """Return TMY3 Date and Time fields.

    TMY3 timestamps label the end of the previous hour: midnight at the end of
    a day is written as 24:00 for that same date. The existing EPW preparation
    follows the same convention with Hour = local_hour + 1.
    """
    dates: list[str] = []
    times: list[str] = []
    for ts in local_index:
        hour = int(ts.hour) + 1
        dates.append(f"{int(ts.month):02d}/{int(ts.day):02d}/{int(ts.year):04d}")
        times.append("24:00" if hour == 24 else f"{hour:02d}:00")
    return dates, times


def _prepare_simstadt_tmy3_dataframe(epw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert an EnergyPlus-like dataframe to a SimStadt-compatible TMY3 table."""
    if not isinstance(epw_df.index, pd.DatetimeIndex):
        raise TypeError("epw_df must have a DatetimeIndex.")

    local_index = epw_df.index
    out = _empty_tmy3_frame(local_index)
    dates, times = _to_tmy3_hour_and_date(local_index)
    out["Date (MM/DD/YYYY)"] = dates
    out["Time (HH:MM)"] = times

    out["GHI (W/m^2)"] = epw_df["Global Horizontal Radiation"].astype(float).clip(lower=0.0).round().astype(int)
    out["DNI (W/m^2)"] = epw_df["Direct Normal Radiation"].astype(float).clip(lower=0.0).round().astype(int)
    out["DHI (W/m^2)"] = epw_df["Diffuse Horizontal Radiation"].astype(float).clip(lower=0.0).round().astype(int)
    out["Dry-bulb (C)"] = epw_df["Dry Bulb Temperature"].astype(float).round(1)

    # Helpful optional fields. SimStadt mainly needs the four columns above,
    # but writing these makes the file more useful for inspection and tooling.
    if "Dew Point Temperature" in epw_df.columns:
        out["Dew-point (C)"] = epw_df["Dew Point Temperature"].astype(float).round(1)
    elif {"Dry Bulb Temperature", "Relative Humidity"}.issubset(epw_df.columns):
        out["Dew-point (C)"] = _dew_point_from_t_rh(
            epw_df["Dry Bulb Temperature"].astype(float),
            epw_df["Relative Humidity"].astype(float),
        ).round(1)

    if "Relative Humidity" in epw_df.columns:
        out["RHum (%)"] = epw_df["Relative Humidity"].astype(float).clip(0, 100).round().astype(int)
    if "Atmospheric Station Pressure" in epw_df.columns:
        # EPW uses Pa, TMY3 expects mbar/hPa.
        out["Pressure (mbar)"] = (epw_df["Atmospheric Station Pressure"].astype(float) / 100.0).round().astype(int)
    if "Wind Direction" in epw_df.columns:
        out["Wdir (degrees)"] = epw_df["Wind Direction"].astype(float).mod(360).round().astype(int)
    if "Wind Speed" in epw_df.columns:
        out["Wspd (m/s)"] = epw_df["Wind Speed"].astype(float).clip(lower=0.0).round(1)
    if "Total Sky Cover" in epw_df.columns:
        sky = epw_df["Total Sky Cover"].astype(float)
        valid = sky.where(sky < EPW_MISSING["SkyCover"], np.nan)
        if not valid.isna().all():
            out["TotCld (tenths)"] = _cloud_cover_tenths(valid).fillna(99).round().astype(int)
            out["OpqCld (tenths)"] = out["TotCld (tenths)"]

    return out[TMY3_DATA_COLUMNS]


def _format_tmy3_value(value: object) -> str:
    try:
        if pd.isna(value):
            return _NAN_TEXT
    except TypeError:
        pass
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return _NAN_TEXT
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return str(value)


def write_simstadt_tmy3(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    lon: float,
    lat: float,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    location_name: str = "HOSTRADA",
    state: str = "",
    station_id: str | int = "999999",
    time_zone: float | None = None,
) -> Path:
    """Write a SimStadt-compatible local Meteonorm/TMY3 ``.tmy3`` file.

    The file keeps the complete TMY3 column layout. The SimStadt-relevant
    fields are written in their standard positions: GHI column E, DNI column H,
    DHI column K and dry-bulb temperature column AF.
    """
    missing = [col for col in TMY3_DATA_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing SimStadt/TMY3 output columns: {missing}")

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    if time_zone is None:
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            time_zone = _time_zone_offset_hours(tz, df.index)
        else:
            time_zone = 0.0
    elevation = 0.0 if altitude is None or math.isnan(float(altitude)) else float(altitude)

    header = [
        str(station_id),
        location_name,
        state,
        _format_number(time_zone, 1),
        _format_number(lat, 6),
        _format_number(lon, 6),
        _format_number(elevation, 1),
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(TMY3_DATA_COLUMNS) + "\n")
        for _, row in df.iterrows():
            f.write(",".join(_format_tmy3_value(row[col]) for col in TMY3_DATA_COLUMNS) + "\n")

    return path


def create_simstadt_weather_dataframe(
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
    """Create a SimStadt/TMY3 dataframe for one HOSTRADA grid point.

    Only the minimum HOSTRADA variables needed by the shared EnergyPlus/EPW
    preparation are downloaded. Existing files in ``cache_dir`` are reused.
    """
    start_ts = _parse_utc_timestamp(start)
    end_ts = _parse_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("'end' must be greater than or equal to 'start'.")

    epw_df = create_energyplus_weather_dataframe(
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
    return _prepare_simstadt_tmy3_dataframe(epw_df)


def create_simstadt_weather_file(
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
    state: str = "",
    station_id: str | int = "999999",
    time_zone: float | None = None,
) -> Path:
    """Create a SimStadt local TMY3 weather file for a point and period.

    Parameters
    ----------
    lon, lat:
        Longitude and latitude in WGS84 / EPSG:4326.
    start, end:
        UTC timestamps or date strings. HOSTRADA data are hourly; output rows
        are cut to the requested period.
    output_file:
        Target path, usually ending in ``.tmy3``.
    altitude:
        Optional site altitude in metres.
    tz:
        Local time zone used for the TMY3 time rows.
    cache_dir:
        Local HOSTRADA cache directory.
    apply_weather_correction:
        If True, use the optional weather correction from ``HostradaDiffuse``.
    include_sky_cover:
        If True, download ``clt`` and write total/opaque cloud cover in tenths.
    location_name, state, station_id, time_zone:
        TMY3 metadata written to the first line.
    """
    df = create_simstadt_weather_dataframe(
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
    return write_simstadt_tmy3(
        df,
        output_file,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        location_name=location_name,
        state=state,
        station_id=station_id,
        time_zone=time_zone,
    )


# Aliases for users searching for the exact SimStadt/Meteonorm/TMY3 wording.
create_simstadt_tmy3_dataframe = create_simstadt_weather_dataframe
create_simstadt_tmy3_file = create_simstadt_weather_file
create_meteonorm_tmy3_dataframe = create_simstadt_weather_dataframe
create_meteonorm_tmy3_file = create_simstadt_weather_file
write_tmy3 = write_simstadt_tmy3
