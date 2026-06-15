#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_IDA_ICE_Weather.py

Create point-based weather files for IDA ICE from HOSTRADA data.

The public helper ``create_ida_ice_weather_file`` writes a whitespace separated
``.prn`` file for a user-defined period. It keeps the HOSTRADA download volume
minimal by requesting only variables that are needed for the selected output.

Default output columns
----------------------
The default format is an IDA-ICE-oriented hourly climate table with the columns::

    Month Day Hour DryBulb_C RelHum_pct DirectNormal_W_m2 DiffuseHorizontal_W_m2 WindDir_deg WindSpeed_m_s GlobalHorizontal_W_m2

``Hour`` is written as 1..24, i.e. the common convention for hourly weather
records where hour 1 denotes the interval ending at 01:00. Set
``hour_convention='zero_based'`` if the target workflow expects 0..23.

Notes
-----
IDA ICE installations and import workflows can differ between versions and
regional climate databases. The generated file intentionally contains a compact
numeric table plus short comment headers so it can be inspected and, if needed,
mapped in the IDA ICE climate-file import dialog.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import CACHE_DIR, extract_multiple_values_for_point

IDA_ICE_DEFAULT_COLUMNS = [
    "Month",
    "Day",
    "Hour",
    "DryBulb_C",
    "RelHum_pct",
    "DirectNormal_W_m2",
    "DiffuseHorizontal_W_m2",
    "WindDir_deg",
    "WindSpeed_m_s",
    "GlobalHorizontal_W_m2",
]


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _required_vars(apply_weather_correction: bool) -> list[str]:
    """Return the minimum HOSTRADA variables needed for the IDA ICE file."""
    # rsds, tas and ps are required for the radiation decomposition helper.
    # hurs, sfcWind and sfcWind_direction are written directly to the IDA table.
    required = ["rsds", "tas", "ps", "hurs", "sfcWind", "sfcWind_direction"]

    if apply_weather_correction:
        # Additional variables used by HostradaDiffuse._weather_correction.
        required.extend(["clt", "tdew", "mixr", "uhi", "psl"])

    return _unique_preserve_order(required)


def _as_utc_index(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
    elif time_col in out.columns:
        idx = pd.to_datetime(out[time_col], utc=True)
        out = out.drop(columns=[time_col])
    else:
        raise KeyError("Input data must have a DatetimeIndex or a 'time' column.")

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    out.index = idx
    out = out.sort_index()
    return out


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return df[column].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _prepare_ida_ice_dataframe(
    data: pd.DataFrame,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    apply_weather_correction: bool,
    hour_convention: Literal["one_based", "zero_based"],
) -> pd.DataFrame:
    if hour_convention not in {"one_based", "zero_based"}:
        raise ValueError("hour_convention must be either 'one_based' or 'zero_based'.")

    data = _as_utc_index(data)

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        tz=tz,
    )
    radiation = model.estimate(data, apply_weather_correction=apply_weather_correction)

    # Use local civil time in the output table if a local timezone is requested.
    local_index = radiation.index.tz_convert(tz)

    ghi = _series_or_default(radiation, "global_radiation", np.nan)
    if ghi.isna().all() and "rsds" in radiation.columns:
        ghi = radiation["rsds"].astype(float)

    temp = _series_or_default(radiation, "temp_2m", np.nan)
    if temp.isna().all() and "tas" in radiation.columns:
        temp = radiation["tas"].astype(float)

    rel_humidity = _series_or_default(radiation, "rh_2m", 50.0)
    wind_speed = _series_or_default(radiation, "wind_speed_10m", 0.0)
    wind_dir = _series_or_default(radiation, "wind_dir_10m", 0.0)

    hour = local_index.hour + 1 if hour_convention == "one_based" else local_index.hour

    out = pd.DataFrame(
        {
            "Month": local_index.month,
            "Day": local_index.day,
            "Hour": hour,
            "DryBulb_C": temp.to_numpy(dtype=float),
            "RelHum_pct": rel_humidity.clip(lower=0.0, upper=100.0).to_numpy(dtype=float),
            "DirectNormal_W_m2": radiation["dni"].clip(lower=0.0).to_numpy(dtype=float),
            "DiffuseHorizontal_W_m2": radiation["dhi"].clip(lower=0.0).to_numpy(dtype=float),
            "WindDir_deg": wind_dir.mod(360.0).to_numpy(dtype=float),
            "WindSpeed_m_s": wind_speed.clip(lower=0.0).to_numpy(dtype=float),
            "GlobalHorizontal_W_m2": ghi.clip(lower=0.0).to_numpy(dtype=float),
        },
        index=local_index,
    )

    return out[IDA_ICE_DEFAULT_COLUMNS]


def write_ida_ice_prn(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    include_header: bool = True,
    float_format: str = "%.3f",
) -> Path:
    """Write an IDA-ICE-oriented whitespace-separated ``.prn`` weather table.

    Parameters
    ----------
    df:
        Dataframe created by ``create_ida_ice_weather_dataframe`` or a dataframe
        with the same columns.
    output_file:
        Target file path, usually ending in ``.prn``.
    include_header:
        If True, write short comment lines and a column-name line before the
        numeric data. Set to False for workflows that require numeric-only PRN
        files.
    float_format:
        Numeric formatting passed to ``DataFrame.to_csv``.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [col for col in IDA_ICE_DEFAULT_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing IDA ICE output columns: {missing}")

    with path.open("w", encoding="utf-8", newline="") as f:
        if include_header:
            f.write("# HOSTRADA weather file for IDA ICE\n")
            f.write("# Columns: " + " ".join(IDA_ICE_DEFAULT_COLUMNS) + "\n")
            f.write("# Units: - - h degC % W/m2 W/m2 deg m/s W/m2\n")
        df.to_csv(
            f,
            sep=" ",
            columns=IDA_ICE_DEFAULT_COLUMNS,
            index=False,
            header=include_header,
            float_format=float_format,
        )

    return path


def create_ida_ice_weather_dataframe(
    lon: float,
    lat: float,
    start: str,
    end: str,
    *,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    cache_dir: str | Path = CACHE_DIR,
    apply_weather_correction: bool = False,
    hour_convention: Literal["one_based", "zero_based"] = "one_based",
) -> pd.DataFrame:
    """Create the IDA ICE weather dataframe for one HOSTRADA grid point.

    Only the minimum HOSTRADA variables required for the selected output are
    downloaded. Existing files in ``cache_dir`` are reused.
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts < start_ts:
        raise ValueError("'end' must be greater than or equal to 'start'.")

    data = extract_multiple_values_for_point(
        vars=_required_vars(apply_weather_correction),
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=Path(cache_dir),
    )

    if data.empty:
        raise ValueError("No HOSTRADA data was returned for the requested period.")

    return _prepare_ida_ice_dataframe(
        data=data,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        apply_weather_correction=apply_weather_correction,
        hour_convention=hour_convention,
    )


def create_ida_ice_weather_file(
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
    hour_convention: Literal["one_based", "zero_based"] = "one_based",
    include_header: bool = True,
    float_format: str = "%.3f",
) -> Path:
    """Create an IDA ICE weather ``.prn`` file for a given point and period.

    Parameters
    ----------
    lon, lat:
        Longitude and latitude in WGS84 / EPSG:4326.
    start, end:
        UTC timestamps or date strings. HOSTRADA data are hourly; all requested
        months intersecting the interval are considered, but the output is cut to
        the exact time interval.
    output_file:
        Target path for the generated ``.prn`` file.
    altitude:
        Optional site altitude in metres, used for solar-position calculation.
    tz:
        Time zone used for the output date/hour columns. The data are read in
        UTC and converted to this timezone before writing.
    cache_dir:
        Local HOSTRADA cache directory.
    apply_weather_correction:
        If True, download a few additional HOSTRADA variables and apply the
        conservative weather correction already implemented in ``HostradaDiffuse``.
    hour_convention:
        ``'one_based'`` writes hours 1..24. ``'zero_based'`` writes 0..23.
    include_header:
        Write comment and column lines. Set to False for numeric-only import.
    float_format:
        Numeric formatting for the output file.

    Returns
    -------
    pathlib.Path
        Path to the generated file.
    """
    df = create_ida_ice_weather_dataframe(
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        altitude=altitude,
        tz=tz,
        cache_dir=cache_dir,
        apply_weather_correction=apply_weather_correction,
        hour_convention=hour_convention,
    )
    return write_ida_ice_prn(
        df,
        output_file,
        include_header=include_header,
        float_format=float_format,
    )
