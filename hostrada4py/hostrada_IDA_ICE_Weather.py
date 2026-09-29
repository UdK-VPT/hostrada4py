#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IDA ICE weather export for hostrada4py.

Creates the normal HOSTRADA_IDA_ICE.prn weather file and, in addition,
a matching HOSTRADA_IDA_ICE.idm location-description file.

The .prn file uses the headerless IDA ICE weather format with seven
whitespace-separated numeric columns (as in the IDA ICE example files):

    hour of year, dry-bulb temperature [degC], relative humidity [%],
    wind direction [deg], wind speed [m/s],
    direct normal radiation [W/m2], diffuse horizontal radiation [W/m2]

The hour of year counts continuously from 1 January 00:00 local standard
time (no daylight saving time); the time zone is written to the .idm file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from hostrada4py import hostrada as hs
from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import extract_multiple_values_for_point


def _unique_preserve_order(values):
    return list(dict.fromkeys(values))


def _required_vars(apply_weather_correction=False, provider=None):
    required = [
        "tas",
        "tdew",
        "hurs",
        "ps",
        "sfcWind",
        "sfcWind_direction",
        "rsds",
        "clt",
    ]
    if apply_weather_correction:
        required.append("uhi")

    available = hs.provider_capabilities(provider).variables
    provider_required = [v for v in required if v in available]
    return _unique_preserve_order(provider_required)


def _ensure_target(output_file, default_name):
    path = Path(output_file or default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


PRN_COLUMNS = [
    "tas",
    "hurs",
    "sfcWind_direction",
    "sfcWind",
    "dni",
    "dhi",
]


def _local_standard_time(df, tz, reference_date=None):
    """Return the data timestamps in local standard time (fixed offset, no DST)."""
    if "time" not in df.columns:
        raise KeyError("Weather dataframe contains no 'time' column.")

    times = pd.DatetimeIndex(pd.to_datetime(df["time"]))
    if times.tz is None:
        times = times.tz_localize("UTC")
    offset_hours = -_ida_time_zone_value(tz, reference_date)
    return times.tz_convert(timezone(timedelta(hours=offset_hours)))


def _prn_dataframe(df, tz, start, end):
    """Build the seven IDA ICE .prn columns on a gap-free hourly grid.

    The grid runs from ``start`` 00:00 to ``end`` + 1 day 00:00 in local
    standard time (both included, like the IDA ICE example files). The hour
    column counts from 1 January 00:00 of the start year.
    """
    missing = [c for c in PRN_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Weather dataframe lacks columns required for IDA ICE: {missing}")

    times = _local_standard_time(df, tz, reference_date=start)
    data = df[PRN_COLUMNS].astype(float).set_axis(times)
    data = data[~data.index.duplicated(keep="first")].sort_index()

    start_local = pd.Timestamp(start).normalize().tz_localize(None).tz_localize(times.tz)
    end_local = pd.Timestamp(end).normalize().tz_localize(None).tz_localize(times.tz) + pd.Timedelta(days=1)
    grid = pd.date_range(start_local, end_local, freq="h")

    data = data.reindex(data.index.union(grid))
    direction = data["sfcWind_direction"].ffill().bfill()
    data = data.interpolate(method="time", limit_direction="both")
    data["sfcWind_direction"] = direction
    data = data.reindex(grid)

    data["hurs"] = data["hurs"].clip(0.0, 100.0)
    for col in ("sfcWind", "dni", "dhi"):
        data[col] = data[col].clip(lower=0.0)

    year_start = start_local.replace(month=1, day=1)
    hours = ((grid - year_start) / pd.Timedelta(hours=1)).round().astype(int)
    data.insert(0, "hour", hours)
    return data.reset_index(drop=True)


def _write_prn(data, path):
    """Write the headerless fixed-width IDA ICE .prn file (CRLF line ends)."""
    lines = [
        f"{int(row[0]):5d}" + "".join(f"{value:8.2f}" for value in row[1:])
        for row in data.itertuples(index=False)
    ]
    with path.open("w", encoding="ascii", newline="") as handle:
        handle.write("\r\n".join(lines) + "\r\n")


def _weather_dataframe(
    lon,
    lat,
    start,
    end,
    variables,
    *,
    tz="Europe/Berlin",
    altitude=0,
    apply_weather_correction=False,
    cache_dir="hostrada_cache",
    cache_strategy=None,
    subset_margin_cells=None,
    provider=None,
    verbose=True,
):
    """Load point weather data and add diffuse/direct solar radiation."""
    df = extract_multiple_values_for_point(
        variables=variables,
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
        provider=provider,
        verbose=verbose,
    )

    if df.empty:
        raise ValueError("No weather data was returned for the requested period.")

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=0.0 if altitude is None else float(altitude),
        tz=tz,
    )
    return model.estimate(
        df,
        apply_weather_correction=apply_weather_correction,
    )


def _escape_idm_string(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _format_idm_number(value, decimals=6):
    value = 0.0 if value is None else float(value)
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0", "+0"} else text


def _ida_time_zone_value(tz, reference_date=None):
    """Convert an IANA timezone to the sign convention used by IDA ICE.

    Example: Europe/Berlin has UTC+1 standard time and is written as -1.00.
    """
    zone = ZoneInfo(str(tz))

    if reference_date is None:
        year = datetime.now().year
    else:
        year = pd.Timestamp(reference_date).year

    sample = datetime(year, 1, 15, 12, 0, tzinfo=zone)
    utc_offset = sample.utcoffset() or timedelta(0)
    dst_offset = sample.dst() or timedelta(0)
    standard_offset = utc_offset - dst_offset

    ida_offset = -standard_offset.total_seconds() / 3600.0
    if abs(ida_offset) < 0.0005:
        ida_offset = 0.0
    return ida_offset


def create_ida_ice_location_file(
    lon,
    lat,
    altitude,
    location_name,
    *,
    tz="Europe/Berlin",
    prn_file="HOSTRADA_IDA_ICE.prn",
    output_file="HOSTRADA_IDA_ICE.idm",
    reference_date=None,
):
    """Create the IDA ICE .idm file using the supplied location metadata."""
    location = _escape_idm_string(location_name or "Weather location")
    prn_name = _escape_idm_string(Path(prn_file).name)
    timezone_value = _ida_time_zone_value(tz, reference_date)

    lines = [
        ";IDA 4.0 Document",
        '(DOCUMENT-HEADER :TYPE ICE-DB :D "This document contains descriptions to ICE weather files produced using hostrada4py.',
        'To use these description, save them to the ICE database.")',
        f'((CLIMATE-DEF :N "{location}" :T CLIMATE-DEF :D "Generated by hostrada4py (https://github.com/UdK-VPT/hostrada4py)") ',
        f'(:PAR :N FILENAME :V "{prn_name}")',
        '(:PAR :N COUNTRY :V "-")',
        f'(:PAR :N STATION :V "{location}")',
        f'(:PAR :N LATITUDE :V {_format_idm_number(lat)})',
        f'(:PAR :N LONGITUDE :V {_format_idm_number(lon)})',
        f'(:PAR :N ELEVATION :V {_format_idm_number(altitude)})',
        f'(:PAR :N TIME_ZONE :V {timezone_value:.2f}))',
        "",
    ]

    path = _ensure_target(output_file, "HOSTRADA_IDA_ICE.idm")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\r\n".join(lines))
    return path


def create_ida_ice_weather_file(
    lon,
    lat,
    start,
    end,
    output_file="HOSTRADA_IDA_ICE.prn",
    *,
    tz="Europe/Berlin",
    altitude=0,
    location_name="Weather location",
    idm_file=None,
    apply_weather_correction=False,
    cache_dir="hostrada_cache",
    cache_strategy=None,
    subset_margin_cells=None,
    provider=None,
    verbose=True,
    **kwargs,
):
    """Create the IDA ICE .prn file and matching HOSTRADA_IDA_ICE.idm file."""
    del kwargs  # accepted for backwards compatibility

    # Fetch one extra day on both sides: the requested local-standard-time
    # period starts/ends one hour off the UTC day boundary (e.g. UTC+1).
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    df = _weather_dataframe(
        lon,
        lat,
        fetch_start,
        fetch_end,
        _required_vars(apply_weather_correction, provider=provider),
        tz=tz,
        altitude=altitude,
        apply_weather_correction=apply_weather_correction,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
        provider=provider,
        verbose=verbose,
    )

    prn_path = _ensure_target(output_file, "HOSTRADA_IDA_ICE.prn")
    _write_prn(_prn_dataframe(df, tz, start, end), prn_path)

    if idm_file is None:
        idm_path = prn_path.with_name("HOSTRADA_IDA_ICE.idm")
    else:
        idm_path = Path(idm_file)

    create_ida_ice_location_file(
        lon=lon,
        lat=lat,
        altitude=altitude,
        location_name=location_name,
        tz=tz,
        prn_file=prn_path.name,
        output_file=idm_path,
        reference_date=start,
    )

    return prn_path


create_weather_file = create_ida_ice_weather_file
