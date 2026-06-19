#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_IDA_ICE_Weather.py

Create point-based weather files for IDA ICE from HOSTRADA data.

The public helper ``create_ida_ice_weather_file`` writes a whitespace separated
``.prn`` file for a user-defined period. The default output format follows the
uploaded IDA ICE reference file ``KALMAR.PRN``.

Default output columns
----------------------
The generated file contains a compact hourly climate table with seven columns::

    Hour DryBulb_C RelHum WindDirect WindSpeed DirectNormal DiffuseHorizontal

``Hour`` is a continuous hour counter starting at 0, as in the IDA ICE example
file. ``DirectNormal`` is the direct normal irradiance (DNI) in W/m2, not the
direct horizontal irradiance. It is calculated explicitly from the direct
horizontal component and the solar zenith angle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import CACHE_DIR, extract_multiple_values_for_point

IDA_ICE_DEFAULT_COLUMNS = [
    "Hour",
    "DryBulb_C",
    "RelHum",
    "WindDirect",
    "WindSpeed",
    "DirectNormal",
    "DiffuseHorizontal",
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


def _calculate_direct_normal_irradiance(
    ghi: pd.Series,
    dhi: pd.Series,
    solar_zenith_deg: pd.Series,
    *,
    min_cos_zenith: float = 1.0e-6,
) -> pd.Series:
    """Calculate direct normal irradiance for IDA ICE.

    HOSTRADA provides global horizontal irradiance. ``HostradaDiffuse`` estimates
    diffuse horizontal irradiance. The direct component on the horizontal plane is
    therefore ``GHI - DHI``. IDA ICE, however, expects direct normal irradiance in
    column 6. This function converts the horizontal direct component to the beam
    normal to the sun rays by dividing by ``cos(solar_zenith)``.

    Values are set to zero at night and when the direct horizontal component is
    not positive. This avoids writing direct-horizontal radiation by mistake and
    keeps the generated PRN file compatible with the KALMAR.PRN column layout.
    """
    ghi = ghi.astype(float).clip(lower=0.0)
    dhi = dhi.astype(float).clip(lower=0.0, upper=ghi)
    direct_horizontal = (ghi - dhi).clip(lower=0.0)

    cos_zenith = pd.Series(
        np.cos(np.radians(solar_zenith_deg.astype(float).to_numpy())),
        index=ghi.index,
        dtype=float,
    )

    dni = direct_horizontal / cos_zenith.where(cos_zenith > min_cos_zenith)
    dni = dni.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return dni.clip(lower=0.0)


def _prepare_ida_ice_dataframe(
    data: pd.DataFrame,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    apply_weather_correction: bool,
) -> pd.DataFrame:
    data = _as_utc_index(data)

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        tz=tz,
    )
    radiation = model.estimate(data, apply_weather_correction=apply_weather_correction)

    # Use local civil time as dataframe index for consistency with the selected
    # output time zone. The IDA ICE file itself uses a continuous hour counter.
    local_index = radiation.index.tz_convert(tz)

    ghi = _series_or_default(radiation, "global_radiation", np.nan)
    if ghi.isna().all() and "rsds" in radiation.columns:
        ghi = radiation["rsds"].astype(float)

    dhi = _series_or_default(radiation, "dhi", 0.0).clip(lower=0.0, upper=ghi.clip(lower=0.0))

    if "solar_zenith_deg" not in radiation.columns:
        raise KeyError("Missing 'solar_zenith_deg'; cannot calculate direct normal irradiance.")

    # IDA ICE expects DirectNormal in column 6. Calculate it here explicitly from
    # direct horizontal irradiance and solar zenith, instead of writing the
    # horizontal direct component.
    dni = _calculate_direct_normal_irradiance(
        ghi=ghi,
        dhi=dhi,
        solar_zenith_deg=radiation["solar_zenith_deg"],
    )

    temp = _series_or_default(radiation, "temp_2m", np.nan)
    if temp.isna().all() and "tas" in radiation.columns:
        temp = radiation["tas"].astype(float)

    rel_humidity = _series_or_default(radiation, "rh_2m", 50.0)
    wind_speed = _series_or_default(radiation, "wind_speed_10m", 0.0)
    wind_dir = _series_or_default(radiation, "wind_dir_10m", 0.0)

    out = pd.DataFrame(
        {
            "Hour": np.arange(len(radiation), dtype=int),
            "DryBulb_C": temp.to_numpy(dtype=float),
            "RelHum": rel_humidity.clip(lower=0.0, upper=100.0).to_numpy(dtype=float),
            "WindDirect": wind_dir.mod(360.0).to_numpy(dtype=float),
            "WindSpeed": wind_speed.clip(lower=0.0).to_numpy(dtype=float),
            "DirectNormal": dni.to_numpy(dtype=float),
            "DiffuseHorizontal": dhi.to_numpy(dtype=float),
        },
        index=local_index,
    )

    return out[IDA_ICE_DEFAULT_COLUMNS]


def write_ida_ice_prn(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    include_header: bool = True,
    float_format: str = "%.2f",
) -> Path:
    """Write an IDA ICE ``.prn`` weather table in KALMAR.PRN-compatible format.

    Parameters
    ----------
    df:
        Dataframe created by ``create_ida_ice_weather_dataframe`` or a dataframe
        with the same columns.
    output_file:
        Target file path, usually ending in ``.prn``.
    include_header:
        If True, write the same comment-style header structure as the KALMAR.PRN
        reference file. No extra pandas column header is written as data.
    float_format:
        Numeric formatting for all non-hour columns.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [col for col in IDA_ICE_DEFAULT_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing IDA ICE output columns: {missing}")

    with path.open("w", encoding="utf-8", newline="") as f:
        if include_header:
            f.write("# HOSTRADA weather file for IDA ICE\n")
            f.write("# Columns: Hour DryBulb_C\tRelHum\tWindDirect\tWindSpeed DirectNormal DiffuseHorizontal\n")
            f.write("# Units: h degC % deg m/s W/m2 W/m2\n")

        for row in df[IDA_ICE_DEFAULT_COLUMNS].itertuples(index=False):
            f.write(
                f"{int(row.Hour):5d} "
                f"{float(row.DryBulb_C):7.2f} "
                f"{float(row.RelHum):7.2f} "
                f"{float(row.WindDirect):7.2f} "
                f"{float(row.WindSpeed):7.2f} "
                f"{float(row.DirectNormal):7.2f} "
                f"{float(row.DiffuseHorizontal):7.2f}\n"
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
    include_header: bool = True,
    float_format: str = "%.2f",
) -> Path:
    """Create an IDA ICE weather ``.prn`` file for a point and period.

    The output follows the KALMAR.PRN-style IDA ICE format::

        Hour DryBulb_C RelHum WindDirect WindSpeed DirectNormal DiffuseHorizontal

    The sixth column is direct normal irradiance (DNI) in W/m2. It is calculated
    explicitly from direct horizontal irradiance and solar zenith angle.

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
        Time zone used for the output index and solar-position calculation. The
        IDA ICE output file itself uses a continuous hour counter starting at 0.
    cache_dir:
        Local HOSTRADA cache directory.
    apply_weather_correction:
        If True, download a few additional HOSTRADA variables and apply the
        conservative weather correction already implemented in ``HostradaDiffuse``.
    include_header:
        Write comment lines compatible with the KALMAR.PRN reference structure.
    float_format:
        Kept for API compatibility; output is formatted to two decimals to match
        the KALMAR.PRN-style fixed-width numeric layout.

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
    )
    return write_ida_ice_prn(
        df,
        output_file,
        include_header=include_header,
        float_format=float_format,
    )
