#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_EnergyPlus_Weather.py

Create point-based EnergyPlus weather files (EPW) from HOSTRADA data.

The public helper ``create_energyplus_weather_file`` writes an EnergyPlus
``.epw`` file for a user-defined point and period. It keeps the HOSTRADA
network traffic minimal by requesting only the variables that are needed for
standard EPW weather fields and for the selected optional refinements.

Default HOSTRADA inputs
----------------------
The default EnergyPlus export downloads only these variables:

    rsds, tas, ps, hurs, sfcWind, sfcWind_direction

``HostradaDiffuse`` derives direct-normal and diffuse-horizontal irradiance
from global-horizontal irradiance. Dew point is calculated locally from dry-bulb
temperature and relative humidity, so the HOSTRADA dew-point file is not
required for the default EPW output.

Optional inputs
---------------
Set ``include_sky_cover=True`` to download ``clt`` and write total/opaque sky
cover instead of EPW missing-value markers. Set ``apply_weather_correction=True``
to use the conservative weather correction implemented in ``HostradaDiffuse``;
this downloads the additional weather variables required by that correction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import CACHE_DIR, extract_multiple_values_for_point

EPW_DATA_COLUMNS = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Data Source and Uncertainty Flags",
    "Dry Bulb Temperature",
    "Dew Point Temperature",
    "Relative Humidity",
    "Atmospheric Station Pressure",
    "Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation",
    "Horizontal Infrared Radiation Intensity",
    "Global Horizontal Radiation",
    "Direct Normal Radiation",
    "Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance",
    "Direct Normal Illuminance",
    "Diffuse Horizontal Illuminance",
    "Zenith Luminance",
    "Wind Direction",
    "Wind Speed",
    "Total Sky Cover",
    "Opaque Sky Cover",
    "Visibility",
    "Ceiling Height",
    "Present Weather Observation",
    "Present Weather Codes",
    "Precipitable Water",
    "Aerosol Optical Depth",
    "Snow Depth",
    "Days Since Last Snowfall",
    "Albedo",
    "Liquid Precipitation Depth",
    "Liquid Precipitation Quantity",
]

EPW_MISSING = {
    "Temperature": 99.9,
    "Humidity": 999.0,
    "Radiation": 9999.0,
    "Illuminance": 999999.0,
    "SkyCover": 99.0,
    "Visibility": 9999.0,
    "CeilingHeight": 99999.0,
    "PrecipitableWater": 999.0,
    "AerosolOpticalDepth": 0.999,
    "SnowDepth": 999.0,
    "DaysSinceSnowfall": 99.0,
    "Albedo": 999.0,
    "LiquidPrecipitationDepth": 999.0,
    "LiquidPrecipitationQuantity": 99.0,
}


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _required_vars(
    apply_weather_correction: bool,
    include_sky_cover: bool,
) -> list[str]:
    """Return the minimum HOSTRADA variables needed for the EPW file."""
    # rsds, tas and ps are needed by HostradaDiffuse. hurs, sfcWind and
    # sfcWind_direction are written directly; dew point is calculated from tas
    # and hurs to avoid downloading tdew in the default path.
    required = ["rsds", "tas", "ps", "hurs", "sfcWind", "sfcWind_direction"]

    if include_sky_cover:
        required.append("clt")

    if apply_weather_correction:
        # Additional variables used by HostradaDiffuse._weather_correction.
        # ``clt`` is included here because the correction uses it when present.
        required.extend(["clt", "tdew", "mixr", "uhi", "psl"])

    return _unique_preserve_order(required)


def _parse_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


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


def _dew_point_from_t_rh(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Approximate dew point in °C using the Magnus formula."""
    temp = temp_c.astype(float)
    rh = rh_pct.astype(float).clip(lower=0.1, upper=100.0)
    a = 17.625
    b = 243.04
    gamma = np.log(rh / 100.0) + (a * temp) / (b + temp)
    return (b * gamma) / (a - gamma)


def _cloud_cover_tenths(series: pd.Series) -> pd.Series:
    cloud = series.astype(float)
    if cloud.max(skipna=True) <= 1.5:
        cloud = cloud * 10.0
    elif cloud.max(skipna=True) > 10.0:
        cloud = cloud / 10.0
    return cloud.round().clip(lower=0.0, upper=10.0)


def _time_zone_offset_hours(tz: str, reference: pd.DatetimeIndex) -> float:
    """Return the EPW LOCATION time-zone offset in hours from GMT.

    EPW weather rows are normally interpreted in local standard time. For the
    common northern-hemisphere DST case, using January 1 avoids accidentally
    writing a daylight-saving offset when the requested period contains only
    summer hours. Users can still override the value with ``time_zone=...``.
    """
    if len(reference) == 0:
        return 0.0
    year = int(reference[0].year)
    jan = pd.DatetimeIndex([pd.Timestamp(year=year, month=1, day=1, hour=12, tz="UTC")])
    local = jan.tz_convert(tz)[0]
    return float(local.utcoffset().total_seconds() / 3600.0)


def _format_number(value: object, decimals: int | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if decimals is None:
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)) and float(value).is_integer():
            return str(int(value))
        return str(value)
    return f"{float(value):.{decimals}f}"


def _format_epw_row(row: pd.Series) -> str:
    values = [
        int(row["Year"]),
        int(row["Month"]),
        int(row["Day"]),
        int(row["Hour"]),
        int(row["Minute"]),
        row["Data Source and Uncertainty Flags"],
        _format_number(row["Dry Bulb Temperature"], 1),
        _format_number(row["Dew Point Temperature"], 1),
        int(round(float(row["Relative Humidity"]))),
        int(round(float(row["Atmospheric Station Pressure"]))),
        int(round(float(row["Extraterrestrial Horizontal Radiation"]))),
        int(round(float(row["Extraterrestrial Direct Normal Radiation"]))),
        int(round(float(row["Horizontal Infrared Radiation Intensity"]))),
        int(round(float(row["Global Horizontal Radiation"]))),
        int(round(float(row["Direct Normal Radiation"]))),
        int(round(float(row["Diffuse Horizontal Radiation"]))),
        int(round(float(row["Global Horizontal Illuminance"]))),
        int(round(float(row["Direct Normal Illuminance"]))),
        int(round(float(row["Diffuse Horizontal Illuminance"]))),
        int(round(float(row["Zenith Luminance"]))),
        int(round(float(row["Wind Direction"]))),
        _format_number(row["Wind Speed"], 1),
        int(round(float(row["Total Sky Cover"]))),
        int(round(float(row["Opaque Sky Cover"]))),
        _format_number(row["Visibility"], 1),
        int(round(float(row["Ceiling Height"]))),
        int(round(float(row["Present Weather Observation"]))),
        row["Present Weather Codes"],
        _format_number(row["Precipitable Water"], 1),
        _format_number(row["Aerosol Optical Depth"], 3),
        int(round(float(row["Snow Depth"]))),
        int(round(float(row["Days Since Last Snowfall"]))),
        _format_number(row["Albedo"], 2),
        _format_number(row["Liquid Precipitation Depth"], 1),
        int(round(float(row["Liquid Precipitation Quantity"]))),
    ]
    return ",".join(str(v) for v in values)


def _prepare_energyplus_dataframe(
    data: pd.DataFrame,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    apply_weather_correction: bool,
    include_sky_cover: bool,
) -> pd.DataFrame:
    data = _as_utc_index(data)

    model = HostradaDiffuse(
        latitude=lat,
        longitude=lon,
        altitude=altitude,
        tz=tz,
    )
    radiation = model.estimate(data, apply_weather_correction=apply_weather_correction)
    local_index = radiation.index.tz_convert(tz)

    ghi = _series_or_default(radiation, "global_radiation", np.nan)
    if ghi.isna().all() and "rsds" in radiation.columns:
        ghi = radiation["rsds"].astype(float)

    temp = _series_or_default(radiation, "temp_2m", np.nan)
    if temp.isna().all() and "tas" in radiation.columns:
        temp = radiation["tas"].astype(float)

    rh = _series_or_default(radiation, "rh_2m", 50.0).clip(lower=0.0, upper=100.0)
    dew = _series_or_default(radiation, "dewpoint_2m", np.nan)
    if dew.isna().all():
        dew = _dew_point_from_t_rh(temp, rh)

    pressure = _series_or_default(radiation, "pressure_station", np.nan)
    if pressure.isna().all() and "ps" in radiation.columns:
        pressure = radiation["ps"].astype(float)
    # HOSTRADA pressure is hPa; EPW expects Pa.
    pressure_pa = pressure * 100.0

    wind_speed = _series_or_default(radiation, "wind_speed_10m", 0.0)
    wind_dir = _series_or_default(radiation, "wind_dir_10m", 0.0)

    if include_sky_cover and "cloud_cover" in radiation.columns:
        sky_cover = _cloud_cover_tenths(radiation["cloud_cover"])
    else:
        sky_cover = pd.Series(EPW_MISSING["SkyCover"], index=radiation.index, dtype=float)

    out = pd.DataFrame(
        {
            "Year": local_index.year,
            "Month": local_index.month,
            "Day": local_index.day,
            "Hour": local_index.hour + 1,
            "Minute": 60,
            "Data Source and Uncertainty Flags": "?9?9?9?9?9?9?9?9?9",
            "Dry Bulb Temperature": temp.to_numpy(dtype=float),
            "Dew Point Temperature": dew.to_numpy(dtype=float),
            "Relative Humidity": rh.to_numpy(dtype=float),
            "Atmospheric Station Pressure": pressure_pa.to_numpy(dtype=float),
            "Extraterrestrial Horizontal Radiation": EPW_MISSING["Radiation"],
            "Extraterrestrial Direct Normal Radiation": EPW_MISSING["Radiation"],
            "Horizontal Infrared Radiation Intensity": EPW_MISSING["Radiation"],
            "Global Horizontal Radiation": ghi.clip(lower=0.0).to_numpy(dtype=float),
            "Direct Normal Radiation": radiation["dni"].clip(lower=0.0).to_numpy(dtype=float),
            "Diffuse Horizontal Radiation": radiation["dhi"].clip(lower=0.0).to_numpy(dtype=float),
            "Global Horizontal Illuminance": EPW_MISSING["Illuminance"],
            "Direct Normal Illuminance": EPW_MISSING["Illuminance"],
            "Diffuse Horizontal Illuminance": EPW_MISSING["Illuminance"],
            "Zenith Luminance": EPW_MISSING["Illuminance"],
            "Wind Direction": wind_dir.mod(360.0).to_numpy(dtype=float),
            "Wind Speed": wind_speed.clip(lower=0.0).to_numpy(dtype=float),
            "Total Sky Cover": sky_cover.to_numpy(dtype=float),
            "Opaque Sky Cover": sky_cover.to_numpy(dtype=float),
            "Visibility": EPW_MISSING["Visibility"],
            "Ceiling Height": EPW_MISSING["CeilingHeight"],
            "Present Weather Observation": 9,
            "Present Weather Codes": "999999999",
            "Precipitable Water": EPW_MISSING["PrecipitableWater"],
            "Aerosol Optical Depth": EPW_MISSING["AerosolOpticalDepth"],
            "Snow Depth": EPW_MISSING["SnowDepth"],
            "Days Since Last Snowfall": EPW_MISSING["DaysSinceSnowfall"],
            "Albedo": EPW_MISSING["Albedo"],
            "Liquid Precipitation Depth": EPW_MISSING["LiquidPrecipitationDepth"],
            "Liquid Precipitation Quantity": EPW_MISSING["LiquidPrecipitationQuantity"],
        },
        index=local_index,
    )

    return out[EPW_DATA_COLUMNS]


def _epw_headers(
    df: pd.DataFrame,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    location_name: str,
    state_province_region: str,
    country: str,
    data_source: str,
    wmo_number: str,
    time_zone: float | None,
) -> list[str]:
    if time_zone is None:
        time_zone = _time_zone_offset_hours(tz, df.index)
    elevation = 0.0 if altitude is None or math.isnan(float(altitude)) else float(altitude)

    if len(df) > 0:
        start = df.index[0]
        end = df.index[-1]
        start_day_name = start.strftime("%A")
        start_md = f"{start.month}/{start.day}"
        end_md = f"{end.month}/{end.day}"
    else:
        start_day_name = "Sunday"
        start_md = "1/1"
        end_md = "12/31"

    return [
        "LOCATION,"
        + ",".join(
            [
                location_name,
                state_province_region,
                country,
                data_source,
                wmo_number,
                _format_number(lat, 6),
                _format_number(lon, 6),
                _format_number(time_zone, 1),
                _format_number(elevation, 1),
            ]
        ),
        "DESIGN CONDITIONS,0",
        "TYPICAL/EXTREME PERIODS,0",
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,Generated by hostrada4py from HOSTRADA hourly grid data",
        "COMMENTS 2,Direct and diffuse radiation are derived from HOSTRADA global horizontal radiation using pvlib erbs_driesse",
        f"DATA PERIODS,1,1,Data,{start_day_name},{start_md},{end_md}",
    ]


def write_energyplus_epw(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    lon: float,
    lat: float,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    location_name: str = "HOSTRADA",
    state_province_region: str = "",
    country: str = "DEU",
    data_source: str = "HOSTRADA",
    wmo_number: str = "999999",
    time_zone: float | None = None,
) -> Path:
    """Write an EnergyPlus ``.epw`` weather file.

    Parameters
    ----------
    df:
        Dataframe created by ``create_energyplus_weather_dataframe`` or a
        dataframe with the same ``EPW_DATA_COLUMNS``.
    output_file:
        Target file path, usually ending in ``.epw``.
    lon, lat:
        WGS84 longitude/latitude used in the EPW LOCATION header.
    altitude:
        Site altitude in metres. Missing values are written as 0 m.
    tz:
        Time zone of the dataframe and LOCATION header.
    time_zone:
        Optional EnergyPlus time-zone offset in hours from GMT. If omitted, the
        smallest UTC offset in ``tz`` over the dataframe is used, which matches
        standard time for DST regions.
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [col for col in EPW_DATA_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing EnergyPlus EPW output columns: {missing}")

    with path.open("w", encoding="utf-8", newline="") as f:
        for line in _epw_headers(
            df=df,
            lon=lon,
            lat=lat,
            altitude=altitude,
            tz=tz,
            location_name=location_name,
            state_province_region=state_province_region,
            country=country,
            data_source=data_source,
            wmo_number=wmo_number,
            time_zone=time_zone,
        ):
            f.write(line + "\n")
        for _, row in df.iterrows():
            f.write(_format_epw_row(row) + "\n")

    return path


def create_energyplus_weather_dataframe(
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
    """Create an EnergyPlus EPW dataframe for one HOSTRADA grid point.

    Only the minimum HOSTRADA variables required for the selected output are
    downloaded. Existing files in ``cache_dir`` are reused.
    """
    start_ts = _parse_utc_timestamp(start)
    end_ts = _parse_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("'end' must be greater than or equal to 'start'.")

    data = extract_multiple_values_for_point(
        vars=_required_vars(
            apply_weather_correction=apply_weather_correction,
            include_sky_cover=include_sky_cover,
        ),
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=Path(cache_dir),
    )

    if data.empty:
        raise ValueError("No HOSTRADA data was returned for the requested period.")

    return _prepare_energyplus_dataframe(
        data=data,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        apply_weather_correction=apply_weather_correction,
        include_sky_cover=include_sky_cover,
    )


def create_energyplus_weather_file(
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
    state_province_region: str = "",
    country: str = "DEU",
    data_source: str = "HOSTRADA",
    wmo_number: str = "999999",
    time_zone: float | None = None,
) -> Path:
    """Create an EnergyPlus ``.epw`` weather file for a point and period.

    Parameters
    ----------
    lon, lat:
        Longitude and latitude in WGS84 / EPSG:4326.
    start, end:
        UTC timestamps or date strings. HOSTRADA data are hourly; all requested
        months intersecting the interval are considered, but the output is cut
        to the exact time interval.
    output_file:
        Target path for the generated ``.epw`` file.
    altitude:
        Optional site altitude in metres, used for solar-position calculation
        and written to the EPW LOCATION header.
    tz:
        Local time zone used for the EPW data rows.
    cache_dir:
        Local HOSTRADA cache directory.
    apply_weather_correction:
        If True, download the additional HOSTRADA variables needed for the
        conservative weather correction already implemented in ``HostradaDiffuse``.
    include_sky_cover:
        If True, download ``clt`` and write total/opaque sky cover in tenths.
        If False, EPW missing-value markers are written for sky cover.
    location_name, state_province_region, country, data_source, wmo_number:
        EPW LOCATION header metadata.
    time_zone:
        Optional EnergyPlus time-zone offset in hours from GMT. If omitted, it
        is derived from ``tz``.

    Returns
    -------
    pathlib.Path
        Path to the generated EPW file.
    """
    df = create_energyplus_weather_dataframe(
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
    return write_energyplus_epw(
        df,
        output_file,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        location_name=location_name,
        state_province_region=state_province_region,
        country=country,
        data_source=data_source,
        wmo_number=wmo_number,
        time_zone=time_zone,
    )


# Short aliases for users who search for the EPW wording directly.
create_epw_weather_dataframe = create_energyplus_weather_dataframe
create_epw_weather_file = create_energyplus_weather_file
write_epw = write_energyplus_epw
