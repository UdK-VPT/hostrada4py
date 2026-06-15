#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hostrada_Polysun_Weather.py

Create point-based Polysun / Vela Solaris weather profile CSV files from
HOSTRADA data.

Polysun weather profiles can be imported as CSV files and require hourly values
for these quantities:

    Gh     global radiation [Wh/m²]
    Dh     diffuse radiation [Wh/m²]
    Lh     long-wave irradiation [Wh/m²]
    Tamb   ambient temperature [°C]
    Vwnd   wind speed [m/s]
    Hrel   relative humidity [%]

HOSTRADA provides hourly mean irradiance in W/m². For hourly rows, the numeric
value is equivalent to Wh/m² over the hour. Direct and diffuse radiation are
derived through the existing HostradaDiffuse pipeline. Long-wave irradiation is
estimated locally from temperature and humidity; an optional cloud-cover
correction downloads only the additional ``clt`` variable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostrada_EnergyPlus_Weather import _dew_point_from_t_rh, _parse_utc_timestamp
from hostrada4py.hostradaPoint import CACHE_DIR, extract_multiple_values_for_point

POLYSUN_WEATHER_COLUMNS = ["Gh", "Dh", "Lh", "Tamb", "Vwnd", "Hrel"]
POLYSUN_WEATHER_COLUMN_DESCRIPTIONS = {
    "Gh": "Global radiation [Wh/m2]",
    "Dh": "Diffuse radiation [Wh/m2]",
    "Lh": "Long-wave irradiation [Wh/m2]",
    "Tamb": "Ambient temperature [degC]",
    "Vwnd": "Wind speed [m/s]",
    "Hrel": "Relative humidity [%]",
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
    *,
    apply_weather_correction: bool,
    include_longwave_cloud_correction: bool,
) -> list[str]:
    """Return the minimum HOSTRADA variables needed for Polysun CSV output."""
    # rsds, tas and ps are needed by HostradaDiffuse; hurs and sfcWind are
    # written directly or used for local long-wave estimation. Dew point is
    # calculated locally to avoid downloading tdew by default.
    required = ["rsds", "tas", "ps", "hurs", "sfcWind"]

    if include_longwave_cloud_correction:
        required.append("clt")

    if apply_weather_correction:
        # Additional variables used by HostradaDiffuse._weather_correction.
        # sfcWind is already part of the base set; sfcWind_direction is not used
        # by the correction or by the Polysun CSV profile.
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
    return out.sort_index()


def _series_or_default(df: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in df.columns:
        return df[column].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def _cloud_cover_fraction(series: pd.Series) -> pd.Series:
    cloud = series.astype(float)
    if cloud.max(skipna=True) > 10.0:
        # HOSTRADA clt is commonly stored as percent.
        cloud = cloud / 100.0
    elif cloud.max(skipna=True) > 1.5:
        # Some weather formats store tenths or oktas.
        cloud = cloud / 10.0
    return cloud.clip(lower=0.0, upper=1.0).fillna(0.0)


def _longwave_irradiation_wh_m2(
    temp_c: pd.Series,
    rh_pct: pd.Series,
    cloud_cover: pd.Series | None = None,
) -> pd.Series:
    """Estimate hourly long-wave irradiation in Wh/m².

    The clear-sky emissivity uses Brutsaert's common vapour-pressure relation.
    If cloud cover is available, a conservative cloud correction is applied.
    For hourly mean data the numeric W/m² value equals Wh/m² for the hour.
    """
    sigma = 5.670374419e-8
    temp_k = temp_c.astype(float) + 273.15
    dew_c = _dew_point_from_t_rh(temp_c.astype(float), rh_pct.astype(float))

    # Saturation vapour pressure at dew point in hPa. The Brutsaert expression
    # needs vapour pressure in kPa and air temperature in K.
    vapour_pressure_hpa = 6.112 * np.exp((17.67 * dew_c) / (dew_c + 243.5))
    vapour_pressure_kpa = vapour_pressure_hpa / 10.0
    emissivity_clear = 1.24 * np.power((vapour_pressure_kpa / temp_k).clip(lower=1e-6), 1.0 / 7.0)

    if cloud_cover is not None:
        cloud = _cloud_cover_fraction(cloud_cover)
        emissivity = emissivity_clear * (1.0 + 0.22 * np.square(cloud))
    else:
        emissivity = emissivity_clear

    emissivity = emissivity.clip(lower=0.0, upper=1.0)
    return pd.Series(emissivity * sigma * np.power(temp_k, 4), index=temp_c.index).clip(lower=0.0)


def _format_csv_value(value: object, *, decimal: str = ".") -> str:
    try:
        if pd.isna(value):
            text = "0"
        elif isinstance(value, (np.integer, int)):
            text = str(int(value))
        else:
            number = float(value)
            if math.isfinite(number) and number.is_integer():
                text = str(int(number))
            else:
                text = f"{number:.6g}"
    except (TypeError, ValueError):
        text = str(value)

    if decimal != ".":
        text = text.replace(".", decimal)
    return text


def _prepare_polysun_dataframe(
    data: pd.DataFrame,
    *,
    lon: float,
    lat: float,
    altitude: float | None,
    tz: str,
    apply_weather_correction: bool,
    include_longwave_cloud_correction: bool,
) -> pd.DataFrame:
    data = _as_utc_index(data)

    model = HostradaDiffuse(latitude=lat, longitude=lon, altitude=altitude, tz=tz)
    radiation = model.estimate(data, apply_weather_correction=apply_weather_correction)
    local_index = radiation.index.tz_convert(tz)

    ghi = _series_or_default(radiation, "global_radiation", np.nan)
    if ghi.isna().all() and "rsds" in radiation.columns:
        ghi = radiation["rsds"].astype(float)

    dhi = _series_or_default(radiation, "dhi", 0.0)
    temp = _series_or_default(radiation, "temp_2m", np.nan)
    if temp.isna().all() and "tas" in radiation.columns:
        temp = radiation["tas"].astype(float)

    rh = _series_or_default(radiation, "rh_2m", 50.0).clip(lower=0.0, upper=100.0)
    wind_speed = _series_or_default(radiation, "wind_speed_10m", 0.0).clip(lower=0.0)

    cloud = None
    if include_longwave_cloud_correction and "cloud_cover" in radiation.columns:
        cloud = radiation["cloud_cover"]
    elif include_longwave_cloud_correction and "clt" in radiation.columns:
        cloud = radiation["clt"]

    longwave = _longwave_irradiation_wh_m2(temp, rh, cloud_cover=cloud)

    out = pd.DataFrame(index=local_index)
    # Polysun expects hourly irradiation values in Wh/m². HOSTRADA hourly means
    # in W/m² are numerically equal for one-hour rows.
    out["Gh"] = ghi.clip(lower=0.0).to_numpy(dtype=float)
    out["Dh"] = dhi.clip(lower=0.0).to_numpy(dtype=float)
    out["Lh"] = longwave.to_numpy(dtype=float)
    out["Tamb"] = temp.to_numpy(dtype=float)
    out["Vwnd"] = wind_speed.to_numpy(dtype=float)
    out["Hrel"] = rh.to_numpy(dtype=float)
    return out[POLYSUN_WEATHER_COLUMNS]


def create_polysun_weather_dataframe(
    lon: float,
    lat: float,
    start: str,
    end: str,
    *,
    altitude: float | None = None,
    tz: str = "Europe/Berlin",
    cache_dir: str | Path = CACHE_DIR,
    apply_weather_correction: bool = False,
    include_longwave_cloud_correction: bool = False,
) -> pd.DataFrame:
    """Create a Polysun weather-profile dataframe for one HOSTRADA point.

    By default only ``rsds``, ``tas``, ``ps``, ``hurs`` and ``sfcWind`` are
    downloaded. ``include_longwave_cloud_correction=True`` additionally requests
    ``clt`` and uses it for the locally estimated long-wave irradiation. The
    returned dataframe has exactly the six Polysun profile columns ``Gh``, ``Dh``,
    ``Lh``, ``Tamb``, ``Vwnd`` and ``Hrel``.
    """
    start_ts = _parse_utc_timestamp(start)
    end_ts = _parse_utc_timestamp(end)
    if end_ts < start_ts:
        raise ValueError("'end' must be greater than or equal to 'start'.")

    data = extract_multiple_values_for_point(
        vars=_required_vars(
            apply_weather_correction=apply_weather_correction,
            include_longwave_cloud_correction=include_longwave_cloud_correction,
        ),
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        cache_dir=Path(cache_dir),
    )

    if data.empty:
        raise ValueError("No HOSTRADA data was returned for the requested period.")

    return _prepare_polysun_dataframe(
        data,
        lon=lon,
        lat=lat,
        altitude=altitude,
        tz=tz,
        apply_weather_correction=apply_weather_correction,
        include_longwave_cloud_correction=include_longwave_cloud_correction,
    )


def write_polysun_csv(
    df: pd.DataFrame,
    output_file: str | Path,
    *,
    delimiter: str = ",",
    decimal: str = ".",
    include_header: bool = True,
    include_metadata: bool = True,
    location_name: str = "HOSTRADA",
    lon: float | None = None,
    lat: float | None = None,
    altitude: float | None = None,
) -> Path:
    """Write a Polysun weather-profile CSV file.

    The default file contains optional comment metadata followed by a header row
    and the six Polysun weather columns. For strict imports, set
    ``include_metadata=False`` to write only the CSV header and numeric rows.
    """
    missing = [col for col in POLYSUN_WEATHER_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing Polysun output columns: {missing}")

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        if include_metadata:
            f.write("# Polysun weather profile generated by hostrada4py from HOSTRADA data\n")
            f.write("# Columns: " + ", ".join(POLYSUN_WEATHER_COLUMN_DESCRIPTIONS[c] for c in POLYSUN_WEATHER_COLUMNS) + "\n")
            if lon is not None and lat is not None:
                f.write(f"# Location: {location_name}, lat={lat:.8g}, lon={lon:.8g}")
                if altitude is not None:
                    f.write(f", altitude={float(altitude):.3g} m")
                f.write("\n")
        if include_header:
            f.write(delimiter.join(POLYSUN_WEATHER_COLUMNS) + "\n")
        for _, row in df[POLYSUN_WEATHER_COLUMNS].iterrows():
            f.write(delimiter.join(_format_csv_value(row[col], decimal=decimal) for col in POLYSUN_WEATHER_COLUMNS) + "\n")

    return path


def create_polysun_weather_file(
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
    include_longwave_cloud_correction: bool = False,
    location_name: str = "HOSTRADA",
    delimiter: str = ",",
    decimal: str = ".",
    include_header: bool = True,
    include_metadata: bool = True,
) -> Path:
    """Create a Polysun / Vela Solaris weather-profile CSV file.

    Parameters are analogous to the other hostrada4py weather exporters. The
    output rows are hourly and contain ``Gh``, ``Dh``, ``Lh``, ``Tamb``, ``Vwnd``
    and ``Hrel`` in the order expected by Polysun weather profiles.
    """
    df = create_polysun_weather_dataframe(
        lon=lon,
        lat=lat,
        start=start,
        end=end,
        altitude=altitude,
        tz=tz,
        cache_dir=cache_dir,
        apply_weather_correction=apply_weather_correction,
        include_longwave_cloud_correction=include_longwave_cloud_correction,
    )
    return write_polysun_csv(
        df,
        output_file,
        delimiter=delimiter,
        decimal=decimal,
        include_header=include_header,
        include_metadata=include_metadata,
        location_name=location_name,
        lon=lon,
        lat=lat,
        altitude=altitude,
    )


# Short aliases for users who search for the vendor/program wording directly.
create_polysun_weather_csv = create_polysun_weather_file
create_polysun_csv_weather_file = create_polysun_weather_file
create_velasolaris_weather_dataframe = create_polysun_weather_dataframe
create_velasolaris_weather_file = create_polysun_weather_file
write_polysun_weather_csv = write_polysun_csv
write_velasolaris_weather_csv = write_polysun_csv
