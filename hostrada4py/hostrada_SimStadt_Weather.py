"""SimStadt weather export in the 68-column TMY3 layout.

* line 1: TMY3 station metadata (68 comma-separated fields; first 7 used)
* line 2: the 68 TMY3 column names
* following lines: one hourly weather record per row

HOSTRADA variables are mapped to the corresponding TMY3 fields.  Fields for
which HOSTRADA does not provide an equivalent are written with explicit TMY3-
style missing values instead of inventing measurements.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from hostrada4py.hostradaDiffuse import HostradaDiffuse
from hostrada4py.hostradaPoint import extract_multiple_values_for_point


TMY3_COLUMNS = [
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
]

if len(TMY3_COLUMNS) != 68:  # guard against accidental edits
    raise RuntimeError("TMY3_COLUMNS must contain exactly 68 columns.")


MISSING = -9990


def ensure_target(output_file, default_name):
    """Return the output path and create its parent directory if needed."""
    path = Path(output_file or default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _utc_times(df):
    if "time" not in df.columns:
        raise KeyError("Weather dataframe contains no 'time' column.")
    times = pd.DatetimeIndex(pd.to_datetime(df["time"]))
    if times.tz is None:
        times = times.tz_localize("UTC")
    else:
        times = times.tz_convert("UTC")
    return times


def _standard_utc_offset_hours(tz, year):
    """Return the non-DST UTC offset required by the TMY3 metadata field.

    ``tz`` may be an IANA zone name (for example ``Europe/Berlin``) or a
    numeric UTC offset.  TMY3 uses local *standard* time, so DST is deliberately
    ignored in the written date/time values.
    """
    if isinstance(tz, (int, float, np.integer, np.floating)):
        return float(tz)

    text = str(tz).strip()
    try:
        return float(text)
    except ValueError:
        pass

    zone = ZoneInfo(text)
    candidates = []
    standard_candidates = []
    for month in range(1, 13):
        dt = datetime(int(year), month, 15, 12, tzinfo=zone)
        offset = dt.utcoffset() or timedelta(0)
        candidates.append(offset.total_seconds() / 3600.0)
        dst = dt.dst() or timedelta(0)
        if dst == timedelta(0):
            standard_candidates.append(offset.total_seconds() / 3600.0)

    if standard_candidates:
        return float(standard_candidates[0])
    # Fallback for unusual zones whose tzdata never reports dst()==0.
    return float(min(candidates, key=abs)) if candidates else 0.0


def _local_standard_times(df, tz):
    times_utc = _utc_times(df)
    if len(times_utc) == 0:
        return times_utc, 0.0
    offset = _standard_utc_offset_hours(tz, times_utc[0].year)
    fixed_tz = timezone(timedelta(hours=offset))
    return times_utc.tz_convert(fixed_tz), offset


def _tmy3_date_time_labels(local_standard_times):
    """Create TMY3 end-of-hour labels, including ``24:00``.

    A midnight timestamp is written as 24:00 on the preceding calendar date.
    """
    dates = []
    hours = []
    for ts in local_standard_times:
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            previous_day = ts - pd.Timedelta(days=1)
            dates.append(previous_day.strftime("%m/%d/%Y"))
            hours.append("24:00")
        else:
            dates.append(ts.strftime("%m/%d/%Y"))
            hours.append(ts.strftime("%H:%M"))
    return dates, hours


def weather_dataframe(
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


def _num(df, name, default=np.nan):
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def _temperature_c(values):
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    # HOSTRADA/DWD is normally already degC.  This also makes CERRA-like
    # Kelvin input safe without changing normal HOSTRADA output.
    if finite.size and np.nanmedian(finite) > 150.0:
        values -= 273.15
    return values


def _relative_humidity_percent(values):
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmax(finite) <= 1.5:
        values *= 100.0
    return np.clip(values, 0.0, 100.0)


def _pressure_mbar(values):
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size and np.nanmedian(finite) > 2000.0:
        values /= 100.0
    return values


def _cloud_tenths(values):
    values = np.asarray(values, dtype=float).copy()
    finite = values[np.isfinite(values)]
    if finite.size:
        vmax = np.nanmax(finite)
        if vmax <= 1.5:          # fraction 0..1
            values *= 10.0
        elif vmax > 10.0:        # percent 0..100
            values /= 10.0
    return np.rint(np.clip(values, 0.0, 10.0))


def _extraterrestrial_radiation(df, times_utc):
    """Return ETR and ETRN in W/m2.

    ETRN uses pvlib when available (pvlib is a hostrada4py dependency).  A
    Spencer-type day-of-year fallback is retained for robustness.  ETR is the
    normal extraterrestrial irradiance projected onto the horizontal plane by
    the solar zenith supplied by ``HostradaDiffuse``.
    """
    try:
        import pvlib

        etrn = np.asarray(
            pvlib.irradiance.get_extra_radiation(times_utc), dtype=float
        )
    except Exception:
        doy = times_utc.dayofyear.to_numpy(dtype=float)
        etrn = 1367.0 * (1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0))

    zenith = _num(df, "solar_zenith", 180.0)
    etr = etrn * np.maximum(np.cos(np.deg2rad(zenith)), 0.0)
    etr = np.where(zenith < 90.0, etr, 0.0)
    # ETRN is the normal irradiance on a surface perpendicular to the sun.
    # It is physically 0 when the sun is below the horizon.
    etrn = np.where(zenith < 90.0, etrn, 0.0)
    return np.rint(np.clip(etr, 0.0, None)), np.rint(np.clip(etrn, 0.0, None))


def _precipitable_water_cm(temp_c, rh_percent):
    try:
        import pvlib

        pw = np.asarray(
            pvlib.atmosphere.gueymard94_pw(temp_c, rh_percent), dtype=float
        )
        return np.where(np.isfinite(pw), pw, MISSING)
    except Exception:
        return np.full(len(temp_c), MISSING, dtype=float)


def _format_float(value, decimals=2, minimum_decimals=0):
    if value is None or not np.isfinite(float(value)):
        return str(MISSING)
    text = f"{float(value):.{decimals}f}"
    if "." in text:
        whole, frac = text.split(".", 1)
        frac = frac.rstrip("0")
        if len(frac) < minimum_decimals:
            frac += "0" * (minimum_decimals - len(frac))
        text = whole if not frac else whole + "." + frac
    return text


def _format_int(value, missing=MISSING):
    if value is None or not np.isfinite(float(value)):
        return str(missing)
    return str(int(np.rint(float(value))))


def _format_metadata_number(value, decimals=6):
    if value is None:
        return ""
    x = float(value)
    if x.is_integer():
        return str(int(x))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def _build_tmy3_rows(
    df,
    *,
    lon,
    lat,
    tz,
    altitude,
    location_name,
    station_id,
    state,
    tmy3_tz=None,
    albedo=0.2,
):
    """Build the metadata row and the 68-column hourly TMY3 rows."""
    if len(df) == 0:
        raise ValueError("Cannot build a TMY3 file from an empty dataframe.")

    times_utc = _utc_times(df)
    local_standard_times, inferred_tz = _local_standard_times(df, tz)
    tz_offset = inferred_tz if tmy3_tz is None else float(tmy3_tz)

    # If tmy3_tz is explicitly overridden, the data labels must use the same
    # fixed offset as the metadata row.
    if tmy3_tz is not None:
        fixed_tz = timezone(timedelta(hours=tz_offset))
        local_standard_times = times_utc.tz_convert(fixed_tz)

    dates, clock = _tmy3_date_time_labels(local_standard_times)

    ghi = np.clip(_num(df, "rsds", 0.0), 0.0, None)
    dni = np.clip(_num(df, "dni", 0.0), 0.0, None)
    dhi = np.clip(_num(df, "dhi", 0.0), 0.0, None)
    etr, etrn = _extraterrestrial_radiation(df, times_utc)

    temp = _temperature_c(_num(df, "tas"))
    dew = _temperature_c(_num(df, "tdew"))
    rh = _relative_humidity_percent(_num(df, "hurs"))
    pressure = _pressure_mbar(_num(df, "ps"))
    wdir = np.mod(_num(df, "sfcWind_direction"), 360.0)
    wspd = np.clip(_num(df, "sfcWind"), 0.0, None)
    total_cloud = _cloud_tenths(_num(df, "clt", 0.0))

    # HOSTRADA contains total cloud cover but no separate opaque cloud cover.
    # Using the same value preserves the measured cloud amount without
    # fabricating a second independent observation.
    opaque_cloud = total_cloud.copy()

    pwat = _precipitable_water_cm(temp, rh)

    # TMY3 fields not supplied by HOSTRADA.  The reference file uses -9990 for
    # unavailable meteorological quantities; source/uncertainty are 0 in those
    # cases.  This is preferable to inventing observations.
    missing = str(MISSING)

    rows = []
    for i in range(len(df)):
        ghi_present = np.isfinite(ghi[i]) and ghi[i] > 0
        dni_present = np.isfinite(dni[i]) and dni[i] > 0
        dhi_present = np.isfinite(dhi[i]) and dhi[i] > 0
        pw_present = np.isfinite(pwat[i]) and pwat[i] != MISSING

        row = [
            dates[i],
            clock[i],
            _format_int(etr[i]),
            _format_int(etrn[i]),
            _format_float(ghi[i], 2, 1),
            "3",
            "10" if ghi_present else "0",
            _format_float(dni[i], 2, 1),
            "3",
            "20" if dni_present else "0",
            _format_float(dhi[i], 2, 1),
            "3",
            "10" if dhi_present else "0",
            # No HOSTRADA illuminance/luminance variables are available.
            missing, "0", "0",
            missing, "0", "0",
            missing, "0", "0",
            missing, "0", "0",
            _format_int(total_cloud[i]), "E", "8",
            _format_int(opaque_cloud[i]), "E", "8",
            _format_float(temp[i], 2, 1), "E", "8",
            _format_float(dew[i], 2, 1), "E", "8",
            _format_int(rh[i]), "E", "8",
            _format_int(pressure[i]), "E", "8",
            _format_int(wdir[i]), "E", "8",
            _format_float(wspd[i], 1, 1), "E", "8",
            missing, "0", "0",
            missing, "0", "0",
            _format_float(pwat[i], 3, 1) if pw_present else missing,
            "E" if pw_present else "0",
            "8" if pw_present else "0",
            # Aerosol optical depth is not part of the HOSTRADA inputs.
            missing, "0", "0",
            _format_float(albedo, 3, 1), "E", "8",
            # HOSTRADA exporter currently has no precipitation variable.
            missing, missing, "0", "0",
        ]
        if len(row) != 68:
            raise RuntimeError(f"Internal TMY3 row has {len(row)} columns instead of 68.")
        rows.append(row)

    metadata = [
        str(station_id),
        str(location_name),
        str(state),
        _format_metadata_number(tz_offset, decimals=2),
        _format_metadata_number(lat, decimals=6),
        _format_metadata_number(lon, decimals=6),
        _format_metadata_number(altitude, decimals=3),
    ] + [""] * (68 - 7)

    return metadata, rows


def create_simstadt_weather_file(
    lon,
    lat,
    start,
    end,
    output_file="HOSTRADA_SimStadt.tmy3",
    *,
    tz="Europe/Berlin",
    altitude=0,
    location_name="HOSTRADA_SimStadt",
    station_id="0",
    state="GM",
    tmy3_tz=None,
    albedo=0.2,
    apply_weather_correction=False,
    cache_dir="hostrada_cache",
    cache_strategy=None,
    subset_margin_cells=None,
    provider=None,
    verbose=True,
    **kwargs,
):
    """Create a SimStadt-compatible TMY3 file with exactly 68 columns.

    Parameters added for the TMY3 metadata row
    ------------------------------------------
    location_name : str
        Station/location name written to metadata field 2.
    station_id : str or int
        TMY3 station identifier.  Defaults to ``0`` like the reference file.
    state : str
        State/country code field.  Defaults to ``GM`` like BERLIN_2022.tmy3.
    tmy3_tz : float, optional
        Fixed TMY3 UTC offset.  If omitted it is derived from ``tz`` using the
        standard (non-DST) offset.  For Europe/Berlin this is ``1``.
    albedo : float
        Ground albedo written to the TMY3 ``Alb`` field; default ``0.2``.
    """
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

    # Extend the fetch window by one extra day so that UTC timestamps that map
    # to the last local-time hours of the requested year are not cut off.
    # Example: for UTC+1, "2023-12-31 23:00 local" = "2023-12-31 22:00 UTC".
    # If the API stops at "2023-12-31 00:00 UTC" we lose the last 23 hours.
    end_extended = (pd.Timestamp(end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    df = weather_dataframe(
        lon,
        lat,
        start,
        end_extended,
        required,
        tz=tz,
        altitude=altitude,
        apply_weather_correction=apply_weather_correction,
        cache_dir=cache_dir,
        cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,
        provider=provider,
        verbose=verbose,
    )

    # Trim the dataframe to the originally requested date range in local standard
    # time so the output file contains exactly the hours of the requested period.
    local_times, _off = _local_standard_times(df, tz)
    start_local = pd.Timestamp(start, tz=local_times.tz)
    end_local   = pd.Timestamp(end, tz=local_times.tz) + pd.Timedelta(days=1)
    mask = (local_times >= start_local) & (local_times < end_local)
    df = df.iloc[mask.to_numpy()].reset_index(drop=True)

    metadata, rows = _build_tmy3_rows(
        df,
        lon=lon,
        lat=lat,
        tz=tz,
        altitude=altitude,
        location_name=location_name,
        station_id=station_id,
        state=state,
        tmy3_tz=tmy3_tz,
        albedo=albedo,
    )

    path = ensure_target(output_file, "HOSTRADA_SimStadt.tmy3")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(metadata)
        writer.writerow(TMY3_COLUMNS)
        writer.writerows(rows)

    return path


create_weather_file = create_simstadt_weather_file
