"""Diffuse/direct irradiance derivation for HOSTRADA-compatible point data."""
from __future__ import annotations
import numpy as np
import pandas as pd

_METADATA = {"unit", "variable_description"}


def _value_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"time", "input_lon", "input_lat", "grid_x_epsg3034", "grid_y_epsg3034",
                "grid_lon", "grid_lat", "X", "Y", "x", "y", "lon", "lat", *_METADATA}
    return [c for c in frame.columns if c not in metadata]


def combine_point_variables(frames) -> pd.DataFrame:
    result = None
    for frame in frames:
        if frame is None or frame.empty:
            continue
        cols = ["time", *_value_columns(frame)]
        piece = frame.loc[:, [c for c in cols if c in frame]].copy()
        result = piece if result is None else result.merge(piece, on="time", how="outer")
    if result is None:
        return pd.DataFrame(columns=["time"])
    result["time"] = pd.to_datetime(result["time"])
    return result.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# Offset from the timestamp to the midpoint of the averaging interval of the
# hourly irradiance. HOSTRADA rsds is labelled at the END of the hour it
# averages (the diurnal centroid lies ~30 min after solar noon, although the
# file's time_bnds claim [t, t+1h]); CERRA de-accumulated rsds is labelled at
# its forecast valid time, i.e. also the end of the hour.
_INTERVAL_MIDPOINT_OFFSET = {
    "end": pd.Timedelta(minutes=-30),
    "center": pd.Timedelta(0),
    "start": pd.Timedelta(minutes=30),
}


class HostradaDiffuse:
    def __init__(self, latitude: float, longitude: float, altitude: float = 0.0,
                 tz: str = "UTC", method: str = "erbs_driesse",
                 timestamp_label: str = "end"):
        if timestamp_label not in _INTERVAL_MIDPOINT_OFFSET:
            raise ValueError(f"timestamp_label must be one of {sorted(_INTERVAL_MIDPOINT_OFFSET)}.")
        self.latitude, self.longitude = float(latitude), float(longitude)
        self.altitude, self.tz, self.method = float(altitude), str(tz), str(method)
        self.timestamp_label = timestamp_label

    def estimate(self, data: pd.DataFrame, ghi_column: str = "rsds",
                 apply_weather_correction: bool = False) -> pd.DataFrame:
        if ghi_column not in data:
            raise KeyError(f"Missing global horizontal irradiance column {ghi_column!r}.")
        out = data.copy()
        times = pd.DatetimeIndex(pd.to_datetime(out["time"]))
        if times.tz is None:
            times = times.tz_localize("UTC")
        # Solar position at the midpoint of the averaging interval; using the
        # timestamp itself gives near-horizon zenith angles for hours that
        # still carry irradiance and hence unrealistic DNI peaks.
        local_times = (times + _INTERVAL_MIDPOINT_OFFSET[self.timestamp_label]).tz_convert(self.tz)
        ghi = pd.to_numeric(out[ghi_column], errors="coerce").fillna(0.0).clip(lower=0.0)
        try:
            import pvlib
            solar = pvlib.solarposition.get_solarposition(
                local_times, self.latitude, self.longitude, altitude=self.altitude
            )
            zenith = solar["zenith"].to_numpy()
            if self.method == "erbs_driesse" and hasattr(pvlib.irradiance, "erbs_driesse"):
                split = pvlib.irradiance.erbs_driesse(ghi.to_numpy(), zenith, local_times)
            else:
                split = pvlib.irradiance.erbs(ghi.to_numpy(), zenith, local_times)
            dhi = np.asarray(split["dhi"], dtype=float)
            dni = np.asarray(split["dni"], dtype=float)
            kt = np.asarray(split.get("kt", np.nan), dtype=float)
        except ImportError:
            # Conservative offline fallback; the package declares pvlib as a normal dependency.
            hour = local_times.hour.to_numpy() + local_times.minute.to_numpy() / 60.0
            daylight = np.clip(np.sin(np.pi * (hour - 6.0) / 12.0), 0.0, None)
            zenith = np.degrees(np.arccos(np.clip(daylight, 0.0, 1.0)))
            kd_guess = np.where(ghi.to_numpy() > 0, 0.35 + 0.35 * (1 - daylight), 0.0)
            dhi = ghi.to_numpy() * np.clip(kd_guess, 0.15, 0.95)
            cosz = np.clip(np.cos(np.radians(zenith)), 0.065, None)
            dni = np.where(daylight > 0, np.maximum(0.0, (ghi.to_numpy() - dhi) / cosz), 0.0)
            kt = np.full(len(out), np.nan)
        if apply_weather_correction:
            correction = np.ones(len(out))
            if "clt" in out:
                cloud = pd.to_numeric(out["clt"], errors="coerce").fillna(0).to_numpy()
                if np.nanmax(cloud, initial=0) > 1.5: cloud = cloud / 100.0
                correction *= 1.0 + 0.18 * np.clip(cloud, 0, 1)
            if "hurs" in out:
                rh = pd.to_numeric(out["hurs"], errors="coerce").fillna(50).to_numpy()
                correction *= 1.0 + 0.05 * np.clip((rh - 70.0) / 30.0, 0, 1)
            if "uhi" in out:
                uhi = pd.to_numeric(out["uhi"], errors="coerce").fillna(0).to_numpy()
                correction *= 1.0 + 0.01 * np.clip(uhi, 0, 5)
            dhi = np.minimum(ghi.to_numpy(), dhi * correction)
            cosz = np.clip(np.cos(np.radians(zenith)), 0.065, None)
            dni = np.where(zenith < 90, np.maximum(0.0, (ghi.to_numpy() - dhi) / cosz), 0.0)
        out["solar_zenith"] = zenith
        out["dhi"] = np.nan_to_num(dhi, nan=0.0, posinf=0.0, neginf=0.0)
        out["dni"] = np.nan_to_num(dni, nan=0.0, posinf=0.0, neginf=0.0)
        ghi_np = ghi.to_numpy(dtype=float)
        dhi_np = out["dhi"].to_numpy(dtype=float)
        kdh = np.zeros_like(ghi_np, dtype=float)
        np.divide(dhi_np,ghi_np,out=kdh,where=ghi_np > 0)
        out["kdh"] = kdh
        out["kt"] = kt
        return out

    calculate = estimate
