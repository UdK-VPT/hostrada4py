#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hostradaDiffuse.py

Calculation of diffuse horizontal irradiance (DHI) from HOSTRADA climate data.

Robust base method:
    - pvlib.irradiance.erbs_driesse

Required minimum inputs:
    - timestamp (DatetimeIndex)
    - local latitude / longitude
    - global radiation (GHI, HOSTRADA variable: rsds)

Optional HOSTRADA refinement inputs:
    - cloud cover (clt)
    - wind speed / direction at 10 m (sfcWind / sfcWind_direction)
    - air temperature at 2 m (tas)
    - dew point temperature at 2 m (tdew)
    - relative humidity at 2 m (hurs)
    - mixing ratio at 2 m (mixr)
    - air pressure at station height (ps)
    - air pressure at sea level (psl)
    - urban heat island intensity (uhi)

The DHI estimate itself is based on erbs_driesse for maximal robustness.
Optionally, a conservative weather-based correction can be applied on top of the
robust base estimate when the additional HOSTRADA variables are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

def _get_pvlib():
    try:
        import pvlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pvlib is required for hostradaDiffuse. Install it with: pip install pvlib"
        ) from exc
    return pvlib


HOSTRADA_TO_STANDARD_COLUMNS = {
    "rsds": "global_radiation",
    "tas": "temp_2m",
    "tdew": "dewpoint_2m",
    "hurs": "rh_2m",
    "mixr": "mixing_ratio_2m",
    "ps": "pressure_station",
    "psl": "pressure_msl",
    "sfcWind": "wind_speed_10m",
    "sfcWind_direction": "wind_dir_10m",
    "clt": "cloud_cover",
    "uhi": "uhi_intensity",
}


@dataclass
class HostradaDiffuse:
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    tz: str = "UTC"

    ghi_col: str = "global_radiation"
    temp_col: str = "temp_2m"
    dew_col: str = "dewpoint_2m"
    rh_col: str = "rh_2m"
    mixr_col: str = "mixing_ratio_2m"
    p_station_col: str = "pressure_station"
    p_msl_col: str = "pressure_msl"
    wind_speed_col: str = "wind_speed_10m"
    wind_dir_col: str = "wind_dir_10m"
    cloud_col: str = "cloud_cover"
    uhi_col: str = "uhi_intensity"

    def _prepare_index(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas.DatetimeIndex.")

        if out.index.tz is None:
            out.index = out.index.tz_localize(self.tz)
        else:
            out.index = out.index.tz_convert(self.tz)
        return out

    def _normalize_cloud(self, s: pd.Series) -> pd.Series:
        s = s.astype(float)
        if s.max(skipna=True) > 1.5:
            s = s / 100.0
        return s.clip(lower=0.0, upper=1.0)

    def _pressure_pa(self, df: pd.DataFrame) -> pd.Series:
        if self.p_station_col in df.columns:
            return df[self.p_station_col].astype(float) * 100.0  # hPa -> Pa
        if self.p_msl_col in df.columns:
            return df[self.p_msl_col].astype(float) * 100.0  # hPa -> Pa
        return pd.Series(101325.0, index=df.index, dtype=float)

    def prepare_hostrada_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename HOSTRADA-native column names to the internal standard names if necessary.
        Existing standard names are preserved.
        """
        out = df.copy()
        rename_map = {
            source: target
            for source, target in HOSTRADA_TO_STANDARD_COLUMNS.items()
            if source in out.columns and target not in out.columns
        }
        if rename_map:
            out = out.rename(columns=rename_map)
        return out

    def estimate(
        self,
        df: pd.DataFrame,
        apply_weather_correction: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate diffuse horizontal irradiance (DHI) from HOSTRADA data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with a DatetimeIndex and at least the GHI column.
        apply_weather_correction : bool, default False
            If True, apply a conservative correction on top of the robust
            erbs_driesse estimate using available HOSTRADA weather variables.

        Returns
        -------
        pd.DataFrame
            Original dataframe plus solar geometry and radiation decomposition.
        """
        out = self.prepare_hostrada_dataframe(df)
        out = self._prepare_index(out)

        if self.ghi_col not in out.columns:
            raise KeyError(
                f"Missing required GHI column '{self.ghi_col}'. "
                "Provide either 'global_radiation' or HOSTRADA 'rsds'."
            )

        ghi = out[self.ghi_col].astype(float).clip(lower=0.0)

        temp = out[self.temp_col].astype(float) if self.temp_col in out.columns else pd.Series(10.0, index=out.index)
        pressure_pa = self._pressure_pa(out)

        pvlib = _get_pvlib()
        solar_pos = pvlib.solarposition.get_solarposition(
            time=out.index,
            latitude=self.latitude,
            longitude=self.longitude,
            altitude=self.altitude,
            pressure=pressure_pa,
            temperature=temp,
            method="nrel_numpy",
        )
        zenith = solar_pos["zenith"].clip(lower=0.0, upper=180.0)
        cosz = pd.Series(np.cos(np.radians(zenith.to_numpy())), index=out.index).clip(lower=0.0)

        comp = pvlib.irradiance.erbs_driesse(
            ghi=ghi,
            zenith=zenith,
            datetime_or_doy=out.index,
        )

        dni = comp["dni"].fillna(0.0).clip(lower=0.0)
        dhi = comp["dhi"].fillna(0.0).clip(lower=0.0, upper=ghi)
        kd_base = (dhi / ghi.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0, upper=1.0)
        kd = kd_base.copy()

        if apply_weather_correction:
            correction = self._weather_correction(out)
            kd = (kd_base * correction).clip(lower=0.0, upper=1.0)
            dhi = (kd * ghi).clip(lower=0.0, upper=ghi)
            dni = ((ghi - dhi) / cosz.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0)
        else:
            dni = ((ghi - dhi) / cosz.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0)
            dhi = (ghi - dni * cosz).clip(lower=0.0, upper=ghi)

        kd = (dhi / ghi.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0, upper=1.0)

        result = out.copy()
        result["solar_zenith_deg"] = zenith
        result["solar_elevation_deg"] = solar_pos["elevation"]
        result["solar_azimuth_deg"] = solar_pos["azimuth"]
        result["dni"] = dni
        result["dhi"] = dhi
        result["kd"] = kd
        result["dhi_method"] = "erbs_driesse+weather" if apply_weather_correction else "erbs_driesse"
        return result

    def _weather_correction(self, df: pd.DataFrame) -> pd.Series:
        """
        Conservative bounded correction factor using available HOSTRADA features.
        This keeps erbs_driesse as the base method and only slightly adjusts kd.
        """
        corr = pd.Series(1.0, index=df.index, dtype=float)

        if self.cloud_col in df.columns:
            cloud = self._normalize_cloud(df[self.cloud_col])
            corr += 0.20 * (cloud - 0.5)

        if self.rh_col in df.columns:
            rh = df[self.rh_col].astype(float)
            rh_n = ((rh - 40.0) / 60.0).clip(lower=0.0, upper=1.0)
            corr += 0.05 * (rh_n - 0.5)

        if self.temp_col in df.columns and self.dew_col in df.columns:
            temp = df[self.temp_col].astype(float)
            dew = df[self.dew_col].astype(float)
            dew_dep = (temp - dew).clip(lower=0.0, upper=20.0)
            dew_n = (1.0 - dew_dep / 20.0).clip(lower=0.0, upper=1.0)
            corr += 0.04 * (dew_n - 0.5)

        if self.mixr_col in df.columns:
            mixr = df[self.mixr_col].astype(float)
            if mixr.median(skipna=True) > 0.2:
                mixr = mixr / 1000.0  # g/kg -> kg/kg
            mixr_n = ((mixr - 0.002) / 0.018).clip(lower=0.0, upper=1.0)
            corr += 0.04 * (mixr_n - 0.5)

        if self.wind_speed_col in df.columns:
            wind = df[self.wind_speed_col].astype(float)
            wind_n = (wind / 12.0).clip(lower=0.0, upper=1.0)
            corr += 0.015 * (wind_n - 0.5)

        if self.uhi_col in df.columns:
            uhi = df[self.uhi_col].astype(float)
            uhi_n = (uhi / 8.0).clip(lower=0.0, upper=1.0)
            corr += 0.02 * (uhi_n - 0.5)

        if self.p_station_col in df.columns and self.p_msl_col in df.columns:
            p_station = df[self.p_station_col].astype(float)
            p_msl = df[self.p_msl_col].astype(float)
            dp_n = ((p_msl - p_station) / 50.0).clip(lower=0.0, upper=1.0)
            corr += 0.01 * (dp_n - 0.5)

        return corr.clip(lower=0.75, upper=1.25)

def combine_point_variables(frames: Iterable[pd.DataFrame], time_col: str = "time") -> pd.DataFrame:
    """
    Combine long-format per-variable point frames returned by hostradaPoint into one
    wide dataframe indexed by time.
    """
    frames = list(frames)
    if not frames:
        return pd.DataFrame()

    wide_frames = []

    # Metadata / coordinates that are NOT considered actual value columns
    excluded_columns = {
        time_col,
        "unit",
        "variable_description",
        "input_lon",
        "input_lat",
        "grid_x_epsg3034",
        "grid_y_epsg3034",
        "grid_lon",
        "grid_lat",
        "X",
        "Y",
        "x",
        "y",
        "lon",
        "lat",
    }

    # Known HOSTRADA value columns
    known_value_columns = {
        "rsds",
        "tas",
        "tdew",
        "hurs",
        "mixr",
        "ps",
        "psl",
        "sfcWind",
        "sfcWind_direction",
        "clt",
        "uhi",
    }

    meta = None

    for df in frames:
        if df.empty:
            continue

        current = df.copy()

        if time_col not in current.columns:
            raise KeyError(f"Missing time column '{time_col}' in one of the input frames.")

        # First, narrow down the candidates
        candidate_cols = [c for c in current.columns if c not in excluded_columns]

        # Preferably known HOSTRADA value columns
        variable_cols = [c for c in candidate_cols if c in known_value_columns]

        # Fallback: if exactly one column remains, use that one
        if len(variable_cols) == 0 and len(candidate_cols) == 1:
            variable_cols = candidate_cols

        if len(variable_cols) != 1:
            raise ValueError(
                "Expected exactly one HOSTRADA value column per frame after excluding "
                f"metadata/coordinates. Found candidates: {candidate_cols}, "
                f"selected: {variable_cols}"
            )

        value_col = variable_cols[0]
        wide_frames.append(current[[time_col, value_col]].copy())

        if meta is None:
            meta_cols = [
                c for c in [
                    "input_lon",
                    "input_lat",
                    "grid_x_epsg3034",
                    "grid_y_epsg3034",
                    "grid_lon",
                    "grid_lat",
                    "X",
                    "Y",
                    "lon",
                    "lat",
                ]
                if c in current.columns
            ]
            if meta_cols:
                meta = current[[time_col] + meta_cols].drop_duplicates(subset=[time_col])

    if not wide_frames:
        return pd.DataFrame()

    result = wide_frames[0]
    for frame in wide_frames[1:]:
        result = result.merge(frame, on=time_col, how="outer")

    if meta is not None:
        result = result.merge(meta, on=time_col, how="left")

    result[time_col] = pd.to_datetime(result[time_col], utc=True)
    result = result.sort_values(time_col).drop_duplicates(subset=[time_col]).set_index(time_col)
    return result

