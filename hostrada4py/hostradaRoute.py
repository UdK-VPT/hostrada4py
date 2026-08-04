#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HOSTRADA/CERRA climate profile along a time-defined route.

The default provider selection is DWD/HOSTRADA; CERRA can be
selected globally or with the optional ``provider`` argument.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from hostrada4py import hostrada as hs
from hostrada4py.hostradaPoint import extract_values_for_point

PathLike = Union[str, Path]
ProgressCallback = Callable[[int, int, pd.Series, str], None]
HOSTRADA_ROUTE_CACHE_REVISION = "2026-08-03-shared-month-cache-v1"

HOSTRADA_VARIABLES = {
    "tas": {"label":"Air temperature (2 m)","short_label":"Air temperature","output_column":"air_temperature_c","display_unit":"°C","conversion":"temperature","cyclic":False},
    "tdew": {"label":"Dew point temperature (2 m)","short_label":"Dew point temperature","output_column":"dew_point_temperature_c","display_unit":"°C","conversion":"temperature","cyclic":False},
    "uhi": {"label":"Urban heat island intensity","short_label":"UHI intensity","output_column":"urban_heat_island_intensity_k","display_unit":"K","conversion":"identity","cyclic":False},
    "sfcWind": {"label":"Wind speed (10 m)","short_label":"Wind speed","output_column":"wind_speed_m_s","display_unit":"m/s","conversion":"identity","cyclic":False},
    "sfcWind_direction": {"label":"Wind direction (10 m)","short_label":"Wind direction","output_column":"wind_direction_deg","display_unit":"°","conversion":"degrees","cyclic":True},
    "rsds": {"label":"Global radiation","short_label":"Global radiation","output_column":"global_radiation_w_m2","display_unit":"W/m²","conversion":"identity","cyclic":False},
    "clt": {"label":"Cloud cover","short_label":"Cloud cover","output_column":"cloud_cover_percent","display_unit":"%","conversion":"percent","cyclic":False},
    "hurs": {"label":"Relative humidity (2 m)","short_label":"Relative humidity","output_column":"relative_humidity_percent","display_unit":"%","conversion":"percent","cyclic":False},
    "mixr": {"label":"Water vapor mixing ratio (2 m)","short_label":"Mixing ratio","output_column":"water_vapor_mixing_ratio","display_unit":"kg/kg","conversion":"identity","cyclic":False},
    "ps": {"label":"Air pressure at station elevation","short_label":"Station pressure","output_column":"surface_pressure_hpa","display_unit":"hPa","conversion":"pressure","cyclic":False},
    "psl": {"label":"Air pressure at sea level","short_label":"Sea-level pressure","output_column":"sea_level_pressure_hpa","display_unit":"hPa","conversion":"pressure","cyclic":False},
}

@dataclass(frozen=True)
class RouteClimateConfig:
    variables: tuple[str, ...] = ("tas",)
    timezone: str = "Europe/Berlin"
    interpolation: str = "linear"
    cache_dir: Path = Path("hostrada_cache")
    cache_strategy: Optional[str] = "full"
    subset_margin_cells: Optional[int] = 0
    coordinate_decimals: int = 6
    provider: object = None

_COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time"),
    "longitude": ("longitude", "lon", "lng", "x"),
    "latitude": ("latitude", "lat", "y"),
}

def _available_codes(provider=None) -> tuple[str, ...]:
    ctx = hs.use_provider(provider) if provider is not None else nullcontext()
    with ctx:
        supported = set(hs.provider_capabilities().variables)
    return tuple(code for code in HOSTRADA_VARIABLES if code in supported)

def available_variables(provider=None) -> pd.DataFrame:
    """Return variables available from the active provider as a table."""
    return pd.DataFrame([
        {"variable": code, "description": HOSTRADA_VARIABLES[code]["label"],
         "output_column": HOSTRADA_VARIABLES[code]["output_column"],
         "display_unit": HOSTRADA_VARIABLES[code]["display_unit"]}
        for code in _available_codes(provider)
    ])

def _validate_variables(variables: Union[str, Iterable[str]], provider=None) -> tuple[str, ...]:
    available = _available_codes(provider)
    if isinstance(variables, str):
        result = available if variables.lower() == "all" else (variables,)
    else:
        result = tuple(variables)
    if not result:
        raise ValueError("At least one HOSTRADA variable must be selected.")
    unknown = [v for v in result if v not in HOSTRADA_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown HOSTRADA variable(s): {', '.join(unknown)}. Allowed: {', '.join(HOSTRADA_VARIABLES)}.")
    unsupported = [v for v in result if v not in available]
    if unsupported:
        name = hs.get_provider(provider).name if provider is not None else hs.get_provider().name
        raise NotImplementedError(f"Provider {name!r} does not provide: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(result))

def _find_column(frame: pd.DataFrame, logical_name: str) -> str:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for alias in _COLUMN_ALIASES[logical_name]:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(f"Column for '{logical_name}' is missing. Supported names: {', '.join(_COLUMN_ALIASES[logical_name])}.")

def _normalise_route_frame(route, timezone):
    frame = route.copy() if isinstance(route, pd.DataFrame) else pd.read_csv(Path(route))
    if frame.empty:
        raise ValueError("The route table does not contain any intermediate points.")
    time_col = _find_column(frame, "timestamp")
    lon_col = _find_column(frame, "longitude")
    lat_col = _find_column(frame, "latitude")
    parsed = pd.to_datetime(frame[time_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError("At least one timestamp is invalid.")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    else:
        parsed = parsed.dt.tz_convert(timezone)
    frame[time_col] = parsed
    frame[lon_col] = pd.to_numeric(frame[lon_col], errors="coerce")
    frame[lat_col] = pd.to_numeric(frame[lat_col], errors="coerce")
    invalid = frame[lon_col].isna() | frame[lat_col].isna() | ~frame[lon_col].between(-180,180) | ~frame[lat_col].between(-90,90)
    if invalid.any():
        raise ValueError("At least one coordinate is invalid.")
    return frame, time_col, lon_col, lat_col

def _prepare_series(data: pd.DataFrame, variable: str):
    if data.empty or variable not in data.columns:
        raise RuntimeError(f"HOSTRADA did not return any values for '{variable}'.")
    times = pd.to_datetime(data["time"], errors="coerce", utc=True)
    values = pd.to_numeric(data[variable], errors="coerce")
    valid = times.notna() & values.notna()
    if not valid.any():
        raise RuntimeError(f"HOSTRADA values for '{variable}' cannot be evaluated.")
    unit = ""
    if "unit" in data.columns:
        units = data.loc[valid, "unit"].dropna()
        if not units.empty:
            unit = str(units.iloc[0])
    series = pd.Series(values[valid].to_numpy(float), index=times[valid])
    return series[~series.index.duplicated(keep="first")].sort_index(), unit

def _interpolate(series, target_utc, method, cyclic=False):
    if series.empty:
        raise RuntimeError("No values are available for interpolation.")
    if method not in {"linear", "nearest"}:
        raise ValueError("interpolation must be either 'linear' or 'nearest'.")
    if method == "nearest" or len(series) == 1:
        pos = int(np.argmin(np.abs((series.index-target_utc).total_seconds())))
        source = series.index[pos]
        return float(series.iloc[pos]), source, source, 0.0
    before, after = series.loc[series.index<=target_utc], series.loc[series.index>=target_utc]
    if before.empty:
        source=series.index[0]; return float(series.iloc[0]),source,source,0.0
    if after.empty:
        source=series.index[-1]; return float(series.iloc[-1]),source,source,0.0
    t0,t1=before.index[-1],after.index[0]; v0,v1=float(before.iloc[-1]),float(after.iloc[0])
    if t0==t1: return v0,t0,t1,0.0
    fraction=(target_utc-t0).total_seconds()/(t1-t0).total_seconds()
    if cyclic:
        delta=((v1-v0+180.0)%360.0)-180.0; value=(v0+fraction*delta)%360.0
    else: value=v0+fraction*(v1-v0)
    return float(value),t0,t1,float(fraction)

def _convert_value(value, source_unit, variable):
    conversion=HOSTRADA_VARIABLES[variable]["conversion"]
    unit=str(source_unit or "").strip().lower().replace("°","")
    if conversion=="temperature": return value-273.15 if unit in {"k","kelvin"} or value>150 else value
    if conversion=="pressure": return value/100.0 if unit in {"pa","pascal","pascals"} or value>2000 else value
    if conversion=="percent": return value*100.0 if 0.0<=value<=1.0 and unit not in {"%","percent"} else value
    if conversion=="degrees": return value%360.0
    return value

class HostradaRouteClimateCalculator:
    def __init__(self, config: Optional[RouteClimateConfig] = None):
        self.config = config or RouteClimateConfig()
        self.variables = _validate_variables(self.config.variables, self.config.provider)
        self._cache = {}
        self._prefetched_months: set[tuple[str, int, int, str]] = set()

    def _provider_name(self) -> str:
        return hs.get_provider_name(self.config.provider)

    def _effective_cache_strategy(self) -> Optional[str]:
        """Use one shared DWD monthly file for all points on a route.

        Point subsets are useful for isolated point requests, but a route creates
        many different point selectors for the same variable and month. Keeping
        the complete DWD month in the cache prevents an identical file from being
        downloaded once per route point. Other providers retain their configured
        strategy.
        """
        strategy = self.config.cache_strategy
        if self._provider_name() == "dwd" and strategy in {None, "subset"}:
            return "full"
        return strategy

    def _prefetch_shared_month_files(self, frame: pd.DataFrame, time_col: str) -> None:
        if self._provider_name() != "dwd":
            return
        if self._effective_cache_strategy() != "full":
            return

        start_utc = pd.Timestamp(frame[time_col].min()).tz_convert("UTC").floor("h")
        end_utc = pd.Timestamp(frame[time_col].max()).tz_convert("UTC").ceil("h")
        if end_utc == start_utc:
            end_utc += pd.Timedelta(hours=1)

        for variable in self.variables:
            for year, month in hs.month_range(start_utc, end_utc):
                key = (variable, year, month, self._provider_name())
                if key in self._prefetched_months:
                    continue
                hs.ensure_month_file(
                    variable,
                    year,
                    month,
                    Path(self.config.cache_dir),
                    verbose=True,
                    provider=self.config.provider,
                )
                self._prefetched_months.add(key)

    def _load_window(self, variable, longitude, latitude, target_utc):
        start_utc = target_utc.floor("h")
        end_utc = target_utc.ceil("h")
        if end_utc == start_utc:
            end_utc += pd.Timedelta(hours=1)
        d = self.config.coordinate_decimals
        key = (
            variable,
            round(float(longitude), d),
            round(float(latitude), d),
            start_utc.isoformat(),
            end_utc.isoformat(),
            str(self.config.provider),
        )
        if key not in self._cache:
            self._cache[key] = extract_values_for_point(
                variable,
                lon=float(longitude),
                lat=float(latitude),
                start=start_utc.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_utc.strftime("%Y-%m-%d %H:%M:%S"),
                cache_dir=Path(self.config.cache_dir),
                cache_strategy=self._effective_cache_strategy(),
                subset_margin_cells=self.config.subset_margin_cells,
                provider=self.config.provider,
                verbose=False,
            )
        return self._cache[key].copy()
    def climate_value_at(self, variable, longitude, latitude, timestamp):
        local_time=pd.Timestamp(timestamp)
        local_time=local_time.tz_localize(self.config.timezone) if local_time.tzinfo is None else local_time.tz_convert(self.config.timezone)
        target_utc=local_time.tz_convert("UTC")
        raw=self._load_window(variable,longitude,latitude,target_utc)
        series,unit=_prepare_series(raw,variable)
        raw_value,t0,t1,fraction=_interpolate(series,target_utc,self.config.interpolation,HOSTRADA_VARIABLES[variable]["cyclic"])
        value=_convert_value(raw_value,unit,variable); first=raw.iloc[0]; meta=HOSTRADA_VARIABLES[variable]
        return {meta["output_column"]:round(value,4),f"{variable}_raw":float(raw_value),f"{variable}_source_unit":unit,
                f"{variable}_display_unit":meta["display_unit"],f"{variable}_time_before_utc":t0.isoformat(),
                f"{variable}_time_after_utc":t1.isoformat(),f"{variable}_interpolation_fraction":round(fraction,6),
                f"{variable}_grid_lon":first.get("grid_lon",np.nan),f"{variable}_grid_lat":first.get("grid_lat",np.nan)}
    def calculate(self, route, output_csv=None, progress_callback=None, continue_on_error=False):
        frame,time_col,lon_col,lat_col=_normalise_route_frame(route,self.config.timezone)
        self._prefetch_shared_month_files(frame, time_col)
        rows=[]; total=len(frame)*len(self.variables); step=0
        for _,row in frame.iterrows():
            result={}; errors=[]
            for variable in self.variables:
                step+=1
                if progress_callback: progress_callback(step,total,row,variable)
                try: result.update(self.climate_value_at(variable,float(row[lon_col]),float(row[lat_col]),row[time_col]))
                except Exception as error:
                    if not continue_on_error: raise RuntimeError(f"Calculation of '{variable}' failed: {error}") from error
                    result[HOSTRADA_VARIABLES[variable]["output_column"]]=np.nan; errors.append(f"{variable}: {error}")
            result["hostrada_error"]=" | ".join(errors); rows.append(result)
        output=pd.concat([frame.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True,exist_ok=True); output.to_csv(output_csv,index=False,encoding="utf-8")
        return output

def calculate_route_climate(route, variables="all", output_csv=None, timezone="Europe/Berlin", interpolation="linear",
                            cache_dir="hostrada_cache", cache_strategy="full", subset_margin_cells=0,
                            continue_on_error=False, progress_callback=None, provider=None, **legacy_kwargs):
    """Calculate one or more climate variables along the route.

    ``output_file`` from an early development version is accepted as an alias,
    but the 0.42.0 public argument remains ``output_csv``.
    """
    if output_csv is None and "output_file" in legacy_kwargs: output_csv=legacy_kwargs.pop("output_file")
    if legacy_kwargs: raise TypeError(f"Unexpected keyword arguments: {', '.join(legacy_kwargs)}")
    selected=_validate_variables(variables,provider)
    calculator=HostradaRouteClimateCalculator(RouteClimateConfig(selected,timezone,interpolation,Path(cache_dir),cache_strategy,subset_margin_cells,6,provider))
    return calculator.calculate(route,output_csv,progress_callback,continue_on_error)

calculate_hostrada_values_along_route=calculate_route_climate
extract_values_for_route=calculate_route_climate

def _build_parser():
    p=argparse.ArgumentParser(description="Calculate HOSTRADA climate variables along a route.")
    p.add_argument("route_csv");p.add_argument("-o","--output",default="route_positions_climate.csv")
    p.add_argument("-v","--variables",nargs="+",default=["all"]);p.add_argument("--timezone",default="Europe/Berlin")
    p.add_argument("--interpolation",choices=["linear","nearest"],default="linear")
    p.add_argument("--cache-strategy",choices=["full","subset","http_range","auto"],default="full")
    p.add_argument("--continue-on-error",action="store_true");p.add_argument("--provider",default=None)
    return p

def main():
    args=_build_parser().parse_args(); variables="all" if args.variables==["all"] else args.variables
    def progress(step,total,_row,variable): print(f"\r{step}/{total}: {variable}",end="",flush=True)
    result=calculate_route_climate(args.route_csv,variables,args.output,args.timezone,args.interpolation,
                                   cache_strategy=args.cache_strategy,continue_on_error=args.continue_on_error,
                                   progress_callback=progress,provider=args.provider)
    print(f"\n{len(result)} route points saved: {Path(args.output).resolve()}")
if __name__=="__main__": main()
