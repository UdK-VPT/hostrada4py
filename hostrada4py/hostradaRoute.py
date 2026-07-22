#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HOSTRADA climate profile along a time-defined route.

The module reads a route CSV containing timestamps, longitude, and latitude,
and adds one or more HOSTRADA climate variables for every intermediate point.

Supported variables
-------------------
tas               Air temperature at 2 m
tdew              Dew point temperature at 2 m
uhi               Urban heat island intensity
sfcWind           Wind speed at 10 m
sfcWind_direction Wind direction at 10 m
rsds              Global radiation
clt               Cloud cover
hurs              Relative humidity at 2 m
mixr              Water vapor mixing ratio at 2 m
ps                Air pressure at station elevation
psl               Air pressure at sea level

HOSTRADA data are hourly. Values between two hours are interpolated either
linearly or by using the temporally nearest value. For cyclic wind direction,
linear interpolation follows the shortest angular path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    from hostrada4py.hostradaPoint import extract_values_for_point
except ImportError as exc:
    raise ImportError(
        "hostrada4py could not be imported. Install the library "
        "or place this module next to the hostrada4py package."
    ) from exc


PathLike = Union[str, Path]
ProgressCallback = Callable[[int, int, pd.Series, str], None]


HOSTRADA_VARIABLES = {
    "tas": {
        "label": "Air temperature (2 m)",
        "short_label": "Air temperature",
        "output_column": "air_temperature_c",
        "display_unit": "°C",
        "conversion": "temperature",
        "cyclic": False,
    },
    "tdew": {
        "label": "Dew point temperature (2 m)",
        "short_label": "Dew point temperature",
        "output_column": "dew_point_temperature_c",
        "display_unit": "°C",
        "conversion": "temperature",
        "cyclic": False,
    },
    "uhi": {
        "label": "Urban heat island intensity",
        "short_label": "UHI intensity",
        "output_column": "urban_heat_island_intensity_k",
        "display_unit": "K",
        "conversion": "identity",
        "cyclic": False,
    },
    "sfcWind": {
        "label": "Wind speed (10 m)",
        "short_label": "Wind speed",
        "output_column": "wind_speed_m_s",
        "display_unit": "m/s",
        "conversion": "identity",
        "cyclic": False,
    },
    "sfcWind_direction": {
        "label": "Wind direction (10 m)",
        "short_label": "Wind direction",
        "output_column": "wind_direction_deg",
        "display_unit": "°",
        "conversion": "degrees",
        "cyclic": True,
    },
    "rsds": {
        "label": "Global radiation",
        "short_label": "Global radiation",
        "output_column": "global_radiation_w_m2",
        "display_unit": "W/m²",
        "conversion": "identity",
        "cyclic": False,
    },
    "clt": {
        "label": "Cloud cover",
        "short_label": "Cloud cover",
        "output_column": "cloud_cover_percent",
        "display_unit": "%",
        "conversion": "percent",
        "cyclic": False,
    },
    "hurs": {
        "label": "Relative humidity (2 m)",
        "short_label": "Relative humidity",
        "output_column": "relative_humidity_percent",
        "display_unit": "%",
        "conversion": "percent",
        "cyclic": False,
    },
    "mixr": {
        "label": "Water vapor mixing ratio (2 m)",
        "short_label": "Mixing ratio",
        "output_column": "water_vapor_mixing_ratio",
        "display_unit": "kg/kg",
        "conversion": "identity",
        "cyclic": False,
    },
    "ps": {
        "label": "Air pressure at station elevation",
        "short_label": "Station pressure",
        "output_column": "surface_pressure_hpa",
        "display_unit": "hPa",
        "conversion": "pressure",
        "cyclic": False,
    },
    "psl": {
        "label": "Air pressure at sea level",
        "short_label": "Sea-level pressure",
        "output_column": "sea_level_pressure_hpa",
        "display_unit": "hPa",
        "conversion": "pressure",
        "cyclic": False,
    },
}


@dataclass(frozen=True)
class RouteClimateConfig:
    variables: tuple[str, ...] = ("tas",)
    timezone: str = "Europe/Berlin"
    interpolation: str = "linear"
    cache_dir: Path = Path("hostrada_cache")
    cache_strategy: Optional[str] = "subset"
    subset_margin_cells: Optional[int] = 0
    coordinate_decimals: int = 6


_COLUMN_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime", "date_time"),
    "longitude": ("longitude", "lon", "lng", "x"),
    "latitude": ("latitude", "lat", "y"),
}


def available_variables() -> pd.DataFrame:
    """Returns all supported HOSTRADA variables as a table."""
    return pd.DataFrame(
        [
            {
                "variable": code,
                "description": meta["label"],
                "output_column": meta["output_column"],
                "display_unit": meta["display_unit"],
            }
            for code, meta in HOSTRADA_VARIABLES.items()
        ]
    )


def _validate_variables(variables: Union[str, Iterable[str]]) -> tuple[str, ...]:
    if isinstance(variables, str):
        if variables.lower() == "all":
            result = tuple(HOSTRADA_VARIABLES)
        else:
            result = (variables,)
    else:
        result = tuple(variables)

    if not result:
        raise ValueError("At least one HOSTRADA variable must be selected.")

    unknown = [variable for variable in result if variable not in HOSTRADA_VARIABLES]
    if unknown:
        raise ValueError(
            f"Unknown HOSTRADA variable(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(HOSTRADA_VARIABLES)}."
        )
    return tuple(dict.fromkeys(result))


def _find_column(frame: pd.DataFrame, logical_name: str) -> str:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for alias in _COLUMN_ALIASES[logical_name]:
        if alias in lookup:
            return lookup[alias]
    raise ValueError(
        f"Column for '{logical_name}' is missing. Supported names: "
        f"{', '.join(_COLUMN_ALIASES[logical_name])}."
    )


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
        parsed = parsed.dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    else:
        parsed = parsed.dt.tz_convert(timezone)

    frame[time_col] = parsed
    frame[lon_col] = pd.to_numeric(frame[lon_col], errors="coerce")
    frame[lat_col] = pd.to_numeric(frame[lat_col], errors="coerce")

    invalid = (
        frame[lon_col].isna()
        | frame[lat_col].isna()
        | ~frame[lon_col].between(-180, 180)
        | ~frame[lat_col].between(-90, 90)
    )
    if invalid.any():
        raise ValueError("At least one coordinate is invalid.")

    return frame, time_col, lon_col, lat_col


def _prepare_series(data: pd.DataFrame, variable: str) -> tuple[pd.Series, str]:
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
        position = int(np.argmin(np.abs((series.index - target_utc).total_seconds())))
        source = series.index[position]
        return float(series.iloc[position]), source, source, 0.0

    before = series.loc[series.index <= target_utc]
    after = series.loc[series.index >= target_utc]
    if before.empty:
        source = series.index[0]
        return float(series.iloc[0]), source, source, 0.0
    if after.empty:
        source = series.index[-1]
        return float(series.iloc[-1]), source, source, 0.0

    t0, t1 = before.index[-1], after.index[0]
    v0, v1 = float(before.iloc[-1]), float(after.iloc[0])
    if t0 == t1:
        return v0, t0, t1, 0.0

    fraction = (target_utc - t0).total_seconds() / (t1 - t0).total_seconds()
    if cyclic:
        delta = ((v1 - v0 + 180.0) % 360.0) - 180.0
        value = (v0 + fraction * delta) % 360.0
    else:
        value = v0 + fraction * (v1 - v0)
    return float(value), t0, t1, float(fraction)


def _convert_value(value: float, source_unit: str, variable: str) -> float:
    meta = HOSTRADA_VARIABLES[variable]
    conversion = meta["conversion"]
    unit = str(source_unit or "").strip().lower().replace("°", "")

    if conversion == "temperature":
        return value - 273.15 if unit in {"k", "kelvin"} or value > 150 else value

    if conversion == "pressure":
        return value / 100.0 if unit in {"pa", "pascal", "pascals"} or value > 2000 else value

    if conversion == "percent":
        return value * 100.0 if 0.0 <= value <= 1.0 and unit not in {"%", "percent"} else value

    if conversion == "degrees":
        return value % 360.0

    return value


class HostradaRouteClimateCalculator:
    """Calculates selected HOSTRADA climate variables for route points."""

    def __init__(self, config: Optional[RouteClimateConfig] = None) -> None:
        self.config = config or RouteClimateConfig()
        self.variables = _validate_variables(self.config.variables)
        self._cache: dict[tuple[str, float, float, str, str], pd.DataFrame] = {}

    def _load_window(self, variable, longitude, latitude, target_utc):
        start_utc = target_utc.floor("h")
        end_utc = target_utc.ceil("h")
        if end_utc == start_utc:
            end_utc += pd.Timedelta(hours=1)

        decimals = self.config.coordinate_decimals
        key = (
            variable,
            round(float(longitude), decimals),
            round(float(latitude), decimals),
            start_utc.isoformat(),
            end_utc.isoformat(),
        )
        if key not in self._cache:
            self._cache[key] = extract_values_for_point(
                variable,
                lon=float(longitude),
                lat=float(latitude),
                start=start_utc.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_utc.strftime("%Y-%m-%d %H:%M:%S"),
                cache_dir=Path(self.config.cache_dir),
                cache_strategy=self.config.cache_strategy,
                subset_margin_cells=self.config.subset_margin_cells,
            )
        return self._cache[key].copy()

    def climate_value_at(self, variable, longitude, latitude, timestamp):
        if variable not in HOSTRADA_VARIABLES:
            raise ValueError(f"Unknown variable: {variable}")

        local_time = pd.Timestamp(timestamp)
        if local_time.tzinfo is None:
            local_time = local_time.tz_localize(self.config.timezone)
        else:
            local_time = local_time.tz_convert(self.config.timezone)
        target_utc = local_time.tz_convert("UTC")

        raw = self._load_window(variable, longitude, latitude, target_utc)
        series, source_unit = _prepare_series(raw, variable)
        raw_value, t0, t1, fraction = _interpolate(
            series,
            target_utc,
            self.config.interpolation,
            cyclic=HOSTRADA_VARIABLES[variable]["cyclic"],
        )
        value = _convert_value(raw_value, source_unit, variable)
        first = raw.iloc[0]

        prefix = variable
        return {
            HOSTRADA_VARIABLES[variable]["output_column"]: round(value, 4),
            f"{prefix}_raw": float(raw_value),
            f"{prefix}_source_unit": source_unit,
            f"{prefix}_display_unit": HOSTRADA_VARIABLES[variable]["display_unit"],
            f"{prefix}_time_before_utc": t0.isoformat(),
            f"{prefix}_time_after_utc": t1.isoformat(),
            f"{prefix}_interpolation_fraction": round(fraction, 6),
            f"{prefix}_grid_lon": first.get("grid_lon", np.nan),
            f"{prefix}_grid_lat": first.get("grid_lat", np.nan),
        }

    def calculate(
        self,
        route,
        output_csv: Optional[PathLike] = None,
        progress_callback: Optional[ProgressCallback] = None,
        continue_on_error: bool = False,
    ) -> pd.DataFrame:
        frame, time_col, lon_col, lat_col = _normalise_route_frame(
            route, self.config.timezone
        )
        rows = []
        total_steps = len(frame) * len(self.variables)
        step = 0

        for _, row in frame.iterrows():
            result = {}
            errors = []
            for variable in self.variables:
                step += 1
                if progress_callback:
                    progress_callback(step, total_steps, row, variable)
                try:
                    result.update(
                        self.climate_value_at(
                            variable,
                            float(row[lon_col]),
                            float(row[lat_col]),
                            row[time_col],
                        )
                    )
                except Exception as error:
                    if not continue_on_error:
                        raise RuntimeError(
                            f"Calculation of '{variable}' failed: {error}"
                        ) from error
                    result[HOSTRADA_VARIABLES[variable]["output_column"]] = np.nan
                    errors.append(f"{variable}: {error}")
            result["hostrada_error"] = " | ".join(errors)
            rows.append(result)

        output = pd.concat(
            [frame.reset_index(drop=True), pd.DataFrame(rows)],
            axis=1,
        )
        if output_csv is not None:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_csv, index=False, encoding="utf-8")
        return output


def calculate_route_climate(
    route,
    variables: Union[str, Sequence[str]] = "all",
    output_csv: Optional[PathLike] = None,
    timezone: str = "Europe/Berlin",
    interpolation: str = "linear",
    cache_dir: PathLike = "hostrada_cache",
    cache_strategy: Optional[str] = "subset",
    subset_margin_cells: Optional[int] = 0,
    continue_on_error: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> pd.DataFrame:
    """Calculates one or more climate variables along the route."""
    selected = _validate_variables(variables)
    calculator = HostradaRouteClimateCalculator(
        RouteClimateConfig(
            variables=selected,
            timezone=timezone,
            interpolation=interpolation,
            cache_dir=Path(cache_dir),
            cache_strategy=cache_strategy,
            subset_margin_cells=subset_margin_cells,
        )
    )
    return calculator.calculate(
        route,
        output_csv=output_csv,
        progress_callback=progress_callback,
        continue_on_error=continue_on_error,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Calculate HOSTRADA climate variables along a route."
    )
    parser.add_argument("route_csv")
    parser.add_argument("-o", "--output", default="route_positions_climate.csv")
    parser.add_argument(
        "-v",
        "--variables",
        nargs="+",
        default=["all"],
        help="Variable codes or 'all'.",
    )
    parser.add_argument("--timezone", default="Europe/Berlin")
    parser.add_argument("--interpolation", choices=["linear", "nearest"], default="linear")
    parser.add_argument(
        "--cache-strategy",
        choices=["full", "subset", "http_range", "auto"],
        default="subset",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args()
    variables = "all" if args.variables == ["all"] else args.variables

    def progress(step, total, _row, variable):
        print(f"\r{step}/{total}: {variable}", end="", flush=True)

    result = calculate_route_climate(
        args.route_csv,
        variables=variables,
        output_csv=args.output,
        timezone=args.timezone,
        interpolation=args.interpolation,
        cache_strategy=args.cache_strategy,
        continue_on_error=args.continue_on_error,
        progress_callback=progress,
    )
    print(f"\n{len(result)} route points saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
