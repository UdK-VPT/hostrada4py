from __future__ import annotations

import calendar
import json
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from .base import ProviderCapabilities, TimeoutValue, WeatherProvider
from .common import (
    SUBSET_MARGIN_CELLS_DEFAULT,
    bbox_3034_to_lonlat,
    cache_key,
    file_lock,
    is_cached_file,
    point_to_epsg3034,
    write_netcdf_atomic,
)

CERRA_DATASET = os.getenv("HOSTRADA_CERRA_DATASET", "reanalysis-cerra-single-levels")
CERRA_GRID_SIZE_M = float(os.getenv("HOSTRADA_CERRA_TARGET_GRID_SIZE", "5500"))
CERRA_ANALYSIS_TIMES = tuple(f"{hour:02d}:00" for hour in range(0, 24, 3))
CERRA_FORECAST_TIMES = CERRA_ANALYSIS_TIMES
CERRA_FORECAST_LEADS = ("1", "2", "3")
CERRA_CONFIG_FILENAME = "cerra_config.json"


def _cerra_config_path() -> Path:
    """Return the package-local CERRA configuration file."""
    return Path(__file__).resolve().parents[1] / CERRA_CONFIG_FILENAME


def _load_cerra_config() -> dict[str, Any]:
    """Read optional CDS credentials from ``hostrada4py/cerra_config.json``.

    Empty values deliberately fall back to the normal cdsapi configuration
    (``~/.cdsapirc`` and environment variables).
    """
    path = _cerra_config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid CERRA configuration file: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"CERRA configuration must be a JSON object: {path}")
    return data


@dataclass(frozen=True)
class CerraVariable:
    cds_variables: tuple[str, ...]
    product_type: str = "analysis"
    short_names: tuple[str, ...] = ()
    long_name: str = ""
    units: str = ""
    accumulated: bool = False


# CDS names are isolated here intentionally. They can be overridden without
# touching the provider or the public API if the CDS catalogue changes labels.
CERRA_VARIABLES: dict[str, CerraVariable] = {
    "tas": CerraVariable(
        ("2m_temperature",), "analysis", ("t2m", "2t"),
        "2 metre air temperature", "degC"
    ),
    "hurs": CerraVariable(
        ("2m_relative_humidity",), "analysis", ("r2", "2r", "rh2m"),
        "2 metre relative humidity", "%"
    ),
    "sfcWind": CerraVariable(
        ("10m_wind_speed",), "analysis", ("si10", "10si", "ws10"),
        "10 metre wind speed", "m s-1"
    ),
    "sfcWind_direction": CerraVariable(
        ("10m_wind_direction",), "analysis", ("wdir10", "10wdir", "wd10"),
        "10 metre wind direction", "degree"
    ),
    "ps": CerraVariable(
        ("surface_pressure",), "analysis", ("sp",),
        "surface air pressure", "hPa"
    ),
    "psl": CerraVariable(
        ("mean_sea_level_pressure",), "analysis", ("msl", "prmsl"),
        "mean sea level pressure", "hPa"
    ),
    "clt": CerraVariable(
        ("total_cloud_cover",), "analysis", ("tcc",),
        "total cloud cover", "%"
    ),
    "rsds": CerraVariable(
        ("surface_solar_radiation_downwards",), "forecast", ("ssrd",),
        "surface downwelling shortwave radiation", "W m-2", accumulated=True
    ),
    # Derived from native CERRA fields.
    "tdew": CerraVariable(
        ("2m_temperature", "2m_relative_humidity"), "analysis",
        (), "2 metre dew-point temperature", "degC"
    ),
    "mixr": CerraVariable(
        ("2m_temperature", "2m_relative_humidity", "surface_pressure"),
        "analysis", (), "water-vapour mixing ratio", "g kg-1"
    ),
}

# Optional environment override, useful if a CDS catalogue deployment uses a
# different display/API label. Example:
# HOSTRADA_CERRA_VARIABLE_OVERRIDES='{"rsds":"surface_solar_radiation_downwards"}'
def _variable_overrides() -> dict[str, str]:
    raw = os.getenv("HOSTRADA_CERRA_VARIABLE_OVERRIDES", "")
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("HOSTRADA_CERRA_VARIABLE_OVERRIDES must be a JSON object")
    return {str(k): str(v) for k, v in parsed.items()}


class CERRAProvider(WeatherProvider):
    """Copernicus European Regional Reanalysis provider.

    The provider downloads CERRA GRIB data through the CDS API, converts its
    native rotated/Lambert grid to a small rectilinear EPSG:3034 target raster,
    normalises variable names and units to HOSTRADA conventions, and stores a
    regular NetCDF cache. Existing hostradaPoint/Area/Route code then sees the
    same shape as a DWD monthly file.
    """

    name = "cerra"

    def __init__(self, client: Any = None, target_grid_size_m: float = CERRA_GRID_SIZE_M):
        self._client = client
        self.target_grid_size_m = float(target_grid_size_m)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            variables=frozenset(CERRA_VARIABLES),
            temporal_resolution="1 hour after normalisation (3-hour analyses interpolated)",
            spatial_resolution_m=self.target_grid_size_m,
            crs="EPSG:3034",
            start="1984-09",
            notes=(
                "CERRA analysis fields are available every three hours.",
                "Hourly forecast accumulations are used for downwelling solar radiation.",
                "uhi is intentionally unsupported because CERRA has no HOSTRADA-equivalent UHI field.",
            ),
        )

    def filename(self, var: str, year: int, month: int) -> str:
        self.require_variable(var)
        return f"{var}_1hr_CERRA-v1_EUR_EPSG3034_{year:04d}{month:02d}.nc"

    def _cds_client(self):
        if self._client is not None:
            return self._client
        try:
            import cdsapi
        except ImportError as exc:
            raise ImportError(
                "CERRA access requires the optional dependency 'cdsapi'. "
                "Install requirements-cerra.txt and configure "
                "hostrada4py/cerra_config.json or ~/.cdsapirc."
            ) from exc

        config = _load_cerra_config()
        kwargs: dict[str, Any] = {}
        url = str(config.get("url", "")).strip()
        key = str(config.get("key", "")).strip()
        if url:
            kwargs["url"] = url
        if key:
            kwargs["key"] = key
        self._client = cdsapi.Client(**kwargs)
        return self._client

    def _spec(self, var: str) -> CerraVariable:
        self.require_variable(var)
        spec = CERRA_VARIABLES[var]
        overrides = _variable_overrides()
        names = tuple(overrides.get(name, name) for name in spec.cds_variables)
        return CerraVariable(
            names,
            spec.product_type,
            spec.short_names,
            spec.long_name,
            spec.units,
            spec.accumulated,
        )

    def build_request(
        self,
        var: str,
        year: int,
        month: int,
        area_lonlat: Optional[tuple[float, float, float, float]] = None,
    ) -> dict[str, Any]:
        """Build a request accepted by the current CDS CERRA process.

        ``area_lonlat`` is retained only for API compatibility. CERRA is stored
        on a Lambert conformal grid and the CDS retrieval process does not
        expose geographic subsetting. The native domain must therefore be
        retrieved and cropped locally.
        """
        del area_lonlat
        spec = self._spec(var)
        request: dict[str, Any] = {
            "variable": list(dict.fromkeys(spec.cds_variables)),
            "level_type": "surface_or_atmosphere",
            "data_type": ["reanalysis"],
            "product_type": spec.product_type,
            "year": [f"{year:04d}"],
            "month": [f"{month:02d}"],
            "day": [f"{day:02d}" for day in range(1, calendar.monthrange(year, month)[1] + 1)],
            "time": list(CERRA_FORECAST_TIMES if spec.product_type == "forecast" else CERRA_ANALYSIS_TIMES),
            "data_format": "grib",
        }
        if spec.product_type == "forecast":
            request["leadtime_hour"] = list(CERRA_FORECAST_LEADS)
        return request

    def _target_axes(
        self,
        selector: Mapping[str, Any],
        margin_cells: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolution = self.target_grid_size_m
        if selector["type"] == "point_epsg3034":
            x = float(selector["x"])
            y = float(selector["y"])
            x0 = round(x / resolution) * resolution
            y0 = round(y / resolution) * resolution
            offsets = np.arange(-margin_cells, margin_cells + 1, dtype=float) * resolution
            return x0 + offsets, y0 + offsets
        if selector["type"] == "bbox_epsg3034":
            minx, miny, maxx, maxy = map(float, selector["bbox"])
            x0 = math.floor(minx / resolution) * resolution - margin_cells * resolution
            x1 = math.ceil(maxx / resolution) * resolution + margin_cells * resolution
            y0 = math.floor(miny / resolution) * resolution - margin_cells * resolution
            y1 = math.ceil(maxy / resolution) * resolution + margin_cells * resolution
            return (
                np.arange(x0, x1 + resolution * 0.5, resolution, dtype=float),
                np.arange(y0, y1 + resolution * 0.5, resolution, dtype=float),
            )
        raise ValueError(f"Unknown CERRA selector: {selector}")

    def _selector_area_lonlat(
        self,
        selector: Mapping[str, Any],
        margin_cells: int,
    ) -> tuple[float, float, float, float]:
        x, y = self._target_axes(selector, margin_cells + 2)
        bbox = (
            float(x.min() - self.target_grid_size_m / 2),
            float(y.min() - self.target_grid_size_m / 2),
            float(x.max() + self.target_grid_size_m / 2),
            float(y.max() + self.target_grid_size_m / 2),
        )
        return bbox_3034_to_lonlat(bbox, pad_deg=0.2)

    def _retrieve_grib(
        self,
        var: str,
        year: int,
        month: int,
        area_lonlat: tuple[float, float, float, float],
        cache_dir: Path,
        verbose: bool,
    ) -> Path:
        # Geographic subsetting is intentionally not sent to CDS. The current
        # CERRA process has no ``area`` input and MARS/MIR fails when one is
        # injected for the native Lambert grid. The requested area is used only
        # after download for local extraction.
        request = self.build_request(var, year, month)
        request_hash = cache_key(request, 20)
        raw_target = (
            Path(cache_dir)
            / self.name
            / "raw"
            / f"{var}_{year:04d}{month:02d}_{request_hash}.grib"
        )
        if is_cached_file(raw_target):
            if verbose:
                print(f"CERRA native-domain cache: {raw_target}")
            return raw_target
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        part = raw_target.with_name(raw_target.name + ".part")
        with file_lock(raw_target.with_name(raw_target.name + ".lock")):
            if is_cached_file(raw_target):
                return raw_target
            if part.exists():
                part.unlink()
            if verbose:
                west, south, east, north = area_lonlat
                print(
                    f"CDS/CERRA request: {var} {year:04d}-{month:02d}, "
                    "native European domain; local subset after download "
                    f"[{west:.4f}, {south:.4f}, {east:.4f}, {north:.4f}]"
                )
                print(
                    "Note: CDS currently does not support area subsetting for "
                    "the native CERRA Lambert grid. The raw monthly file can be "
                    "large, but is cached and reused for other locations."
                )
            try:
                self._cds_client().retrieve(CERRA_DATASET, request, str(part))
                if not part.exists():
                    raise RuntimeError(
                        "The CDS API completed without creating the requested GRIB file."
                    )
                if zipfile.is_zipfile(part):
                    # CDS may return one GRIB per selected field. A GRIB file is
                    # a sequence of messages, so concatenating members preserves
                    # all fields and is directly readable by cfgrib.
                    with zipfile.ZipFile(part) as archive:
                        members = [
                            name for name in archive.namelist()
                            if not name.endswith("/")
                            and name.lower().endswith((".grib", ".grb", ".grib2"))
                        ]
                        if not members:
                            raise RuntimeError(
                                "The CDS response was a ZIP archive without GRIB files."
                            )
                        combined = part.with_name(part.name + ".combined")
                        with combined.open("wb") as destination:
                            for name in members:
                                with archive.open(name) as source:
                                    while True:
                                        chunk = source.read(1024 * 1024)
                                        if not chunk:
                                            break
                                        destination.write(chunk)
                    part.unlink()
                    combined.replace(raw_target)
                else:
                    with part.open("rb") as stream:
                        magic = stream.read(4)
                    if magic != b"GRIB":
                        raise RuntimeError(
                            "The CDS response is neither GRIB nor a ZIP containing GRIB "
                            f"(file signature {magic!r})."
                        )
                    part.replace(raw_target)
            except Exception:
                if part.exists():
                    part.unlink()
                raise
        return raw_target

    @staticmethod
    def _open_grib_datasets(path: Path) -> list[xr.Dataset]:
        """Open CERRA GRIB files without triggering xarray's step decoder bug.

        xarray versions up to 2025.6.x can fail while decoding the GRIB
        ``step`` coordinate when cfgrib supplies both a timedelta-like unit and
        a ``dtype`` attribute. Keeping ``step`` numeric is sufficient here:
        CERRA lead times are expressed in hours and are handled explicitly in
        :meth:`_forecast_to_valid_time`.
        """
        try:
            import cfgrib
        except ImportError as exc:
            raise ImportError(
                "Reading CERRA GRIB requires 'cfgrib' and the ecCodes runtime."
            ) from exc

        datasets = list(
            cfgrib.open_datasets(
                str(path),
                backend_kwargs={"indexpath": ""},
                decode_timedelta=False,
                #decode_timedelta=True,
            )
        )

        # With timedelta decoding disabled, cfgrib/xarray may leave ``dtype``
        # in a coordinate's attributes. It is serialization metadata rather
        # than user metadata and can otherwise cause the same collision when
        # the normalized dataset is written to NetCDF.
        for ds in datasets:
            for variable in ds.variables.values():
                variable.attrs.pop("dtype", None)

        return datasets

    @staticmethod
    def _pick_dataarray(datasets: Iterable[xr.Dataset], candidates: Sequence[str]) -> xr.DataArray:
        datasets = list(datasets)
        for candidate in candidates:
            for ds in datasets:
                if candidate in ds.data_vars:
                    return ds[candidate]
        all_vars = [(ds, name) for ds in datasets for name in ds.data_vars]
        if len(all_vars) == 1:
            ds, name = all_vars[0]
            return ds[name]
        candidate_text = ", ".join(candidates) or "<derived variable>"
        available = [name for _, name in all_vars]
        raise KeyError(
            f"Could not identify CERRA field ({candidate_text}). Available GRIB variables: {available}"
        )

    @staticmethod
    def _spatial_coordinates(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        lat = da.coords.get("latitude")
        lon = da.coords.get("longitude")
        if lat is None:
            lat = da.coords.get("lat")
        if lon is None:
            lon = da.coords.get("lon")
        if lat is None or lon is None:
            raise KeyError(
                f"CERRA field has no latitude/longitude coordinates: {list(da.coords)}"
            )
        return lat, lon

    @staticmethod
    def _forecast_to_valid_time(da: xr.DataArray, accumulated: bool) -> xr.DataArray:
        if "step" not in da.dims:
            return da
        if "time" not in da.dims:
            raise KeyError("Forecast CERRA field has a step dimension but no reference time.")
        spatial_dims = [dim for dim in da.dims if dim not in {"time", "step"}]
        work = da.transpose("time", "step", *spatial_dims)
        values = np.asarray(work.values, dtype=float)
        if accumulated:
            units = str(da.attrs.get("units", "")).lower()
            if "j" in units or "joule" in units or not units:
                differences = np.diff(values, axis=1, prepend=np.zeros_like(values[:, :1]))
                differences = np.maximum(differences, 0.0)
                step_values = np.asarray(work.coords["step"].values)
                if np.issubdtype(step_values.dtype, np.timedelta64):
                    seconds = step_values.astype("timedelta64[s]").astype(float)
                else:
                    seconds = np.asarray(step_values, dtype=float) * 3600.0
                durations = np.diff(seconds, prepend=0.0)
                durations[durations <= 0] = 3600.0
                reshape = (1, len(durations)) + (1,) * len(spatial_dims)
                values = differences / durations.reshape(reshape)
        if "valid_time" in work.coords:
            valid = np.asarray(work.coords["valid_time"].values)
        else:
            ref = np.asarray(work.coords["time"].values)[:, None]
            step = np.asarray(work.coords["step"].values)[None, :]
            valid = ref + step
        flat_values = values.reshape((-1,) + tuple(values.shape[2:]))
        flat_time = valid.reshape(-1)
        coords: dict[str, Any] = {"time": flat_time}
        for dim in spatial_dims:
            if dim in work.coords:
                coords[dim] = work.coords[dim]
        for name in ("latitude", "longitude", "lat", "lon"):
            if name in work.coords and set(work.coords[name].dims).issubset(spatial_dims):
                coords[name] = work.coords[name]
        result = xr.DataArray(
            flat_values,
            dims=("time", *spatial_dims),
            coords=coords,
            attrs=dict(da.attrs),
            name=da.name,
        )
        result = result.sortby("time")
        times = pd.Index(np.asarray(result.time.values))
        keep = ~times.duplicated(keep="first")
        return result.isel(time=np.flatnonzero(keep))

    @staticmethod
    def _to_hostrada_units(var: str, da: xr.DataArray) -> xr.DataArray:
        result = da.astype(float)
        units = str(da.attrs.get("units", "")).lower()
        sample = np.asarray(result.values)
        finite = sample[np.isfinite(sample)]
        median = float(np.median(finite)) if finite.size else float("nan")
        maximum = float(np.max(finite)) if finite.size else float("nan")
        if var == "tas":
            if units in {"k", "kelvin"} or median > 100:
                result = result - 273.15
            result.attrs["units"] = "degC"
        elif var == "hurs":
            if maximum <= 1.5:
                result = result * 100.0
            result = result.clip(0.0, 100.0)
            result.attrs["units"] = "%"
        elif var == "clt":
            if maximum <= 1.5:
                result = result * 100.0
            result = result.clip(0.0, 100.0)
            result.attrs["units"] = "%"
        elif var in {"ps", "psl"}:
            # Existing hostrada4py evaluation/export code uses HOSTRADA pressure
            # in hPa and performs any target-format conversion itself.
            if "kpa" in units:
                result = result * 10.0
            elif units in {"pa", "pascal", "pascals"} or median > 2000:
                result = result / 100.0
            result.attrs["units"] = "hPa"
        elif var == "sfcWind_direction":
            result = result % 360.0
            result.attrs["units"] = "degree"
        elif var == "sfcWind":
            result = result.clip(min=0.0)
            result.attrs["units"] = "m s-1"
        elif var == "rsds":
            result = result.clip(min=0.0)
            result.attrs["units"] = "W m-2"
        return result

    @staticmethod
    def _dewpoint_celsius(temperature_c: xr.DataArray, relative_humidity_percent: xr.DataArray) -> xr.DataArray:
        rh = relative_humidity_percent.clip(0.1, 100.0)
        alpha = np.log(rh / 100.0) + (17.625 * temperature_c) / (243.04 + temperature_c)
        dew = (243.04 * alpha) / (17.625 - alpha)
        dew.attrs.update(units="degC", long_name="2 metre dew-point temperature")
        return dew

    @staticmethod
    def _mixing_ratio_gkg(
        temperature_c: xr.DataArray,
        relative_humidity_percent: xr.DataArray,
        pressure_hpa: xr.DataArray,
    ) -> xr.DataArray:
        saturation_hpa = 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
        vapour_hpa = (relative_humidity_percent / 100.0) * saturation_hpa
        ratio = 1000.0 * 0.622 * vapour_hpa / (pressure_hpa - vapour_hpa).clip(min=0.01)
        ratio.attrs.update(units="g kg-1", long_name="water-vapour mixing ratio")
        return ratio

    @staticmethod
    def _hourly_analysis(da: xr.DataArray, var: str, year: int, month: int) -> xr.DataArray:
        last_day = calendar.monthrange(year, month)[1]
        target = pd.date_range(
            f"{year:04d}-{month:02d}-01 00:00",
            f"{year:04d}-{month:02d}-{last_day:02d} 23:00",
            freq="h",
        )
        da = da.sortby("time")
        if var == "sfcWind_direction":
            radians = np.deg2rad(da)
            sin_component = np.sin(radians).interp(
                time=target, kwargs={"fill_value": "extrapolate"}
            )
            cos_component = np.cos(radians).interp(
                time=target, kwargs={"fill_value": "extrapolate"}
            )
            result = (np.rad2deg(np.arctan2(sin_component, cos_component)) + 360.0) % 360.0
            result.attrs = dict(da.attrs)
            return result
        return da.interp(time=target, kwargs={"fill_value": "extrapolate"})

    @staticmethod
    def _hourly_forecast(da: xr.DataArray, year: int, month: int) -> xr.DataArray:
        """Return a complete hourly month from forecast valid times.

        CERRA short forecasts provide the requested month from lead times 1--3,
        which normally leaves only 00 UTC on the first day without a preceding
        forecast cycle. Exact hourly values are retained; an edge value is
        interpolated/extrapolated only where a boundary sample is absent.
        """
        last_day = calendar.monthrange(year, month)[1]
        target = pd.date_range(
            f"{year:04d}-{month:02d}-01 00:00",
            f"{year:04d}-{month:02d}-{last_day:02d} 23:00",
            freq="h",
        )
        result = da.sortby("time").interp(
            time=target, kwargs={"fill_value": "extrapolate"}
        )
        result.attrs = dict(da.attrs)
        return result

    def _sample_native_to_epsg3034(
        self,
        da: xr.DataArray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> xr.DataArray:
        """Sample only the required native CERRA cells before time processing.

        This avoids loading/interpolating the complete 1069 x 1069 European
        grid for a point or small polygon. The native monthly GRIB remains in
        the shared raw cache, while the returned array contains only the target
        EPSG:3034 cells.
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise ImportError(
                "CERRA grid normalisation requires scipy (scipy.spatial.cKDTree)."
            ) from exc

        lat_coord, lon_coord = self._spatial_coordinates(da)
        spatial_dims = list(lat_coord.dims)
        if len(spatial_dims) != 2 or set(lon_coord.dims) != set(spatial_dims):
            raise ValueError(
                "Expected two-dimensional CERRA latitude/longitude coordinates, "
                f"found latitude={lat_coord.dims}, longitude={lon_coord.dims}."
            )
        if not all(dim in da.dims for dim in spatial_dims):
            raise ValueError(
                f"CERRA coordinate dimensions {spatial_dims} are absent from {da.dims}."
            )

        # Align the 2-D coordinates with the selected spatial dimension order.
        lat_values = np.asarray(lat_coord.transpose(*spatial_dims).values, dtype=float)
        lon_values = np.asarray(lon_coord.transpose(*spatial_dims).values, dtype=float)

        transformer = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        target_lon, target_lat = transformer.transform(x_grid, y_grid)

        source_valid = np.isfinite(lat_values) & np.isfinite(lon_values)
        if not source_valid.any():
            raise ValueError("CERRA latitude/longitude grid contains no finite points.")
        tree = cKDTree(self._lonlat_xyz(lon_values[source_valid], lat_values[source_valid]))
        _, nearest_valid = tree.query(
            self._lonlat_xyz(
                np.asarray(target_lon, dtype=float).ravel(),
                np.asarray(target_lat, dtype=float).ravel(),
            ),
            k=1,
        )
        source_flat = np.flatnonzero(source_valid.ravel())[nearest_valid]
        row_idx, col_idx = np.unravel_index(source_flat, lat_values.shape)

        # Load only the smallest native rectangle enclosing all requested cells.
        row0, row1 = int(row_idx.min()), int(row_idx.max()) + 1
        col0, col1 = int(col_idx.min()), int(col_idx.max()) + 1
        non_spatial = [dim for dim in da.dims if dim not in spatial_dims]
        ordered = da.transpose(*non_spatial, *spatial_dims)
        subset = ordered.isel(
            {spatial_dims[0]: slice(row0, row1), spatial_dims[1]: slice(col0, col1)}
        )
        values = np.asarray(subset.values)
        local_rows = (row_idx - row0).reshape(len(y_axis), len(x_axis))
        local_cols = (col_idx - col0).reshape(len(y_axis), len(x_axis))
        sampled = values[..., local_rows, local_cols]

        coords: dict[str, Any] = {
            "X": np.asarray(x_axis, dtype=float),
            "Y": np.asarray(y_axis, dtype=float),
            "lon": (("Y", "X"), np.asarray(target_lon, dtype=float)),
            "lat": (("Y", "X"), np.asarray(target_lat, dtype=float)),
        }
        for name, coord in ordered.coords.items():
            if name in {"latitude", "longitude", "lat", "lon"}:
                continue
            if set(coord.dims).issubset(non_spatial):
                coords[name] = coord

        result = xr.DataArray(
            sampled,
            dims=(*non_spatial, "Y", "X"),
            coords=coords,
            attrs=dict(da.attrs),
            name=da.name,
        )
        result["X"].attrs.update(units="m", standard_name="projection_x_coordinate")
        result["Y"].attrs.update(units="m", standard_name="projection_y_coordinate")
        result["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        result["lat"].attrs.update(units="degrees_north", standard_name="latitude")
        return result

    def _normalise_native_datasets(
        self,
        datasets: Sequence[xr.Dataset],
        var: str,
        year: int,
        month: int,
        x_axis: Optional[np.ndarray] = None,
        y_axis: Optional[np.ndarray] = None,
    ) -> xr.DataArray:
        spec = self._spec(var)

        def field(cds_name: str, preferred: Sequence[str]) -> xr.DataArray:
            generic = {
                "2m_temperature": ("t2m", "2t"),
                "2m_relative_humidity": ("r2", "2r", "rh2m"),
                "surface_pressure": ("sp",),
                "10m_wind_speed": ("si10", "10si", "ws10"),
                "10m_wind_direction": ("wdir10", "10wdir", "wd10"),
                "mean_sea_level_pressure": ("msl", "prmsl"),
                "total_cloud_cover": ("tcc",),
                "surface_solar_radiation_downwards": ("ssrd",),
            }
            candidates = tuple(preferred) + generic.get(cds_name, ()) + (cds_name,)
            selected = self._pick_dataarray(datasets, candidates)
            if x_axis is not None and y_axis is not None:
                selected = self._sample_native_to_epsg3034(selected, x_axis, y_axis)
            return selected

        if var in {"tdew", "mixr"}:
            tas = self._forecast_to_valid_time(field("2m_temperature", ()), False)
            tas = self._to_hostrada_units("tas", tas)
            hurs = self._forecast_to_valid_time(field("2m_relative_humidity", ()), False)
            hurs = self._to_hostrada_units("hurs", hurs)
            if var == "tdew":
                result = self._dewpoint_celsius(tas, hurs)
            else:
                ps = self._forecast_to_valid_time(field("surface_pressure", ()), False)
                ps = self._to_hostrada_units("ps", ps)
                result = self._mixing_ratio_gkg(tas, hurs, ps)
        else:
            result = field(spec.cds_variables[0], spec.short_names)
            result = self._forecast_to_valid_time(result, spec.accumulated)
            result = self._to_hostrada_units(var, result)

        if spec.product_type == "analysis":
            result = self._hourly_analysis(result, var, year, month)
        else:
            result = self._hourly_forecast(result, year, month)
            if var == "rsds":
                result = result.clip(min=0.0)
        result.name = var
        result.attrs.update(
            long_name=spec.long_name,
            units=spec.units,
            source="Copernicus CERRA",
            hostrada4py_canonical_variable=var,
        )
        return result

    @staticmethod
    def _lonlat_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        lon_rad = np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        cos_lat = np.cos(lat_rad)
        return np.column_stack(
            (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad))
        )

    def _regrid_to_epsg3034(
        self,
        da: xr.DataArray,
        var: str,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> xr.Dataset:
        # The production path samples the native grid before temporal
        # interpolation. In that case the data already use the target axes.
        if "Y" in da.dims and "X" in da.dims:
            result = xr.Dataset(
                {var: da.transpose("time", "Y", "X")},
                attrs={
                    "Conventions": "CF-1.8",
                    "source": "Copernicus CERRA via Climate Data Store",
                    "provider": self.name,
                    "hostrada4py_crs": "EPSG:3034",
                    "hostrada4py_cell_size": self.target_grid_size_m,
                    "hostrada4py_spatial_method": "nearest neighbour from native CERRA grid",
                    "hostrada4py_temporal_method": (
                        "linear/circular interpolation from 3-hour analysis"
                        if self._spec(var).product_type == "analysis"
                        else "hourly forecast accumulation de-accumulation with boundary completion"
                    ),
                },
            )
            return result

        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise ImportError(
                "CERRA grid normalisation requires scipy (scipy.spatial.cKDTree)."
            ) from exc

        lat_coord, lon_coord = self._spatial_coordinates(da)
        spatial_dims = [dim for dim in da.dims if dim != "time"]
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected a two-dimensional CERRA field, found {da.dims}")
        da = da.transpose("time", *spatial_dims)

        lat_values = np.asarray(lat_coord.values, dtype=float)
        lon_values = np.asarray(lon_coord.values, dtype=float)
        if lat_values.ndim == lon_values.ndim == 1:
            lon_values, lat_values = np.meshgrid(lon_values, lat_values)
        if lat_values.shape != tuple(da.shape[1:]) or lon_values.shape != tuple(da.shape[1:]):
            # cfgrib coordinates may be attached in the reverse spatial order.
            if lat_values.T.shape == tuple(da.shape[1:]):
                lat_values = lat_values.T
                lon_values = lon_values.T
            else:
                raise ValueError(
                    "CERRA latitude/longitude shape does not match the data field: "
                    f"data={da.shape[1:]}, lat={lat_values.shape}, lon={lon_values.shape}"
                )

        transformer = Transformer.from_crs("EPSG:3034", "EPSG:4326", always_xy=True)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        target_lon, target_lat = transformer.transform(x_grid, y_grid)

        source_valid = np.isfinite(lat_values) & np.isfinite(lon_values)
        tree = cKDTree(self._lonlat_xyz(lon_values[source_valid], lat_values[source_valid]))
        _, nearest_valid = tree.query(
            self._lonlat_xyz(np.asarray(target_lon).ravel(), np.asarray(target_lat).ravel()),
            k=1,
        )
        source_flat_indices = np.flatnonzero(source_valid.ravel())[nearest_valid]
        values = np.asarray(da.values)
        flat = values.reshape(values.shape[0], -1)
        target_values = flat[:, source_flat_indices].reshape(
            values.shape[0], len(y_axis), len(x_axis)
        )

        result = xr.Dataset(
            {
                var: xr.DataArray(
                    target_values,
                    dims=("time", "Y", "X"),
                    attrs=dict(da.attrs),
                )
            },
            coords={
                "time": np.asarray(da.time.values),
                "X": np.asarray(x_axis, dtype=float),
                "Y": np.asarray(y_axis, dtype=float),
                "lon": (("Y", "X"), np.asarray(target_lon, dtype=float)),
                "lat": (("Y", "X"), np.asarray(target_lat, dtype=float)),
            },
            attrs={
                "Conventions": "CF-1.8",
                "source": "Copernicus CERRA via Climate Data Store",
                "provider": self.name,
                "hostrada4py_crs": "EPSG:3034",
                "hostrada4py_cell_size": self.target_grid_size_m,
                "hostrada4py_spatial_method": "nearest neighbour from native CERRA grid",
                "hostrada4py_temporal_method": (
                    "linear/circular interpolation from 3-hour analysis"
                    if self._spec(var).product_type == "analysis"
                    else "hourly forecast accumulation de-accumulation with boundary completion"
                ),
            },
        )
        result["X"].attrs.update(units="m", standard_name="projection_x_coordinate")
        result["Y"].attrs.update(units="m", standard_name="projection_y_coordinate")
        result["lon"].attrs.update(units="degrees_east", standard_name="longitude")
        result["lat"].attrs.update(units="degrees_north", standard_name="latitude")
        return result

    @staticmethod
    def _default_selector_from_env() -> Optional[dict[str, Any]]:
        raw = os.getenv("HOSTRADA_CERRA_DEFAULT_AREA", "").strip()
        if not raw:
            return None
        values = [float(value.strip()) for value in raw.split(",")]
        if len(values) != 4:
            raise ValueError(
                "HOSTRADA_CERRA_DEFAULT_AREA must be west,south,east,north in EPSG:4326."
            )
        west, south, east, north = values
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3034", always_xy=True)
        x, y = transformer.transform(
            [west, west, east, east], [south, north, south, north]
        )
        return {
            "type": "bbox_epsg3034",
            "bbox": (min(x), min(y), max(x), max(y)),
        }

    def ensure_month_file(
        self,
        var: str,
        year: int,
        month: int,
        cache_dir: Path,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        selector: Optional[Mapping[str, Any]] = None,
        subset_mode: Optional[str] = None,
        subset_margin_cells: Optional[int] = None,
        timeout: Optional[TimeoutValue] = None,
        retries: Optional[int] = None,
        backoff: Optional[float] = None,
        verbose: bool = True,
    ) -> Path:
        del subset_mode, timeout, retries, backoff  # CDS manages transport/retries.
        self.require_variable(var)
        if year < 1984 or (year == 1984 and month < 9):
            raise ValueError("CERRA starts in September 1984.")
        selector = dict(selector or self._default_selector_from_env() or {})
        if not selector:
            raise ValueError(
                "CERRA downloads must be spatially bounded. Use "
                "ensure_month_file_for_point/for_bbox or set "
                "HOSTRADA_CERRA_DEFAULT_AREA=west,south,east,north."
            )
        margin = (
            SUBSET_MARGIN_CELLS_DEFAULT
            if subset_margin_cells is None
            else max(0, int(subset_margin_cells))
        )
        x_axis, y_axis = self._target_axes(selector, margin)
        area = self._selector_area_lonlat(selector, margin)
        params = {
            "version": 1,
            "provider": self.name,
            "var": var,
            "year": year,
            "month": month,
            "selector": selector,
            "margin": margin,
            "grid_size": self.target_grid_size_m,
            "area": area,
        }
        target = (
            Path(cache_dir)
            / self.name
            / var
            / f"{Path(self.filename(var, year, month)).stem}.{cache_key(params)}.nc"
        )
        if is_cached_file(target):
            if verbose:
                print(f"CERRA cache: {target}")
            return target

        with file_lock(target.with_name(target.name + ".lock")):
            if is_cached_file(target):
                return target
            raw = self._retrieve_grib(var, year, month, area, Path(cache_dir), verbose)
            datasets = self._open_grib_datasets(raw)
            try:
                native = self._normalise_native_datasets(
                    datasets, var, year, month, x_axis=x_axis, y_axis=y_axis
                )
                normalised = self._regrid_to_epsg3034(native, var, x_axis, y_axis)
                # Cache complete monthly files. The existing point/area modules
                # apply the exact start/end slice while reading; storing only a
                # first caller's interval would make later wider requests reuse
                # an incomplete cache entry.
                return write_netcdf_atomic(normalised.load(), target)
            finally:
                for ds in datasets:
                    ds.close()

    # Convenience helpers used in tests and advanced integrations.
    def selector_for_point(self, lon: float, lat: float) -> dict[str, Any]:
        x, y = point_to_epsg3034(lon, lat)
        return {"type": "point_epsg3034", "x": x, "y": y}
