from __future__ import annotations

import calendar
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import xarray as xr

from .base import ProviderCapabilities, TimeoutValue, WeatherProvider
from .common import (
    DOWNLOAD_BACKOFF,
    DOWNLOAD_RETRIES,
    SUBSET_MARGIN_CELLS_DEFAULT,
    cache_key,
    download_file,
    is_cached_file,
    point_to_epsg3034,
    subset_rectilinear_dataset,
    write_netcdf_atomic,
)

BASE_URLS = {
    "tas": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/air_temperature_mean",
    "uhi": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/urban_heat_island_intensity",
    "sfcWind": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/wind_speed",
    "sfcWind_direction": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/wind_direction",
    "rsds": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/radiation_downwelling",
    "clt": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/cloud_cover",
    "hurs": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/humidity_relative",
    "mixr": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/humidity_mixing_ratio",
    "ps": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/pressure_surface",
    "psl": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/pressure_sealevel",
    "tdew": "https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/dew_point",
}

FILE_PREFIXES = {
    "tas": "tas_1hr_HOSTRADA-v1-0_BE_gn",
    "uhi": "uhi_1hr_HOSTRADA-v1-0_BE_gn",
    "sfcWind": "sfcWind_1hr_HOSTRADA-v1-0_BE_gn",
    "sfcWind_direction": "sfcWind_direction_1hr_HOSTRADA-v1-0_BE_gn",
    "rsds": "rsds_1hr_HOSTRADA-v1-0_BE_gn",
    "clt": "clt_1hr_HOSTRADA-v1-0_BE_gn",
    "hurs": "hurs_1hr_HOSTRADA-v1-0_BE_gn",
    "mixr": "mixr_1hr_HOSTRADA-v1-0_BE_gn",
    "ps": "ps_1hr_HOSTRADA-v1-0_BE_gn",
    "psl": "psl_1hr_HOSTRADA-v1-0_BE_gn",
    "tdew": "tdew_1hr_HOSTRADA-v1-0_BE_gn",
}

# Kept as a misspelled alias because the original module exported this name.
FILE_PREFIXS = FILE_PREFIXES


class DWDHostradaProvider(WeatherProvider):
    name = "dwd"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.name,
            variables=frozenset(BASE_URLS),
            temporal_resolution="1 hour",
            spatial_resolution_m=1000.0,
            crs="EPSG:3034",
            start="1995-01",
            notes=("Original DWD HOSTRADA backend",),
        )

    def filename(self, var: str, year: int, month: int) -> str:
        self.require_variable(var)
        last_day = calendar.monthrange(year, month)[1]
        return (
            f"{FILE_PREFIXES[var]}_{year:04d}{month:02d}0100-"
            f"{year:04d}{month:02d}{last_day:02d}23.nc"
        )

    def url(self, var: str, year: int, month: int) -> str:
        return f"{BASE_URLS[var]}/{self.filename(var, year, month)}"

    @staticmethod
    def _normalise_mode(mode: Optional[str]) -> str:
        value = (mode or os.getenv("HOSTRADA_NETCDF_SUBSET_MODE", "full")).strip().lower()
        aliases = {
            "0": "full", "false": "full", "no": "full", "off": "full",
            "1": "subset", "true": "subset", "yes": "subset", "on": "subset",
            "range": "http_range", "remote": "http_range",
            "remote_subset": "http_range", "http-range": "http_range",
        }
        value = aliases.get(value, value)
        if value not in {"full", "subset", "http_range", "auto"}:
            raise ValueError(
                f"Unknown HOSTRADA NetCDF subset mode '{value}'. "
                "Use full, subset, http_range or auto."
            )
        return value

    def _full_file(
        self,
        var: str,
        year: int,
        month: int,
        cache_dir: Path,
        timeout: Optional[TimeoutValue],
        retries: int,
        backoff: float,
        verbose: bool,
    ) -> Path:
        target = Path(cache_dir) / self.filename(var, year, month)
        if is_cached_file(target):
            if verbose:
                print(f"Cache: {target}")
            return target
        url = self.url(var, year, month)
        if verbose:
            print(f"Download: {url}")
        return download_file(
            url,
            target,
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )

    def _remote_subset(
        self,
        url: str,
        target: Path,
        var: str,
        selector: Mapping[str, Any],
        margin: int,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Path:
        import fsspec  # optional

        engine = os.getenv("HOSTRADA_HTTP_RANGE_ENGINE", "h5netcdf")
        block_size = int(os.getenv("HOSTRADA_HTTP_RANGE_BLOCK_SIZE", str(2 * 1024 * 1024)))
        opened = fsspec.open(
            url,
            mode="rb",
            block_size=block_size,
            cache_type="bytes",
            headers={"User-Agent": "hostrada4py/http-range-subsetter"},
        )
        with opened.open() as fileobj:
            with xr.open_dataset(fileobj, engine=engine) as ds:
                subset = subset_rectilinear_dataset(
                    ds, var, selector, margin_cells=margin, start=start, end=end
                )
        return write_netcdf_atomic(subset, target)

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
        self.require_variable(var)
        retries = DOWNLOAD_RETRIES if retries is None else int(retries)
        backoff = DOWNLOAD_BACKOFF if backoff is None else float(backoff)
        mode = self._normalise_mode(subset_mode)
        if selector is None or mode == "full":
            return self._full_file(
                var, year, month, Path(cache_dir), timeout, retries, backoff, verbose
            )

        margin = (
            SUBSET_MARGIN_CELLS_DEFAULT
            if subset_margin_cells is None
            else max(0, int(subset_margin_cells))
        )
        params = {
            "version": 2,
            "provider": self.name,
            "var": var,
            "year": year,
            "month": month,
            "selector": dict(selector),
            "start": str(start or ""),
            "end": str(end or ""),
            "margin": margin,
        }
        target = (
            Path(cache_dir)
            / "subsets"
            / str(selector["type"]).replace("_epsg3034", "")
            / f"{Path(self.filename(var, year, month)).stem}.{cache_key(params)}.nc"
        )
        if is_cached_file(target):
            if verbose:
                print(f"Subset cache: {target}")
            return target

        if mode in {"http_range", "auto"}:
            try:
                if verbose:
                    print(f"HTTP-range subset: {self.url(var, year, month)}")
                return self._remote_subset(
                    self.url(var, year, month),
                    target,
                    var,
                    selector,
                    margin,
                    start,
                    end,
                )
            except Exception as exc:  # noqa: BLE001
                fallback = os.getenv("HOSTRADA_NETCDF_SUBSET_FALLBACK", "1").lower() not in {
                    "0", "false", "no", "off"
                }
                if mode == "http_range" and not fallback:
                    raise RuntimeError("HTTP-range NetCDF subsetting failed") from exc
                warnings.warn(
                    "HTTP-range subsetting failed; falling back to a full monthly "
                    f"download and local subset. Last error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        source = self._full_file(
            var, year, month, Path(cache_dir), timeout, retries, backoff, verbose
        )
        if verbose:
            print(f"Create subset cache: {target}")
        with xr.open_dataset(source) as ds:
            subset = subset_rectilinear_dataset(
                ds, var, selector, margin_cells=margin, start=start, end=end
            )
        result = write_netcdf_atomic(subset, target)
        drop_full = os.getenv("HOSTRADA_DROP_FULL_AFTER_SUBSET", "0").lower() in {
            "1", "true", "yes", "on"
        }
        if drop_full:
            try:
                source.unlink()
            except OSError:
                pass
        return result

    def required_month_files(self, variables, start, end, cache_dir):
        # Preserve the exact original cache layout for the default provider.
        from .common import month_range

        result: list[Path] = []
        seen: set[str] = set()
        for var in variables:
            if var in seen:
                continue
            seen.add(var)
            self.require_variable(var)
            for year, month in month_range(start, end):
                result.append(Path(cache_dir) / self.filename(var, year, month))
        return result
