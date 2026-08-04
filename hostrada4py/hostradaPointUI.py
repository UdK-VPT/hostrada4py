"""Interactive controls for selecting and downloading HOSTRADA point data.

The module contains the address search, Leaflet map, period controls and
HOSTRADA download callback used by ``hostradaPoint.ipynb``.  It deliberately
updates the notebook namespace so that the established variables ``lat``,
``lon``, ``start_UTC``, ``end_UTC``, ``df`` and ``fn`` remain available to all
following notebook cells.
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path
from collections.abc import Callable, MutableMapping
from typing import Any

import pandas as pd
import requests
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import Map, Marker, basemaps

from . import hostrada as hs


GERMANY_BOUNDS = {
    "lon_min": 5.5,
    "lon_max": 15.6,
    "lat_min": 47.0,
    "lat_max": 55.5,
}

# ISO 3166-1 alpha-2 country codes used by OpenStreetMap Nominatim when the
# active provider is CERRA.  Restricting the geocoder to European countries
# avoids equally named places elsewhere while preserving cross-border search.
EUROPE_COUNTRY_CODES = (
    "al", "ad", "at", "by", "be", "ba", "bg", "hr", "cy", "cz",
    "dk", "ee", "fi", "fr", "de", "gr", "hu", "is", "ie", "it",
    "xk", "lv", "li", "lt", "lu", "mt", "md", "mc", "me", "nl",
    "mk", "no", "pl", "pt", "ro", "ru", "sm", "rs", "sk", "si",
    "es", "se", "ch", "tr", "ua", "gb", "va", "fo", "gi", "gg",
    "im", "je",
)

HOUR_OPTIONS = [(f"{hour:02d}:00 UTC", hour) for hour in range(24)]

HOSTRADA_POINT_UI_API_VERSION = "2.2-resilient-geocoder"

PHOTON_SEARCH_URL = os.getenv(
    "HOSTRADA_PHOTON_SEARCH_URL", "https://photon.komoot.io/api/"
).strip()
NOMINATIM_SEARCH_URL = os.getenv(
    "HOSTRADA_NOMINATIM_SEARCH_URL",
    "https://nominatim.openstreetmap.org/search",
).strip()
GEOCODER_ORDER = tuple(
    item.strip().lower()
    for item in os.getenv("HOSTRADA_GEOCODER_ORDER", "photon,nominatim").split(",")
    if item.strip()
)
GEOCODER_MIN_INTERVAL_SECONDS = max(
    1.05, float(os.getenv("HOSTRADA_GEOCODER_MIN_INTERVAL_SECONDS", "1.1"))
)
GEOCODER_RATE_LIMIT_COOLDOWN_SECONDS = max(
    30.0, float(os.getenv("HOSTRADA_GEOCODER_RATE_LIMIT_COOLDOWN_SECONDS", "60"))
)
GEOCODER_USER_AGENT = os.getenv(
    "HOSTRADA_GEOCODER_USER_AGENT",
    "hostrada4py/0.42 interactive-address-search",
).strip()


class GeocoderRateLimited(RuntimeError):
    """Raised when a public geocoder asks the client to slow down."""


class GeocoderUnavailable(RuntimeError):
    """Raised when all configured geocoders failed."""


class HostradaPointUI:
    """Build and manage the interactive HOSTRADA point-data controls.

    Parameters
    ----------
    namespace:
        Usually ``globals()`` from the notebook.  The class keeps the original
        notebook variables synchronized with widget selections and downloads.
    extractor:
        Callable compatible with ``hostradaPoint.extract_values_for_point``.
    initial_lat, initial_lon, initial_address:
        Initial climate location.
    initial_start, initial_end:
        Initial UTC timestamps in ``YYYY-MM-DDTHH:MM`` format.
    initial_output_directory, initial_output_filename:
        Default directory and optional file name for the exported CSV file.
        Relative directories are resolved against the notebook working directory.
    """

    def __init__(
        self,
        namespace: MutableMapping[str, Any],
        extractor: Callable[..., pd.DataFrame],
        *,
        initial_lat: float = 52.51712,
        initial_lon: float = 13.32259,
        initial_address: str = "Einsteinufer 43–53, 10587 Berlin",
        initial_start: str = "2025-01-01T00:00",
        initial_end: str = "2025-12-31T23:00",
        initial_output_directory: str | os.PathLike[str] = ".",
        initial_output_filename: str | None = None,
    ) -> None:
        self.namespace = namespace
        self.extractor = extractor
        self._geocode_results: list[dict[str, Any]] = []
        self._geocode_cache: dict[tuple[str, str], tuple[list[dict[str, Any]], str]] = {}
        self._last_geocoder_request_at = 0.0
        self._geocoder_cooldown_until: dict[str, float] = {}
        self._updating_marker = False
        self._updating_output_filename = False
        self._custom_output_filename = initial_output_filename is not None
        self._initial_output_directory = str(initial_output_directory)
        self._initial_output_filename = initial_output_filename

        self._initial_start = pd.Timestamp(initial_start)
        self._initial_end = pd.Timestamp(initial_end)
        if self._initial_end < self._initial_start:
            raise ValueError("initial_end must not be earlier than initial_start")

        self.namespace.update(
            {
                "lon": float(initial_lon),
                "lat": float(initial_lat),
                "location_label": initial_address,
                "start_UTC": self._format_timestamp(self._initial_start),
                "end_UTC": self._format_timestamp(self._initial_end),
                "df": None,
                "fn": self._filename(),
                "output_directory": str(
                    self._normalise_output_directory(self._initial_output_directory)
                ),
                "output_filename": (
                    self._initial_output_filename or self._filename()
                ),
            }
        )

        self._build_location_controls(initial_address)
        self._build_period_controls()
        self._build_download_controls()
        self._connect_callbacks()
        self._update_period()
        self._refresh_location_status(initial_address)
        self._build_layout()
        self._publish_widget_references()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    @property
    def lat(self) -> float:
        return float(self.namespace["lat"])

    @property
    def lon(self) -> float:
        return float(self.namespace["lon"])

    @property
    def climate_variable(self) -> str:
        return str(self.namespace["HOSTRADA_VAR"])

    def _filename(self, variable: str | None = None) -> str:
        variable = str(variable or self.namespace.get("HOSTRADA_VAR", "tas"))
        return f"HOSTRADA_{variable}.csv"

    @staticmethod
    def _normalise_output_directory(value: str | os.PathLike[str]) -> Path:
        raw_value = os.path.expandvars(str(value).strip())
        directory = Path(raw_value or ".").expanduser()
        if not directory.is_absolute():
            directory = Path.cwd() / directory
        return directory.resolve(strict=False)

    def _normalise_output_filename(self, value: str) -> str:
        filename = value.strip() or self._filename()
        if any(separator in filename for separator in ("/", "\\")):
            raise ValueError(
                "Please enter only a file name. Use the directory field for the path."
            )
        if filename in {".", ".."}:
            raise ValueError("Please enter a valid CSV file name.")
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        return filename

    def _resolve_output_path(self, *, create_directory: bool = False) -> Path:
        directory = self._normalise_output_directory(self.output_directory_input.value)
        filename = self._normalise_output_filename(self.output_filename_input.value)
        if create_directory:
            directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / filename
        self.namespace.update(
            {
                "output_directory": str(directory),
                "output_filename": filename,
                "fn": str(output_path),
            }
        )
        return output_path

    @staticmethod
    def _inside_germany(latitude: float, longitude: float) -> bool:
        return (
            GERMANY_BOUNDS["lat_min"] <= float(latitude) <= GERMANY_BOUNDS["lat_max"]
            and GERMANY_BOUNDS["lon_min"]
            <= float(longitude)
            <= GERMANY_BOUNDS["lon_max"]
        )

    def _inside_provider_domain(self, latitude: float, longitude: float) -> bool:
        """Preserve the DWD Germany guard; allow the CERRA European domain."""
        if hs.get_provider_name() == "dwd":
            return self._inside_germany(latitude, longitude)
        return 20.0 <= float(latitude) <= 80.0 and -45.0 <= float(longitude) <= 75.0

    @staticmethod
    def _provider_name() -> str:
        return hs.get_provider_name()

    def _location_scope_label(self) -> str:
        return "Europe" if self._provider_name() == "cerra" else "Germany"

    def _location_scope_description(self) -> str:
        if self._provider_name() == "cerra":
            return "the European CERRA domain"
        return "Germany"

    def _geocoder_countrycodes(self) -> str:
        if self._provider_name() == "cerra":
            return ",".join(EUROPE_COUNTRY_CODES)
        return "de"

    def _reset_address_results(self, message: str = "Search for an address first") -> None:
        """Reset stale geocoder state before a new query or provider change.

        The dropdown value must pass through ``None`` between searches.  Without
        this reset, two consecutive searches whose first result both use index
        0 do not emit a value-change event, so the second address is not applied.
        """
        self._geocode_results = []
        self.address_results.disabled = True
        self.address_results.options = [(message, None)]
        self.address_results.value = None

    def refresh_provider_scope(self) -> None:
        """Refresh address-search guidance after the notebook changes provider."""
        scope = self._location_scope_label()
        if hasattr(self, "address_results"):
            self._reset_address_results(
                f"Search for an address in {scope}"
            )
        self.address_status.value = (
            f"Enter an address in <b>{scope}</b> and click "
            "<b>Search address</b>, or click directly on the map. "
            "The marker can also be dragged."
        )
        if hasattr(self, "address_scope_note"):
            self.address_scope_note.value = (
                "<small>Address search uses OpenStreetMap data via Photon "
                "with Nominatim as a fallback. Requests are cached and rate-limited. "
                f"The selected point is restricted to {html.escape(self._location_scope_description())}."
                "</small>"
            )
        if not self._inside_provider_domain(self.lat, self.lon):
            self.address_status.value += (
                "<br><span style='color:#b00020'><b>The currently selected "
                f"point is outside {html.escape(self._location_scope_description())}. "
                "Choose a new address or map location before downloading.</b></span>"
            )

    @staticmethod
    def _format_utc(selected_date: Any, selected_hour: int) -> str:
        return f"{selected_date:%Y-%m-%d}T{int(selected_hour):02d}:00"

    @staticmethod
    def _format_timestamp(timestamp: pd.Timestamp) -> str:
        return timestamp.strftime("%Y-%m-%dT%H:%M")

    # ------------------------------------------------------------------
    # Location selection
    # ------------------------------------------------------------------
    def _build_location_controls(self, initial_address: str) -> None:
        self.address_input = widgets.Text(
            value=initial_address,
            description="Address:",
            placeholder="Street, house number, postal code, city",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="690px"),
        )
        self.address_search_button = widgets.Button(
            description="Search address",
            icon="search",
            button_style="info",
            layout=widgets.Layout(width="180px"),
        )
        self.address_results = widgets.Dropdown(
            options=[("Search for an address first", None)],
            value=None,
            description="Results:",
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="875px"),
        )
        self.address_status = widgets.HTML()
        self.location_status = widgets.HTML()

        self.location_map = Map(
            basemap=basemaps.OpenStreetMap.Mapnik,
            center=(self.lat, self.lon),
            zoom=13,
            scroll_wheel_zoom=True,
            layout=widgets.Layout(width="100%", height="520px"),
        )
        self.location_marker = Marker(
            location=(self.lat, self.lon),
            draggable=True,
            title="Selected HOSTRADA location – drag to move",
        )
        try:
            self.location_map.add(self.location_marker)
        except (AttributeError, TypeError):
            self.location_map.add_layer(self.location_marker)
        self.refresh_provider_scope()

    def _refresh_location_status(self, source: str | None = None) -> None:
        source_text = f" &nbsp; <b>Source:</b> {html.escape(source)}" if source else ""
        self.location_status.value = (
            "<b>Selected coordinates:</b> "
            f"<code>lat = {self.lat:.6f}</code>, "
            f"<code>lon = {self.lon:.6f}</code>{source_text}"
        )

    def _set_location(
        self,
        latitude: float,
        longitude: float,
        *,
        source: str = "Map",
        zoom: int | None = None,
    ) -> bool:
        latitude = float(latitude)
        longitude = float(longitude)
        if not self._inside_provider_domain(latitude, longitude):
            scope = html.escape(self._location_scope_description())
            self.address_status.value = (
                "<span style='color:#b00020'><b>The selected point is outside "
                f"{scope}. Please choose a location within {scope}.</b></span>"
            )
            return False

        self.namespace.update(
            {
                "lat": latitude,
                "lon": longitude,
                "location_label": source,
            }
        )

        self._updating_marker = True
        try:
            self.location_marker.location = (latitude, longitude)
        finally:
            self._updating_marker = False

        self.location_map.center = (latitude, longitude)
        if zoom is not None:
            self.location_map.zoom = int(zoom)
        self._refresh_location_status(source)
        return True

    @staticmethod
    def _normalise_geocode_query(query: str) -> str:
        return " ".join(query.casefold().split())

    def _geocoder_headers(self) -> dict[str, str]:
        return {
            "User-Agent": GEOCODER_USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "de,en;q=0.8",
        }

    def _wait_for_geocoder_slot(self) -> None:
        elapsed = time.monotonic() - self._last_geocoder_request_at
        delay = GEOCODER_MIN_INTERVAL_SECONDS - elapsed
        if delay > 0:
            time.sleep(delay)

    def _request_geocoder_json(
        self,
        service: str,
        url: str,
        *,
        params: Any,
    ) -> Any:
        now = time.monotonic()
        cooldown_until = self._geocoder_cooldown_until.get(service, 0.0)
        if now < cooldown_until:
            seconds = max(1, int(round(cooldown_until - now)))
            raise GeocoderRateLimited(
                f"{service} is temporarily rate-limited; retry in about {seconds} s"
            )

        self._wait_for_geocoder_slot()
        try:
            response = requests.get(
                url,
                params=params,
                headers=self._geocoder_headers(),
                timeout=20,
            )
        finally:
            self._last_geocoder_request_at = time.monotonic()

        status_code = int(getattr(response, "status_code", 200))
        if status_code == 429:
            retry_after = None
            headers = getattr(response, "headers", {}) or {}
            try:
                retry_after = float(headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
            cooldown = max(
                GEOCODER_RATE_LIMIT_COOLDOWN_SECONDS,
                retry_after or 0.0,
            )
            self._geocoder_cooldown_until[service] = time.monotonic() + cooldown
            raise GeocoderRateLimited(
                f"{service} returned HTTP 429 and is paused for {int(cooldown)} s"
            )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _photon_label(properties: dict[str, Any], fallback: str) -> str:
        parts: list[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if text and text.casefold() not in {part.casefold() for part in parts}:
                parts.append(text)

        name = properties.get("name")
        street = properties.get("street")
        house_number = properties.get("housenumber")
        if name:
            add(name)
        if street:
            add(f"{street} {house_number}".strip())
        elif house_number:
            add(house_number)

        locality = properties.get("city") or properties.get("town") or properties.get("village")
        postcode = properties.get("postcode")
        if postcode or locality:
            add(" ".join(str(value) for value in (postcode, locality) if value).strip())

        for key in ("district", "county", "state", "country"):
            add(properties.get(key))
        return ", ".join(parts) or fallback

    def _search_photon(self, query: str) -> list[dict[str, Any]]:
        params: list[tuple[str, Any]] = [
            ("q", query),
            ("limit", 12),
            ("lang", "de"),
        ]
        if self._provider_name() == "dwd":
            params.extend(
                [
                    ("countrycode", "DE"),
                    (
                        "bbox",
                        f"{GERMANY_BOUNDS['lon_min']},{GERMANY_BOUNDS['lat_min']},"
                        f"{GERMANY_BOUNDS['lon_max']},{GERMANY_BOUNDS['lat_max']}",
                    ),
                ]
            )
        else:
            params.append(("bbox", "-45,20,75,80"))

        payload = self._request_geocoder_json(
            "Photon", PHOTON_SEARCH_URL, params=params
        )
        features = payload.get("features", []) if isinstance(payload, dict) else []
        allowed_countries = set(EUROPE_COUNTRY_CODES)
        results: list[dict[str, Any]] = []
        for feature in features:
            try:
                properties = feature.get("properties", {}) or {}
                coordinates = feature["geometry"]["coordinates"]
                longitude = float(coordinates[0])
                latitude = float(coordinates[1])
                country_code = str(properties.get("countrycode", "")).lower()
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if self._provider_name() == "dwd" and country_code not in {"", "de"}:
                continue
            if (
                self._provider_name() == "cerra"
                and country_code
                and country_code not in allowed_countries
            ):
                continue
            if not self._inside_provider_domain(latitude, longitude):
                continue
            results.append(
                {
                    "lat": latitude,
                    "lon": longitude,
                    "label": self._photon_label(properties, query),
                }
            )
            if len(results) >= 8:
                break
        return results

    def _search_nominatim(self, query: str) -> list[dict[str, Any]]:
        payload = self._request_geocoder_json(
            "Nominatim",
            NOMINATIM_SEARCH_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "addressdetails": 1,
                "countrycodes": self._geocoder_countrycodes(),
                "limit": 8,
            },
        )
        if not isinstance(payload, list):
            return []
        results: list[dict[str, Any]] = []
        for item in payload:
            try:
                latitude = float(item["lat"])
                longitude = float(item["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            if not self._inside_provider_domain(latitude, longitude):
                continue
            results.append(
                {
                    "lat": latitude,
                    "lon": longitude,
                    "label": item.get("display_name", query),
                }
            )
        return results

    def _geocode_address(
        self, query: str
    ) -> tuple[list[dict[str, Any]], str, bool]:
        cache_key = (self._provider_name(), self._normalise_geocode_query(query))
        cached = self._geocode_cache.get(cache_key)
        if cached is not None:
            results, service = cached
            return [dict(item) for item in results], service, True

        errors: list[str] = []
        searchers = {
            "photon": ("Photon", self._search_photon),
            "nominatim": ("Nominatim", self._search_nominatim),
        }
        for configured_name in GEOCODER_ORDER:
            entry = searchers.get(configured_name)
            if entry is None:
                continue
            service, searcher = entry
            try:
                results = searcher(query)
            except (GeocoderRateLimited, requests.RequestException, ValueError) as exc:
                errors.append(f"{service}: {exc}")
                continue
            if results:
                self._geocode_cache[cache_key] = ([dict(item) for item in results], service)
                return results, service, False

        if errors:
            raise GeocoderUnavailable("; ".join(errors))
        self._geocode_cache[cache_key] = ([], "")
        return [], "", False

    def _search_address(self, _button: Any = None) -> None:
        query = self.address_input.value.strip()
        if not query:
            self.address_status.value = (
                "<span style='color:#b00020'><b>Please enter an address.</b></span>"
            )
            return

        # Clear the previous dropdown selection before issuing the request.
        # This guarantees that selecting result index 0 fires again on every
        # search, including two consecutive searches with a first result.
        self._reset_address_results("Searching …")
        self.address_search_button.disabled = True
        self.address_search_button.description = "Searching …"
        scope = self._location_scope_label()
        self.address_status.value = (
            f"Searching the address in {html.escape(scope)} using an "
            "OpenStreetMap address service …"
        )

        try:
            self._geocode_results, geocoder_name, from_cache = self._geocode_address(query)

            if not self._geocode_results:
                self.address_results.options = [("No matching address found", None)]
                self.address_results.value = None
                self.address_results.disabled = True
                self.address_status.value = (
                    "<span style='color:#b00020'><b>No matching address in "
                    f"{html.escape(scope)} was found. Refine the address or choose "
                    "the point on the map.</b></span>"
                )
                return

            result_options = []
            for index, result in enumerate(self._geocode_results):
                label = str(result["label"])
                if len(label) > 120:
                    label = label[:117] + "…"
                result_options.append((label, index))

            self.address_results.options = [
                ("Select an address result", None),
                *result_options,
            ]
            self.address_results.value = None
            self.address_results.disabled = False
            self.address_results.value = 0
            cache_note = " (cache)" if from_cache else ""
            self.address_status.value = (
                f"<b>{len(self._geocode_results)} result(s) found via "
                f"{html.escape(geocoder_name)}{cache_note}.</b> "
                "Select the desired address from the list."
            )
        except GeocoderUnavailable as exc:
            self.address_results.options = [("Address search temporarily unavailable", None)]
            self.address_results.value = None
            self.address_results.disabled = True
            self.address_status.value = (
                "<span style='color:#b00020'><b>Address search is temporarily "
                "unavailable.</b> The public geocoding services rejected or could "
                f"not answer the request ({html.escape(str(exc))}). Wait before "
                "trying again, or select the location on the map.</span>"
            )
        except (TypeError, ValueError, KeyError) as exc:
            self.address_status.value = (
                "<span style='color:#b00020'><b>The address result could not be "
                f"processed:</b> {html.escape(str(exc))}</span>"
            )
        finally:
            self.address_search_button.disabled = False
            self.address_search_button.description = "Search address"

    def _select_address_result(self, change: dict[str, Any]) -> None:
        if change.get("name") != "value" or change.get("new") is None:
            return
        index = int(change["new"])
        if not 0 <= index < len(self._geocode_results):
            return
        result = self._geocode_results[index]
        if self._set_location(
            result["lat"],
            result["lon"],
            source=f"Address: {result['label']}",
            zoom=15,
        ):
            self.address_status.value = (
                "<span style='color:#187b20'><b>Address selected.</b></span> "
                "You can refine the location by clicking on the map or dragging "
                "the marker."
            )

    def _select_location_on_map(self, **kwargs: Any) -> None:
        if kwargs.get("type") != "click":
            return
        coordinates = kwargs.get("coordinates")
        if not coordinates or len(coordinates) < 2:
            return
        latitude, longitude = coordinates[:2]
        if self._set_location(latitude, longitude, source="Map click"):
            self.address_status.value = (
                "<span style='color:#187b20'><b>Map location selected.</b></span> "
                "The marker can be dragged for fine adjustment."
            )

    def _marker_location_changed(self, change: dict[str, Any]) -> None:
        if self._updating_marker or change.get("name") != "location":
            return
        new_location = change.get("new")
        if not new_location or len(new_location) < 2:
            return
        latitude, longitude = new_location[:2]
        if self._inside_provider_domain(latitude, longitude):
            self.namespace.update(
                {
                    "lat": float(latitude),
                    "lon": float(longitude),
                    "location_label": "Dragged marker",
                }
            )
            self._refresh_location_status("Dragged marker")
            self.address_status.value = (
                "<span style='color:#187b20'><b>Marker location selected.</b></span>"
            )
        else:
            self._set_location(
                self.lat,
                self.lon,
                source=str(self.namespace["location_label"]),
            )
            self.address_status.value = (
                "<span style='color:#b00020'><b>The marker must remain within "
                f"{html.escape(self._location_scope_description())}.</b></span>"
            )

    # ------------------------------------------------------------------
    # Period selection
    # ------------------------------------------------------------------
    def _build_period_controls(self) -> None:
        self.start_date_picker = widgets.DatePicker(
            description="Start date:",
            value=self._initial_start.date(),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px"),
        )
        self.start_hour_dropdown = widgets.Dropdown(
            options=HOUR_OPTIONS,
            value=int(self._initial_start.hour),
            description="Start hour:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="245px"),
        )
        self.end_date_picker = widgets.DatePicker(
            description="End date:",
            value=self._initial_end.date(),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="280px"),
        )
        self.end_hour_dropdown = widgets.Dropdown(
            options=HOUR_OPTIONS,
            value=int(self._initial_end.hour),
            description="End hour:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="245px"),
        )
        self.period_status = widgets.HTML()

    def _update_period(self, change: dict[str, Any] | None = None) -> None:
        del change
        if self.start_date_picker.value is None or self.end_date_picker.value is None:
            self.download_button.disabled = True
            self.period_status.value = (
                "<span style='color:#b00020'><b>Please select both dates.</b></span>"
            )
            return

        start_utc = self._format_utc(
            self.start_date_picker.value,
            self.start_hour_dropdown.value,
        )
        end_utc = self._format_utc(
            self.end_date_picker.value,
            self.end_hour_dropdown.value,
        )
        self.namespace.update({"start_UTC": start_utc, "end_UTC": end_utc})

        start_timestamp = pd.Timestamp(start_utc)
        end_timestamp = pd.Timestamp(end_utc)
        valid = end_timestamp >= start_timestamp
        self.download_button.disabled = not valid

        if valid:
            hours = int((end_timestamp - start_timestamp).total_seconds() // 3600) + 1
            self.period_status.value = (
                f"<b>Selected period:</b> <code>{start_utc}</code> to "
                f"<code>{end_utc}</code> ({hours:,} hourly values maximum)"
            )
        else:
            self.period_status.value = (
                "<span style='color:#b00020'><b>The end must not be earlier than "
                "the start.</b></span>"
            )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    def _build_download_controls(self) -> None:
        default_directory = str(
            self._normalise_output_directory(self._initial_output_directory)
        )
        default_filename = self._initial_output_filename or self._filename()

        self.output_directory_input = widgets.Text(
            value=default_directory,
            description="Output directory:",
            placeholder="Directory for the CSV file",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="760px"),
        )
        self.output_filename_input = widgets.Text(
            value=default_filename,
            description="File name:",
            placeholder=self._filename(),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="560px"),
        )
        self.reset_output_filename_button = widgets.Button(
            description="Use default name",
            icon="refresh",
            layout=widgets.Layout(width="190px"),
            tooltip="Use the default file name for the selected climate variable",
        )
        self.output_path_status = widgets.HTML()
        self.download_button = widgets.Button(
            description="Download and save CSV",
            icon="download",
            button_style="success",
            layout=widgets.Layout(width="260px", height="42px"),
        )
        self.download_output = widgets.Output()
        self._update_output_path_status()

    def _update_output_path_status(self, change: dict[str, Any] | None = None) -> None:
        del change
        try:
            output_path = self._resolve_output_path(create_directory=False)
            self.output_path_status.value = (
                "<b>CSV output:</b> "
                f"<code>{html.escape(str(output_path))}</code>"
            )
        except (OSError, ValueError) as exc:
            self.output_path_status.value = (
                "<span style='color:#b00020'><b>Invalid output path:</b> "
                f"{html.escape(str(exc))}</span>"
            )

    def _output_filename_changed(self, change: dict[str, Any]) -> None:
        if self._updating_output_filename or change.get("name") != "value":
            return
        self._custom_output_filename = True
        self._update_output_path_status()

    def _output_directory_changed(self, change: dict[str, Any]) -> None:
        if change.get("name") == "value":
            self._update_output_path_status()

    def _reset_output_filename(self, _button: Any = None) -> None:
        self._custom_output_filename = False
        self._updating_output_filename = True
        try:
            self.output_filename_input.value = self._filename()
        finally:
            self._updating_output_filename = False
        self._update_output_path_status()

    def _climate_variable_changed(self, change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        if not self._custom_output_filename:
            self._updating_output_filename = True
            try:
                self.output_filename_input.value = self._filename(str(change["new"]))
            finally:
                self._updating_output_filename = False
        self._update_output_path_status()

    def _download_selected_data(self, _button: Any = None) -> None:
        self._update_period()
        if self.download_button.disabled:
            return

        try:
            output_path = self._resolve_output_path(create_directory=True)
        except (OSError, ValueError) as exc:
            self.download_output.clear_output(wait=True)
            with self.download_output:
                print(f"Invalid output path: {exc}")
            self._update_output_path_status()
            return

        self.download_button.disabled = True
        self.download_button.description = "Downloading …"
        self.download_output.clear_output(wait=True)

        with self.download_output:
            print(
                f"Downloading {self.climate_variable} for "
                f"lat={self.lat:.6f}, lon={self.lon:.6f} from "
                f"{self.namespace['start_UTC']} to {self.namespace['end_UTC']} …"
            )
            try:
                dataframe = self.extractor(
                    var=self.climate_variable,
                    lon=self.lon,
                    lat=self.lat,
                    start=self.namespace["start_UTC"],
                    end=self.namespace["end_UTC"],
                )
                dataframe.to_csv(output_path, index=False)
                self.namespace.update(
                    {
                        "df": dataframe,
                        "fn": str(output_path),
                        "output_directory": str(output_path.parent),
                        "output_filename": output_path.name,
                    }
                )
                print(f"{len(dataframe)} rows written to {output_path}.")
                print("The following analysis cells can now be executed.")
            except Exception as exc:  # keep notebook interaction responsive
                print(f"Download failed: {type(exc).__name__}: {exc}")
                print(
                    "Check the selected period, network connection and HOSTRADA "
                    "data availability."
                )
            finally:
                self.download_button.description = "Download and save CSV"
                self._update_period()
                self._update_output_path_status()

    # ------------------------------------------------------------------
    # Wiring and layout
    # ------------------------------------------------------------------
    def _connect_callbacks(self) -> None:
        self.address_search_button.on_click(self._search_address)
        # ipywidgets 7/8 exposes on_submit on Text (deprecated in some releases
        # but still functional).  Keep button search as the primary path and
        # support Enter where available.
        on_submit = getattr(self.address_input, "on_submit", None)
        if callable(on_submit):
            on_submit(self._search_address)
        self.address_results.observe(self._select_address_result, names="value")
        self.location_map.on_interaction(self._select_location_on_map)
        self.location_marker.observe(self._marker_location_changed, names="location")
        self.download_button.on_click(self._download_selected_data)
        self.output_directory_input.observe(
            self._output_directory_changed, names="value"
        )
        self.output_filename_input.observe(
            self._output_filename_changed, names="value"
        )
        self.reset_output_filename_button.on_click(self._reset_output_filename)

        climate_dropdown = self.namespace.get("climate_dropdown")
        if hasattr(climate_dropdown, "observe"):
            climate_dropdown.observe(self._climate_variable_changed, names="value")

        for control in (
            self.start_date_picker,
            self.start_hour_dropdown,
            self.end_date_picker,
            self.end_hour_dropdown,
        ):
            control.observe(self._update_period, names="value")

    def _build_layout(self) -> None:
        self.address_scope_note = widgets.HTML()
        self.refresh_provider_scope()
        self.location_box = widgets.VBox(
            [
                widgets.HTML("<h3>1. Select climate location</h3>"),
                widgets.HBox([self.address_input, self.address_search_button]),
                self.address_results,
                self.address_status,
                self.location_status,
                self.location_map,
                self.address_scope_note,
            ]
        )
        self.period_box = widgets.VBox(
            [
                widgets.HTML("<h3>2. Select download period</h3>"),
                widgets.HBox([self.start_date_picker, self.start_hour_dropdown]),
                widgets.HBox([self.end_date_picker, self.end_hour_dropdown]),
                self.period_status,
            ]
        )
        self.download_box = widgets.VBox(
            [
                widgets.HTML("<h3>3. Download and save selected HOSTRADA data</h3>"),
                self.output_directory_input,
                widgets.HBox(
                    [
                        self.output_filename_input,
                        self.reset_output_filename_button,
                    ]
                ),
                self.output_path_status,
                self.download_button,
                self.download_output,
            ]
        )
        self.widget = widgets.VBox(
            [self.location_box, self.period_box, self.download_box]
        )

    def _publish_widget_references(self) -> None:
        """Keep the original widget variable names available in the notebook."""
        names = (
            "address_input",
            "address_search_button",
            "address_results",
            "address_status",
            "address_scope_note",
            "location_status",
            "location_map",
            "location_marker",
            "start_date_picker",
            "start_hour_dropdown",
            "end_date_picker",
            "end_hour_dropdown",
            "period_status",
            "output_directory_input",
            "output_filename_input",
            "reset_output_filename_button",
            "output_path_status",
            "download_button",
            "download_output",
            "location_box",
            "period_box",
            "download_box",
        )
        self.namespace.update({name: getattr(self, name) for name in names})

    def show(self) -> None:
        """Additive convenience alias; the upstream notebook uses ``display(ui.widget)``."""
        display(self.widget)


def create_hostrada_point_ui(
    namespace: MutableMapping[str, Any],
    extractor: Callable[..., pd.DataFrame],
    **kwargs: Any,
) -> HostradaPointUI:
    """Create the interactive point selector used by the notebook."""
    return HostradaPointUI(namespace=namespace, extractor=extractor, **kwargs)
