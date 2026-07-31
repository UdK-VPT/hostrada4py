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
from pathlib import Path
from collections.abc import Callable, MutableMapping
from typing import Any

import pandas as pd
import requests
import ipywidgets as widgets
from ipyleaflet import Map, Marker, basemaps


GERMANY_BOUNDS = {
    "lon_min": 5.5,
    "lon_max": 15.6,
    "lat_min": 47.0,
    "lat_max": 55.5,
}

HOUR_OPTIONS = [(f"{hour:02d}:00 UTC", hour) for hour in range(24)]

HOSTRADA_POINT_UI_API_VERSION = "2.0-csv-export"


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
        self.address_status = widgets.HTML(
            value=(
                "Enter an address in Germany and click <b>Search address</b>, "
                "or click directly on the map. The marker can also be dragged."
            )
        )
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
        if not self._inside_germany(latitude, longitude):
            self.address_status.value = (
                "<span style='color:#b00020'><b>The selected point is outside "
                "Germany. Please choose a location within Germany.</b></span>"
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

    def _search_address(self, _button: Any = None) -> None:
        query = self.address_input.value.strip()
        if not query:
            self.address_status.value = (
                "<span style='color:#b00020'><b>Please enter an address.</b></span>"
            )
            return

        self.address_search_button.disabled = True
        self.address_search_button.description = "Searching …"
        self.address_status.value = "Searching the address in OpenStreetMap Nominatim …"

        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": query,
                    "format": "jsonv2",
                    "addressdetails": 1,
                    "countrycodes": "de",
                    "limit": 8,
                },
                headers={
                    "User-Agent": (
                        "hostrada4py-jupyter/1.0 "
                        "(interactive climate location selector)"
                    )
                },
                timeout=20,
            )
            response.raise_for_status()
            raw_results = response.json()

            self._geocode_results = [
                {
                    "lat": float(item["lat"]),
                    "lon": float(item["lon"]),
                    "label": item.get("display_name", query),
                }
                for item in raw_results
                if self._inside_germany(item["lat"], item["lon"])
            ]

            if not self._geocode_results:
                self.address_results.options = [("No matching address found", None)]
                self.address_results.value = None
                self.address_results.disabled = True
                self.address_status.value = (
                    "<span style='color:#b00020'><b>No matching address in Germany "
                    "was found. Refine the address or choose the point on the map."
                    "</b></span>"
                )
                return

            result_options = []
            for index, result in enumerate(self._geocode_results):
                label = str(result["label"])
                if len(label) > 120:
                    label = label[:117] + "…"
                result_options.append((label, index))

            self.address_results.disabled = False
            self.address_results.options = result_options
            self.address_results.value = 0
            self.address_status.value = (
                f"<b>{len(self._geocode_results)} result(s) found.</b> "
                "Select the desired address from the list."
            )
        except requests.RequestException as exc:
            self.address_results.options = [("Address search unavailable", None)]
            self.address_results.value = None
            self.address_results.disabled = True
            self.address_status.value = (
                "<span style='color:#b00020'><b>Address search failed:</b> "
                f"{html.escape(str(exc))}. You can still select the location on "
                "the map.</span>"
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
        if self._inside_germany(latitude, longitude):
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
                "Germany.</b></span>"
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
        self.location_box = widgets.VBox(
            [
                widgets.HTML("<h3>1. Select climate location</h3>"),
                widgets.HBox([self.address_input, self.address_search_button]),
                self.address_results,
                self.address_status,
                self.location_status,
                self.location_map,
                widgets.HTML(
                    "<small>Address search uses OpenStreetMap Nominatim. "
                    "The selected point is restricted to Germany.</small>"
                ),
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


def create_hostrada_point_ui(
    namespace: MutableMapping[str, Any],
    extractor: Callable[..., pd.DataFrame],
    **kwargs: Any,
) -> HostradaPointUI:
    """Create the interactive point selector used by the notebook."""
    return HostradaPointUI(namespace=namespace, extractor=extractor, **kwargs)
