"""Full interactive inputs used by ``hostradaGenerateWeatherFiles.ipynb``.

All selected values and widgets are published into the supplied notebook
namespace, preserving the original notebook workflow.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from datetime import date
from typing import Any

import pandas as pd
import requests
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import Map, Marker, basemaps

from . import hostrada as hs

HOUR_OPTIONS = [(f"{hour:02d}:00 UTC", hour) for hour in range(24)]


class WeatherLocationInput:
    def __init__(
        self,
        namespace: MutableMapping[str, Any],
        *,
        initial_lon: float = 13.32259,
        initial_lat: float = 52.51712,
        initial_location_name: str = "Berlin",
        initial_altitude: float = 34.0,
        show: bool = True,
    ) -> None:
        self.namespace = namespace
        self.results: list[dict[str, Any]] = []
        self._moving = False
        center = (float(initial_lat), float(initial_lon))
        self.address = widgets.Text(
            value=initial_location_name,
            description="Address:",
            layout=widgets.Layout(width="650px"),
        )
        self.search_button = widgets.Button(description="Search", icon="search", button_style="info")
        self.search_results = widgets.Dropdown(description="Matches:", layout=widgets.Layout(width="760px"))
        self.location_name = widgets.Text(value=initial_location_name, description="Location:", layout=widgets.Layout(width="460px"))
        self.longitude = widgets.FloatText(value=float(initial_lon), description="Longitude:")
        self.latitude = widgets.FloatText(value=float(initial_lat), description="Latitude:")
        self.altitude = widgets.FloatText(value=float(initial_altitude), description="Altitude m:")
        self.map = Map(center=center, zoom=6 if hs.get_provider().name != "dwd" else 7,
                       basemap=basemaps.OpenStreetMap.Mapnik,
                       scroll_wheel_zoom=True,
                       layout=widgets.Layout(height="470px", width="100%"))
        self.marker = Marker(location=center, draggable=True, title=initial_location_name)
        self.map.add(self.marker)
        self.status = widgets.HTML()
        self.widget = widgets.VBox([
            widgets.HTML("<h3>1. Select weather-file location</h3>"),
            widgets.HBox([self.address, self.search_button]),
            self.search_results,
            widgets.HBox([self.location_name, self.altitude]),
            widgets.HBox([self.longitude, self.latitude]),
            self.status,
            self.map,
            widgets.HTML("<small>Search with OpenStreetMap/Photon or drag the marker. "
                         "DWD is restricted to Germany; CERRA supports Europe.</small>"),
        ])
        self.search_button.on_click(self._search)
        self.search_results.observe(self._select_result, names="value")
        self.marker.observe(self._marker_moved, names="location")
        self.longitude.observe(self._coordinates_changed, names="value")
        self.latitude.observe(self._coordinates_changed, names="value")
        self.location_name.observe(self._publish, names="value")
        self.altitude.observe(self._publish, names="value")
        self._publish()
        namespace.update(
            location_input=self,
            location_input_widget=self.widget,
            location_map=self.map,
            location_marker=self.marker,
            address_input=self.address,
            address_search_button=self.search_button,
            address_results=self.search_results,
            lon_widget=self.longitude,
            lat_widget=self.latitude,
            altitude_widget=self.altitude,
            location_name_widget=self.location_name,
        )
        if show:
            display(self.widget)

    def _valid(self, lat: float, lon: float) -> bool:
        if hs.get_provider().name == "dwd":
            return 47.0 <= lat <= 55.5 and 5.5 <= lon <= 15.6
        return 20.0 <= lat <= 80.0 and -45.0 <= lon <= 75.0

    def _publish(self, _change: Any = None) -> None:
        self.namespace.update(
            selected_lon=float(self.longitude.value),
            selected_lat=float(self.latitude.value),
            lon=float(self.longitude.value),
            lat=float(self.latitude.value),
            longitude=float(self.longitude.value),
            latitude=float(self.latitude.value),
            location_name=self.location_name.value.strip() or "Weather location",
            altitude=float(self.altitude.value),
        )
        valid = self._valid(float(self.latitude.value), float(self.longitude.value))
        self.status.value = (
            f"<b>Selected:</b> {self.latitude.value:.5f}, {self.longitude.value:.5f}; "
            f"{self.altitude.value:.1f} m"
            if valid else
            "<span style='color:#b00020'><b>The point is outside the active provider domain.</b></span>"
        )

    def _coordinates_changed(self, _change: Any = None) -> None:
        if self._moving:
            return
        self._moving = True
        self.marker.location = (float(self.latitude.value), float(self.longitude.value))
        self.map.center = self.marker.location
        self._moving = False
        self._publish()

    def _marker_moved(self, change: dict[str, Any]) -> None:
        if self._moving:
            return
        self._moving = True
        lat, lon = change["new"]
        self.latitude.value = float(lat)
        self.longitude.value = float(lon)
        self._moving = False
        self._publish()

    def _search(self, _button: Any = None) -> None:
        query = self.address.value.strip()
        if not query:
            self.status.value = "<span style='color:#b00020'>Enter an address.</span>"
            return
        try:
            response = requests.get(
                "https://photon.komoot.io/api",
                params={"q": query, "limit": 8, "lang": "en"},
                headers={"User-Agent": "hostrada4py-weather-input/0.42.0"},
                timeout=30,
            )
            response.raise_for_status()
            self.results = response.json().get("features", [])
            options = []
            for index, feature in enumerate(self.results):
                props = feature.get("properties", {})
                label = ", ".join(str(value) for value in (
                    props.get("name"), props.get("street"), props.get("postcode"),
                    props.get("city") or props.get("county"), props.get("country"),
                ) if value)
                options.append((label or query, index))
            self.search_results.options = options
            if options:
                self.search_results.value = options[0][1]
            else:
                self.status.value = "<span style='color:#8a4b00'>No address found.</span>"
        except Exception as exc:
            self.status.value = f"<span style='color:#b00020'>Address search failed: {exc}</span>"

    def _select_result(self, change: dict[str, Any]) -> None:
        if change["new"] is None or not self.results:
            return
        feature = self.results[int(change["new"])]
        lon, lat = feature["geometry"]["coordinates"]
        props = feature.get("properties", {})
        name = props.get("name") or props.get("city") or self.address.value
        self.location_name.value = str(name)
        self.longitude.value = float(lon)
        self.latitude.value = float(lat)


class WeatherPeriodInput:
    def __init__(
        self,
        namespace: MutableMapping[str, Any],
        *,
        initial_start: str = "2025-01-01T00:00:00",
        initial_end: str = "2025-12-31T23:00:00",
        show: bool = True,
    ) -> None:
        self.namespace = namespace
        start = pd.Timestamp(initial_start)
        end = pd.Timestamp(initial_end)
        self.start_date = widgets.DatePicker(value=start.date(), description="Start date:")
        self.start_hour = widgets.Dropdown(options=HOUR_OPTIONS, value=int(start.hour), description="Start hour:")
        self.end_date = widgets.DatePicker(value=end.date(), description="End date:")
        self.end_hour = widgets.Dropdown(options=HOUR_OPTIONS, value=int(end.hour), description="End hour:")
        self.timezone = widgets.Dropdown(
            options=["UTC", "Europe/Berlin", "Europe/London", "Europe/Paris", "Europe/Vienna",
                     "Europe/Oslo", "Europe/Stockholm", "Europe/Helsinki", "Europe/Madrid", "Europe/Rome"],
            value="Europe/Berlin", description="Output timezone:", layout=widgets.Layout(width="420px")
        )
        self.status = widgets.HTML()
        self.widget = widgets.VBox([
            widgets.HTML("<h3>2. Select period</h3>"),
            widgets.HBox([self.start_date, self.start_hour]),
            widgets.HBox([self.end_date, self.end_hour]),
            self.timezone, self.status,
        ])
        for control in (self.start_date, self.start_hour, self.end_date, self.end_hour, self.timezone):
            control.observe(self._publish, names="value")
        self._publish()
        namespace.update(
            weather_period_input=self,
            weather_period_widget=self.widget,
            start_date_picker=self.start_date,
            start_hour_dropdown=self.start_hour,
            end_date_picker=self.end_date,
            end_hour_dropdown=self.end_hour,
            timezone_dropdown=self.timezone,
        )
        if show:
            display(self.widget)

    def _timestamp(self, selected_date: date | None, hour: int) -> pd.Timestamp | None:
        if selected_date is None:
            return None
        return pd.Timestamp(selected_date) + pd.Timedelta(hours=int(hour))

    def _publish(self, _change: Any = None) -> None:
        start = self._timestamp(self.start_date.value, self.start_hour.value)
        end = self._timestamp(self.end_date.value, self.end_hour.value)
        if start is None or end is None or end < start:
            self.status.value = "<span style='color:#b00020'>Select a valid period.</span>"
            return
        start_text = start.strftime("%Y-%m-%dT%H:%M:%S")
        end_text = end.strftime("%Y-%m-%dT%H:%M:%S")
        hours = int((end - start).total_seconds() / 3600) + 1
        self.namespace.update(
            selected_start=start_text, selected_end=end_text,
            start_UTC=start_text, end_UTC=end_text,
            start_utc=start_text, end_utc=end_text,
            timezone=self.timezone.value, tz=self.timezone.value,
        )
        self.status.value = f"<b>{hours:,}</b> hourly values from {start_text} to {end_text} UTC."


class WeatherFileFormatInput:
    def __init__(self, namespace: MutableMapping[str, Any], *, show=True):
        self.namespace = namespace
        self.formats = widgets.SelectMultiple(
            options=[("EnergyPlus EPW", "energyplus"), ("IDA ICE PRN", "ida_ice"),
                     ("Polysun CSV", "polysun"), ("SimStadt TMY3", "simstadt"),
                     ("BuildingSystems MOS", "buildingsystems")],
            value=("energyplus",), description="Formats:",
            layout=widgets.Layout(width="430px", height="150px"),
        )
        self.output_directory = widgets.Text(value="weather_files", description="Directory:", layout=widgets.Layout(width="620px"))
        self.apply_weather_correction = widgets.Checkbox(value=False, description="Weather-correct diffuse radiation")
        self.widget = widgets.VBox([
            widgets.HTML("<h3>3. Select output formats</h3>"),
            self.formats, self.output_directory, self.apply_weather_correction,
        ])
        for control in (self.formats, self.output_directory, self.apply_weather_correction):
            control.observe(self._publish, names="value")
        self._publish()
        namespace.update(
            weather_file_format_input=self,
            weather_file_format_widget=self.widget,
            weather_formats_widget=self.formats,
            weather_output_directory_widget=self.output_directory,
        )
        if show:
            display(self.widget)

    def _publish(self, _change: Any = None) -> None:
        self.namespace.update(
            weather_formats=tuple(self.formats.value),
            output_directory=self.output_directory.value,
            apply_weather_correction=bool(self.apply_weather_correction.value),
        )


def setup_location_input(namespace, initial_lon=13.32259, initial_lat=52.51712,
                         initial_location_name="Berlin", initial_altitude=34.0, **kwargs):
    return WeatherLocationInput(
        namespace,
        initial_lon=initial_lon,
        initial_lat=initial_lat,
        initial_location_name=initial_location_name,
        initial_altitude=initial_altitude,
        **kwargs,
    )


def setup_weather_period_input(namespace, initial_start="2025-01-01T00:00:00",
                               initial_end="2025-12-31T23:00:00", **kwargs):
    return WeatherPeriodInput(namespace, initial_start=initial_start, initial_end=initial_end, **kwargs
    )                            


def setup_weather_file_format_input(namespace, **kwargs):
    return WeatherFileFormatInput(namespace, **kwargs).widget
