"""Interactive inputs used by ``hostradaGenerateWeatherFiles.ipynb``.

The functions in this module intentionally write the selected values and widget
objects into the notebook namespace passed by the caller.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, MutableMapping


ALEXANDERPLATZ_LAT = 52.521918
ALEXANDERPLATZ_LON = 13.413215


def _require_namespace(namespace: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Validate and return the mutable notebook namespace."""
    if not hasattr(namespace, "__setitem__"):
        raise TypeError("namespace must be a mutable mapping, for example globals()")
    return namespace


def setup_location_input(namespace: MutableMapping[str, Any]) -> None:
    """Display the map/address input and maintain the original notebook globals.

    Parameters
    ----------
    namespace:
        The notebook global namespace, normally supplied as ``globals()``.
    """
    ns = _require_namespace(namespace)

    ns["ALEXANDERPLATZ_LAT"] = ALEXANDERPLATZ_LAT
    ns["ALEXANDERPLATZ_LON"] = ALEXANDERPLATZ_LON
    ns["selected_lat"] = ALEXANDERPLATZ_LAT
    ns["selected_lon"] = ALEXANDERPLATZ_LON
    ns["selected_address"] = "Berlin Alexanderplatz"

    try:
        import json
        import urllib
        import urllib.error
        import urllib.parse
        import urllib.request

        from IPython.display import display
        from ipyleaflet import LayersControl, Map, Marker, basemaps
        from ipywidgets import Button, FloatText, HBox, HTML, Layout, Text, VBox

        lat_widget = FloatText(
            value=ns["selected_lat"],
            description="Latitude",
            step=0.000001,
            layout=Layout(width="260px"),
        )
        lon_widget = FloatText(
            value=ns["selected_lon"],
            description="Longitude",
            step=0.000001,
            layout=Layout(width="260px"),
        )
        address_widget = Text(
            value=ns["selected_address"],
            placeholder="Enter address, e.g. Hardenbergstraße 33, Berlin",
            description="Address",
            layout=Layout(width="620px"),
        )
        search_button = Button(
            description="Search Address",
            button_style="primary",
            tooltip="Search for an address using OpenStreetMap",
            icon="search",
            layout=Layout(width="160px"),
        )
        status_widget = HTML(value="")

        weather_location_marker = Marker(
            location=(ns["selected_lat"], ns["selected_lon"]),
            draggable=True,
            title="Selected Location",
        )

        weather_location_map = Map(
            center=(ALEXANDERPLATZ_LAT, ALEXANDERPLATZ_LON),
            zoom=13,
            basemap=basemaps.OpenStreetMap.Mapnik,
            scroll_wheel_zoom=True,
            layout={"height": "520px", "width": "100%"},
        )
        weather_location_map.add(weather_location_marker)
        weather_location_map.add(LayersControl(position="topright"))

        ns["_updating_widgets"] = False

        def _set_selected_location(
            lat: float,
            lon: float,
            update_marker: bool = True,
            center_map: bool = False,
            zoom: int | None = None,
            address: str | None = None,
        ) -> None:
            """Synchronize marker, widgets, and the selected location globals."""
            ns["selected_lat"] = float(lat)
            ns["selected_lon"] = float(lon)
            if address is not None:
                ns["selected_address"] = str(address)

            ns["_updating_widgets"] = True
            try:
                lat_widget.value = ns["selected_lat"]
                lon_widget.value = ns["selected_lon"]
                if address is not None:
                    address_widget.value = ns["selected_address"]
            finally:
                ns["_updating_widgets"] = False

            if update_marker:
                weather_location_marker.location = (
                    ns["selected_lat"],
                    ns["selected_lon"],
                )
            if center_map:
                weather_location_map.center = (
                    ns["selected_lat"],
                    ns["selected_lon"],
                )
                if zoom is not None:
                    weather_location_map.zoom = zoom

        def _on_marker_moved(change: dict[str, Any]) -> None:
            lat, lon = change["new"]
            _set_selected_location(lat, lon, update_marker=False)
            status_widget.value = (
                "Gewählter Standort: "
                f"lat={ns['selected_lat']:.6f}, lon={ns['selected_lon']:.6f}"
            )

        def _on_map_clicked(**kwargs: Any) -> None:
            if kwargs.get("type") == "click":
                lat, lon = kwargs.get("coordinates")
                _set_selected_location(lat, lon, update_marker=True)
                status_widget.value = (
                    "Gewählter Standort: "
                    f"lat={ns['selected_lat']:.6f}, lon={ns['selected_lon']:.6f}"
                )

        def _on_widget_changed(change: dict[str, Any]) -> None:
            del change
            if not ns["_updating_widgets"]:
                _set_selected_location(
                    lat_widget.value,
                    lon_widget.value,
                    update_marker=True,
                    center_map=True,
                )
                status_widget.value = (
                    "Gewählter Standort: "
                    f"lat={ns['selected_lat']:.6f}, lon={ns['selected_lon']:.6f}"
                )

        def _geocode_address(address: str, *, timeout: int = 15) -> tuple[float, float, str]:
            """Search OSM/Nominatim and return ``(lat, lon, display_name)``."""
            query = address.strip()
            if not query:
                raise ValueError("Please, enter a address.")

            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                }
            )
            url = f"https://nominatim.openstreetmap.org/search?{params}"
            request = urllib.request.Request(
                url,
                headers={
                    # Nominatim requires an identifiable user agent.
                    "User-Agent": "hostrada4py-weather-notebook/1.0 (Jupyter Notebook)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))

            if not payload:
                raise LookupError(f"Keine Koordinaten für diese Adresse gefunden: {query}")

            hit = payload[0]
            return float(hit["lat"]), float(hit["lon"]), hit.get("display_name", query)

        def search_address(_: Any = None) -> None:
            """Search an address and place the marker at the resulting location."""
            search_button.disabled = True
            status_widget.value = "Search for an address using OpenStreetMap/Nominatim ..."
            try:
                lat, lon, label = _geocode_address(address_widget.value)
                _set_selected_location(
                    lat,
                    lon,
                    update_marker=True,
                    center_map=True,
                    zoom=15,
                    address=label,
                )
                status_widget.value = (
                    "Found: "
                    f"{label}<br>lat={ns['selected_lat']:.6f}, "
                    f"lon={ns['selected_lon']:.6f}"
                )
            except Exception as exc:
                status_widget.value = (
                    "<b>Address search failed:</b> "
                    f"{type(exc).__name__}: {exc}<br>"
                    "You can still select the location using a map or by entering coordinates."
                )
            finally:
                search_button.disabled = False

        weather_location_marker.observe(_on_marker_moved, names="location")
        weather_location_map.on_interaction(_on_map_clicked)
        lat_widget.observe(_on_widget_changed, names="value")
        lon_widget.observe(_on_widget_changed, names="value")
        search_button.on_click(search_address)
        address_widget.on_submit(search_address)

        # Preserve the names that were previously created directly in the cell.
        ns.update(
            {
                # Imports that were previously exposed by the notebook cell.
                "display": display,
                "Map": Map,
                "Marker": Marker,
                "basemaps": basemaps,
                "LayersControl": LayersControl,
                "FloatText": FloatText,
                "HBox": HBox,
                "VBox": VBox,
                "HTML": HTML,
                "Text": Text,
                "Button": Button,
                "Layout": Layout,
                "json": json,
                "urllib": urllib,
                "lat_widget": lat_widget,
                "lon_widget": lon_widget,
                "address_widget": address_widget,
                "search_button": search_button,
                "status_widget": status_widget,
                "weather_location_marker": weather_location_marker,
                "weather_location_map": weather_location_map,
                "_set_selected_location": _set_selected_location,
                "_on_marker_moved": _on_marker_moved,
                "_on_map_clicked": _on_map_clicked,
                "_on_widget_changed": _on_widget_changed,
                "_geocode_address": _geocode_address,
                "search_address": search_address,
            }
        )

        display(
            VBox(
                [
                    HTML(
                        "<b>Select Location:</b> Click on the map, move the marker, "
                        "or search for an address."
                    ),
                    HBox([address_widget, search_button]),
                    status_widget,
                    weather_location_map,
                    HBox([lat_widget, lon_widget]),
                    HTML(
                        "The following cells use <code>selected_lon</code> and "
                        "<code>selected_lat</code>."
                    ),
                ]
            )
        )

    except ImportError as exc:
        from IPython.display import Markdown, display

        ns.update({"display": display, "Markdown": Markdown})
        display(
            Markdown(
                "**Hint:** The interactive map requires `ipyleaflet` and `ipywidgets`. "
                "Install these packages using, for example, "
                "`pip install ipyleaflet ipywidgets`.\n\n"
                "Until then, the coordinates for Berlin-Alexanderplatz will be used "
                "as the default."
            )
        )
        print(f"Map widget not loaded: {exc}")


def setup_weather_period_input(namespace: MutableMapping[str, Any]) -> None:
    """Display the weather-period input and maintain the original globals."""
    ns = _require_namespace(namespace)
    ns["date"] = date
    ns["datetime"] = datetime

    # Default period: full year 2025
    ns["selected_start"] = "2025-01-01 00:00"
    ns["selected_end"] = "2025-12-31 23:00"
    ns["selected_start_datetime"] = datetime(2025, 1, 1, 0, 0)
    ns["selected_end_datetime"] = datetime(2025, 12, 31, 23, 0)

    try:
        from IPython.display import display
        from ipywidgets import DatePicker, Dropdown, HBox, HTML, Layout, VBox

        start_date_widget = DatePicker(
            description="Start date",
            value=date(2025, 1, 1),
            layout=Layout(width="260px"),
        )
        start_hour_widget = Dropdown(
            description="Start hour",
            options=[(f"{hour:02d}:00", hour) for hour in range(24)],
            value=0,
            layout=Layout(width="220px"),
        )
        end_date_widget = DatePicker(
            description="End date",
            value=date(2025, 12, 31),
            layout=Layout(width="260px"),
        )
        end_hour_widget = Dropdown(
            description="End hour",
            options=[(f"{hour:02d}:00", hour) for hour in range(24)],
            value=23,
            layout=Layout(width="220px"),
        )
        period_status_widget = HTML(value="")

        def _update_selected_period(*_: Any) -> None:
            """Synchronize date/hour widgets and weather-period globals."""
            if start_date_widget.value is None or end_date_widget.value is None:
                period_status_widget.value = "<b>Bitte Start- und Enddatum auswählen.</b>"
                return

            ns["selected_start_datetime"] = datetime.combine(
                start_date_widget.value,
                datetime.min.time(),
            ).replace(
                hour=int(start_hour_widget.value),
                minute=0,
                second=0,
                microsecond=0,
            )

            ns["selected_end_datetime"] = datetime.combine(
                end_date_widget.value,
                datetime.min.time(),
            ).replace(
                hour=int(end_hour_widget.value),
                minute=0,
                second=0,
                microsecond=0,
            )

            ns["selected_start"] = ns["selected_start_datetime"].strftime("%Y-%m-%d %H:%M")
            ns["selected_end"] = ns["selected_end_datetime"].strftime("%Y-%m-%d %H:%M")

            if ns["selected_start_datetime"] > ns["selected_end_datetime"]:
                period_status_widget.value = (
                    "<b style='color:#b00020'>Unvalid time period:</b> "
                    f"Start {ns['selected_start']} liegt nach Ende {ns['selected_end']}."
                )
            else:
                period_status_widget.value = (
                    f"Gewählter Zeitraum: <b>{ns['selected_start']}</b> bis "
                    f"<b>{ns['selected_end']}</b>"
                )

        for widget in [
            start_date_widget,
            start_hour_widget,
            end_date_widget,
            end_hour_widget,
        ]:
            widget.observe(_update_selected_period, names="value")

        _update_selected_period()

        # Preserve the names that were previously created directly in the cell.
        ns.update(
            {
                # Imports that were previously exposed by the notebook cell.
                "display": display,
                "DatePicker": DatePicker,
                "Dropdown": Dropdown,
                "HBox": HBox,
                "VBox": VBox,
                "HTML": HTML,
                "Layout": Layout,
                "start_date_widget": start_date_widget,
                "start_hour_widget": start_hour_widget,
                "end_date_widget": end_date_widget,
                "end_hour_widget": end_hour_widget,
                "period_status_widget": period_status_widget,
                "_update_selected_period": _update_selected_period,
            }
        )

        display(
            VBox(
                [
                    HTML(
                        "<b>Select time period:</b> Choose start and end date/hour "
                        "for the generated weather file."
                    ),
                    HBox([start_date_widget, start_hour_widget]),
                    HBox([end_date_widget, end_hour_widget]),
                    period_status_widget,
                    HTML(
                        "The following cells use <code>selected_start</code> and "
                        "<code>selected_end</code>."
                    ),
                ]
            )
        )

    except ImportError as exc:
        from IPython.display import Markdown, display

        ns.update({"display": display, "Markdown": Markdown})
        display(
            Markdown(
                "**Hint:** The interactive time-period selection requires `ipywidgets`. "
                "Install it using, for example, `pip install ipywidgets`.\n\n"
                "Until then, the default period `2025-01-01 00:00` to "
                "`2025-12-31 23:00` will be used."
            )
        )
        print(f"Time-period widget not loaded: {exc}")
