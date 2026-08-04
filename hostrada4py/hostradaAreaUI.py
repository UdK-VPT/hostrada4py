"""Interactive area and period selection for ``hostradaArea.ipynb``.

The module keeps the notebook variables ``polygon_points``, ``selected_area_name``,
``custom_polygon_points``, ``start_utc`` and ``end_utc`` synchronized through the
namespace passed to :class:`HostradaAreaUI`.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Sequence

import pandas as pd
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (
    Map,
    Polygon as LeafletPolygon,
    Polyline as LeafletPolyline,
    CircleMarker,
    basemaps,
)
from shapely.geometry import Polygon

from . import hostrada as hs


class HostradaAreaUI:
    """Leaflet-based area selector and UTC-period selector for HOSTRADA."""

    CUSTOM_AREA = "__custom_polygon__"
    GERMANY_CENTER = (51.1657, 10.4515)
    GERMANY_BOUNDS = (5.5, 15.6, 47.0, 55.5)

    def __init__(
        self,
        notebook_namespace: MutableMapping[str, Any],
        cities_module: Any,
        regions_module: Any,
        *,
        initial_area_name: str = "City – Berlin",
        initial_start_utc: str = "2025-01-01T00:00:00",
        initial_end_utc: str = "2025-01-31T23:00:00",
    ) -> None:
        self.namespace = notebook_namespace
        self.cities_module = cities_module
        self.regions_module = regions_module

        self.predefined_areas = {
            **{
                f"City – {name}": points
                for name, points in cities_module.CITY_POLYGONS.items()
            },
            **{
                f"Region – {name}": points
                for name, points in regions_module.REGIONS_POLYGONS.items()
            },
        }
        if not self.predefined_areas:
            raise ValueError("No predefined city or region polygons are available.")

        if initial_area_name not in self.predefined_areas:
            initial_area_name = next(iter(self.predefined_areas))

        self.polygon_points = self._normalise_polygon(
            self.predefined_areas[initial_area_name]
        )
        self.selected_area_name = initial_area_name
        self.custom_polygon_points: list[tuple[float, float]] | None = None
        self._area_layer = None

        self._drawing_active = False
        self._drawing_points: list[tuple[float, float]] = []
        self._draft_line = None
        self._draft_markers: list[Any] = []
        self._period_updating = False

        self._initial_start = pd.Timestamp(initial_start_utc)
        self._initial_end = pd.Timestamp(initial_end_utc)
        if self._initial_end < self._initial_start:
            raise ValueError("initial_end_utc must not be earlier than initial_start_utc.")

        self._build_widgets(initial_area_name)
        self._connect_callbacks()
        self._publish_widget_references()
        self._set_active_polygon(self.polygon_points, self.selected_area_name)
        self._set_download_period()

    def _build_widgets(self, initial_area_name: str) -> None:
        self.area_dropdown = widgets.Dropdown(
            options=[(name, name) for name in self.predefined_areas]
            + [("Custom polygon – click points on map", self.CUSTOM_AREA)],
            value=initial_area_name,
            description="Area:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="520px"),
        )
        self.redraw_button = widgets.Button(
            description="Draw new custom polygon",
            icon="pencil",
            tooltip=(
                "Remove the current polygon and define a new polygon "
                "by clicking on the map"
            ),
            layout=widgets.Layout(width="250px"),
        )
        self.undo_button = widgets.Button(
            description="Undo last point",
            icon="undo",
            disabled=True,
            tooltip="Remove the most recently set polygon vertex",
            layout=widgets.Layout(width="190px"),
        )
        self.close_polygon_button = widgets.Button(
            description="Close polygon",
            icon="check",
            button_style="success",
            disabled=True,
            tooltip=(
                "Finish the polygon (alternative to clicking the green first point)"
            ),
            layout=widgets.Layout(width="180px"),
        )
        self.area_status = widgets.HTML()

        self.area_map = Map(
            basemap=basemaps.OpenStreetMap.Mapnik,
            center=self.GERMANY_CENTER,
            zoom=6,
            scroll_wheel_zoom=True,
            double_click_zoom=False,
            close_popup_on_click=False,
            layout=widgets.Layout(width="100%", height="600px"),
        )

        self.start_date_picker = widgets.DatePicker(
            description="Start date:",
            value=self._initial_start.date(),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        self.start_hour_dropdown = widgets.Dropdown(
            options=[(f"{hour:02d}:00 UTC", hour) for hour in range(24)],
            value=int(self._initial_start.hour),
            description="Start hour:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px"),
        )
        self.end_date_picker = widgets.DatePicker(
            description="End date:",
            value=self._initial_end.date(),
            style={"description_width": "initial"},
            layout=widgets.Layout(width="260px"),
        )
        self.end_hour_dropdown = widgets.Dropdown(
            options=[(f"{hour:02d}:00 UTC", hour) for hour in range(24)],
            value=int(self._initial_end.hour),
            description="End hour:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="230px"),
        )
        self.period_status = widgets.HTML()

        self.area_widget = widgets.VBox(
            [
                widgets.HBox([self.area_dropdown, self.redraw_button]),
                widgets.HBox([self.undo_button, self.close_polygon_button]),
                self.area_status,
                self.area_map,
            ]
        )
        self.period_widget = widgets.VBox(
            [
                widgets.HTML("<b>HOSTRADA download period (UTC)</b>"),
                widgets.HBox([self.start_date_picker, self.start_hour_dropdown]),
                widgets.HBox([self.end_date_picker, self.end_hour_dropdown]),
                self.period_status,
            ]
        )
        self.widget = widgets.VBox([self.area_widget, self.period_widget])

    def _connect_callbacks(self) -> None:
        self.area_dropdown.observe(self._on_area_selected, names="value")
        self.redraw_button.on_click(self._start_redraw)
        self.undo_button.on_click(self._undo_last_point)
        self.close_polygon_button.on_click(self._finish_custom_polygon)
        self.area_map.on_interaction(self._on_map_interaction)

        for period_widget in (
            self.start_date_picker,
            self.start_hour_dropdown,
            self.end_date_picker,
            self.end_hour_dropdown,
        ):
            period_widget.observe(self._set_download_period, names="value")

    def _publish_widget_references(self) -> None:
        """Expose the former notebook names for backward compatibility."""
        references = {
            "PREDEFINED_AREAS": self.predefined_areas,
            "CUSTOM_AREA": self.CUSTOM_AREA,
            "area_dropdown": self.area_dropdown,
            "redraw_button": self.redraw_button,
            "undo_button": self.undo_button,
            "close_polygon_button": self.close_polygon_button,
            "area_status": self.area_status,
            "area_map": self.area_map,
            "start_date_picker": self.start_date_picker,
            "start_hour_dropdown": self.start_hour_dropdown,
            "end_date_picker": self.end_date_picker,
            "end_hour_dropdown": self.end_hour_dropdown,
            "period_status": self.period_status,
        }
        self.namespace.update(references)
        self._sync_area_namespace()

    def _sync_area_namespace(self) -> None:
        self.namespace["polygon_points"] = self.polygon_points
        self.namespace["selected_area_name"] = self.selected_area_name
        self.namespace["custom_polygon_points"] = self.custom_polygon_points

    def _map_add(self, item: Any) -> None:
        """Compatibility helper for older and newer ipyleaflet versions."""
        try:
            self.area_map.add(item)
        except (AttributeError, TypeError):
            self.area_map.add_layer(item)

    def _map_remove(self, item: Any) -> None:
        """Remove a map layer without failing when it is already absent."""
        if item is None:
            return
        try:
            self.area_map.remove(item)
        except (AttributeError, TypeError, ValueError):
            try:
                self.area_map.remove_layer(item)
            except (AttributeError, TypeError, ValueError):
                pass

    @staticmethod
    def _normalise_polygon(
        points: Sequence[Sequence[float]],
    ) -> list[tuple[float, float]]:
        normalised = [(float(lon), float(lat)) for lon, lat in points]
        if len(normalised) < 3:
            raise ValueError("A polygon needs at least three points.")
        if normalised[0] != normalised[-1]:
            normalised.append(normalised[0])
        return normalised

    def _inside_provider_domain(self, lon: float, lat: float) -> bool:
        if hs.get_provider_name() == "dwd":
            min_lon, max_lon, min_lat, max_lat = self.GERMANY_BOUNDS
            return min_lon <= float(lon) <= max_lon and min_lat <= float(lat) <= max_lat
        return -45.0 <= float(lon) <= 75.0 and 20.0 <= float(lat) <= 80.0

    def _validate_custom_polygon(
        self, points: Sequence[Sequence[float]]
    ) -> list[tuple[float, float]]:
        points = self._normalise_polygon(points)
        outside = [
            (lon, lat)
            for lon, lat in points
            if not self._inside_provider_domain(lon, lat)
        ]
        if outside:
            raise ValueError("Please define the polygon within Germany.")

        geometry = Polygon(points)
        if geometry.is_empty or geometry.area == 0:
            raise ValueError("The polygon has no area.")
        if not geometry.is_valid:
            raise ValueError(
                "The polygon is invalid, for example because edges intersect. "
                "Please redraw it without crossing lines."
            )
        return points

    def _show_polygon(
        self, points: Sequence[Sequence[float]], *, fit: bool = True
    ) -> None:
        self._map_remove(self._area_layer)
        self._area_layer = LeafletPolygon(
            locations=[(lat, lon) for lon, lat in points],
            color="#1f77b4",
            fill_color="#1f77b4",
            fill_opacity=0.12,
            weight=3,
        )
        self._map_add(self._area_layer)
        if fit:
            lons = [lon for lon, _ in points]
            lats = [lat for _, lat in points]
            self.area_map.fit_bounds(
                [[min(lats), min(lons)], [max(lats), max(lons)]]
            )

    def _set_active_polygon(
        self,
        points: Sequence[Sequence[float]],
        name: str,
        *,
        fit: bool = True,
    ) -> None:
        self.polygon_points = self._normalise_polygon(points)
        self.selected_area_name = name
        self._sync_area_namespace()
        self._show_polygon(self.polygon_points, fit=fit)
        self.area_status.value = (
            f"<b>Selected:</b> {self.selected_area_name} &nbsp; "
            f"({len(self.polygon_points) - 1} polygon vertices)"
        )

    @staticmethod
    def _drawing_instruction() -> str:
        return (
            "<b>Custom polygon:</b> Click the desired vertices on the map. "
            "The <span style='color:#168821'><b>green first point</b></span> "
            "is larger. After at least three points, click this green point "
            "again to close the polygon. The <b>Close polygon</b> button "
            "provides the same action."
        )

    def _clear_draft_layers(self) -> None:
        self._map_remove(self._draft_line)
        self._draft_line = None
        for marker in self._draft_markers:
            self._map_remove(marker)
        self._draft_markers = []

    def _first_point_clicked(self, lat: float, lon: float) -> bool:
        if not self._drawing_points:
            return False
        first_lon, first_lat = self._drawing_points[0]
        return (
            abs(float(lat) - first_lat) <= 0.00025
            and abs(float(lon) - first_lon) <= 0.00025
        )

    def _on_first_marker_click(self, *_args: Any, **_kwargs: Any) -> None:
        if self._drawing_active and len(self._drawing_points) >= 3:
            self._finish_custom_polygon()

    def _refresh_draft_layers(self) -> None:
        self._clear_draft_layers()

        if len(self._drawing_points) >= 2:
            self._draft_line = LeafletPolyline(
                locations=[(lat, lon) for lon, lat in self._drawing_points],
                color="#d62728",
                weight=3,
                dash_array="6, 6",
            )
            self._map_add(self._draft_line)

        for index, (lon, lat) in enumerate(self._drawing_points):
            first = index == 0
            marker = CircleMarker(
                location=(lat, lon),
                radius=11 if first else 6,
                color="#168821" if first else "#d62728",
                fill_color="#32cd32" if first else "#ffffff",
                fill_opacity=0.95,
                weight=4 if first else 2,
            )
            if first:
                marker.on_click(self._on_first_marker_click)
            self._map_add(marker)
            self._draft_markers.append(marker)

        self.undo_button.disabled = len(self._drawing_points) == 0
        self.close_polygon_button.disabled = len(self._drawing_points) < 3

    def _begin_custom_drawing(self, *, clear_existing: bool = True) -> None:
        if clear_existing:
            self.custom_polygon_points = None
        self.polygon_points = None
        self.selected_area_name = "Custom polygon"
        self._drawing_active = True
        self._drawing_points = []
        self._map_remove(self._area_layer)
        self._area_layer = None
        self._clear_draft_layers()
        self.undo_button.disabled = True
        self.close_polygon_button.disabled = True
        self._sync_area_namespace()
        self.area_status.value = self._drawing_instruction()

    def _finish_custom_polygon(self, *_args: Any) -> None:
        if not self._drawing_active:
            return
        try:
            self.custom_polygon_points = self._validate_custom_polygon(
                self._drawing_points
            )
            self._drawing_active = False
            self._clear_draft_layers()
            self.undo_button.disabled = True
            self.close_polygon_button.disabled = True
            self._sync_area_namespace()

            if self.area_dropdown.value != self.CUSTOM_AREA:
                self.area_dropdown.value = self.CUSTOM_AREA
            else:
                self._set_active_polygon(
                    self.custom_polygon_points, "Custom polygon"
                )
        except Exception as exc:
            self.area_status.value = f"<b>Polygon not accepted:</b> {exc}"

    def _on_map_interaction(self, **event: Any) -> None:
        if not self._drawing_active or event.get("type") != "click":
            return

        coordinates = event.get("coordinates")
        if coordinates is None or len(coordinates) < 2:
            return
        lat, lon = map(float, coordinates[:2])

        if len(self._drawing_points) >= 3 and self._first_point_clicked(lat, lon):
            self._finish_custom_polygon()
            return

        if not self._inside_provider_domain(lon, lat):
            self.area_status.value = (
                "<b>Point not accepted:</b> Please click within Germany. "
                + self._drawing_instruction()
            )
            return

        self._drawing_points.append((lon, lat))
        self._refresh_draft_layers()
        self.area_status.value = (
            self._drawing_instruction()
            + f" &nbsp; <b>Points set:</b> {len(self._drawing_points)}"
        )

    def _undo_last_point(self, _button: Any) -> None:
        if not self._drawing_active or not self._drawing_points:
            return
        self._drawing_points.pop()
        self._refresh_draft_layers()
        self.area_status.value = (
            self._drawing_instruction()
            + f" &nbsp; <b>Points set:</b> {len(self._drawing_points)}"
        )

    def _on_area_selected(self, change: MutableMapping[str, Any]) -> None:
        selected = change["new"]
        self._clear_draft_layers()
        self._drawing_active = False

        if selected == self.CUSTOM_AREA:
            if self.custom_polygon_points is None:
                self._begin_custom_drawing(clear_existing=False)
            else:
                self._set_active_polygon(
                    self.custom_polygon_points, "Custom polygon"
                )
        else:
            self.undo_button.disabled = True
            self.close_polygon_button.disabled = True
            self._set_active_polygon(self.predefined_areas[selected], selected)

    def _start_redraw(self, _button: Any) -> None:
        self.custom_polygon_points = None
        self._sync_area_namespace()
        if self.area_dropdown.value != self.CUSTOM_AREA:
            self.area_dropdown.value = self.CUSTOM_AREA
        else:
            self._begin_custom_drawing(clear_existing=False)
        self.area_map.center = self.GERMANY_CENTER
        self.area_map.zoom = 6

    @staticmethod
    def _period_timestamp(date_value: Any, hour_value: int) -> pd.Timestamp | None:
        if date_value is None:
            return None
        return pd.Timestamp(date_value) + pd.Timedelta(hours=int(hour_value))

    def _set_download_period(
        self, change: MutableMapping[str, Any] | None = None
    ) -> None:
        if self._period_updating:
            return

        start_timestamp = self._period_timestamp(
            self.start_date_picker.value, self.start_hour_dropdown.value
        )
        end_timestamp = self._period_timestamp(
            self.end_date_picker.value, self.end_hour_dropdown.value
        )

        if start_timestamp is None or end_timestamp is None:
            self.period_status.value = (
                "<span style='color:#b00020'><b>"
                "Please select both dates.</b></span>"
            )
            return

        if end_timestamp < start_timestamp:
            self._period_updating = True
            changed_widget = (
                change.get("owner")
                if change is not None and hasattr(change, "get")
                else None
            )
            if changed_widget in (
                self.start_date_picker,
                self.start_hour_dropdown,
            ):
                self.end_date_picker.value = start_timestamp.date()
                self.end_hour_dropdown.value = int(start_timestamp.hour)
                end_timestamp = start_timestamp
            else:
                self.start_date_picker.value = end_timestamp.date()
                self.start_hour_dropdown.value = int(end_timestamp.hour)
                start_timestamp = end_timestamp
            self._period_updating = False

        start_utc = start_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        end_utc = end_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        self.namespace["start_utc"] = start_utc
        self.namespace["end_utc"] = end_utc

        number_of_hours = (
            int((end_timestamp - start_timestamp).total_seconds() / 3600) + 1
        )
        self.period_status.value = (
            f"<b>Download period:</b> {start_timestamp:%d.%m.%Y %H:%M} UTC "
            f"to {end_timestamp:%d.%m.%Y %H:%M} UTC "
            f"&nbsp; ({number_of_hours:,} hourly time steps)"
        )

    def show(self) -> None:
        """Display the complete area and period selection interface."""
        display(self.widget)


def create_area_ui(
    notebook_namespace: MutableMapping[str, Any],
    cities_module: Any,
    regions_module: Any,
    **kwargs: Any,
) -> HostradaAreaUI:
    """Create and display a :class:`HostradaAreaUI` instance."""
    ui = HostradaAreaUI(
        notebook_namespace=notebook_namespace,
        cities_module=cities_module,
        regions_module=regions_module,
        **kwargs,
    )
    ui.show()
    return ui
