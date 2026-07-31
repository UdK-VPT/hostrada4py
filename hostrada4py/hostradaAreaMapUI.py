"""Interactive OSM/Leaflet result-map display for ``hostradaArea.ipynb``.

The module builds the date/time selectors from the timestamps that are actually
available in the downloaded HOSTRADA GeoDataFrame. It keeps the variables used
by the original notebook synchronized in the supplied notebook namespace.
"""

from __future__ import annotations

from typing import Any, MutableMapping

import pandas as pd
import ipywidgets as widgets
from IPython.display import display


class HostradaAreaMapUI:
    """Display one HOSTRADA time slice in an interactive Leaflet/OSM map."""

    VARIABLE_TITLES = {
        "tas": "Air temperature in °C",
        "uhi": "Urban Heat Island Intensity in °C",
        "sfcWind": "Wind speed in m/s",
        "sfcWind_direction": "Wind direction in degree",
        "rsds": "Global radiation in W/m2",
        "clt": "Cloud cover in eighth",
        "hurs": "Relative humidity in percent",
        "tdew": "Dew point temperature in °C",
        "mixr": "Water vapor mixing ratio in g H20/kg dry air",
    }

    def __init__(
        self,
        notebook_namespace: MutableMapping[str, Any],
        hostrada_area_module: Any,
        *,
        original_default_time: str = "2025-01-08T12:00:00",
        output_directory: str = "./html",
        show_cell_values: bool = True,
        decimals: int = 1,
        fill_opacity: float = 0.2,
        value_label_color: str = "black",
        reverse_colormap: bool = False,
    ) -> None:
        self.namespace = notebook_namespace
        self.hostrada_area_module = hostrada_area_module
        self.original_default_time = original_default_time
        self.output_directory = output_directory
        self.show_cell_values = show_cell_values
        self.decimals = decimals
        self.fill_opacity = fill_opacity
        self.value_label_color = value_label_color
        self.reverse_colormap = reverse_colormap

        self.gdf = self._require_downloaded_data()
        self.available_map_time_strings = self._available_time_strings(self.gdf)
        self.available_map_dates = sorted(
            {value[:10] for value in self.available_map_time_strings}
        )
        self.initial_map_time = (
            original_default_time
            if original_default_time in self.available_map_time_strings
            else self.available_map_time_strings[0]
        )

        self._build_widgets()
        self._connect_callbacks()
        self._publish_widget_references()
        self._update_map_time_options()

        if self.initial_map_time in self._map_times_for_date(
            self.map_date_dropdown.value
        ):
            self.map_time_dropdown.value = self.initial_map_time
        self._update_map_time_status()

    def _require_downloaded_data(self) -> Any:
        if "gdf" not in self.namespace:
            raise NameError(
                "The variable 'gdf' does not exist. Run the HOSTRADA download "
                "cell before creating the climate map."
            )

        gdf = self.namespace["gdf"]
        if gdf is None or len(gdf) == 0:
            raise ValueError(
                "The downloaded data set is empty. Run the download cell with "
                "a valid area and time period before creating the climate map."
            )
        if "time" not in gdf.columns:
            raise KeyError("The downloaded data set has no 'time' column.")
        return gdf

    @staticmethod
    def _available_time_strings(gdf: Any) -> list[str]:
        available_times = pd.to_datetime(
            gdf["time"], utc=True, errors="coerce"
        ).dropna()
        time_strings = sorted(
            {
                value.strftime("%Y-%m-%dT%H:%M:%S")
                for value in available_times
            }
        )
        if not time_strings:
            raise ValueError(
                "No valid timestamps are available in the downloaded data set."
            )
        return time_strings

    def _build_widgets(self) -> None:
        self.map_date_dropdown = widgets.Dropdown(
            options=[
                (pd.Timestamp(value).strftime("%d.%m.%Y"), value)
                for value in self.available_map_dates
            ],
            value=self.initial_map_time[:10],
            description="Map date:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="270px"),
        )
        self.map_time_dropdown = widgets.Dropdown(
            description="Map time:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="245px"),
        )
        self.map_update_button = widgets.Button(
            description="Update climate map",
            icon="map",
            button_style="primary",
            tooltip=(
                "Create the two-dimensional climate map for the selected UTC time"
            ),
            layout=widgets.Layout(width="230px"),
        )
        self.map_time_status = widgets.HTML()
        self.map_output = widgets.Output()

        self.widget = widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.map_date_dropdown,
                        self.map_time_dropdown,
                        self.map_update_button,
                    ]
                ),
                self.map_time_status,
                self.map_output,
            ]
        )

    def _connect_callbacks(self) -> None:
        self.map_date_dropdown.observe(
            self._update_map_time_options, names="value"
        )
        self.map_time_dropdown.observe(
            self._update_map_time_status, names="value"
        )
        self.map_update_button.on_click(self.render_climate_map)

    def _publish_widget_references(self) -> None:
        """Expose the former notebook variables for backward compatibility."""
        self.namespace.update(
            {
                "_available_map_time_strings": self.available_map_time_strings,
                "_available_map_dates": self.available_map_dates,
                "_original_default_time": self.original_default_time,
                "_initial_map_time": self.initial_map_time,
                "map_date_dropdown": self.map_date_dropdown,
                "map_time_dropdown": self.map_time_dropdown,
                "map_update_button": self.map_update_button,
                "map_time_status": self.map_time_status,
                "map_output": self.map_output,
            }
        )

    def _map_times_for_date(self, date_string: str) -> list[str]:
        return [
            value
            for value in self.available_map_time_strings
            if value.startswith(f"{date_string}T")
        ]

    def _update_map_time_status(self, change: Any = None) -> None:
        del change
        if self.map_time_dropdown.value:
            selected = pd.Timestamp(self.map_time_dropdown.value)
            self.map_time_status.value = (
                f"<b>Selected map time:</b> "
                f"{selected:%d.%m.%Y %H:%M} UTC"
            )

    def _update_map_time_options(self, change: Any = None) -> None:
        del change
        values = self._map_times_for_date(self.map_date_dropdown.value)
        if not values:
            self.map_time_dropdown.options = []
            self.map_time_status.value = (
                "<b>No map timestamps are available for this date.</b>"
            )
            return

        previous_value = self.map_time_dropdown.value
        previous_clock_time = (
            previous_value[11:19] if previous_value else "12:00:00"
        )

        preferred_value = (
            f"{self.map_date_dropdown.value}T{previous_clock_time}"
        )
        if preferred_value not in values:
            noon_value = f"{self.map_date_dropdown.value}T12:00:00"
            preferred_value = noon_value if noon_value in values else values[0]

        self.map_time_dropdown.options = [
            (pd.Timestamp(value).strftime("%H:%M UTC"), value)
            for value in values
        ]
        self.map_time_dropdown.value = preferred_value
        self._update_map_time_status()

    def _current_variable(self) -> str:
        if "HOSTRADA_VAR" not in self.namespace:
            raise NameError(
                "The variable 'HOSTRADA_VAR' does not exist. Select a climate "
                "variable before creating the climate map."
            )
        return str(self.namespace["HOSTRADA_VAR"])

    def _current_title(self, variable: str) -> str:
        return self.VARIABLE_TITLES.get(variable, "unknown")

    def render_climate_map(self, _button: Any = None) -> Any:
        """Create and display the map for the currently selected UTC timestamp."""
        del _button
        if not self.map_time_dropdown.value:
            raise ValueError("No map timestamp is selected.")

        variable = self._current_variable()
        utc = self.map_time_dropdown.value
        timestamp = pd.Timestamp(utc)
        safe_timestamp = timestamp.strftime("%Y%m%dT%H%M%S")
        title = self._current_title(variable)

        save_html = (
            f"{self.output_directory.rstrip('/')}/"
            f"HOSTRADA_{variable}_{safe_timestamp}.html"
        )

        leaflet_map = self.hostrada_area_module.make_leaflet_map_timepoint(
            gdf_or_df=self.gdf,
            var=variable,
            time_utc=timestamp,
            show_cell_values=self.show_cell_values,
            decimals=self.decimals,
            fill_opacity=self.fill_opacity,
            value_label_color=self.value_label_color,
            title=title,
            subtitle=timestamp.strftime("%d.%m.%Y %H:%M UTC"),
            reverse_colormap=self.reverse_colormap,
            save_html=save_html,
        )

        self.namespace.update(
            {
                "utc": utc,
                "timestamp": timestamp,
                "safe_timestamp": safe_timestamp,
                "title": title,
                "leaflet_map": leaflet_map,
            }
        )

        self.map_output.clear_output(wait=True)
        with self.map_output:
            display(leaflet_map)
        return leaflet_map

    def show(self, *, render_initial_map: bool = True) -> None:
        """Display the selectors and optionally render the initial map."""
        display(self.widget)
        if render_initial_map:
            self.render_climate_map()


def create_area_map_ui(
    notebook_namespace: MutableMapping[str, Any],
    hostrada_area_module: Any,
    **kwargs: Any,
) -> HostradaAreaMapUI:
    """Create and display a :class:`HostradaAreaMapUI` instance."""
    ui = HostradaAreaMapUI(
        notebook_namespace=notebook_namespace,
        hostrada_area_module=hostrada_area_module,
        **kwargs,
    )
    ui.show()
    return ui
