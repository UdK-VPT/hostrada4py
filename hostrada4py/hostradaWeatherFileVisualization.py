"""Interactive result browser for generated weather files and DataFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display


def read_weather_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".epw":
        names = [
            "year", "month", "day", "hour", "minute", "flags", "tas", "tdew", "hurs", "ps",
            "extraterrestrial_horizontal", "extraterrestrial_direct", "horizontal_ir", "rsds", "dni", "dhi",
            "global_illuminance", "direct_illuminance", "diffuse_illuminance", "zenith_luminance",
            "sfcWind_direction", "sfcWind", "total_sky_cover", "opaque_sky_cover", "visibility",
            "ceiling_height", "present_weather_observation", "present_weather_codes", "precipitable_water",
            "aerosol_optical_depth", "snow_depth", "days_since_last_snow", "albedo", "liquid_precipitation_depth",
            "liquid_precipitation_quantity",
        ]
        frame = pd.read_csv(path, skiprows=8, header=None, names=names)
        frame["time"] = pd.to_datetime(dict(
            year=frame.year, month=frame.month, day=frame.day,
            hour=(frame.hour - 1).clip(lower=0), minute=0,
        ), errors="coerce")
        return frame
    for separator in (",", ";", "\t", r"\s+"):
        try:
            frame = pd.read_csv(path, sep=separator, comment="#", engine="python")
            if len(frame.columns) > 1:
                return frame
        except Exception:
            continue
    return pd.read_csv(path)


def _time_index(frame: pd.DataFrame) -> pd.DatetimeIndex | None:
    for column in ("time", "timestamp", "datetime", "Date/Time", "date_time"):
        if column in frame:
            index = pd.DatetimeIndex(pd.to_datetime(frame[column], errors="coerce"))
            if index.notna().any():
                return index
    if {"year", "month", "day", "hour"} <= set(frame):
        return pd.DatetimeIndex(pd.to_datetime(dict(
            year=frame.year, month=frame.month, day=frame.day,
            hour=pd.to_numeric(frame.hour, errors="coerce").fillna(1).astype(int).sub(1).clip(lower=0),
        ), errors="coerce"))
    return None


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.select_dtypes(include=[np.number]).columns)


def visualize_weather_file(weather_file, columns=None, max_rows=8760):
    frame = weather_file if isinstance(weather_file, pd.DataFrame) else read_weather_table(weather_file)
    numeric = frame.select_dtypes("number")
    if columns:
        numeric = numeric[[column for column in columns if column in numeric]]
    index = _time_index(frame)
    if index is not None:
        numeric = numeric.set_axis(index)
    return numeric.iloc[:max_rows].plot(figsize=(12, 5), grid=True)


class WeatherFileVisualization:
    def __init__(self, weather_file=None, *, namespace=None, show=True):
        self.namespace = namespace
        self.weather_file = weather_file
        self.frame: pd.DataFrame | None = None
        self._build_widgets()
        self.load_button.on_click(self.load)
        self.variable.observe(self.draw, names="value")
        self.view.observe(self.draw, names="value")
        self.resample.observe(self.draw, names="value")
        if weather_file is not None:
            self.load()
        if show:
            self.show()

    def _build_widgets(self):
        initial_path = "" if isinstance(self.weather_file, pd.DataFrame) or self.weather_file is None else str(self.weather_file)
        self.path = widgets.Text(value=initial_path, description="Weather file:", layout=widgets.Layout(width="720px"))
        self.load_button = widgets.Button(description="Load / refresh", icon="refresh", button_style="info")
        self.variable = widgets.SelectMultiple(description="Variables:", layout=widgets.Layout(width="430px", height="150px"))
        self.view = widgets.ToggleButtons(options=[("Time series", "series"), ("Daily profile", "profile"),
                                                   ("Monthly means", "monthly"), ("Heatmap", "heatmap"),
                                                   ("Distribution", "hist"), ("Table", "table")],
                                          value="series", description="View:")
        self.resample = widgets.Dropdown(options=[("Hourly", None), ("Daily mean", "D"),
                                                   ("Weekly mean", "W"), ("Monthly mean", "MS")],
                                         value=None, description="Aggregation:")
        self.status = widgets.HTML()
        self.output = widgets.Output(layout=widgets.Layout(min_height="420px"))
        self.widget = widgets.VBox([
            widgets.HTML("<h3>Weather-file result explorer</h3>"),
            widgets.HBox([self.path, self.load_button]), self.variable,
            widgets.HBox([self.view, self.resample]), self.status, self.output,
        ])

    def show(self):
        display(self.widget)
        return self

    def load(self, _button=None):
        try:
            source = self.weather_file
            if self.path.value.strip():
                source = self.path.value.strip()
            if isinstance(source, pd.DataFrame):
                frame = source.copy()
            else:
                frame = read_weather_table(source)
            self.frame = frame
            columns = numeric_columns(frame)
            preferred = [column for column in ("tas", "tdew", "hurs", "rsds", "dni", "dhi", "sfcWind") if column in columns]
            self.variable.options = columns
            self.variable.value = tuple(preferred[:4] or columns[: min(4, len(columns))])
            self.status.value = f"<b>{len(frame):,}</b> records and <b>{len(columns)}</b> numeric columns loaded."
            if self.namespace is not None:
                self.namespace.update(weather_visualization=self, weather_dataframe=frame)
            self.draw()
        except Exception as exc:
            self.status.value = f"<span style='color:#b00020'>Could not load weather data: {exc}</span>"

    def _selected_frame(self):
        if self.frame is None:
            return None
        columns = [column for column in self.variable.value if column in self.frame]
        if not columns:
            return None
        values = self.frame[columns].apply(pd.to_numeric, errors="coerce")
        index = _time_index(self.frame)
        if index is not None:
            values.index = index
            values = values[~values.index.isna()]
            if self.resample.value:
                values = values.resample(self.resample.value).mean()
        return values

    def draw(self, _change=None):
        values = self._selected_frame()
        if values is None or values.empty:
            return
        with self.output:
            clear_output(wait=True)
            import matplotlib.pyplot as plt
            view = self.view.value
            if view == "series":
                ax = values.plot(figsize=(13, 5), linewidth=1)
                ax.grid(True, alpha=0.3); ax.set_xlabel("Time"); plt.show()
            elif view == "profile":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                profile = values.groupby(values.index.hour).mean()
                ax = profile.plot(figsize=(11, 5), marker="o")
                ax.set_xlabel("Hour of day"); ax.set_xticks(range(24)); ax.grid(True, alpha=0.3); plt.show()
            elif view == "monthly":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                monthly = values.groupby(values.index.month).mean()
                ax = monthly.plot(kind="bar", figsize=(12, 5))
                ax.set_xlabel("Month"); ax.grid(True, axis="y", alpha=0.3); plt.show()
            elif view == "heatmap":
                if not isinstance(values.index, pd.DatetimeIndex):
                    print("A time column is required for this view.")
                    return
                column = values.columns[0]
                pivot = pd.DataFrame({"date": values.index.date, "hour": values.index.hour,
                                      "value": values[column].to_numpy()}).pivot_table(
                                          index="date", columns="hour", values="value", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(13, 6))
                image = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest")
                ax.set_xlabel("Hour"); ax.set_ylabel("Date index"); ax.set_title(column)
                fig.colorbar(image, ax=ax); plt.show()
            elif view == "hist":
                axes = values.plot.hist(bins=40, alpha=0.55, figsize=(11, 5))
                axes.grid(True, alpha=0.3); plt.show()
            else:
                display(values.head(250))
                display(values.describe().T)


def setup_weather_file_visualization(namespace=None, weather_file=None, **kwargs):
    if weather_file is None and namespace is not None:
        weather_file = namespace.get("weather_file") or namespace.get("output_file") or namespace.get("weather_dataframe")
    return WeatherFileVisualization(weather_file, namespace=namespace, **kwargs)
