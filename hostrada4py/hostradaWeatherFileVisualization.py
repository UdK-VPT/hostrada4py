"""Visualization helpers for generated HOSTRADA weather files.

This module contains the logic previously embedded in the visualization cell of
``hostradaGenerateWeatherFiles.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import display, HTML, clear_output

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except Exception as _plotly_error:
    _PLOTLY_AVAILABLE = False
    _PLOTLY_IMPORT_ERROR = _plotly_error

try:
    import ipywidgets as widgets
    _WIDGETS_AVAILABLE = True
except Exception as _widgets_error:
    _WIDGETS_AVAILABLE = False
    _WIDGETS_IMPORT_ERROR = _widgets_error


_NOTEBOOK_NAMESPACE = None
_EXPLICIT_START_DATETIME = None


def configure_notebook_context(namespace=None, start_datetime=None):
    """Configure access to the notebook variables used by the visualizations.

    Parameters
    ----------
    namespace:
        Notebook namespace, normally supplied as ``globals()``. It is read each
        time a generated file is loaded, so a newly selected start date is used
        without re-importing this module.
    start_datetime:
        Optional explicit start date/time. When supplied, it takes precedence
        over values in the notebook namespace.
    """
    global _NOTEBOOK_NAMESPACE, _EXPLICIT_START_DATETIME
    if namespace is not None and not hasattr(namespace, "get"):
        raise TypeError("namespace must be a mapping, for example globals()")
    _NOTEBOOK_NAMESPACE = namespace
    _EXPLICIT_START_DATETIME = start_datetime
    return None


def _current_start_timestamp():
    """Return the selected start timestamp used to reconstruct file time axes."""
    if _EXPLICIT_START_DATETIME is not None:
        try:
            return pd.Timestamp(_EXPLICIT_START_DATETIME)
        except Exception:
            pass

    if _NOTEBOOK_NAMESPACE is not None:
        for name in ("selected_start_datetime", "selected_start"):
            if name in _NOTEBOOK_NAMESPACE:
                try:
                    return pd.Timestamp(_NOTEBOOK_NAMESPACE[name])
                except Exception:
                    pass

    return pd.Timestamp("2025-01-01 00:00")


def _hourly_index(n_rows):
    return pd.date_range(start=_current_start_timestamp(), periods=int(n_rows), freq="h")


def _read_ida_ice_weather(path):
    """Read a headerless seven-column IDA ICE PRN weather file.

    HOSTRADA writes the columns in this fixed order::

        Hour DryBulb_C RelHum WindDirect WindSpeed DirectNormal DiffuseHorizontal

    The file has no header.  It is therefore essential to use ``header=None``;
    otherwise pandas consumes the first data row as column names and the old
    fallback selected column four (wind direction) as the temperature.
    """
    path = Path(path)
    columns = [
        "Hour",
        "DryBulb_C",
        "RelHum",
        "WindDirect",
        "WindSpeed",
        "DirectNormal",
        "DiffuseHorizontal",
    ]

    # The generated PRN is fixed-width but whitespace-separated parsing is
    # robust for both the exact HOSTRADA layout and compatible IDA ICE files.
    raw = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        comment="#",
        header=None,
        skip_blank_lines=True,
    )
    if raw.shape[1] < len(columns):
        raise ValueError(
            f"IDA-ICE-Datei {path} enthält nur {raw.shape[1]} Spalten; "
            f"erwartet werden mindestens {len(columns)}."
        )

    # Ignore unexpected trailing fields, but never infer the temperature column
    # from labels or fallback positions. DryBulb_C is exactly column 2.
    df = raw.iloc[:, :len(columns)].copy()
    df.columns = columns

    temperature = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    diffuse_horizontal = pd.to_numeric(df.iloc[:, 6], errors="coerce")
    if temperature.notna().sum() == 0:
        raise ValueError(f"Keine numerischen Außentemperaturwerte in Spalte 2 von {path} gefunden.")

    return pd.DataFrame({
        "time": _hourly_index(len(df)),
        "temperature_C": temperature.to_numpy(),
        "global_radiation": diffuse_horizontal.to_numpy(),
    }), "W/m²"


def _read_polysun_weather(path):
    path = Path(path)
    df = pd.read_csv(path, comment="#")
    temperature_col = "Tamb" if "Tamb" in df.columns else df.columns[3]
    radiation_col = "Gh" if "Gh" in df.columns else df.columns[0]
    return pd.DataFrame({
        "time": _hourly_index(len(df)),
        "temperature_C": pd.to_numeric(df[temperature_col], errors="coerce"),
        "global_radiation": pd.to_numeric(df[radiation_col], errors="coerce"),
    }), "Wh/m²"


def _read_energyplus_weather(path):
    path = Path(path)
    epw_columns = [
        "Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags",
        "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity",
        "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation",
        "Extraterrestrial Direct Normal Radiation", "Horizontal Infrared Radiation Intensity",
        "Global Horizontal Radiation", "Direct Normal Radiation", "Diffuse Horizontal Radiation",
        "Global Horizontal Illuminance", "Direct Normal Illuminance", "Diffuse Horizontal Illuminance",
        "Zenith Luminance", "Wind Direction", "Wind Speed", "Total Sky Cover", "Opaque Sky Cover",
        "Visibility", "Ceiling Height", "Present Weather Observation", "Present Weather Codes",
        "Precipitable Water", "Aerosol Optical Depth", "Snow Depth", "Days Since Last Snowfall",
        "Albedo", "Liquid Precipitation Depth", "Liquid Precipitation Quantity",
    ]
    df = pd.read_csv(path, skiprows=8, header=None, names=epw_columns)
    hour_zero_based = pd.to_numeric(df["Hour"], errors="coerce").fillna(1).astype(int) - 1
    hour_zero_based = hour_zero_based.clip(lower=0, upper=23)
    time = pd.to_datetime({
        "year": pd.to_numeric(df["Year"], errors="coerce").astype(int),
        "month": pd.to_numeric(df["Month"], errors="coerce").astype(int),
        "day": pd.to_numeric(df["Day"], errors="coerce").astype(int),
        "hour": hour_zero_based,
    })
    return pd.DataFrame({
        "time": time,
        "temperature_C": pd.to_numeric(df["Dry Bulb Temperature"], errors="coerce"),
        "global_radiation": pd.to_numeric(df["Global Horizontal Radiation"], errors="coerce"),
    }), "W/m²"


def _read_simstadt_tmy3_weather(path):
    path = Path(path)
    df = pd.read_csv(path, skiprows=1)
    temperature_col = "Dry-bulb (C)" if "Dry-bulb (C)" in df.columns else df.columns[31]
    radiation_col = "GHI (W/m^2)" if "GHI (W/m^2)" in df.columns else df.columns[4]
    return pd.DataFrame({
        "time": _hourly_index(len(df)),
        "temperature_C": pd.to_numeric(df[temperature_col], errors="coerce"),
        "global_radiation": pd.to_numeric(df[radiation_col], errors="coerce"),
    }), "W/m²"


def _read_buildingsystems_csv_weather(path):
    path = Path(path)
    columns = [
        "Time_h", "Time_s", "TAirRef_degC", "RelHum_pct",
        "GlobalHorizontalRadiation_W_m2", "DiffuseHorizontalRadiation_W_m2",
        "CloudCover_okta", "WindSpeed_m_s", "WindDirection_deg",
    ]
    df = pd.read_csv(path, comment="#", header=None)
    if len(df) > 0 and str(df.iloc[0, 0]).strip() == "Time_h":
        df = df.iloc[1:].reset_index(drop=True)
    df = df.iloc[:, :len(columns)]
    df.columns = columns[:df.shape[1]]
    if "Time_s" in df.columns:
        seconds = pd.to_numeric(df["Time_s"], errors="coerce")
        time = _current_start_timestamp() + pd.to_timedelta(seconds - seconds.iloc[0], unit="s")
    else:
        time = _hourly_index(len(df))
    return pd.DataFrame({
        "time": time,
        "temperature_C": pd.to_numeric(df["TAirRef_degC"], errors="coerce"),
        "global_radiation": pd.to_numeric(df["GlobalHorizontalRadiation_W_m2"], errors="coerce"),
    }), "W/m²"


WEATHER_FILE_CONFIGS = {
    "ida_ice": {
        "label": "IDA ICE (*.prn)",
        "file_name": "HOSTRADA_IDA_ICE.prn",
        "reader": _read_ida_ice_weather,
        "title": "IDA ICE: Outdoor air temperature and global radiation",
    },
    "polysun": {
        "label": "Polysun (*.csv)",
        "file_name": "HOSTRADA_Polysun.csv",
        "reader": _read_polysun_weather,
        "title": "Polysun: Outdoor air temperature and global radiation",
    },
    "energyplus": {
        "label": "EnergyPlus (*.epw)",
        "file_name": "HOSTRADA_EnergyPlus.epw",
        "reader": _read_energyplus_weather,
        "title": "EnergyPlus: Outdoor air temperature and global radiation",
    },
    "simstadt": {
        "label": "SimStadt (*.tmy3)",
        "file_name": "HOSTRADA_SimStadt.tmy3",
        "reader": _read_simstadt_tmy3_weather,
        "title": "SimStadt: Outdoor air temperature and global radiation",
    },
    "buildingsystems_csv": {
        "label": "BuildingSystems CSV (*.csv)",
        "file_name": "HOSTRADA_BuildingSystems.csv",
        "reader": _read_buildingsystems_csv_weather,
        "title": "BuildingSystems CSV: Outdoor air temperature and global radiationg",
    },
}


def available_generated_weather_files():
    """Return a dict of generated weather files that currently exist in the notebook folder."""
    return {
        key: cfg for key, cfg in WEATHER_FILE_CONFIGS.items()
        if Path(cfg["file_name"]).exists()
    }


def create_weather_timeseries_figure(file_key):
    """Create a zoomable Plotly figure for one generated weather file."""
    if not _PLOTLY_AVAILABLE:
        raise ImportError(f"Plotly ist nicht verfügbar: {_PLOTLY_IMPORT_ERROR}")
    if file_key not in WEATHER_FILE_CONFIGS:
        raise KeyError(f"Unknown weather data type: {file_key}")
    cfg = WEATHER_FILE_CONFIGS[file_key]
    path = Path(cfg["file_name"])
    if not path.exists():
        raise FileNotFoundError(
            f"{cfg['file_name']} was not found. Please run the appropriate generation cell first."
        )
    data, radiation_unit = cfg["reader"](path)
    data = data.dropna(subset=["time"])
    if data.empty:
        raise ValueError(f"No usable time series data found in {cfg['file_name']}.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["temperature_C"],
            name="Outdoor air temperature [°C]",
            mode="lines",
            hovertemplate="%{x}<br>Outdoor air temperature: %{y:.2f} °C<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["time"],
            y=data["global_radiation"],
            name=f"Global radiation [{radiation_unit}]",
            mode="lines",
            hovertemplate=f"%{{x}}<br>Global radiation: %{{y:.2f}} {radiation_unit}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=cfg["title"],
        xaxis_title="Zeit",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode="zoom",
        height=560,
        margin=dict(l=70, r=90, t=95, b=75),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        rangeselector=dict(buttons=[
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="7d", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(step="all", label="alles"),
        ]),
    )
    fig.update_yaxes(title_text="Outdoor air temperature [°C]", secondary_y=False)
    fig.update_yaxes(title_text=f"Global radiation [{radiation_unit}]", secondary_y=True)
    return fig


def display_plotly_html(fig):
    """Display Plotly as explicit HTML instead of relying on the active notebook renderer."""
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "scrollZoom": True,
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )
    display(HTML(html))


def plot_weather_file(file_key):
    """Display one weather file by key: ida_ice, polysun, energyplus, simstadt, buildingsystems_csv."""
    fig = create_weather_timeseries_figure(file_key)
    display_plotly_html(fig)
    return fig


def plot_all_weather_files():
    """Display all generated weather files that exist in the notebook folder."""
    existing = available_generated_weather_files()
    if not existing:
        display(HTML(
            "<b>No weather files found yet.</b><br>"
            "Please run at least one generation cell first. "
            "Then restart this visualization cell."
        ))
        return []
    figures = []
    for file_key in existing:
        try:
            display(HTML(f"<h4>{WEATHER_FILE_CONFIGS[file_key]['label']}</h4>"))
            figures.append(plot_weather_file(file_key))
        except Exception as exc:
            display(HTML(f"<b>{WEATHER_FILE_CONFIGS[file_key]['file_name']}:</b> Visualization not possible: {exc}"))
    return figures


def show_weather_file_status():
    rows = []
    for key, cfg in WEATHER_FILE_CONFIGS.items():
        path = Path(cfg["file_name"])
        status = "available" if path.exists() else "not found"
        size = f"{path.stat().st_size / 1024:.1f} kB" if path.exists() else "-"
        rows.append(f"<tr><td><code>{key}</code></td><td>{cfg['file_name']}</td><td>{status}</td><td>{size}</td></tr>")
    display(HTML(
        "<b>Status of the weather files in the current notebook directory</b>"
        "<table><tr><th>Schlüssel</th><th>Datei</th><th>Status</th><th>Größe</th></tr>"
        + "".join(rows) + "</table>"
    ))


def display_weather_file_selector():
    """Display selector if ipywidgets is available. Direct function calls below work without widgets."""
    if not _PLOTLY_AVAILABLE:
        display(HTML(
            "<b>Plotly ist not installed.</b><br>"
            "Please run it once: <code>pip install plotly</code>"
        ))
        return None
    show_weather_file_status()
    if not _WIDGETS_AVAILABLE:
        display(HTML(
            "<b>ipywidgets ist nicht installiert.</b><br>"
            "The direct plot calls in the following cells still work, for example:"
            "<code>plot_weather_file('energyplus')</code>."
        ))
        return None

    existing = available_generated_weather_files()
    option_items = [("All available weather files", "__all__")]
    for key, cfg in WEATHER_FILE_CONFIGS.items():
        suffix = " ✓" if key in existing else " (not yet generated)"
        option_items.append((cfg["label"] + suffix, key))

    dropdown = widgets.Dropdown(
        options=option_items,
        value="__all__",
        description="File:",
        layout=widgets.Layout(width="560px"),
    )
    button = widgets.Button(
        description="Show visualization",
        button_style="primary",
        icon="line-chart",
        layout=widgets.Layout(width="240px"),
    )
    output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="8px"))

    def _render_selection(_=None):
        with output:
            clear_output(wait=True)
            if dropdown.value == "__all__":
                plot_all_weather_files()
            else:
                try:
                    plot_weather_file(dropdown.value)
                except Exception as exc:
                    cfg = WEATHER_FILE_CONFIGS[dropdown.value]
                    display(HTML(
                        f"<b>{cfg['file_name']}</b> cannot be displayed at this time.<br>"
                        f"{exc}"
                    ))

    button.on_click(_render_selection)
    ui = widgets.VBox([
        widgets.HTML(
            "<b>Visualization of the generated weather data</b><br>"
            "Select a single file or all available files. "
        ),
        widgets.HBox([dropdown, button]),
        output,
    ])
    if _NOTEBOOK_NAMESPACE is not None:
        _NOTEBOOK_NAMESPACE["weather_file_selector_ui"] = ui
    display(ui)
    _render_selection()
    return None

def setup_weather_file_visualization(namespace=None, start_datetime=None):
    """Configure notebook context and display the generated-weather selector.

    When a notebook namespace is supplied, the public functions that existed in
    the original notebook cell are injected into that namespace. Existing calls
    such as ``plot_weather_file("ida_ice")`` therefore continue to work.
    """
    configure_notebook_context(namespace=namespace, start_datetime=start_datetime)
    if namespace is not None:
        for name in __all__:
            namespace[name] = globals()[name]
    return display_weather_file_selector()


__all__ = [
    "WEATHER_FILE_CONFIGS",
    "available_generated_weather_files",
    "configure_notebook_context",
    "create_weather_timeseries_figure",
    "display_plotly_html",
    "display_weather_file_selector",
    "plot_all_weather_files",
    "plot_weather_file",
    "setup_weather_file_visualization",
    "show_weather_file_status",
]
