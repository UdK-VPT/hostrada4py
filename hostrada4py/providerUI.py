"""Small additive provider selector for the original interactive notebooks.

DWD/HOSTRADA remains the default.  Selecting CERRA changes only the data
provider; the notebook controls, maps and result visualisations remain the same.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterable

import ipywidgets as widgets
from IPython.display import display

from . import hostrada as hs

_VARIABLE_LABELS = {
    "tas": "Air temperature (tas)",
    "uhi": "Urban heat island intensity (uhi)",
    "sfcWind": "Wind speed (sfcWind)",
    "sfcWind_direction": "Wind direction (sfcWind_direction)",
    "rsds": "Global horizontal irradiance (rsds)",
    "clt": "Cloud cover (clt)",
    "hurs": "Relative humidity (hurs)",
    "mixr": "Mixing ratio (mixr)",
    "ps": "Surface pressure (ps)",
    "psl": "Mean sea-level pressure (psl)",
    "tdew": "Dew-point temperature (tdew)",
}


def variable_options(provider: str | None = None) -> list[tuple[str, str]]:
    capabilities = hs.provider_capabilities(provider)
    order = tuple(_VARIABLE_LABELS)
    return [(_VARIABLE_LABELS[name], name) for name in order if name in capabilities.variables]


class ProviderSelector:
    """Synchronise provider selection with a notebook namespace."""

    def __init__(
        self,
        namespace: MutableMapping[str, Any],
        *,
        initial: str | None = None,
        climate_dropdown: Any | None = None,
        title: str = "Data source",
        show: bool = True,
    ) -> None:
        self.namespace = namespace
        self.climate_dropdown = climate_dropdown
        current = initial or hs.get_provider().name
        if current not in hs.available_providers():
            current = "dwd"
        self.dropdown = widgets.Dropdown(
            options=[("DWD HOSTRADA – Germany, 1 km", "dwd"),
                     ("Copernicus CERRA – Europe, about 5.5 km", "cerra")],
            value=current,
            description="Provider:",
            layout=widgets.Layout(width="560px"),
            style={"description_width": "100px"},
        )
        self.status = widgets.HTML()
        self.widget = widgets.VBox([
            widgets.HTML(f"<h3>{title}</h3>"), self.dropdown, self.status
        ])
        self.dropdown.observe(self._changed, names="value")
        self._apply(current)
        namespace.update(
            provider_selector=self,
            provider_dropdown=self.dropdown,
            provider_status=self.status,
            provider=current,
        )
        if show:
            display(self.widget)

    def _apply(self, name: str) -> None:
        hs.set_default_provider(name)
        self.namespace["provider"] = name
        caps = hs.provider_capabilities(name)
        vars_text = ", ".join(caps.variables)
        if name == "dwd":
            note = "Original HOSTRADA behaviour; Germany-only and UHI available."
        else:
            note = "Europe-wide CERRA backend; UHI is unavailable. CDS credentials are required."
        self.status.value = (
            f"<b>Active:</b> {name.upper()} &nbsp; "
            f"<b>grid:</b> {caps.spatial_resolution_m/1000:g} km &nbsp; "
            f"<b>variables:</b> {vars_text}<br><small>{note}</small>"
        )
        dropdown = self.climate_dropdown or self.namespace.get("climate_dropdown")
        if dropdown is not None and hasattr(dropdown, "options"):
            options = variable_options(name)
            old_value = getattr(dropdown, "value", None)
            dropdown.options = options
            values = [value for _, value in options]
            dropdown.value = old_value if old_value in values else values[0]
            self.namespace["HOSTRADA_VAR"] = dropdown.value

    def _changed(self, change: dict[str, Any]) -> None:
        self._apply(str(change["new"]))


def create_provider_selector(
    namespace: MutableMapping[str, Any],
    *,
    initial: str | None = None,
    climate_dropdown: Any | None = None,
    title: str = "0. Select data source",
    show: bool = True,
) -> ProviderSelector:
    return ProviderSelector(
        namespace,
        initial=initial,
        climate_dropdown=climate_dropdown,
        title=title,
        show=show,
    )
