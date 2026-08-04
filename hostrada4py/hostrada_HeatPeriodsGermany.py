"""Heat-period helpers and the hottest documented periods in Germany.

The ranking used by ``hostradaHeatPeriods.ipynb`` follows the maximum measured
station temperature within each period. It is not an official DWD heat-wave
intensity index. Historical periods are retained from the original project;
2025 and 2026 entries were updated on 2026-08-04 from DWD reports and DWD-based
station observations.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


DATA_AS_OF = "2026-08-04T18:00:00Z"
RANKING_BASIS = "maximum measured station temperature"


def identify_heat_periods(data, temperature_column="tas", threshold=30.0, minimum_hours=72):
    df = data.copy()
    df["time"] = pd.to_datetime(df["time"])
    hot = pd.to_numeric(df[temperature_column], errors="coerce") >= threshold
    groups = (hot != hot.shift()).cumsum()
    rows = []
    for _, group in df[hot].groupby(groups[hot]):
        if len(group) >= minimum_hours:
            rows.append(
                {
                    "start": group["time"].min(),
                    "end": group["time"].max(),
                    "hours": len(group),
                    "maximum": group[temperature_column].max(),
                    "mean": group[temperature_column].mean(),
                }
            )
    return pd.DataFrame(rows)


def heat_period_metrics(data, temperature_column="tas", threshold=30.0):
    values = pd.to_numeric(data[temperature_column], errors="coerce")
    return {
        "hours_above_threshold": int((values >= threshold).sum()),
        "maximum_temperature": float(values.max()),
        "mean_temperature": float(values.mean()),
    }


def calculate_heating_degree_days(
    data,
    temperature_column="tas",
    base_temperature=20.0,
    heating_limit=15.0,
):
    df = data.copy()
    df["time"] = pd.to_datetime(df["time"])
    daily = df.set_index("time")[temperature_column].resample("D").mean()
    return float((base_temperature - daily[daily < heating_limit]).clip(lower=0).sum())


extract_heat_periods = identify_heat_periods


def _event(
    start: str,
    end: str,
    name: str,
    longitude: float,
    latitude: float,
    station_url: str,
    maximum_temperature_c: float,
    record_date: str,
    *,
    status: str = "historisch",
    event_source_url: str | None = None,
    ongoing: bool = False,
) -> dict[str, Any]:
    return {
        "core_period": {"start": start, "end": end, "ongoing": ongoing},
        "maximum_temperature_c": maximum_temperature_c,
        "record_date": record_date,
        "status": status,
        "event_source_url": event_source_url,
        "measurement_sites": [
            {
                "name": name,
                "longitude": longitude,
                "latitude": latitude,
                "source_url": station_url,
            }
        ],
    }


# Sorted by maximum_temperature_c (descending). Equal maxima retain the
# chronological/project order used by the original notebook.
_HEAT_EVENTS = [
    _event(
        "2026-06-18T00:00",
        "2026-06-28T23:00",
        "Möckern-Drewitz",
        12.1641,
        52.2174,
        "https://climateexplorer.app/dwd/hourly/sachsen-anhalt/moeckern-drewitz/",
        41.8,
        "2026-06-27",
        status="vorläufig verifiziert",
        event_source_url="https://www.dwd.de/DE/leistungen/besondereereignisse/temperatur/20260713_hitzewelle-deutschland.pdf?__blob=publicationFile&v=3",
    ),
    _event(
        "2019-07-22T00:00",
        "2019-07-26T23:00",
        "Duisburg-Baerl",
        6.7018,
        51.5088,
        "https://meteostat.net/de/station/D3670",
        41.2,
        "2019-07-25",
    ),
    _event(
        "2026-07-27T00:00",
        "2026-07-31T23:00",
        "Bernburg/Saale (Nord)",
        11.7109,
        51.8218,
        "https://meteostat.net/de/station/D0445",
        40.5,
        "2026-07-30",
        status="vorläufig",
        event_source_url="https://www.dwd.de/DE/presse/pressemitteilungen/DE/2026/20260730_deutschlandwetter_juli_news.html",
    ),
    _event(
        "2015-06-30T00:00",
        "2015-07-06T23:00",
        "Kitzingen",
        10.1781,
        49.7363,
        "https://meteostat.net/de/station/D2600",
        40.3,
        "2015-07-05",
    ),
    _event(
        "2015-08-03T00:00",
        "2015-08-08T23:00",
        "Kitzingen",
        10.1781,
        49.7363,
        "https://meteostat.net/de/station/D2600",
        40.3,
        "2015-08-07",
    ),
    _event(
        "2003-08-01T00:00",
        "2003-08-13T23:00",
        "Karlsruhe",
        8.3667,
        49.0333,
        "https://meteostat.net/de/station/10727",
        40.2,
        "2003-08-09",
    ),
    _event(
        "2022-07-17T00:00",
        "2022-07-20T23:00",
        "Hamburg-Neuwiedenthal",
        9.8957,
        53.4777,
        "https://meteostat.net/de/station/D1981",
        40.1,
        "2022-07-20",
    ),
    _event(
        "2012-08-17T00:00",
        "2012-08-21T23:00",
        "Dresden-Hosterwitz",
        13.8470,
        51.0221,
        "https://meteostat.net/de/station/D1050",
        39.8,
        "2012-08-20",
    ),
    _event(
        "2019-06-24T00:00",
        "2019-06-30T23:00",
        "Bernburg/Saale (Nord)",
        11.7109,
        51.8218,
        "https://meteostat.net/de/station/D0445",
        39.6,
        "2019-06-30",
    ),
    _event(
        "2018-07-23T00:00",
        "2018-08-08T23:00",
        "Bernburg/Saale (Nord)",
        11.7109,
        51.8218,
        "https://meteostat.net/de/station/D0445",
        39.5,
        "2018-07-31",
    ),
    _event(
        "2007-07-12T00:00",
        "2007-07-16T23:00",
        "Holzdorf",
        13.1833,
        51.7667,
        "https://meteostat.net/de/station/10476",
        39.2,
        "2007-07-16",
    ),
    _event(
        "2022-06-15T00:00",
        "2022-06-19T23:00",
        "Dresden-Strehlen",
        13.7750,
        51.0248,
        "https://meteostat.net/de/station/D1051",
        39.2,
        "2022-06-19",
    ),
    _event(
        "2025-06-28T00:00",
        "2025-07-03T23:00",
        "Kitzingen",
        10.1781,
        49.7363,
        "https://meteostat.net/de/station/D2600",
        39.1,
        "2025-07-02",
        status="DWD-Jahresbilanz",
        event_source_url="https://www.dwd.de/DE/presse/pressemitteilungen/DE/2025/20251230_pm_jahr-2025_news.html",
    ),
    _event(
        "2006-07-09T00:00",
        "2006-07-27T23:00",
        "Bernburg/Saale (Nord)",
        11.7109,
        51.8218,
        "https://meteostat.net/de/station/D0445",
        38.9,
        "2006-07-20",
    ),
    _event(
        "2010-07-03T00:00",
        "2010-07-21T23:00",
        "Bendorf",
        7.5833,
        50.4167,
        "https://meteostat.net/de/station/10515",
        38.8,
        "2010-07-12",
    ),
    _event(
        "2023-07-08T00:00",
        "2023-07-17T23:00",
        "Möhrendorf-Kleinseebach",
        11.0074,
        49.6497,
        "https://meteostat.net/de/station/D1279",
        38.8,
        "2023-07-15",
    ),
    _event(
        "2026-08-03T00:00",
        "2026-08-04T18:00",
        "Saarbrücken-Burbach",
        6.9351,
        49.2406,
        "https://meteostat.net/de/station/D6217",
        38.7,
        "2026-08-03",
        status="laufend, vorläufig",
        event_source_url="https://www.dwd.de/DE/wetter/thema_des_tages/2026/8/3.html",
        ongoing=True,
    ),
    _event(
        "2013-07-25T00:00",
        "2013-07-29T23:00",
        "Rheinfelden",
        7.7721,
        47.5590,
        "https://climateexplorer.app/dwd/hourly/baden-wuerttemberg/rheinfelden/",
        38.6,
        "2013-07-27",
    ),
    _event(
        "2020-08-07T00:00",
        "2020-08-11T23:00",
        "Trier-Petrisberg",
        6.6667,
        49.7500,
        "https://meteostat.net/de/station/10609",
        38.6,
        "2020-08-09",
    ),
    _event(
        "2016-08-24T00:00",
        "2016-08-28T23:00",
        "Saarbrücken-Burbach",
        6.9351,
        49.2406,
        "https://meteostat.net/de/station/D6217",
        37.9,
        "2016-08-27",
    ),
]


heat_periods_germany = {
    "as_of": DATA_AS_OF,
    "ranking_basis": RANKING_BASIS,
    "events": [dict(event, rank=rank) for rank, event in enumerate(_HEAT_EVENTS, start=1)],
}


def get_heat_period(rank: int) -> dict[str, Any]:
    """Return a defensive copy of the heat-period event with the given rank."""
    if not isinstance(rank, int) or not 1 <= rank <= len(heat_periods_germany["events"]):
        raise ValueError(f"rank must be an integer between 1 and {len(heat_periods_germany['events'])}")
    return deepcopy(heat_periods_germany["events"][rank - 1])


_STATUS_TRANSLATIONS = {
    "historisch": "historical",
    "vorläufig verifiziert": "provisionally verified",
    "vorläufig": "provisional",
    "DWD-Jahresbilanz": "DWD annual summary",
    "laufend, vorläufig": "ongoing, provisional",
}


def format_status(event: dict[str, Any], language: str = "de") -> str:
    """Return the event status in German or English."""
    status = str(event["status"])
    if language.lower().startswith("en"):
        return _STATUS_TRANSLATIONS.get(status, status)
    return status


def format_core_period(event: dict[str, Any], language: str = "de") -> str:
    """Format an event core period in compact German or English notation."""
    start = pd.Timestamp(event["core_period"]["start"])
    end = pd.Timestamp(event["core_period"]["end"])
    ongoing = event["core_period"].get("ongoing")
    if language.lower().startswith("en"):
        if ongoing:
            return f"{start:%d %b %Y}–ongoing"
        if start.year == end.year and start.month == end.month:
            return f"{start:%d}–{end:%d %b %Y}"
        if start.year == end.year:
            return f"{start:%d %b}–{end:%d %b %Y}"
        return f"{start:%d %b %Y}–{end:%d %b %Y}"
    if ongoing:
        return f"{start:%d.%m.%Y}–laufend"
    if start.year == end.year and start.month == end.month:
        return f"{start.day:02d}.–{end.day:02d}.{end.month:02d}.{end.year}"
    if start.year == end.year:
        return f"{start.day:02d}.{start.month:02d}.–{end.day:02d}.{end.month:02d}.{end.year}"
    return f"{start:%d.%m.%Y}–{end:%d.%m.%Y}"


def heat_period_options(language: str = "de") -> list[tuple[str, int]]:
    """Return human-readable labels for an ``ipywidgets.Select`` widget."""
    options = []
    for event in heat_periods_germany["events"]:
        site = event["measurement_sites"][0]
        label = (
            f"{event['rank']:02d} | {format_core_period(event, language)} | "
            f"{event['maximum_temperature_c']:.1f} °C | {site['name']}"
        )
        options.append((label, event["rank"]))
    return options


def heat_periods_dataframe(language: str = "de") -> pd.DataFrame:
    """Return the German ranking as a notebook-friendly DataFrame."""
    rows = []
    for event in heat_periods_germany["events"]:
        site = event["measurement_sites"][0]
        if language.lower().startswith("en"):
            row = {
                "Rank": event["rank"],
                "Core period": format_core_period(event, language),
                "Maximum [°C]": event["maximum_temperature_c"],
                "Country": "Germany",
                "Station": site["name"],
                "Latitude [°N]": site["latitude"],
                "Longitude [°E]": site["longitude"],
                "Data status": format_status(event, language),
            }
        else:
            row = {
                "Rang": event["rank"],
                "Kernphase": format_core_period(event, language),
                "Maximum [°C]": event["maximum_temperature_c"],
                "Messstation": site["name"],
                "Breite [°N]": site["latitude"],
                "Länge [°E]": site["longitude"],
                "Datenstatus": event["status"],
            }
        rows.append(row)
    return pd.DataFrame(rows)
