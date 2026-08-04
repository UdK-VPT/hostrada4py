"""European heat-period catalogue and helper functions for HOSTRADA/CERRA.

The catalogue is deliberately constrained to events from September 1984 onward,
matching the temporal start of the Copernicus European Regional ReAnalysis
(CERRA). Events are ranked by the highest documented station air temperature
within the selected core period. This is a curated event ranking, not an
official pan-European heat-wave intensity index.

The catalogue was reviewed on 2026-08-04. Summer 2026 events are not included
because the CERRA archive available at that date did not yet cover them.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

import pandas as pd


DATA_AS_OF = "2026-08-04T19:00:00Z"
CERRA_START = "1984-09-01T00:00:00Z"
CERRA_ARCHIVE_CHECKED_THROUGH = "2026-03-31T23:00:00Z"
RANKING_BASIS = "maximum documented station air temperature within the event core period"
TOP_N = 20


def identify_heat_periods(
    data,
    temperature_column: str = "tas",
    threshold: float = 30.0,
    minimum_hours: int = 72,
):
    """Identify consecutive hot periods in an hourly time series."""
    df = data.copy()
    df["time"] = pd.to_datetime(df["time"])
    values = pd.to_numeric(df[temperature_column], errors="coerce")
    hot = values >= threshold
    groups = (hot != hot.shift()).cumsum()
    rows = []
    for _, group in df[hot].groupby(groups[hot]):
        if len(group) >= minimum_hours:
            group_values = pd.to_numeric(group[temperature_column], errors="coerce")
            rows.append(
                {
                    "start": group["time"].min(),
                    "end": group["time"].max(),
                    "hours": len(group),
                    "maximum": float(group_values.max()),
                    "mean": float(group_values.mean()),
                }
            )
    return pd.DataFrame(rows)


def rank_heat_periods(
    periods: pd.DataFrame,
    *,
    maximum_column: str = "maximum",
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """Return the hottest periods sorted by their maximum temperature."""
    if maximum_column not in periods.columns:
        raise KeyError(f"Missing maximum-temperature column: {maximum_column}")
    ranked = periods.copy()
    ranked[maximum_column] = pd.to_numeric(ranked[maximum_column], errors="coerce")
    ranked = ranked.dropna(subset=[maximum_column]).sort_values(
        [maximum_column, "start"], ascending=[False, True]
    )
    ranked = ranked.head(top_n).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def heat_period_metrics(data, temperature_column: str = "tas", threshold: float = 30.0):
    values = pd.to_numeric(data[temperature_column], errors="coerce")
    return {
        "hours_above_threshold": int((values >= threshold).sum()),
        "maximum_temperature": float(values.max()),
        "mean_temperature": float(values.mean()),
    }


def calculate_heating_degree_days(
    data,
    temperature_column: str = "tas",
    base_temperature: float = 20.0,
    heating_limit: float = 15.0,
):
    df = data.copy()
    df["time"] = pd.to_datetime(df["time"])
    daily = df.set_index("time")[temperature_column].resample("D").mean()
    return float((base_temperature - daily[daily < heating_limit]).clip(lower=0).sum())


extract_heat_periods = identify_heat_periods


def _station_reference(latitude: float, longitude: float) -> str:
    return (
        "https://www.openstreetmap.org/"
        f"?mlat={latitude:.5f}&mlon={longitude:.5f}#map=9/{latitude:.5f}/{longitude:.5f}"
    )


def _event(
    start: str,
    end: str,
    station: str,
    country: str,
    longitude: float,
    latitude: float,
    maximum_temperature_c: float,
    record_date: str,
    *,
    status: str,
    event_source_url: str,
    region: str,
) -> dict[str, Any]:
    return {
        "core_period": {"start": start, "end": end, "ongoing": False},
        "maximum_temperature_c": maximum_temperature_c,
        "record_date": record_date,
        "status": status,
        "event_source_url": event_source_url,
        "region": region,
        "measurement_sites": [
            {
                "name": station,
                "country": country,
                "longitude": longitude,
                "latitude": latitude,
                "source_url": _station_reference(latitude, longitude),
            }
        ],
    }


# Candidate events are sorted programmatically below. Core periods are compact
# windows intended for CERRA point extraction, not formal continent-wide onset
# and end dates. Sources prioritise WMO, Copernicus and national services.
_CANDIDATE_EVENTS = [
    _event(
        "2021-08-08T00:00", "2021-08-15T23:00", "Floridia (Syracuse)", "Italy",
        15.1536, 37.0833, 48.8, "2021-08-11",
        status="WMO verified",
        event_source_url="https://wmo.int/news/media-centre/wmo-confirms-verification-of-new-continental-european-temperature-record",
        region="Mediterranean Europe",
    ),
    _event(
        "2007-06-24T00:00", "2007-06-30T23:00", "Nea Filadelfeia", "Greece",
        23.7390, 38.0370, 47.5, "2007-06-26",
        status="national network record",
        event_source_url="https://en.wikipedia.org/wiki/2007_European_heatwave",
        region="South-eastern Europe",
    ),
    _event(
        "2003-07-29T00:00", "2003-08-14T23:00", "Amareleja", "Portugal",
        -7.2260, 38.2097, 47.4, "2003-08-01",
        status="IPMA official record",
        event_source_url="https://www.ipma.pt/en/media/noticias/news.detail.jsp?f=ema-amareleja.html&y=2015",
        region="Western and Central Europe",
    ),
    _event(
        "2023-07-17T00:00", "2023-07-25T23:00", "Sestu", "Italy",
        9.0920, 39.2980, 47.3, "2023-07-24",
        status="national network report",
        event_source_url="https://wmo.int/news/media-centre/july-2023-set-be-hottest-month-record",
        region="Mediterranean Europe",
    ),
    _event(
        "2017-07-12T00:00", "2017-07-14T23:00", "Montoro", "Spain",
        -4.3810, 38.0240, 47.3, "2017-07-13",
        status="AEMET verified",
        event_source_url="https://www.rtve.es/noticias/20170714/cordoba-registra-temperatura-mas-alta-durante-ola-calor-espana-con-469-grados/1581533.shtml",
        region="Iberian Peninsula",
    ),
    _event(
        "1994-07-03T00:00", "1994-07-05T23:00", "Murcia–Alfonso X", "Spain",
        -1.1307, 37.9922, 47.2, "1994-07-04",
        status="historical AEMET record",
        event_source_url="https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/Olas_calor/Olas_Calor_Actualizacion_Junio_2019.pdf",
        region="Iberian Peninsula",
    ),
    _event(
        "2012-08-08T00:00", "2012-08-11T23:00", "Mengíbar", "Spain",
        -3.8080, 37.9700, 47.1, "2012-08-10",
        status="national network report",
        event_source_url="https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/Olas_calor/Olas_Calor_Actualizacion_Junio_2019.pdf",
        region="Iberian Peninsula",
    ),
    _event(
        "2022-07-08T00:00", "2022-07-18T23:00", "Pinhão", "Portugal",
        -7.5460, 41.1900, 47.0, "2022-07-14",
        status="IPMA national network",
        event_source_url="https://wmo.int/news/media-centre/heatwaves-and-wildfires-scorch-europe-africa-and-asia",
        region="Western Europe",
    ),
    _event(
        "2023-08-07T00:00", "2023-08-12T23:00", "Valencia Airport", "Spain",
        -0.4730, 39.4900, 46.8, "2023-08-10",
        status="AEMET official station",
        event_source_url="https://wmo.int/news/media-centre/july-2023-set-be-hottest-month-record",
        region="Western Mediterranean",
    ),
    _event(
        "2018-08-01T00:00", "2018-08-06T23:00", "Alvega", "Portugal",
        -8.0090, 39.4630, 46.8, "2018-08-04",
        status="IPMA official station",
        event_source_url="https://www.ipma.pt/resources.www/docs/im.publicacoes/edicoes.online/20180924/QyzZvZwgxxBnLFiHkSkX/cli_20180801_20180831_pcl_mm_co_pt.pdf",
        region="Iberian Peninsula",
    ),
    _event(
        "2025-06-19T00:00", "2025-07-04T23:00", "Mora", "Portugal",
        -8.1650, 38.9430, 46.6, "2025-06-27",
        status="WMO annual summary",
        event_source_url="https://wmo.int/sites/default/files/2026-03/State_of_the_Global_Climate_2025_Extreme_Supplement.pdf",
        region="Western Europe",
    ),
    _event(
        "2020-09-02T00:00", "2020-09-05T23:00", "Athalassa", "Cyprus",
        33.3970, 35.1410, 46.2, "2020-09-04",
        status="national record",
        event_source_url="https://climate.copernicus.eu/surface-air-temperature-september-2020",
        region="Eastern Mediterranean",
    ),
    _event(
        "2019-06-24T00:00", "2019-06-30T23:00", "Vérargues", "France",
        4.1000, 43.7170, 46.0, "2019-06-28",
        status="Météo-France verified",
        event_source_url="https://wmo.int/media/news/european-heatwave-sets-new-temperature-records",
        region="Western and Central Europe",
    ),
    _event(
        "2025-08-03T00:00", "2025-08-18T23:00", "Jerez Airport", "Spain",
        -6.0600, 36.7440, 45.8, "2025-08-17",
        status="AEMET annual summary",
        event_source_url="https://www.aemet.es/es/noticias/2026/01/resumen_anual_2025",
        region="South-western Europe",
    ),
    _event(
        "2010-07-30T00:00", "2010-08-02T23:00", "Athalassa", "Cyprus",
        33.3970, 35.1410, 45.6, "2010-08-01",
        status="national network record",
        event_source_url="https://en.wikipedia.org/wiki/Climate_of_Cyprus",
        region="Eastern Mediterranean",
    ),
    _event(
        "2024-06-11T00:00", "2024-06-14T23:00", "Astromeritis", "Cyprus",
        33.0370, 35.1280, 45.3, "2024-06-13",
        status="national network report",
        event_source_url="https://www.worldweatherattribution.org/deadly-mediterranean-heatwave-would-not-have-occurred-without-human-induced-climate-change/",
        region="Eastern Mediterranean",
    ),
    _event(
        "2015-06-27T00:00", "2015-07-22T23:00", "Córdoba Airport", "Spain",
        -4.8490, 37.8420, 45.2, "2015-07-06",
        status="AEMET official station",
        event_source_url="https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/Olas_calor/Olas_Calor_Actualizacion_Junio_2019.pdf",
        region="Western and Central Europe",
    ),
    _event(
        "2016-08-06T00:00", "2016-08-10T23:00", "Lousã", "Portugal",
        -8.2490, 40.1160, 45.0, "2016-08-07",
        status="IPMA national network",
        event_source_url="https://www.ipma.pt/en/oclima/extremos.clima/index.jsp",
        region="Iberian Peninsula",
    ),
    _event(
        "2017-06-14T00:00", "2017-06-18T23:00", "Córdoba Airport", "Spain",
        -4.8490, 37.8420, 44.9, "2017-06-17",
        status="AEMET official station",
        event_source_url="https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/Olas_calor/Olas_Calor_Actualizacion_Junio_2019.pdf",
        region="Iberian Peninsula",
    ),
    _event(
        "2013-07-25T00:00", "2013-07-30T23:00", "Córdoba Airport", "Spain",
        -4.8490, 37.8420, 44.8, "2013-07-28",
        status="historical station report",
        event_source_url="https://www.aemet.es/documentos/es/conocermas/recursos_en_linea/publicaciones_y_estudios/estudios/Olas_calor/Olas_Calor_Actualizacion_Junio_2019.pdf",
        region="Iberian Peninsula",
    ),
    # Additional documented candidates retained so the top-20 calculation is explicit.
    _event(
        "2021-06-25T00:00", "2021-07-01T23:00", "Montoro", "Spain",
        -4.3810, 38.0240, 44.7, "2021-06-28",
        status="national network report",
        event_source_url="https://climate.copernicus.eu/surface-air-temperature-june-2021",
        region="Iberian Peninsula",
    ),
    _event(
        "2024-07-09T00:00", "2024-07-24T23:00", "Gythio", "Greece",
        22.5650, 36.7610, 44.5, "2024-07-19",
        status="national network report",
        event_source_url="https://climate.copernicus.eu/c3s-seasonal-lookback-summer-2024",
        region="South-eastern Europe",
    ),
    _event(
        "2022-06-11T00:00", "2022-06-18T23:00", "Andújar", "Spain",
        -4.0500, 38.0390, 44.2, "2022-06-17",
        status="AEMET official station",
        event_source_url="https://wmo.int/news/media-centre/heatwaves-and-wildfires-scorch-europe-africa-and-asia",
        region="South-western Europe",
    ),
]


def _rank_catalogue(events: Iterable[dict[str, Any]], top_n: int = TOP_N) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda event: (
            -float(event["maximum_temperature_c"]),
            pd.Timestamp(event["record_date"]),
            event["measurement_sites"][0]["name"],
        ),
    )[:top_n]
    return [dict(deepcopy(event), rank=rank) for rank, event in enumerate(ordered, start=1)]


heat_periods_europe = {
    "as_of": DATA_AS_OF,
    "cerra_start": CERRA_START,
    "cerra_archive_checked_through": CERRA_ARCHIVE_CHECKED_THROUGH,
    "ranking_basis": RANKING_BASIS,
    "events": _rank_catalogue(_CANDIDATE_EVENTS),
}


def get_heat_period(rank: int) -> dict[str, Any]:
    """Return a defensive copy of the European heat-period event at ``rank``."""
    events = heat_periods_europe["events"]
    if not isinstance(rank, int) or not 1 <= rank <= len(events):
        raise ValueError(f"rank must be an integer between 1 and {len(events)}")
    return deepcopy(events[rank - 1])


def format_core_period(event: dict[str, Any], language: str = "en") -> str:
    """Format a core period in English ISO-style or compact German notation."""
    start = pd.Timestamp(event["core_period"]["start"])
    end = pd.Timestamp(event["core_period"]["end"])
    if language.lower().startswith("de"):
        if start.year == end.year and start.month == end.month:
            return f"{start.day:02d}.–{end.day:02d}.{end.month:02d}.{end.year}"
        if start.year == end.year:
            return f"{start.day:02d}.{start.month:02d}.–{end.day:02d}.{end.month:02d}.{end.year}"
        return f"{start:%d.%m.%Y}–{end:%d.%m.%Y}"
    if start.year == end.year and start.month == end.month:
        return f"{start:%d}–{end:%d %b %Y}"
    if start.year == end.year:
        return f"{start:%d %b}–{end:%d %b %Y}"
    return f"{start:%d %b %Y}–{end:%d %b %Y}"


def format_status(event: dict[str, Any], language: str = "en") -> str:
    """Return the catalogue status; European entries are stored in English."""
    return str(event["status"])


def heat_period_options(language: str = "en") -> list[tuple[str, int]]:
    """Return labels suitable for an ``ipywidgets.Select`` widget."""
    options = []
    for event in heat_periods_europe["events"]:
        site = event["measurement_sites"][0]
        label = (
            f"{event['rank']:02d} | {format_core_period(event, language)} | "
            f"{event['maximum_temperature_c']:.1f} °C | {site['name']}, {site['country']}"
        )
        options.append((label, event["rank"]))
    return options


def heat_periods_dataframe(language: str = "en") -> pd.DataFrame:
    """Return the European ranking as a notebook-friendly DataFrame."""
    rows = []
    for event in heat_periods_europe["events"]:
        site = event["measurement_sites"][0]
        if language.lower().startswith("de"):
            row = {
                "Rang": event["rank"],
                "Kernphase": format_core_period(event, language),
                "Maximum [°C]": event["maximum_temperature_c"],
                "Land": site["country"],
                "Messstation": site["name"],
                "Breite [°N]": site["latitude"],
                "Länge [°E]": site["longitude"],
                "Datenstatus": event["status"],
            }
        else:
            row = {
                "Rank": event["rank"],
                "Core period": format_core_period(event, language),
                "Maximum [°C]": event["maximum_temperature_c"],
                "Country": site["country"],
                "Station": site["name"],
                "Latitude [°N]": site["latitude"],
                "Longitude [°E]": site["longitude"],
                "Data status": event["status"],
            }
        rows.append(row)
    return pd.DataFrame(rows)
