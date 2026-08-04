"""Full interactive route designer used by ``hostradaRoute.ipynb``.

The selected weather provider also defines the address-search domain: DWD
accepts routes with start and destination in Germany, whereas CERRA supports
addresses across Europe. The same provider is then used for climate extraction.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import folium
import pandas as pd
import requests
from IPython.display import clear_output, display
from ipywidgets import (
    Button, Dropdown, FloatText, IntText, Layout, Output, Text, VBox, HBox,
    HTML, SelectMultiple,
)
from pyproj import Geod

from .hostradaRoute import calculate_route_climate, available_variables

OSRM_BASE_URL = "https://router.project-osrm.org"
PHOTON_URL = "https://photon.komoot.io/api"
VBB_TRANSIT_URL = "https://v6.vbb.transport.rest/journeys"
GEOD = Geod(ellps="WGS84")
ROUTE_APP_PROVIDER_REVISION = "2026-08-03-provider-before-route-v1"

# Photon returns ISO 3166-1 alpha-2 country codes in ``properties.countrycode``.
# The list intentionally covers sovereign European states plus Kosovo.
EUROPE_COUNTRY_CODES = frozenset({
    "AD", "AL", "AT", "AX", "BA", "BE", "BG", "BY", "CH", "CY",
    "CZ", "DE", "DK", "EE", "ES", "FI", "FO", "FR", "GB", "GG",
    "GI", "GR", "HR", "HU", "IE", "IM", "IS", "IT", "JE", "LI",
    "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO",
    "PL", "PT", "RO", "RS", "RU", "SE", "SI", "SJ", "SK", "SM",
    "TR", "UA", "VA", "XK", "AM", "AZ", "GE",
})

# Bounding boxes are supplied to Photon to improve ranking. Country-code
# validation below remains the authoritative domain check.
GERMANY_BBOX = (5.5, 47.0, 15.5, 55.2)
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.5)


def _normalise_provider(provider: str | None) -> str:
    name = str(provider or "dwd").strip().lower()
    return "cerra" if name == "cerra" else "dwd"


def provider_route_domain(provider: str | None) -> dict[str, object]:
    name = _normalise_provider(provider)
    if name == "cerra":
        return {
            "provider": name,
            "label": "Europe",
            "country_codes": EUROPE_COUNTRY_CODES,
            "bbox": EUROPE_BBOX,
        }
    return {
        "provider": name,
        "label": "Germany",
        "country_codes": frozenset({"DE"}),
        "bbox": GERMANY_BBOX,
    }


def _climate_variable_options(provider: str | None) -> list[tuple[str, str]]:
    """Return ``(label, code)`` options from the route climate table."""
    table = available_variables(provider)
    if hasattr(table, "iterrows") and "variable" in table:
        options = []
        for _, row in table.iterrows():
            code = str(row["variable"])
            label = str(row.get("description", code))
            options.append((label, code))
        return options
    return [(str(value), str(value)) for value in table]


@dataclass(frozen=True)
class RoutePoint:
    index: int
    timestamp: datetime
    longitude: float
    latitude: float
    distance_m: float
    mode: str = ""


@dataclass
class RouteAppDefaults:
    start_address: str = "Einsteinufer 43-53, 10587 Berlin"
    destination_address: str = "Potsdam Hauptbahnhof"
    start_time: str = ""
    average_speed_kmh: float = 50.0
    interval_minutes: int = 15
    profile: str = "driving"
    language: str = "de"
    output_file: str = "route_positions.csv"
    climate_output_file: str = "route_climate.csv"
    variables: Sequence[str] | str = "all"
    provider: str | None = None
    otp_url: str = "http://localhost:8080/otp/gtfs/v1"


def parse_start_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        result = result.astimezone()
    return result


def geocode_address(
    address: str,
    language: str = "de",
    provider: str | None = None,
) -> tuple[float, float, str]:
    """Resolve an address inside the spatial domain of the selected provider."""
    if not address.strip():
        raise ValueError("The address must not be empty.")

    domain = provider_route_domain(provider)
    bbox = domain["bbox"]
    response = requests.get(
        PHOTON_URL,
        params={
            "q": address.strip(),
            "limit": 8,
            "lang": language,
            "bbox": ",".join(str(value) for value in bbox),
        },
        headers={
            "User-Agent": "hostrada4py-route-notebook/0.42.0",
            "Accept": "application/json",
        },
        timeout=30,
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    allowed = domain["country_codes"]

    feature = None
    for candidate in features:
        props = candidate.get("properties", {})
        country_code = str(props.get("countrycode") or "").upper()
        if country_code in allowed:
            feature = candidate
            break

    if feature is None:
        if features:
            raise RuntimeError(
                f"Address found, but not inside the {domain['label']} route domain "
                f"for provider {str(domain['provider']).upper()}: {address}"
            )
        raise RuntimeError(
            f"Address not found inside the {domain['label']} route domain: {address}"
        )

    lon, lat = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})
    street = " ".join(
        str(value)
        for value in (props.get("street"), props.get("housenumber"))
        if value
    )
    label = ", ".join(
        str(value)
        for value in (
            props.get("name"),
            street or None,
            props.get("postcode"),
            props.get("city") or props.get("district") or props.get("county"),
            props.get("state"),
            props.get("country"),
        )
        if value
    )
    return float(lat), float(lon), label or address


def request_osrm_route(start, destination, profile="driving"):
    start_lat, start_lon = start
    dest_lat, dest_lon = destination
    profile_map = {"cycling": "cycling", "walking": "walking", "driving": "driving"}
    profile = profile_map.get(profile, "driving")
    coords = f"{start_lon:.8f},{start_lat:.8f};{dest_lon:.8f},{dest_lat:.8f}"
    response = requests.get(
        f"{OSRM_BASE_URL}/route/v1/{profile}/{coords}",
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message", "No route found."))
    route = data["routes"][0]
    geometry = [(float(lon), float(lat)) for lon, lat in route["geometry"]["coordinates"]]
    return geometry, float(route["distance"]), float(route["duration"])



def decode_polyline(encoded: str, precision: int = 5) -> list[tuple[float, float]]:
    """Decode an encoded polyline and return ``[(lon, lat), ...]``."""
    coordinates, index, latitude, longitude = [], 0, 0, 0
    factor = 10 ** precision
    while index < len(encoded):
        values = []
        for _ in range(2):
            result, shift = 0, 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            values.append(~(result >> 1) if result & 1 else result >> 1)
        latitude += values[0]
        longitude += values[1]
        coordinates.append((longitude / factor, latitude / factor))
    return coordinates


OTP_PLAN_QUERY = """
query Plan($fromPlace: String!, $toPlace: String!, $date: String!, $time: String!) {
  plan(fromPlace: $fromPlace, toPlace: $toPlace, date: $date, time: $time,
       arriveBy: false, numItineraries: 1,
       transportModes: [WALK, RAIL, SUBWAY, TRAM, BUS]) {
    itineraries { startTime endTime duration legs {
      mode startTime endTime duration distance
      from { name lat lon } to { name lat lon }
      route { shortName longName } legGeometry { points }
    }}
    messageStrings
  }
}
"""


def _epoch_ms(value, timezone_hint):
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone_hint)


def request_otp_route(start, destination, start_time: datetime, otp_url: str):
    """Request a public-transport itinerary from an OTP2 GraphQL endpoint."""
    start_lat, start_lon = start
    destination_lat, destination_lon = destination
    response = requests.post(
        otp_url.rstrip('/'),
        json={
            'query': OTP_PLAN_QUERY,
            'variables': {
                'fromPlace': f'Start::{start_lat},{start_lon}',
                'toPlace': f'Destination::{destination_lat},{destination_lon}',
                'date': start_time.strftime('%Y-%m-%d'),
                'time': start_time.strftime('%H:%M:%S'),
            },
        },
        headers={'Accept': 'application/json'},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get('errors'):
        raise RuntimeError('; '.join(error.get('message', str(error)) for error in payload['errors']))
    plan = (payload.get('data') or {}).get('plan') or {}
    itineraries = plan.get('itineraries') or []
    if not itineraries:
        raise RuntimeError('OTP returned no public-transport itinerary.')
    result = []
    for leg in itineraries[0].get('legs', []):
        encoded = (leg.get('legGeometry') or {}).get('points')
        geometry = decode_polyline(encoded) if encoded else []
        if not geometry:
            origin, target = leg.get('from') or {}, leg.get('to') or {}
            geometry = [(float(origin['lon']), float(origin['lat'])),
                        (float(target['lon']), float(target['lat']))]
        route = leg.get('route') or {}
        result.append({
            'mode': leg.get('mode', 'TRANSIT'),
            'start_time': _epoch_ms(leg['startTime'], start_time.tzinfo),
            'end_time': _epoch_ms(leg['endTime'], start_time.tzinfo),
            'geometry': geometry,
            'route_name': route.get('shortName') or route.get('longName') or '',
        })
    return result


def _parse_transit_time(value: str, timezone_hint):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone_hint)
    return result.astimezone(timezone_hint)


def _location_coordinates(location):
    location = (location or {}).get('location') or (location or {})
    if location.get('latitude') is None or location.get('longitude') is None:
        return None
    return float(location['longitude']), float(location['latitude'])


def _transit_geometry(polyline, origin, destination):
    geometry = polyline or {}
    if geometry.get('type') == 'Feature':
        geometry = geometry.get('geometry') or {}
    coordinates = geometry.get('coordinates') or [] if geometry.get('type') == 'LineString' else []
    parsed = [(float(item[0]), float(item[1])) for item in coordinates
              if isinstance(item, (list, tuple)) and len(item) >= 2]
    if len(parsed) >= 2:
        return parsed
    start = _location_coordinates(origin)
    end = _location_coordinates(destination)
    if start and end:
        return [start, end]
    raise RuntimeError('Transit leg contains no usable geometry.')


def request_public_transit_route(start, destination, start_time, start_address,
                                 destination_address, language='de'):
    """Use the public VBB/HAFAS endpoint as an OTP-free transit fallback."""
    start_lat, start_lon = start
    destination_lat, destination_lon = destination
    response = requests.get(
        VBB_TRANSIT_URL,
        params={
            'from.latitude': start_lat, 'from.longitude': start_lon,
            'from.address': start_address,
            'to.latitude': destination_lat, 'to.longitude': destination_lon,
            'to.address': destination_address,
            'departure': start_time.isoformat(timespec='seconds'),
            'results': 1, 'stopovers': 'true', 'polylines': 'true',
            'language': language, 'pretty': 'false',
        },
        headers={'User-Agent': 'hostrada4py-route-notebook/0.42.0', 'Accept': 'application/json'},
        timeout=60,
    )
    response.raise_for_status()
    journeys = response.json().get('journeys') or []
    if not journeys:
        raise RuntimeError('The public transit service returned no itinerary.')
    result = []
    for leg in journeys[0].get('legs', []):
        departure = leg.get('departure') or leg.get('plannedDeparture')
        arrival = leg.get('arrival') or leg.get('plannedArrival')
        if not departure or not arrival:
            continue
        line = leg.get('line') or {}
        result.append({
            'mode': 'WALK' if leg.get('walking') else str(line.get('mode') or line.get('product') or 'TRANSIT').upper(),
            'start_time': _parse_transit_time(departure, start_time.tzinfo),
            'end_time': _parse_transit_time(arrival, start_time.tzinfo),
            'geometry': _transit_geometry(leg.get('polyline'), leg.get('origin'), leg.get('destination')),
            'route_name': line.get('name') or '',
        })
    if not result:
        raise RuntimeError('The transit response contained no usable legs.')
    return result


def request_train_route_with_fallback(start, destination, start_time, otp_url,
                                      start_address, destination_address, language='de'):
    otp_error = None
    if otp_url and otp_url.strip():
        try:
            return request_otp_route(start, destination, start_time, otp_url), 'OpenTripPlanner'
        except Exception as error:
            otp_error = error
    try:
        return request_public_transit_route(
            start, destination, start_time, start_address, destination_address, language
        ), 'VBB transport.rest'
    except Exception as fallback_error:
        prefix = f'OTP: {otp_error}. ' if otp_error is not None else ''
        raise RuntimeError(prefix + f'Public transit fallback failed: {fallback_error}') from fallback_error

def build_segment_table(coordinates):
    lengths, starts, cumulative = [], [], 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coordinates, coordinates[1:]):
        _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
        starts.append(cumulative)
        lengths.append(max(float(distance), 0.0))
        cumulative += max(float(distance), 0.0)
    if not lengths:
        raise ValueError("The route needs at least two coordinates.")
    return lengths, starts, cumulative


def position_at_distance(coordinates, lengths, starts, distance_m):
    total = starts[-1] + lengths[-1]
    distance_m = min(max(float(distance_m), 0.0), total)
    if math.isclose(distance_m, total, abs_tol=1e-6):
        return coordinates[-1]
    for i, (segment_start, segment_length) in enumerate(zip(starts, lengths)):
        if segment_length > 0 and distance_m <= segment_start + segment_length:
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]
            azimuth, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
            lon, lat, _ = GEOD.fwd(lon1, lat1, azimuth, distance_m - segment_start)
            return float(lon), float(lat)
    return coordinates[-1]


def generate_constant_speed_points(geometry, start_time, speed_kmh, interval_seconds):
    if speed_kmh <= 0:
        raise ValueError("Average speed must be greater than zero.")
    if interval_seconds <= 0:
        raise ValueError("Interval must be greater than zero.")
    lengths, starts, total = build_segment_table(geometry)
    speed_mps = float(speed_kmh) / 3.6
    total_seconds = total / speed_mps
    elapsed = [i * interval_seconds for i in range(math.floor(total_seconds / interval_seconds) + 1)]
    if not elapsed or not math.isclose(elapsed[-1], total_seconds, abs_tol=1e-9):
        elapsed.append(total_seconds)
    points = []
    for index, seconds in enumerate(elapsed):
        distance = min(speed_mps * seconds, total)
        lon, lat = position_at_distance(geometry, lengths, starts, distance)
        points.append(RoutePoint(index, start_time + timedelta(seconds=seconds), lon, lat, distance))
    return points, total



def generate_transit_points(legs, interval_seconds):
    """Sample transit legs according to their actual timetable times."""
    if not legs:
        raise ValueError('No transit legs were supplied.')
    start, end = legs[0]['start_time'], legs[-1]['end_time']
    total_seconds = max((end - start).total_seconds(), 0)
    elapsed = [i * interval_seconds for i in range(math.floor(total_seconds / interval_seconds) + 1)]
    if not elapsed or elapsed[-1] != total_seconds:
        elapsed.append(total_seconds)
    tables, cumulative = [], 0.0
    for leg in legs:
        lengths, starts, distance = build_segment_table(leg['geometry'])
        tables.append((lengths, starts, distance, cumulative))
        cumulative += distance
    points = []
    for index, seconds in enumerate(elapsed):
        timestamp = start + timedelta(seconds=seconds)
        leg_index = next((i for i, leg in enumerate(legs)
                          if leg['start_time'] <= timestamp <= leg['end_time']), None)
        if leg_index is None:
            previous = [i for i, leg in enumerate(legs) if leg['end_time'] < timestamp]
            leg_index = previous[-1] if previous else 0
            leg = legs[leg_index]
            lon, lat = leg['geometry'][-1]
            distance = tables[leg_index][3] + tables[leg_index][2]
            mode = 'WAIT'
        else:
            leg = legs[leg_index]
            lengths, starts, leg_distance, distance_before = tables[leg_index]
            duration = max((leg['end_time'] - leg['start_time']).total_seconds(), 1)
            fraction = min(max((timestamp - leg['start_time']).total_seconds() / duration, 0), 1)
            local_distance = leg_distance * fraction
            lon, lat = position_at_distance(leg['geometry'], lengths, starts, local_distance)
            distance = distance_before + local_distance
            mode = leg['mode']
        points.append(RoutePoint(index, timestamp, lon, lat, distance, mode))
    return points, cumulative

def points_to_dataframe(points):
    return pd.DataFrame([{
        "point_index": point.index,
        "timestamp": point.timestamp.isoformat(timespec="seconds"),
        "longitude": point.longitude,
        "latitude": point.latitude,
        "distance_m": round(point.distance_m, 2),
        "mode": point.mode,
    } for point in points])


def create_route_map(geometry, points, start_label, destination_label):
    geometries = geometry if geometry and isinstance(geometry[0], tuple) and len(geometry[0]) == 2 and isinstance(geometry[0][1], str) else [(geometry, "Route")]
    all_coordinates = [coordinate for coordinates, _label in geometries for coordinate in coordinates]
    start_lon, start_lat = all_coordinates[0]
    end_lon, end_lat = all_coordinates[-1]
    route_map = folium.Map(location=[start_lat, start_lon], zoom_start=9, control_scale=True)
    for coordinates, label in geometries:
        folium.PolyLine([[lat, lon] for lon, lat in coordinates], weight=5, opacity=0.85,
                        tooltip=label).add_to(route_map)
    folium.Marker([start_lat, start_lon], tooltip="Start", popup=start_label,
                  icon=folium.Icon(icon="play")).add_to(route_map)
    folium.Marker([end_lat, end_lon], tooltip="Destination", popup=destination_label,
                  icon=folium.Icon(icon="stop")).add_to(route_map)
    for point in points:
        folium.CircleMarker(
            [point.latitude, point.longitude], radius=4, weight=1, fill=True,
            tooltip=point.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            popup=f"Point {point.index}<br>{point.distance_m/1000:.2f} km",
        ).add_to(route_map)
    route_map.fit_bounds([[lat, lon] for lon, lat in all_coordinates])
    return route_map


class RouteNotebookApp:
    """Address, routing, map, climate extraction and result display in one UI."""

    def __init__(self, defaults=None, *, show=False):
        self.defaults = defaults or RouteAppDefaults()
        self.route_dataframe = None
        self.climate_dataframe = None
        self.route_map = None
        self._build_widgets()
        self.set_provider(self.defaults.provider or "dwd")
        self.route_button.on_click(self._run_route)
        self.climate_button.on_click(self._run_climate)
        if show:
            self.show()

    def _build_widgets(self):
        d = self.defaults
        default_start = d.start_time or datetime.now().astimezone().replace(second=0, microsecond=0).isoformat(timespec="minutes")
        self.start = Text(value=d.start_address, description="Start:", layout=Layout(width="850px"))
        self.destination = Text(value=d.destination_address, description="Destination:", layout=Layout(width="850px"))
        self.start_time = Text(value=default_start, description="Start time:", layout=Layout(width="520px"))
        self.speed = FloatText(value=d.average_speed_kmh, description="Average km/h:")
        self.interval = IntText(value=d.interval_minutes, description="Interval min:")
        self.profile = Dropdown(options=[("Car", "driving"), ("Bicycle", "cycling"),
                                         ("Walking", "walking"), ("Train / public transport", "train")], value=d.profile, description="Profile:")
        self.language = Text(value=d.language, description="Language:")
        self.output_file = Text(value=d.output_file, description="Route CSV:", layout=Layout(width="520px"))
        self.otp_url = Text(value=d.otp_url, description="OTP URL:", layout=Layout(width="720px"))
        self.climate_output_file = Text(value=d.climate_output_file, description="Climate CSV:", layout=Layout(width="520px"))
        options = _climate_variable_options(d.provider)
        initial = tuple(value for _, value in options[:4])
        self.variables = SelectMultiple(options=options, value=initial, description="Variables:",
                                        layout=Layout(width="480px", height="180px"))
        self.route_button = Button(description="1. Calculate route", button_style="success", icon="map")
        self.climate_button = Button(description="2. Extract climate", button_style="primary", icon="cloud-download")
        self.domain_status = HTML()
        self.status = HTML()
        self.output = Output(layout=Layout(min_height="420px"))
        self.container = VBox([
            HTML("<h3>Route definition</h3>"), self.domain_status, self.start, self.destination,
            HBox([self.start_time, self.profile]), HBox([self.speed, self.interval, self.language]),
            self.output_file, self.otp_url, self.route_button,
            self.status, self.output,
        ])

    def set_provider(self, provider: str | None):
        """Update route domain and climate variables for a provider."""
        name = _normalise_provider(provider)
        self.defaults.provider = name
        domain = provider_route_domain(name)
        if name == "dwd":
            text = (
                "DWD/HOSTRADA: start and destination must be in Germany. "
                "Weather data outside Germany are unavailable."
            )
        else:
            text = (
                "CERRA: start and destination may be selected across Europe. "
                "Long routes can require several monthly or spatial cache files. "
                "European train routing requires a suitable OpenTripPlanner endpoint."
            )
        self.domain_status.value = (
            f"<b>Route domain:</b> {domain['label']} &nbsp; "
            f"<b>provider:</b> {name.upper()}<br><small>{text}</small>"
        )
        self.refresh_variables()
        return name

    def refresh_variables(self):
        """Refresh the climate menu after the notebook provider changes."""
        old = tuple(self.variables.value)
        options = _climate_variable_options(self.defaults.provider)
        values = [value for _, value in options]
        self.variables.options = options
        selected = tuple(value for value in old if value in values)
        self.variables.value = selected or tuple(values[: min(4, len(values))])
        return values

    def show(self):
        display(self.container)
        return self

    def _run_route(self, _button=None):
        with self.output:
            clear_output(wait=True)
            try:
                start_time = parse_start_time(self.start_time.value)
                print("Geocoding start address …")
                start_lat, start_lon, start_label = geocode_address(self.start.value, self.language.value or "de", self.defaults.provider)
                time.sleep(0.2)
                print("Geocoding destination …")
                dest_lat, dest_lon, destination_label = geocode_address(self.destination.value, self.language.value or "de", self.defaults.provider)
                print("Calculating route …")
                interval_seconds = int(self.interval.value) * 60
                if self.profile.value == "train":
                    legs, engine = request_train_route_with_fallback(
                        (start_lat, start_lon), (dest_lat, dest_lon), start_time,
                        self.otp_url.value, self.start.value, self.destination.value,
                        self.language.value or "de",
                    )
                    points, distance = generate_transit_points(legs, interval_seconds)
                    geometry = [(leg["geometry"], f'{leg["mode"]} {leg["route_name"]}'.strip()) for leg in legs]
                    routed_duration = (points[-1].timestamp - points[0].timestamp).total_seconds()
                    print(f"Transit engine: {engine}; timetable times are used.")
                else:
                    route_geometry, _routed_distance, routed_duration = request_osrm_route(
                        (start_lat, start_lon), (dest_lat, dest_lon), self.profile.value
                    )
                    points, distance = generate_constant_speed_points(
                        route_geometry, start_time, float(self.speed.value), interval_seconds
                    )
                    geometry = route_geometry
                frame = points_to_dataframe(points)
                target = Path(self.output_file.value or "route_positions.csv")
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(target, index=False)
                self.route_dataframe = frame
                self.route_map = create_route_map(geometry, points, start_label, destination_label)
                self.status.value = (
                    f"<b>Route ready:</b> {distance/1000:.1f} km, {len(frame)} sample points. "
                    f"OSRM estimate: {routed_duration/60:.0f} min."
                )
                display(self.route_map)
                display(frame.head(20))
            except Exception as exc:
                self.status.value = f"<span style='color:#b00020'><b>Route error:</b> {exc}</span>"
                raise

    def _run_climate(self, _button=None):
        with self.output:
            clear_output(wait=True)
            if self.route_dataframe is None:
                print("Calculate the route first.")
                return
            variables = tuple(self.variables.value) or "all"
            route = self.route_dataframe.rename(columns={
                "longitude": "lon", "latitude": "lat", "timestamp": "time"
            })
            print("Extracting climate data for route points …")
            result = calculate_route_climate(
                route, variables=variables,
                output_file=self.climate_output_file.value or "route_climate.csv",
                provider=self.defaults.provider,
            )
            self.climate_dataframe = result
            self.status.value = f"<b>Climate extraction complete:</b> {len(result)} route points."
            display(result.head(30))
            display(result.select_dtypes("number").describe().T)

    def _plot_climate(self, _button=None):
        with self.output:
            clear_output(wait=True)
            if self.climate_dataframe is None or self.climate_dataframe.empty:
                print("Extract climate data first.")
                return
            import matplotlib.pyplot as plt
            frame = self.climate_dataframe.copy()
            x = frame.get("distance_m", pd.Series(range(len(frame)))) / 1000.0
            columns = [column for column in self.variables.value if column in frame]
            if not columns:
                print("No selected climate columns are present.")
                return
            ax = frame.set_index(x)[columns].plot(figsize=(12, 5), marker="o", markersize=2)
            ax.set_xlabel("Distance along route [km]")
            ax.grid(True, alpha=0.3)
            plt.show()
            display(frame[[c for c in ["timestamp", "longitude", "latitude", "distance_m", *columns] if c in frame]].head(100))

    def calculate(self, route):
        """Historic compatibility method for programmatically supplied routes."""
        result = calculate_route_climate(
            route,
            variables=self.defaults.variables,
            output_file=self.climate_output_file.value,
            provider=self.defaults.provider,
        )
        self.climate_dataframe = result
        return result


RouteLeafletApp = RouteNotebookApp


def launch_route_app(defaults=None):
    return RouteNotebookApp(defaults).show()
