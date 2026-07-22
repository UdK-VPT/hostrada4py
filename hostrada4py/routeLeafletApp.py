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
from ipywidgets import Button, Dropdown, FloatText, IntText, Layout, Output, Text, VBox
from pyproj import Geod


OSRM_BASE_URL = "https://router.project-osrm.org"
PHOTON_URL = "https://photon.komoot.io/api"
VBB_TRANSIT_URL = "https://v6.vbb.transport.rest/journeys"
GEOD = Geod(ellps="WGS84")


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
    start_address: str = "Raabestraße 13, 10405 Berlin"
    destination_address: str = "Klaushagen 33, 17268 Boitzenburger Land"
    start_time: str = ""
    average_speed_kmh: float = 50.0
    interval_minutes: int = 5
    profile: str = "driving"
    language: str = "de"
    output_file: str = "route_positions.csv"
    otp_url: str = "http://localhost:8080/otp/gtfs/v1"


def parse_start_time(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Der Startzeitpunkt muss eine Zeitzone enthalten.")
    return result


def geocode_address(address: str, language: str = "de") -> tuple[float, float, str]:
    if not address.strip():
        raise ValueError("Die Adresse darf nicht leer sein.")
    response = requests.get(
        PHOTON_URL,
        params={"q": address.strip(), "limit": 1, "lang": language},
        headers={"User-Agent": "osm-route-otp-notebook/3.0", "Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError(f"Adresse nicht gefunden: {address}")
    feature = features[0]
    lon, lat = feature["geometry"]["coordinates"]
    p = feature.get("properties", {})
    street = " ".join(str(v) for v in (p.get("street"), p.get("housenumber")) if v)
    label = ", ".join(str(v) for v in (
        p.get("name"), street or None, p.get("postcode"),
        p.get("city") or p.get("district") or p.get("county"),
        p.get("state"), p.get("country")
    ) if v)
    return float(lat), float(lon), label or address


def decode_polyline(encoded: str, precision: int = 5) -> list[tuple[float, float]]:
    """Dekodiert Google Encoded Polyline; Rückgabe als [(lon, lat), ...]."""
    coords, index, lat, lon = [], 0, 0, 0
    factor = 10 ** precision
    while index < len(encoded):
        values = []
        for _ in range(2):
            result, shift = 0, 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            values.append(~(result >> 1) if result & 1 else result >> 1)
        lat += values[0]
        lon += values[1]
        coords.append((lon / factor, lat / factor))
    return coords


def request_osrm_route(start, destination, profile="driving"):
    start_lat, start_lon = start
    dest_lat, dest_lon = destination
    coords = f"{start_lon:.8f},{start_lat:.8f};{dest_lon:.8f},{dest_lat:.8f}"
    response = requests.get(
        f"{OSRM_BASE_URL}/route/v1/{profile}/{coords}",
        params={"overview": "full", "geometries": "geojson", "steps": "false"},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(f"OSRM-Fehler: {data.get('message', 'Keine Route gefunden.')}")
    route = data["routes"][0]
    geometry = [(float(lon), float(lat)) for lon, lat in route["geometry"]["coordinates"]]
    return geometry, float(route["distance"]), float(route["duration"])


OTP_PLAN_QUERY = """
query Plan($fromPlace: String!, $toPlace: String!, $date: String!, $time: String!) {
  plan(
    fromPlace: $fromPlace
    toPlace: $toPlace
    date: $date
    time: $time
    arriveBy: false
    numItineraries: 1
    transportModes: [WALK, RAIL, SUBWAY, TRAM, BUS]
  ) {
    itineraries {
      startTime
      endTime
      duration
      legs {
        mode
        startTime
        endTime
        duration
        distance
        from { name lat lon }
        to { name lat lon }
        route { shortName longName }
        legGeometry { points }
      }
    }
    messageStrings
  }
}
"""


def _epoch_ms_to_datetime(value, timezone_hint) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone_hint)


def request_otp_route(start, destination, start_time: datetime, otp_url: str):
    """Fragt eine OTP-2-GTFS-GraphQL-Instanz ab und liefert Legs."""
    start_lat, start_lon = start
    dest_lat, dest_lon = destination
    variables = {
        "fromPlace": f"Start::{start_lat},{start_lon}",
        "toPlace": f"Ziel::{dest_lat},{dest_lon}",
        "date": start_time.strftime("%Y-%m-%d"),
        "time": start_time.strftime("%H:%M:%S"),
    }
    try:
        response = requests.post(
            otp_url.rstrip("/"),
            json={"query": OTP_PLAN_QUERY, "variables": variables},
            headers={"Accept": "application/json"},
            timeout=90,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"OpenTripPlanner ist unter {otp_url} nicht erreichbar. "
            "Starte zuerst eine OTP-Instanz mit passenden OSM- und GTFS-Daten."
        ) from exc

    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        details = "; ".join(error.get("message", str(error)) for error in payload["errors"])
        raise RuntimeError(f"OTP-GraphQL-Fehler: {details}")

    plan = (payload.get("data") or {}).get("plan") or {}
    itineraries = plan.get("itineraries") or []
    if not itineraries:
        messages = plan.get("messageStrings") or []
        raise RuntimeError("OTP hat keine Bahnverbindung gefunden. " + " ".join(messages))

    itinerary = itineraries[0]
    legs_out = []
    for leg in itinerary.get("legs", []):
        points_encoded = (leg.get("legGeometry") or {}).get("points")
        geometry = decode_polyline(points_encoded) if points_encoded else []
        if not geometry:
            origin, target = leg.get("from") or {}, leg.get("to") or {}
            geometry = [
                (float(origin["lon"]), float(origin["lat"])),
                (float(target["lon"]), float(target["lat"])),
            ]
        route = leg.get("route") or {}
        legs_out.append({
            "mode": leg.get("mode", ""),
            "start_time": _epoch_ms_to_datetime(leg["startTime"], start_time.tzinfo),
            "end_time": _epoch_ms_to_datetime(leg["endTime"], start_time.tzinfo),
            "distance_m": float(leg.get("distance") or 0),
            "geometry": geometry,
            "route_name": route.get("shortName") or route.get("longName") or "",
            "from_name": (leg.get("from") or {}).get("name", ""),
            "to_name": (leg.get("to") or {}).get("name", ""),
        })
    return legs_out



def _parse_iso_datetime(value: str, timezone_hint) -> datetime:
    if not value:
        raise ValueError("Eine Fahrplanzeit fehlt in der Transit-Antwort.")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone_hint)
    return result.astimezone(timezone_hint)


def _location_coordinates(location) -> tuple[float, float] | None:
    if not isinstance(location, dict):
        return None
    location = location.get("location") or location
    lat = location.get("latitude")
    lon = location.get("longitude")
    if lat is None or lon is None:
        return None
    return float(lon), float(lat)


def _transport_rest_geometry(polyline, origin, destination):
    """Liest eine Friendly-Public-Transport-Format-Polyline robust ein."""
    coordinates = []

    if isinstance(polyline, dict):
        feature = polyline
        if feature.get("type") == "Feature":
            feature = feature.get("geometry") or {}
        if feature.get("type") == "LineString":
            coordinates = feature.get("coordinates") or []
        elif isinstance(feature.get("features"), list):
            for item in feature["features"]:
                geometry = (item or {}).get("geometry") or {}
                if geometry.get("type") == "LineString":
                    coordinates.extend(geometry.get("coordinates") or [])

    parsed = []
    for coordinate in coordinates:
        if isinstance(coordinate, (list, tuple)) and len(coordinate) >= 2:
            parsed.append((float(coordinate[0]), float(coordinate[1])))

    if len(parsed) >= 2:
        return parsed

    start = _location_coordinates(origin)
    end = _location_coordinates(destination)
    if start and end:
        return [start, end]

    raise RuntimeError("Eine Transit-Teilstrecke enthält keine nutzbare Geometrie.")


def request_public_transit_route(
    start,
    destination,
    start_time: datetime,
    start_address: str,
    destination_address: str,
    language: str = "de",
    transit_url: str = VBB_TRANSIT_URL,
):
    """Öffentliche VBB/HAFAS-Auskunft als Fallback ohne lokalen OTP-Server."""
    start_lat, start_lon = start
    dest_lat, dest_lon = destination

    params = {
        "from.latitude": start_lat,
        "from.longitude": start_lon,
        "from.address": start_address,
        "to.latitude": dest_lat,
        "to.longitude": dest_lon,
        "to.address": destination_address,
        "departure": start_time.isoformat(timespec="seconds"),
        "results": 1,
        "stopovers": "true",
        "polylines": "true",
        "language": language,
        "pretty": "false",
    }

    response = requests.get(
        transit_url,
        params=params,
        headers={
            "User-Agent": "osm-route-transit-notebook/4.0",
            "Accept": "application/json",
        },
        timeout=90,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "Die öffentliche Transit-API meldet zu viele Anfragen. "
            "Bitte später erneut versuchen."
        )

    response.raise_for_status()
    payload = response.json()
    journeys = payload.get("journeys") if isinstance(payload, dict) else None

    if not journeys:
        raise RuntimeError(
            "Die öffentliche Transit-API hat keine Verbindung gefunden. "
            "Prüfe insbesondere Datum, Uhrzeit und die regionale Abdeckung."
        )

    journey = journeys[0]
    legs_out = []

    for leg in journey.get("legs", []):
        origin = leg.get("origin") or {}
        destination_leg = leg.get("destination") or {}

        departure = (
            leg.get("departure")
            or leg.get("plannedDeparture")
            or leg.get("scheduledDeparture")
        )
        arrival = (
            leg.get("arrival")
            or leg.get("plannedArrival")
            or leg.get("scheduledArrival")
        )

        if not departure or not arrival:
            continue

        line = leg.get("line") or {}
        walking = bool(leg.get("walking"))
        mode = "WALK" if walking else (
            str(line.get("mode") or line.get("product") or "TRANSIT").upper()
        )

        geometry = _transport_rest_geometry(
            leg.get("polyline"),
            origin,
            destination_leg,
        )

        legs_out.append({
            "mode": mode,
            "start_time": _parse_iso_datetime(departure, start_time.tzinfo),
            "end_time": _parse_iso_datetime(arrival, start_time.tzinfo),
            "distance_m": float(leg.get("distance") or 0),
            "geometry": geometry,
            "route_name": line.get("name") or "",
            "from_name": origin.get("name") or "",
            "to_name": destination_leg.get("name") or "",
        })

    if not legs_out:
        raise RuntimeError(
            "Die Transit-Antwort enthielt keine auswertbaren Teilstrecken."
        )

    return legs_out


def request_train_route_with_fallback(
    start,
    destination,
    start_time,
    otp_url,
    start_address,
    destination_address,
    language="de",
):
    """Nutzt OTP, falls erreichbar, sonst automatisch die öffentliche VBB-API."""
    otp_error = None

    if otp_url and otp_url.strip():
        try:
            return request_otp_route(
                start,
                destination,
                start_time,
                otp_url.strip(),
            ), "OpenTripPlanner"
        except Exception as error:
            otp_error = error

    try:
        return request_public_transit_route(
            start,
            destination,
            start_time,
            start_address,
            destination_address,
            language=language,
        ), "VBB transport.rest"
    except Exception as fallback_error:
        details = (
            f"OTP: {otp_error}. " if otp_error is not None else ""
        )
        raise RuntimeError(
            details
            + "Auch der öffentliche Transit-Fallback ist fehlgeschlagen: "
            + str(fallback_error)
        ) from fallback_error


def build_segment_table(coordinates):
    lengths, starts, cumulative = [], [], 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coordinates, coordinates[1:]):
        _, _, distance = GEOD.inv(lon1, lat1, lon2, lat2)
        starts.append(cumulative)
        lengths.append(max(distance, 0.0))
        cumulative += max(distance, 0.0)
    return lengths, starts, cumulative


def position_at_distance(coordinates, lengths, starts, distance_m):
    total = starts[-1] + lengths[-1]
    distance_m = min(max(distance_m, 0.0), total)
    if math.isclose(distance_m, total, abs_tol=1e-6):
        return coordinates[-1]
    for i, (segment_start, segment_length) in enumerate(zip(starts, lengths)):
        if segment_length > 0 and distance_m <= segment_start + segment_length:
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]
            azimuth, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
            lon, lat, _ = GEOD.fwd(lon1, lat1, azimuth, distance_m - segment_start)
            return lon, lat
    return coordinates[-1]


def generate_constant_speed_points(geometry, start_time, speed_kmh, interval_seconds):
    lengths, starts, total = build_segment_table(geometry)
    speed_mps = speed_kmh / 3.6
    total_seconds = total / speed_mps
    elapsed = [i * interval_seconds for i in range(math.floor(total_seconds / interval_seconds) + 1)]
    if not math.isclose(elapsed[-1], total_seconds, abs_tol=1e-9):
        elapsed.append(total_seconds)
    points = []
    for i, seconds in enumerate(elapsed):
        distance = min(speed_mps * seconds, total)
        lon, lat = position_at_distance(geometry, lengths, starts, distance)
        points.append(RoutePoint(i, start_time + timedelta(seconds=seconds), lon, lat, distance))
    return points, total


def generate_otp_points(legs, interval_seconds):
    """Erzeugt Punkte anhand der tatsächlichen Fahrplanzeiten der OTP-Legs."""
    if not legs:
        raise ValueError("OTP hat keine Teilstrecken geliefert.")
    start, end = legs[0]["start_time"], legs[-1]["end_time"]
    elapsed = [i * interval_seconds for i in range(math.floor((end-start).total_seconds()/interval_seconds)+1)]
    if elapsed[-1] != (end-start).total_seconds():
        elapsed.append((end-start).total_seconds())

    leg_tables, cumulative_distance = [], 0.0
    for leg in legs:
        lengths, starts, geom_len = build_segment_table(leg["geometry"])
        leg_tables.append((lengths, starts, geom_len, cumulative_distance))
        cumulative_distance += geom_len

    result = []
    for index, seconds in enumerate(elapsed):
        timestamp = start + timedelta(seconds=seconds)
        leg_index = next((i for i, leg in enumerate(legs) if leg["start_time"] <= timestamp <= leg["end_time"]), None)
        if leg_index is None:
            previous = [i for i, leg in enumerate(legs) if leg["end_time"] < timestamp]
            leg_index = previous[-1] if previous else 0
            leg = legs[leg_index]
            lon, lat = leg["geometry"][-1]
            distance = leg_tables[leg_index][3] + leg_tables[leg_index][2]
            mode = "WAIT"
        else:
            leg = legs[leg_index]
            lengths, starts, geom_len, distance_before = leg_tables[leg_index]
            duration = max((leg["end_time"] - leg["start_time"]).total_seconds(), 1)
            fraction = min(max((timestamp - leg["start_time"]).total_seconds() / duration, 0), 1)
            local_distance = geom_len * fraction
            lon, lat = position_at_distance(leg["geometry"], lengths, starts, local_distance)
            distance = distance_before + local_distance
            mode = leg["mode"]
        result.append(RoutePoint(index, timestamp, lon, lat, distance, mode))
    return result, cumulative_distance


def points_to_dataframe(points):
    return pd.DataFrame([{
        "point_index": p.index,
        "timestamp": p.timestamp.isoformat(timespec="seconds"),
        "longitude": p.longitude,
        "latitude": p.latitude,
        "distance_m": round(p.distance_m, 2),
        "mode": p.mode,
    } for p in points])


def create_route_map(geometries, points, start_label, destination_label):
    """Erzeugt eine Folium-Karte aus (Geometrie, Beschriftung)-Paaren."""
    if not geometries:
        raise ValueError("Es wurde keine Routengeometrie übergeben.")

    all_geometry = [
        coordinate
        for geometry, _label in geometries
        for coordinate in geometry
    ]

    if not all_geometry:
        raise ValueError("Die Routengeometrie enthält keine Koordinaten.")

    start_lon, start_lat = all_geometry[0]
    end_lon, end_lat = all_geometry[-1]

    route_map = folium.Map(
        location=[start_lat, start_lon],
        zoom_start=10,
        control_scale=True,
    )

    for geometry, label in geometries:
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in geometry],
            weight=5,
            opacity=0.8,
            tooltip=label,
        ).add_to(route_map)

    folium.Marker(
        location=[start_lat, start_lon],
        tooltip="Start",
        popup=f"<b>Start</b><br>{start_label}",
        icon=folium.Icon(icon="play"),
    ).add_to(route_map)

    folium.Marker(
        location=[end_lat, end_lon],
        tooltip="Ziel",
        popup=f"<b>Ziel</b><br>{destination_label}",
        icon=folium.Icon(icon="stop"),
    ).add_to(route_map)

    for p in points:
        folium.CircleMarker(
            location=[p.latitude, p.longitude],
            radius=4,
            weight=1,
            fill=True,
            fill_opacity=0.9,
            tooltip=f"{p.timestamp.strftime('%H:%M:%S')} · {p.mode or 'Route'}",
            popup=(
                f"<b>Punkt {p.index}</b><br>"
                f"Zeit: {p.timestamp.isoformat(timespec='seconds')}<br>"
                f"Modus: {p.mode or '-'}<br>"
                f"Distanz: {p.distance_m / 1000:.2f} km"
            ),
        ).add_to(route_map)

    route_map.fit_bounds(
        [[lat, lon] for lon, lat in all_geometry]
    )
    return route_map


class RouteNotebookApp:
    def __init__(self, defaults=None):
        self.defaults = defaults or RouteAppDefaults()
        self._build_widgets()
        self.run_button.on_click(self._run)

    def _build_widgets(self):
        d = self.defaults
        self.start = Text(value=d.start_address, description="Start:", layout=Layout(width="850px"))
        self.destination = Text(value=d.destination_address, description="Ziel:", layout=Layout(width="850px"))
        default_start_time = d.start_time or datetime.now().astimezone().replace(
            second=0, microsecond=0
        ).isoformat(timespec="minutes")
        self.start_time = Text(
            value=default_start_time,
            description="Startzeit:",
            layout=Layout(width="500px"),
        )
        self.speed = FloatText(value=d.average_speed_kmh, description="Ø km/h:")
        self.interval = IntText(value=d.interval_minutes, description="Intervall min:")
        self.profile = Dropdown(options=[("Auto","driving"),("Fahrrad","cycling"),("Zu Fuß","walking"),("Bahn/ÖPNV","train")],
                                value=d.profile, description="Profil:")
        self.language = Text(value=d.language, description="Sprache:")
        self.output_file = Text(value=d.output_file, description="CSV-Datei:", layout=Layout(width="500px"))
        self.otp_url = Text(value=d.otp_url, description="OTP-URL:", layout=Layout(width="750px"))
        self.run_button = Button(description="Route berechnen", button_style="success", icon="map")
        self.output = Output()
        self.container = VBox([self.start, self.destination, self.start_time, self.speed, self.interval,
                               self.profile, self.language, self.output_file, self.otp_url,
                               self.run_button, self.output])

    def show(self):
        display(self.container)
        return self

    def _run(self, _button=None):
        with self.output:
            clear_output()
            print("Berechnung gestartet ...")
            try:
                start_time = parse_start_time(self.start_time.value)
                print("Geocodiere Startadresse ...")
                start_lat, start_lon, start_label = geocode_address(self.start.value, self.language.value or "de")
                time.sleep(1)
                print("Geocodiere Zieladresse ...")
                dest_lat, dest_lon, dest_label = geocode_address(self.destination.value, self.language.value or "de")
                interval_seconds = int(self.interval.value) * 60

                if self.profile.value == "train":
                    print("Berechne Bahn-/ÖPNV-Verbindung ...")
                    legs, transit_engine = request_train_route_with_fallback(
                        (start_lat, start_lon),
                        (dest_lat, dest_lon),
                        start_time,
                        self.otp_url.value,
                        self.start.value,
                        self.destination.value,
                        language=self.language.value or "de",
                    )
                    print(f"Verwendete Transit-Engine: {transit_engine}")
                    if transit_engine != "OpenTripPlanner":
                        print(
                            "Lokales OTP war nicht erreichbar; "
                            "öffentlicher Transit-Fallback wird verwendet."
                        )
                    points, distance = generate_otp_points(
                        legs,
                        interval_seconds,
                    )
                    geometries = [
                        (
                            leg["geometry"],
                            f'{leg["mode"]} {leg["route_name"]}'.strip(),
                        )
                        for leg in legs
                    ]
                    duration = (
                        points[-1].timestamp - points[0].timestamp
                    ).total_seconds()
                    print(
                        "Die Bahnzeitpunkte verwenden Fahrplanzeiten; "
                        "Ø km/h wird ignoriert."
                    )
                else:
                    print("Berechne Straßenroute mit OSRM ...")
                    geometry, _, _ = request_osrm_route((start_lat,start_lon), (dest_lat,dest_lon), self.profile.value)
                    points, distance = generate_constant_speed_points(
                        geometry, start_time, float(self.speed.value), interval_seconds
                    )
                    geometries = [(geometry, "OSRM-Straßenroute")]
                    duration = (points[-1].timestamp - points[0].timestamp).total_seconds()

                dataframe = points_to_dataframe(points)
                output_path = Path(self.output_file.value or "route_positions.csv")
                dataframe.to_csv(output_path, index=False, encoding="utf-8")

                print(f"\nStart: {start_label}\nZiel: {dest_label}")
                print(f"Distanz der Geometrie: {distance/1000:.2f} km")
                print(f"Dauer: {duration/60:.1f} min")
                print(f"CSV-Punkte: {len(points)}")
                print(f"CSV: {output_path.resolve()}\n")
                display(dataframe.head(10))
                display(create_route_map(geometries, points, start_label, dest_label))
            except Exception as error:
                print(f"Fehler: {type(error).__name__}: {error}")


def launch_route_app(defaults=None):
    return RouteNotebookApp(defaults).show()
