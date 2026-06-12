#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rough bounding polygons for selected regions in Germany.

Format:
    [(longitude, latitude), ...]

Notes:
- The polygons are intentionally only rough outlines / bounding polygons.
- They are not intended to represent exact administrative boundaries.
- The points are chosen so that the respective city area is completely enclosed.
- Coordinate order is always: (lon, lat)
"""

REGIONS_POLYGONS = {
    "Gemeinde Boitzenburger Land": [
        (13.18, 53.38),
        (13.70, 53.39),
        (13.79, 53.26),
        (13.72, 53.07),
        (13.45, 53.00),
        (13.20, 53.09),
        (13.12, 53.24),
        (13.18, 53.38),
    ],
    "Landkreis Uckermark": [
        (12.92, 53.42),
        (13.30, 53.64),
        (14.02, 53.64),
        (14.55, 53.50),
        (14.56, 53.08),
        (14.40, 52.74),
        (13.90, 52.66),
        (13.08, 52.72),
        (12.88, 53.06),
        (12.92, 53.42),
    ],
    "Berlin und Potsdam": [
        (12.78, 52.53),
        (12.95, 52.70),
        (13.42, 52.76),
        (13.88, 52.66),
        (13.91, 52.40),
        (13.76, 52.20),
        (13.24, 52.16),
        (12.88, 52.23),
        (12.76, 52.39),
        (12.78, 52.53),
    ],

    # Berlin districts 2026 (rough bounding polygons)
    "Bezirk Mitte": [
        (13.30, 52.55),
        (13.31, 52.58),
        (13.37, 52.59),
        (13.44, 52.56),
        (13.46, 52.52),
        (13.44, 52.49),
        (13.38, 52.49),
        (13.33, 52.50),
        (13.30, 52.55),
    ],
    "Bezirk Friedrichshain-Kreuzberg": [
        (13.36, 52.50),
        (13.39, 52.53),
        (13.45, 52.54),
        (13.51, 52.53),
        (13.52, 52.50),
        (13.49, 52.47),
        (13.42, 52.47),
        (13.37, 52.48),
        (13.36, 52.50),
    ],
    "Bezirk Pankow": [
        (13.35, 52.54),
        (13.39, 52.61),
        (13.39, 52.67),
        (13.50, 52.69),
        (13.59, 52.66),
        (13.58, 52.58),
        (13.54, 52.52),
        (13.46, 52.52),
        (13.43, 52.53),
        (13.35, 52.54),
    ],
    "Bezirk Charlottenburg-Wilmersdorf": [
        (13.17, 52.53),
        (13.20, 52.56),
        (13.29, 52.56),
        (13.34, 52.53),
        (13.34, 52.47),
        (13.31, 52.44),
        (13.23, 52.43),
        (13.18, 52.47),
        (13.17, 52.53),
    ],
    "Bezirk Spandau": [
        (13.07, 52.57),
        (13.09, 52.64),
        (13.18, 52.64),
        (13.25, 52.58),
        (13.25, 52.50),
        (13.20, 52.43),
        (13.11, 52.43),
        (13.07, 52.50),
        (13.07, 52.57),
    ],
    "Bezirk Steglitz-Zehlendorf": [
        (13.08, 52.45),
        (13.17, 52.50),
        (13.30, 52.49),
        (13.36, 52.46),
        (13.35, 52.40),
        (13.28, 52.34),
        (13.16, 52.34),
        (13.09, 52.38),
        (13.08, 52.45),
    ],
    "Bezirk Tempelhof-Schöneberg": [
        (13.30, 52.49),
        (13.36, 52.51),
        (13.44, 52.50),
        (13.48, 52.46),
        (13.47, 52.40),
        (13.41, 52.39),
        (13.34, 52.41),
        (13.30, 52.45),
        (13.30, 52.49),
    ],
    "Bezirk Neukölln": [
        (13.39, 52.49),
        (13.46, 52.50),
        (13.53, 52.48),
        (13.56, 52.43),
        (13.53, 52.38),
        (13.45, 52.37),
        (13.39, 52.40),
        (13.37, 52.45),
        (13.39, 52.49),
    ],
    "Bezirk Treptow-Köpenick": [
        (13.44, 52.49),
        (13.53, 52.51),
        (13.66, 52.50),
        (13.76, 52.46),
        (13.78, 52.38),
        (13.70, 52.32),
        (13.55, 52.32),
        (13.45, 52.36),
        (13.40, 52.43),
        (13.44, 52.49),
    ],
    "Bezirk Marzahn-Hellersdorf": [
        (13.48, 52.58),
        (13.55, 52.62),
        (13.66, 52.61),
        (13.70, 52.55),
        (13.65, 52.49),
        (13.56, 52.49),
        (13.50, 52.52),
        (13.48, 52.58),
    ],
    "Bezirk Lichtenberg": [
        (13.43, 52.55),
        (13.49, 52.59),
        (13.56, 52.58),
        (13.61, 52.53),
        (13.58, 52.48),
        (13.51, 52.47),
        (13.45, 52.49),
        (13.43, 52.55),
    ],
    "Bezirk Reinickendorf": [
        (13.18, 52.62),
        (13.22, 52.68),
        (13.34, 52.67),
        (13.41, 52.62),
        (13.40, 52.56),
        (13.34, 52.53),
        (13.24, 52.54),
        (13.18, 52.58),
        (13.18, 52.62),
    ],
}

# Variables for easier access
boitzenburgerland_polygon = REGIONS_POLYGONS["Gemeinde Boitzenburger Land"]
uckermark_polygon = REGIONS_POLYGONS["Landkreis Uckermark"]
berlin_potsdam_polygon = REGIONS_POLYGONS["Berlin und Potsdam"]

# Berlin districts, 2026
berlin_mitte_polygon = REGIONS_POLYGONS["Bezirk Mitte"]
berlin_friedrichshain_kreuzberg_polygon = REGIONS_POLYGONS["Bezirk Friedrichshain-Kreuzberg"]
berlin_pankow_polygon = REGIONS_POLYGONS["Bezirk Pankow"]
berlin_charlottenburg_wilmersdorf_polygon = REGIONS_POLYGONS["Bezirk Charlottenburg-Wilmersdorf"]
berlin_spandau_polygon = REGIONS_POLYGONS["Bezirk Spandau"]
berlin_steglitz_zehlendorf_polygon = REGIONS_POLYGONS["Bezirk Steglitz-Zehlendorf"]
berlin_tempelhof_schoeneberg_polygon = REGIONS_POLYGONS["Bezirk Tempelhof-Schöneberg"]
berlin_neukoelln_polygon = REGIONS_POLYGONS["Bezirk Neukölln"]
berlin_treptow_koepenick_polygon = REGIONS_POLYGONS["Bezirk Treptow-Köpenick"]
berlin_marzahn_hellersdorf_polygon = REGIONS_POLYGONS["Bezirk Marzahn-Hellersdorf"]
berlin_lichtenberg_polygon = REGIONS_POLYGONS["Bezirk Lichtenberg"]
berlin_reinickendorf_polygon = REGIONS_POLYGONS["Bezirk Reinickendorf"]

berlin_districts_2026_polygons = {
    name: REGIONS_POLYGONS[name]
    for name in [
        "Bezirk Mitte",
        "Bezirk Friedrichshain-Kreuzberg",
        "Bezirk Pankow",
        "Bezirk Charlottenburg-Wilmersdorf",
        "Bezirk Spandau",
        "Bezirk Steglitz-Zehlendorf",
        "Bezirk Tempelhof-Schöneberg",
        "Bezirk Neukölln",
        "Bezirk Treptow-Köpenick",
        "Bezirk Marzahn-Hellersdorf",
        "Bezirk Lichtenberg",
        "Bezirk Reinickendorf",
    ]
}

