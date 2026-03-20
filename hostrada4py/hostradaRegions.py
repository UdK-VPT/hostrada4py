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
}

# Variables for easier access
boitzenburgerland_polygon = REGIONS_POLYGONS["Gemeinde Boitzenburger Land"]
uckermark_polygon = REGIONS_POLYGONS["Landkreis Uckermark"]
