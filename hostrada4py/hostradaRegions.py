"""Predefined regional polygons for notebook compatibility."""

def _box(w,s,e,n): return [(w,s),(e,s),(e,n),(w,n),(w,s)]
REGIONS_POLYGONS = {
    "Germany": _box(5.5,47.0,15.6,55.5),
    "Berlin": _box(13.05,52.32,13.78,52.70),
    "Brandenburg": _box(11.2,51.3,14.8,53.6),
    "North Rhine-Westphalia": _box(5.8,50.3,9.5,52.6),
    "Bavaria": _box(8.9,47.2,13.9,50.6),
    "Baden-Württemberg": _box(7.4,47.5,10.5,49.8),
    "Lower Saxony": _box(6.5,51.2,11.6,54.0),
    "Saxony": _box(11.8,50.1,15.1,51.7),
    "Hamburg": _box(9.72,53.40,10.33,53.74),
}
germany_polygon=REGIONS_POLYGONS["Germany"]
berlin_polygon=REGIONS_POLYGONS["Berlin"]
brandenburg_polygon=REGIONS_POLYGONS["Brandenburg"]
bavaria_polygon=REGIONS_POLYGONS["Bavaria"]
baden_wuerttemberg_polygon=REGIONS_POLYGONS["Baden-Württemberg"]
north_rhine_westphalia_polygon=REGIONS_POLYGONS["North Rhine-Westphalia"]
lower_saxony_polygon=REGIONS_POLYGONS["Lower Saxony"]
saxony_polygon=REGIONS_POLYGONS["Saxony"]
hamburg_polygon=REGIONS_POLYGONS["Hamburg"]

# Historic regional selections used in the original notebook.
boitzenburgerland_polygon = _box(13.45, 53.15, 13.75, 53.38)
uckermark_polygon = _box(13.15, 52.95, 14.65, 53.55)
