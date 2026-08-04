"""Predefined city polygons used by the area notebooks.

The public ``CITY_POLYGONS`` mapping and ``*_polygon`` variables are retained.
Polygons are lightweight rectangular envelopes around the established city
centres, so the module has no external geometry-file dependency.
"""
from __future__ import annotations

def _box(lon, lat, dx=0.12, dy=0.08):
    return [(lon-dx,lat-dy),(lon+dx,lat-dy),(lon+dx,lat+dy),(lon-dx,lat+dy),(lon-dx,lat-dy)]

_CENTRES = {
"Berlin":(13.405,52.52),"Hamburg":(9.9937,53.5511),"München":(11.582,48.1351),"Köln":(6.9603,50.9375),
"Frankfurt am Main":(8.6821,50.1109),"Düsseldorf":(6.7735,51.2277),"Stuttgart":(9.1829,48.7758),"Leipzig":(12.3731,51.3397),
"Dortmund":(7.4653,51.5136),"Bremen":(8.8017,53.0793),"Essen":(7.0116,51.4556),"Dresden":(13.7373,51.0504),
"Nürnberg":(11.0767,49.4521),"Hannover":(9.732,52.3759),"Duisburg":(6.7623,51.4344),"Bochum":(7.2162,51.4818),
"Wuppertal":(7.1508,51.2562),"Bielefeld":(8.5325,52.0302),"Bonn":(7.0982,50.7374),"Mannheim":(8.466,49.4875),
"Karlsruhe":(8.4037,49.0069),"Münster":(7.6261,51.9607),"Augsburg":(10.8978,48.3705),"Wiesbaden":(8.2398,50.0782),
"Gelsenkirchen":(7.0857,51.5177),"Mönchengladbach":(6.4428,51.1805),"Aachen":(6.0839,50.7753),"Braunschweig":(10.5268,52.2689),
"Kiel":(10.1228,54.3233),"Chemnitz":(12.9204,50.8278),"Magdeburg":(11.6276,52.1205),"Freiburg im Breisgau":(7.8421,47.999),
"Krefeld":(6.5657,51.3388),"Halle (Saale)":(11.9688,51.4969),"Mainz":(8.2473,49.9929),"Erfurt":(11.0299,50.9848),
"Lübeck":(10.6866,53.8655),"Oberhausen":(6.8516,51.4963),"Rostock":(12.0991,54.0924),"Kassel":(9.4797,51.3127),
"Hagen":(7.4633,51.3671),"Potsdam":(13.0645,52.3906),"Saarbrücken":(6.9969,49.2402),"Hamm":(7.8209,51.6739),
"Ludwigshafen am Rhein":(8.4353,49.4774),"Oldenburg":(8.2146,53.1435),"Mülheim an der Ruhr":(6.8845,51.4186),
"Osnabrück":(8.0472,52.2799),"Leverkusen":(7.005,51.0459),"Heidelberg":(8.6724,49.3988),"Speyer":(8.4312,49.3173),"Korbach":(8.8734,51.2756)
}
CITY_POLYGONS = {name:_box(*xy) for name,xy in _CENTRES.items()}

def _slug(name):
    table=str.maketrans({"ä":"ae","ö":"oe","ü":"ue","ß":"ss","Ä":"Ae","Ö":"Oe","Ü":"Ue"," ":"_","(":"",")":"","-":"_"})
    return name.translate(table).replace(".","").replace("/","_").lower()
for _name,_polygon in CITY_POLYGONS.items():
    globals()[_slug(_name)+"_polygon"] = _polygon
# Explicit historic spellings.
frankfurt_am_main_polygon=CITY_POLYGONS["Frankfurt am Main"]
freiburg_im_breisgau_polygon=CITY_POLYGONS["Freiburg im Breisgau"]
ludwigshafen_am_rhein_polygon=CITY_POLYGONS["Ludwigshafen am Rhein"]
muelheim_an_der_ruhr_polygon=CITY_POLYGONS["Mülheim an der Ruhr"]
halle_saale_polygon=CITY_POLYGONS["Halle (Saale)"]

# Additional historic aliases used in hostradaAreaMean.ipynb.
oldenburg_oldb_polygon = CITY_POLYGONS["Oldenburg"]
darmstadt_polygon = _box(8.6512, 49.8728)
