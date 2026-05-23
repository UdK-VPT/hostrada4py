# hostrada4py
hostrada4Py is a Python library with access and for evaluation of the HOSTRADA weather data from the DWD (Deutscher Wetterdienst).

![Berlin_UHI_HOSTRADA](https://github.com/UdK-VPT/hostrada4py/blob/main/img/Berlin_UHI_HOSTRADA.png)

## Installation

```pip install requests xarray pandas typing geopandas pyproj shapely numpy matplotlib seaborn pvlib```

## The HOSTRADA project of DWD
The high-resolution hourly grid data set (HOSTRADA) for Germany published by DWD is a climatological reference data set. With a spatial resolution of one square kilometer and a temporal resolution of one hour, it provides a wide range of meteorological parameters for the land surfaces of the Federal Republic of Germany since 1995.

For most meteorological parameters, HOSTRADA is based on the interpolation of station data, taking into account satellite and climate model data to calculate a consistent data set. The intensity of the urban heat island (UHI) is included in the data set, allowing for improved representation of temperature distributions in regions with significant orographic features.

The hourly grid data for the HOSTRADA parameters and UHI intensity can be downloaded on a monthly basis from the Open Data section of the DWD Climate Data Center (https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/). It has a spatial resolution of 1 km x 1 km, covering a total area of 720 km x 938 km, and is available in NetCDF format in EPSG:3034 projection (ETRS89 / LCC Europe) and in UTC format.

The HOSTRADA data set is expanded monthly and contains the following variables:

* Cloud cover
* Wind speed and direction (at a height of 10 m)
* Air and dew point temperature (at a height of 2 m)
* Relative humidity (at a height of 2 m)
* Air pressure at station height and sea level
* Water vapor mixing ratio (at a height of 2 m)
* Global radiation and direct radiation
* Urban Heat Island Intensity (UHI)

## The Python library hostrada4py
The folder hostrada4py includes some Python files which simplifies the access to HOSTRADA weather data. hostrada.py contains functions for downloading the desired NetCFD files from the DWD server. The functions contained in hostradaPoint.py support the evaluation of weather data at a climate location specified by longitude and latitude, whereas hostradaArea.py provides functions that enable the evaluation of weather data in a grid with a resolution of 1km x 1km, which is defined by a polygon.

For the largest 50 German cities polygons are predefind in the file hostrada4py/hostradaCities.py.

## Notebooks
The notebooks hostradaPoint.ipynb, hostradaArea.ipynb, hostradaAreaMean.ipynb and hostradaPopint_diffRad.ipynb illustrate the use of the Python functions of hostrada4py.


## Diffuse horizontal irradiance (DHI)
The HOSTRADA dataset contains values for total radiation, but not for the diffuse radiation included in it.

A robust DHI estimate can be calculated directly from HOSTRADA point data with `pvlib` and the `erbs_driesse` decomposition model. The function `extract_diffuse_radiation_for_point` downloads the required HOSTRADA variables for a point, combines them, computes the solar position from time, longitude, and latitude, and returns `dhi`, `dni`, and `kd`.

```python
from hostrada4py import extract_diffuse_radiation_for_point

df = extract_diffuse_radiation_for_point(
    lon=13.405,
    lat=52.52,
    start="2024-01-01T00:00:00",
    end="2024-01-02T23:00:00",
    tz="Europe/Berlin",
)

print(df[["rsds", "dhi", "dni", "kd"]].head())
```

If additional HOSTRADA variables are available, a conservative weather-based correction can optionally be enabled with `apply_weather_correction=True`.
