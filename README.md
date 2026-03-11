# hostrada4py
hostrada4Py is a Python library with access and for evaluation of the HOSTRADA weather data from the DWD (Deutscher Wetterdienst).

![Berlin_UHI_HOSTRADA](https://github.com/UdK-VPT/hostrada4py/blob/main/img/Berlin_UHI_HOSTRADA.png)

## Installation

```pip install calendar requests pathlib xarray pandas typing geopandas pyproj shapely numpy```

## The HOSTRADA project of DWD
The high-resolution hourly grid data set (HOSTRADA) for Germany published by DWD is a climatological reference data set. With a spatial resolution of one square kilometer and a temporal resolution of one hour, it provides a wide range of meteorological parameters for the land surfaces of the Federal Republic of Germany since 1995.

For most meteorological parameters, HOSTRADA is based on the interpolation of station data, taking into account satellite and climate model data to calculate a consistent data set. The intensity of the urban heat island (UHI) is included in the data set, allowing for improved representation of temperature distributions in regions with significant orographic features.

The hourly grid data for the HOSTRADA parameters and UHI intensity can be downloaded on a monthly basis from the Open Data section of the DWD Climate Data Center (https://opendata.dwd.de/climate_environment/CDC/grids_germany/hourly/hostrada/). It has a spatial resolution of 1 km x 1 km, covering a total area of 720 km x 938 km, and is available in NetCDF format in EPSG:3034 projection (ETRS89 / LCC Europe) and in UTC format.

The HOSTRADA data set is expanded monthly and contains the following variables:

* Cloud cover
* Wind speed and direction (at a height of 10 m)
* Air and dew point temperature (at a height of 2 m)
* Absolute and relative humidity (at a height of 2 m)
* Air pressure at station height and sea level
* Global radiation and direct radiation
* Incoming and outgoing terrestrial heat radiation

## The Python library hostrada4py
The folder hostrada4py includes some Python files which simplifies the access to HOSTRADA weather data. hostrada.py contains functions for downloading the desired NetCFD files from the DWD server. The functions contained in hostradaPoint.py support the evaluation of weather data at a climate location specified by longitude and latitude, whereas hostradaArea.py provides functions that enable the evaluation of weather data in a grid with a resolution of 1km x 1km, which is defined by a polygon. 

## Notebooks
The notebooks hostradaPoint.ipynb and hostradaArea.ipynb illustrate the use of the Python functions of hostrada4py.
