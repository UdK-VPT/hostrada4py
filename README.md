# hostrada4py
hostrada4Py is a Python library with access and for evaluation of the HOSTRADA weather data from the DWD (Deutscher Wetterdienst).

![Berlin_UHI_HOSTRADA](https://github.com/UdK-VPT/hostrada4py/blob/main/img/Berlin_UHI_HOSTRADA.png)

## Installation

```bash
pip install numpy pandas requests xarray netcdf4 pyproj shapely geopandas branca folium ipython ipywidgets ipyleaflet matplotlib seaborn plotly pvlib fsspec h5netcdf
```

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
The folder hostrada4py includes some Python files which simplifies the access to HOSTRADA weather data. **hostrada.py** contains functions for downloading the desired NetCFD files from the DWD server. The functions contained in **hostradaPoint.py** support the evaluation of weather data at a climate location specified by longitude and latitude, whereas **hostradaArea.py** provides functions that enable the evaluation of weather data in a grid with a resolution of 1km x 1km, which is defined by a polygon.

For the largest 50 German cities and a couple of regions polygons are predefined in the files **hostrada4py/hostradaCities.py** and **hostrada4py/hostradaRegions.py**.

## Notebooks
The notebooks 

* **hostradaPoint.ipynb** (time series of weather data for one location),
* **hostradaArea.ipynb** (2D fields of weather data),
* **hostradaAreaMean.ipynb** (mean values of 2D fields of weather data),
* **hostradaPoint-diffRad.ipynb** (calculation of the diffuse radiation based of HOSTRADA values for one location)
* **hostradaHeatingDegreeDays.ipynb** (calculation of heating degree days based of HOSTRADA values for one location) and
* **hostradaGenerateWeatherFiles.ipynb** (generation of weather data files based of HOSTRADA values for one location for different simulation programs - IDA ICE, Polysun, EnergyPlus, SimStadt and the Modelica library BuildingSystems). 

illustrate the use of the Python functions of hostrada4py.


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

## Optional NetCDF subsetting cache

For long weather-file exports `hostrada4py` can reduce the amount of data
that is read and kept locally by creating spatial/temporal NetCDF subsets for a
point or polygon request. The default behaviour caches the full monthly HOSTRADA files.

Enable the subset mode from Python before starting an export:

```python
import os
os.environ["HOSTRADA_NETCDF_SUBSET_MODE"] = "subset"
os.environ["HOSTRADA_NETCDF_SUBSET_MARGIN_CELLS"] = "1"
```

or use it directly with the point/area helpers:

```python
from hostrada4py.hostradaPoint import extract_values_for_point

extract_values_for_point(
    "tas",
    lon=13.405,
    lat=52.52,
    start="2023-01-01 00:00",
    end="2023-01-31 23:00",
    cache_strategy="subset",
)
```

Available modes:

- `full`: original full-file download and cache (default)
- `subset`: full download if needed, then write and reuse a small local subset
- `http_range`: best-effort HTTP byte-range/chunked access, then local subset
- `auto`: try `http_range`, then fall back to `subset`

The DWD OpenData HOSTRADA endpoint is a static file service. A true server-side
NetCDF subset is only possible when the remote endpoint and local optional
packages support chunked/range access or an OPeNDAP/THREDDS/NCSS-like service.
Therefore `http_range` is intentionally optional and falls back to the robust
full download unless `HOSTRADA_NETCDF_SUBSET_FALLBACK=0` is set.

Optional packages for `http_range` mode:

```bash
pip install fsspec h5netcdf
```

If disk space is more important than keeping the original full cache, the full
monthly file can be removed after the subset file has been created:

```bash
export HOSTRADA_DROP_FULL_AFTER_SUBSET=1
```
