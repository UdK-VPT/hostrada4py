"""Polysun weather export."""
from hostrada4py import hostrada as hs
from ._weather_common import weather_dataframe,ensure_target,local_time,unique

def _unique_preserve_order(v):return unique(v)
def _required_vars(apply_weather_correction=False):
    required=["tas","hurs","sfcWind","sfcWind_direction","rsds","clt"]
    if apply_weather_correction:required.append("uhi")
    available=hs.provider_capabilities().variables
    provider_required=[variable for variable in required if variable in available]
    return _unique_preserve_order(provider_required)
def create_polysun_weather_file(lon,lat,start,end,output_file="HOSTRADA_Polysun.csv",*,tz="Europe/Berlin",altitude=0,
        apply_weather_correction=False,cache_dir="hostrada_cache",cache_strategy=None,subset_margin_cells=None,provider=None,verbose=True,**kwargs):
    df=weather_dataframe(lon,lat,start,end,_required_vars(apply_weather_correction),tz=tz,altitude=altitude,
        apply_weather_correction=apply_weather_correction,cache_dir=cache_dir,cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,provider=provider,verbose=verbose)
    times=local_time(df,tz);out=df.copy();out.insert(0,"DateTime",times.strftime("%d.%m.%Y %H:%M"))
    path=ensure_target(output_file,"HOSTRADA_Polysun.csv");out.to_csv(path,sep=";",index=False);return path
create_weather_file=create_polysun_weather_file
