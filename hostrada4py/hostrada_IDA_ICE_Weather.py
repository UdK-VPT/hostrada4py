"""IDA ICE PRN export."""
from hostrada4py import hostrada as hs
from ._weather_common import weather_dataframe,ensure_target,local_time,unique

def _unique_preserve_order(v):return unique(v)
def _required_vars(apply_weather_correction=False):
    required=["tas","tdew","hurs","ps","sfcWind","sfcWind_direction","rsds","clt"]
    if apply_weather_correction:required.append("uhi")
    available=hs.provider_capabilities().variables
    provider_required=[variable for variable in required if variable in available]
    return _unique_preserve_order(provider_required)
def create_ida_ice_weather_file(lon,lat,start,end,output_file="HOSTRADA_IDA_ICE.prn",*,tz="Europe/Berlin",altitude=0,
        apply_weather_correction=False,cache_dir="hostrada_cache",cache_strategy=None,subset_margin_cells=None,provider=None,verbose=True,**kwargs):
    df=weather_dataframe(lon,lat,start,end,_required_vars(apply_weather_correction),tz=tz,altitude=altitude,
        apply_weather_correction=apply_weather_correction,cache_dir=cache_dir,cache_strategy=cache_strategy,
        subset_margin_cells=subset_margin_cells,provider=provider,verbose=verbose)
    times=local_time(df,tz); out=df.copy(); out.insert(0,"Date",times.strftime("%Y-%m-%d"));out.insert(1,"Time",times.strftime("%H:%M"))
    cols=["Date","Time","tas","hurs","sfcWind","sfcWind_direction","rsds","dhi","dni","clt","ps"]
    path=ensure_target(output_file,"HOSTRADA_IDA_ICE.prn");out[[c for c in cols if c in out]].to_csv(path,sep="\t",index=False);return path
create_weather_file=create_ida_ice_weather_file
