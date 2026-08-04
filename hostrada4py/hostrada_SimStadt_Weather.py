"""SimStadt/TMY3-style weather export."""
from ._weather_common import weather_dataframe,ensure_target,local_time

def create_simstadt_weather_file(lon,lat,start,end,output_file="HOSTRADA_SimStadt.tmy3",*,tz="Europe/Berlin",altitude=0,
        apply_weather_correction=False,cache_dir="hostrada_cache",cache_strategy=None,subset_margin_cells=None,provider=None,verbose=True,**kwargs):
    required=["tas","tdew","hurs","ps","sfcWind","sfcWind_direction","rsds","clt"]
    df=weather_dataframe(lon,lat,start,end,required,tz=tz,altitude=altitude,apply_weather_correction=apply_weather_correction,
        cache_dir=cache_dir,cache_strategy=cache_strategy,subset_margin_cells=subset_margin_cells,provider=provider,verbose=verbose)
    times=local_time(df,tz);out=df.copy();out.insert(0,"Date (MM/DD/YYYY)",times.strftime("%m/%d/%Y"));out.insert(1,"Time (HH:MM)",times.strftime("%H:%M"))
    path=ensure_target(output_file,"HOSTRADA_SimStadt.tmy3");out.to_csv(path,index=False);return path
create_weather_file=create_simstadt_weather_file
