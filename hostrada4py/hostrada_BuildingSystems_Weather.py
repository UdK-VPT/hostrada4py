"""Modelica BuildingSystems weather table export."""
from ._weather_common import weather_dataframe,ensure_target

def create_buildingsystems_weather_file(lon,lat,start,end,output_file="HOSTRADA_BuildingSystems.mos",*,tz="UTC",altitude=0,
        apply_weather_correction=False,cache_dir="hostrada_cache",cache_strategy=None,subset_margin_cells=None,provider=None,verbose=True,**kwargs):
    required=["tas","tdew","hurs","ps","sfcWind","sfcWind_direction","rsds","clt"]
    df=weather_dataframe(lon,lat,start,end,required,tz=tz,altitude=altitude,apply_weather_correction=apply_weather_correction,
        cache_dir=cache_dir,cache_strategy=cache_strategy,subset_margin_cells=subset_margin_cells,provider=provider,verbose=verbose)
    out=df.copy();out.insert(0,"seconds",range(0,len(out)*3600,3600))
    path=ensure_target(output_file,"HOSTRADA_BuildingSystems.mos")
    with path.open("w",encoding="utf-8") as f:
        f.write("#1\n");f.write(f"double tab1({len(out)},{len(out.columns)})\n");out.to_csv(f,index=False,header=False)
    return path
create_weather_file=create_buildingsystems_weather_file

# Original notebook/API alias preserved.
create_buildingsystems_csv_weather_file = create_buildingsystems_weather_file
