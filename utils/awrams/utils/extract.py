
### SpatialAggregateCellProcessor uses cell area inside shape to perform aggregation
### osgeo.ogr is required for cell area calculations

from awrams.utils.ts.processing import *
from awrams.utils.extents import from_multiple

import os

def extract(var_name,dataset,extent_map,period,agg_method='mean'):

    cmap = CellMapProcessor()

    collectors = {}
    for name, extent in list(extent_map.items()):
        if extent.cell_count == 1:
            collector = TimeSeriesCollector(period)
            target = collector
        else:
            agg = SpatialAggregateCellProcessor(period,extent,mode=agg_method)
            collector = TimeSeriesCollector(period)
            target = ProcessChain([agg,collector])

        cmap.add_target(extent,target)
        collectors[name] = collector

    processors = {var_name: [cmap]}

    e = from_multiple(extent_map)

    prunner = ProcessRunner({var_name: dataset},processors,period,e)

    prunner.run()


    _d = {}
    for k,v in list(collectors.items()):
        _d[str(k)] = v.data
    df = pd.DataFrame.from_dict(_d)

    return df

def localise_extent_to_ncfile(extent,ncfile):
    from awrams.utils.io.netcdf_wrapper import geospatial_reference_from_nc
    import netCDF4 as nc

#     h5py_cleanup_nc_mess(show_log=True)
    with nc.Dataset(ncfile) as ncd:
        gr = geospatial_reference_from_nc(ncd)

    return extent.translate_to_origin(gr)
