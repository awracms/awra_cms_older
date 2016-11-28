'''
Catchments
==========

Provides functionality for relating catchment polygons to the gridded spatial domain, including

* Running the simulation over one or more catchments
* Interrogating results at the catchment level

+++ Polygons more generally?

'''
# from osgeo import ogr, osr
from .metatypes import ObjectDict, AutoCallSelector, pretty_null
from .extents import bounds_ref, Extent, from_boundary_coords, from_boundary_offset, from_cell_offset, CELLSIZE, _LONGLAT_TO_AEA, _LONGLAT_PROJ,build_transform
#from .utils import norm_name

import numpy as np
import pandas as pd
import copy

from .helpers import quantize, aquantize
from .settings import CATCHMENT_SHAPEFILE #pylint: disable=no-name-in-module


class LocationSelector(object):
    def __init__(self,target):
        self._target = target
        self._build_names()

    def by_id(self,id):
        '''
        Select a location by id number
        '''
        self._target._set(self._source.get_by_id(id),self._mode)

    def __repr__(self):
        return "Select using %s.by_name or %s.by_id"%(self._mode,self._mode)

    def _build_names(self):
        '''
        build our 'by name' table
        '''
        from functools import partial

        name_d = AutoCallSelector()
        name_d.__doc__ = 'Select a %s from the tab completable name list'%self._mode
        for _ in self._source.records:
            full_name = self._source._name_for(_)
            def get_c(record):
                # +++ Should really be calling a function on the target
                # that logs the catchment change
                bounds = self._source.extent_from_record(record)
                self._target._set(bounds,self._mode)
                return pretty_null
            name_d[full_name] = partial(get_c,record=_)

        self.by_name = name_d


def norm_name(name):
    import re
    return re.sub('[^a-zA-Z0-9_]','',name.replace('@','_at_'))


def catchment_from_record(record,namer,idder):
    geometry = record.GetGeometryRef()
    sh_bounds = geometry.GetEnvelope()

    min_lon = quantize(sh_bounds[0]+0.025,CELLSIZE,np.floor)
    min_lat = quantize(sh_bounds[2]+0.025,CELLSIZE,np.floor)
    max_lon = quantize(sh_bounds[1]+0.025,CELLSIZE,np.floor)
    max_lat = quantize(sh_bounds[3]+0.025,CELLSIZE,np.floor)

    bounds_r = bounds_ref(min_lat,min_lon,max_lat,max_lon)

    name = namer(record)
    rec_id = idder(record)

    return Catchment(name,rec_id,bounds_r,None,geometry)


class Catchment(Extent):
    def __init__(self,name,record_id,bounds_ref,parent_ref=None,geometry=None,compute_areas=True):
        '''
        Create an AWRA Extents object from the shapefile record supplied
        '''

        self.name = name
        self.id = record_id #int(record['StationID'])

        super(Catchment,self).__init__(parent_ref=parent_ref)
        ### turn instance into a GeoBounds type extent
        from_boundary_coords(*(bounds_ref.bounds_args()),extent=self)

        if geometry is not None:
            self.compute_masks(geometry, isect_only=not compute_areas)

    def translate_to_origin(self,new_origin):
        # Mask calculation is really expensive!
        # Avoid recalculation during translation
        t_ext = Catchment(self.name,self.id,self.geospatial_reference(),new_origin)
        t_ext.mask = self.mask
        t_ext.areas = self.areas
        t_ext.cell_count = self.cell_count
        t_ext.area = self.area
        return t_ext

    def compute_masks(self,geometry,trans_isect=False,isect_only=False):
        '''
        Compute both the binary cell mask and the area mask
        (ie the area of each cell in m2 actually covered by the catchment)

        '''

        self.mask = self.mask.copy()

        areas = np.zeros(self.shape)
        self.areas = np.ma.MaskedArray(data=areas,mask=self.mask)

        for cell in self:
            c = from_cell_offset(cell[0],cell[1],parent_ref=self.parent_ref)
            cpoly = c.to_polygon()
            lcell = self.localise_cell(cell)
            if (trans_isect):
                t_geom = geometry.Clone()
                t_geom.Transform(_LONGLAT_TO_AEA)
                cpoly.Transform(_LONGLAT_TO_AEA)
                if cpoly.Intersects(t_geom):
                    if not isect_only:
                        isect = cpoly.Intersection(t_geom)
                        self.areas[lcell[0],lcell[1]] = isect.Area()
                else:
                    self.mask[lcell[0],lcell[1]] = True
            else:
                if cpoly.Intersects(geometry):
                    if not isect_only:
                        isect = cpoly.Intersection(geometry)
                        isect.Transform(_LONGLAT_TO_AEA)
                        self.areas[lcell[0],lcell[1]] = isect.Area()
                else:
                    self.mask[lcell[0],lcell[1]] = True

        self.areas.mask = self.mask
        self.cell_count = (self.mask == False).sum()
        self.area = self.areas.sum()

    def to_dict(self):
        return {'type': 'catchment', 'id': self.id}

    def __repr__(self):
        return "%s: %s" % (self.id, self.name)


class CatchmentNotFoundError(Exception):
    pass

class CatchmentDB(object):
    key_field = 'StationID'
    name_fields = ['GaugeName', 'RiverName']
    name_format = '%s (%s)'

    def __init__(self,shp_file=CATCHMENT_SHAPEFILE):
        '''
        Create a catchment database from the specified shapefile
        '''
        from osgeo import ogr
        ds = ogr.Open(shp_file)

        if ds is None:
            raise Exception("Can't open shapefile from %s"%shp_file)

        layer = ds.GetLayer()

        self.records = []

        for i in range(0,layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            self.records.append(feature)
        self.key_list = feature.keys()
        self._build_names()

    def _name_for(self,record):
        return map_catchment_name(record)

    def _build_names(self):
        from functools import partial

        name_d = ObjectDict()
        for c in self.records:
            full_name = self._name_for(c)
            #if not name_d.has_key(r_name):
            #    name_d[r_name] = ObjectDict()
            def get_c(record):
                return self.catchment_from_record(record)
            name_d[full_name] = partial(get_c,record=c)

        self.by_name = name_d

    def field_values(self,field):
        '''
        Return a list of values for a particular field, for all catchments
        '''
        values = []
        for record in self.records:
            values.append(record[field])
        return values

    def get_by_id(self,id_to_find):
        '''
        Return a catchment by it's id number
        '''
        id_to_find = int(id_to_find)
        for record in self.records:
            if int(record[self.key_field]) == id_to_find:
                return self.catchment_from_record(record)
        raise CatchmentNotFoundError(id_to_find)

    def list(self):
        '''
        List available catchments
        '''
        out = {}
        for record in self.records:
            name = self.name_format % tuple([record[name] for name in self.name_fields]) #record['GaugeName'], record['RiverName'])
            out[int(record['StationID'])] = name
        return out

    def catchment_from_record(self,record):
        return catchment_from_record(record,
                                     namer=lambda x:'%s (%s)' % tuple([x[name] for name in self.name_fields]), #x['GaugeName'], x['RiverName']),
                                     idder=lambda x:int(x[self.key_field]))

    def extent_from_record(self,record):
        return self.catchment_from_record(record)

class SubCatchmentDB(CatchmentDB):
    key_field = 'ID_updated'
    def __init__(self, shp_file):
        super(SubCatchmentDB, self).__init__(shp_file)
        # self.key_field = 'ID_updated'
        self.id_list = [str(int(c[self.key_field])) for c in self.records]

    def _name_for(self,record):
        return 'c'+str(int(record[self.key_field]))

    def catchment_from_record(self,record):
        return catchment_from_record(record,namer=lambda x:self._name_for(x),idder=lambda x:int(x[self.key_field]))

class CatchmentSelector(LocationSelector):
    '''
    Select a catchment by name or id
    '''
    def __init__(self,source, target):
        self._mode = 'catchment'
        self._source = source
        super(CatchmentSelector,self).__init__(target)

    def __repr__(self):
        return "Select using catchment.by_name or catchment.by_id"

class DBSelector(object):
    '''
    Select a catchment by name or id
    '''
    def __init__(self,db, target):
        self._db = db
        self._target = target
        #self.by_id = catchment_db.by_id
        self.get_by = ObjectDict()
        self._build_selectors()


    def _build_selectors(self):
        '''
        build our 'by name' table
        '''
        from functools import partial

        def get_and_set(func):
            self._target._set(func(),'catchment')

        for field, subdict in list(self._db.get_by.items()):
            getter = ObjectDict()
            self.get_by[field] = getter

            for k,v in list(subdict.items()):
                getter[k] = partial(get_and_set,v)

class ShapefileDB(object):
    def __init__(self,shp_file=CATCHMENT_SHAPEFILE):
        '''
        Create a catchment database from the specified shapefile
        '''
        from osgeo import ogr
        ds = ogr.Open(shp_file)

        if ds is None:
            raise Exception("Can't open shapefile from %s"%shp_file)

        layer = ds.GetLayer()

        self._records = []

        for i in range(0,layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            self._records.append(feature)

        self.get_by = ObjectDict()

        self._name_builder = None

    def _build_getter(self,getter_name,field_mapper):
        from functools import partial

        name_d = ObjectDict()
        for c in self._records:
            name = field_mapper(c)
            def get_c(record):
                '''
                Return the selected item
                '''
                return extent_from_record(record,self._name_builder)
            name_d[name] = partial(get_c,record=c)
        self.get_by[getter_name] = name_d

    def _get_records_df(self):
        import pandas as pd
        records = ([r.items() for r in self._records])
        return pd.DataFrame(records)

    def _get_by_field(self,field,value,as_extent=True,first_only=False):
        if not first_only:
            out = [] #+++ Multiple return logic..?
        for r in self._records:
            if r[field] == value:
                if as_extent:
                    return extent_from_record(r,self._name_builder)
                else:
                    return r

class Geometry:
    '''
    Thin wrapper for OGR geometry featuring easy area translations
    '''
    def __init__(self,ogr_geometry):
        self.geo = ogr_geometry

    @property
    def area(self):
        '''
        Return the area in meters square
        '''
        g = self.geo.Clone()
        g.Transform(_LONGLAT_TO_AEA)
        return g.Area()

class GeoGrid:
    def __init__(self,lats,lons,mode='center'):
        '''
        Container for rectangular lat/lon grids
        '''
        self.lats = lats
        self.lons = lons
        self.mode = mode

    def to_corners(self):
        '''
        Best effort resampling of irregular grids midpoints
        to shared corner grid polygons
        '''
        def gen_corners(series):
            step = (series[-1]-series[0])/(len(series)-1)
            midpts = (series[:-1]+series[1:])*0.5
            out = [series[0]-step*0.5] + list(midpts) + [series[-1]+step*0.5]
            return np.array(out)

        lats = gen_corners(self.lats)
        lons = gen_corners(self.lons)
        return GeoGrid(lats,lons,'corner')

def polygon_from_cell(lats,lons):
    from osgeo import ogr
    poly = ogr.Geometry(ogr.wkbLinearRing)
    poly.AddPoint(lons[0],lats[0])
    poly.AddPoint(lons[0],lats[1])
    poly.AddPoint(lons[1],lats[1])
    poly.AddPoint(lons[1],lats[0])
    poly.AddPoint(lons[0],lats[0])
    c_poly = ogr.Geometry(ogr.wkbPolygon)
    c_poly.AddGeometry(poly)
    return c_poly

def intersect_geogrid(geogrid,geometry):
    '''
    Return the intersection data for a GeoGrid and an item of OGR geometry
    '''

    sref = geometry.GetSpatialReference()
    p4_rep = sref.ExportToProj4()
    LL_TRANSFORM = build_transform(p4_rep,_LONGLAT_PROJ)

    ll_geo = geometry.Clone()
    ll_geo.Transform(LL_TRANSFORM)

    # Minimum enclosing envelope
    env = ll_geo.GetEnvelope()

    ggc = geogrid.to_corners()

    lidx = (env[0] >= ggc.lons).sum()-1
    ridx = (env[1] > ggc.lons).sum()-1
    tidx = (env[3] <= ggc.lats).sum()-1
    bidx = (env[2] < ggc.lats).sum()-1

    #dsm.variables.aet[0,tidx:bidx+1,lidx:ridx+1]

    shape = ((bidx-tidx)+1,(ridx-lidx)+1)

    areas = np.zeros(shape)

    #+++
    # This can be slow for large polygons; needs refactor
    # to use subdivision ala compute_areas; probably
    # this code should be shared and less tied to AWRAL extents

    for i, lat_i in enumerate(range(tidx,bidx+1)):
        for j, lon_i in enumerate(range(lidx,ridx+1)):
            lats = ggc.lats[lat_i:lat_i+2]
            lons = ggc.lons[lon_i:lon_i+2]
            cpoly = polygon_from_cell(lats,lons)
            if cpoly.Intersects(ll_geo):
                isect = cpoly.Intersection(ll_geo)
                isect.Transform(_LONGLAT_TO_AEA)
                areas[i,j] = isect.Area()
            else:
                areas[i,j] = 0.0

    out = {}
    out['areas'] = areas
    out['lat_idx'] = slice(tidx,bidx+1)
    out['lon_idx'] = slice(lidx,ridx+1)
    out['grid'] = GeoGrid(geogrid.lats[out['lat_idx']],geogrid.lons[out['lon_idx']])
    return ObjectDict(out)



def extent_from_record(record,name_builder=None,intersection_only=False):
    geometry = record.GetGeometryRef()

    sref = geometry.GetSpatialReference()
    p4_rep = sref.ExportToProj4()

    #AEA_TRANSFORM = build_transform(p4_rep,AEA_AUS_PROJ)
    LL_TRANSFORM = build_transform(p4_rep,_LONGLAT_PROJ)

    ll_geo = geometry.Clone()
    ll_geo.Transform(LL_TRANSFORM)

    sh_bounds = ll_geo.GetEnvelope()

    min_lon = quantize(sh_bounds[0]+0.025,CELLSIZE,np.floor)
    min_lat = quantize(sh_bounds[2]+0.025,CELLSIZE,np.floor)
    max_lon = quantize(sh_bounds[1]+0.025,CELLSIZE,np.floor)
    max_lat = quantize(sh_bounds[3]+0.025,CELLSIZE,np.floor)

    bounds_r = bounds_ref(min_lat,min_lon,max_lat,max_lon)

    #name = '%s (%s)' % (record['GaugeName'], record['RiverName'])
    #rec_id = int(record['StationID'])
    meta = {}
    for k,v in record.items().items():
        meta[k] = v

    e = from_boundary_coords(*bounds_r.bounds_args(),compute_areas=False)

    areas = compute_areas(e,ll_geo,intersection_only)

    e.areas = areas
    e.area = areas.sum()
    e.mask = areas.mask
    e.cell_count = (areas.mask == False).sum()
    e.meta = meta
    if name_builder is not None:
        e.display_name = name_builder(record)

    return e

def compute_areas(bounds,geometry,isect_only=False):
    '''
    Given a from_boundary_offset and an item of ogr geometry, calculate the areas
    of intersecting cells
    '''

    # +++
    # Still questionable regarding where to perform intersection
    # if inputs have different coordinate systems...
    area_data = np.zeros(bounds.shape)
    areas = np.ma.MaskedArray(data=area_data,mask=bounds.mask.copy())

    subex = subdivide_extent(bounds)

    for e in subex:
        # poly_env = e.poly_envelope()
        poly_env = e.to_polygon()
        if geometry.Contains(poly_env):
            for cell in e:
                # cpoly = cell.to_polygon()
                cpoly = from_cell_offset(cell[0],cell[1],parent_ref=e.parent_ref).to_polygon()
                lcell = bounds.localise_cell(cell)

                if isect_only:
                    areas[lcell[0],lcell[1]] = 1
                else:
                    cpoly.Transform(_LONGLAT_TO_AEA)
                    areas[lcell[0],lcell[1]] = cpoly.Area()
        else:
            l_geom = geometry.Intersection(poly_env)
            for cell in e:
                # cpoly = cell.to_polygon()
                cpoly = from_cell_offset(cell[0],cell[1],parent_ref=e.parent_ref).to_polygon()
                lcell = bounds.localise_cell(cell)

                if cpoly.Intersects(l_geom):
                    if isect_only:
                        areas[lcell[0],lcell[1]] = 1
                    else:
                        if l_geom.Contains(cpoly):
                            cpoly.Transform(_LONGLAT_TO_AEA)
                            areas[lcell[0],lcell[1]] = cpoly.Area()
                        else:
                            isect = cpoly.Intersection(l_geom)
                            isect.Transform(_LONGLAT_TO_AEA)
                            areas[lcell[0],lcell[1]] = isect.Area()
                else:
                    areas.mask[lcell[0],lcell[1]] = True

    return areas

def subdivide_extent(extent,max_cells=None):

    #+++
    #Works well empirically on a range of catchments,
    #may be some pathological cases...
    if max_cells is None:
        max_cells = round(np.sqrt(np.mean(extent.shape)))

    max_cells = int(np.min([extent.x_size,extent.y_size,max_cells]))

    out = []

    x_off = extent.x_min
    y_off = extent.y_min

    l_ex = extent.translate_localise_origin()

    x_start = [0] + (max_cells * np.arange(1,np.floor((l_ex.x_max)/max_cells)+1)).tolist()
    x_stop = (np.array(x_start[1:])-1).tolist() + [l_ex.x_max]

    y_start = [0] + (max_cells * np.arange(1,np.floor((l_ex.y_max)/max_cells)+1)).tolist()
    y_stop = (np.array(y_start[1:])-1).tolist() + [l_ex.y_max]
    
    for y in range(len(y_start)):
        for x in range(len(x_start)):
            mask = extent.mask[y_start[y]:y_stop[y]+1,x_start[x]:x_stop[x]+1]
            e = from_boundary_offset(y_start[y]+y_off,x_start[x]+x_off,y_stop[y]+y_off,x_stop[x]+x_off,None,False,mask)
            if e.cell_count > 0:
                out.append(e)

    return out


def map_catchment_name(record):
    if not record['RiverName']:
        r_name = "Unknown"
    else:
        r_name = norm_name(record['RiverName'])
    g_name = norm_name(record['GaugeName'])
    full_name = r_name + '_' + g_name
    return full_name

def field_getter(field):
    def get_field(record):
        return record[field]
    return get_field

def name_getter(field):
    def get_field(record):
        return norm_name(record[field])
    return get_field

def mask_bounds(mask):
    '''
    Return a slice index of the unmasked subset of mask
    '''
    y_range = list(range(mask.shape[0]))
    x_range = list(range(mask.shape[1]))

    for y in y_range:
        unmasked = (mask[y]==False).sum()
        if unmasked:
            y_min = y
