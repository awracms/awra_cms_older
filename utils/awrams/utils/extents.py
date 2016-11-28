import numpy as _np
from re import search as _search

from .helpers import load_mask, load_mask_grid, DEFAULT_AWRAL_MASK
from awrams.utils import helpers as _helpers
from awrams.utils import metatypes as _metatypes

CELLSIZE = 0.05

# continental_mask = load_mask()
# mask_grid = load_mask_grid()

class _ExtentBuilder(object):
    """
    Factory object used for recreating extents from persistent
    descriptions
    """

    def __init__(self,catchment_db):
        self.cdb = catchment_db

    def build_from_dict(self,extdict):
        '''
        Recreate an extent from a dictionary described as follows

        type: 'cell', 'bounds', 'catchment', 'all'
        lat/lon (for cell)
        lat_origin/lon_origin,x_size,y_size (bounds)
        id (for catchment)
        '''

        if extdict['type'] == 'cell':
            return from_cell_coords(extdict['lat'],extdict['lon'])
        elif extdict['type'] == 'bounds':
            return from_boundary_coords(extdict['lat0'],extdict['lon0'],extdict['lat1'],extdict['lon1'])
        elif extdict['type'] == 'catchment':
            return self.cdb.get_by_id(extdict['id'])
        elif extdict['type'] == 'all':
            return default()
        elif extdict['type'] == 'unknown':
            return default()
        raise Exception("Cannot interpret extent type (%s)"%extdict['type'])


class _ExtentsSelector(object):
    """
    Used to select an extent of a particular kind
    """

    def __init__(self,target):
        from .catchments import CatchmentSelector, CatchmentDB
        self._target = target
        self.catchment = CatchmentSelector(CatchmentDB(),self._target)

    def cell(self,lat,lon):
        """
        Select the extent as the given cell (lat,lon)
        """
        self._target._mode = 'cell'
        self._target._value = from_cell_coords(lat,lon)

    def _cell(self,x,y):
        """
        Select the extent as the given cell (x,y) (ie Row, Column)
        """
        self._target._mode = 'cell'
        self._target._value = from_cell_offset(x,y)

    def bounds(self,lat0,lon0,lat1,lon1):
        """
        Select the extent defined by the bounding box (lat0,lon0) <-> (lat1,lon1) INCLUSIVE.
        """
        self._target._mode = 'bounds'
        self._target._value = from_boundary_coords(lat0,lon0,lat1,lon1)

    def _bounds(self,x0,y0,x1,y1):
        """
        Select the extent defined by the bounding box (x0,y0) <-> (x1,y1) INCLUSIVE.
        """
        self._target._mode = 'bounds'
        self._target._value = from_boundary_offset(x0,y0,x1,y1)

    def all(self):
        """
        Select the entire extent available (eg continental for AWRA-L)
        """
        self._target._mode = 'all'
        self._target._value = default()

    def __call__(self,extent):
        """
        Select the supplied extent
        """
        if not isinstance(extent,Extent):
            raise Exception("%s is not an AWRAMS Extent" % extent)
        self._target._mode = 'user_supplied'
        self._target._value = extent


def build_transform(src_str,dest_str):
    '''
    Build a CoordinateTransformation using the supplied src and
    destination proj4 strings
    '''
    import osr
    src_ref = osr.SpatialReference()
    res = src_ref.ImportFromProj4(src_str)
    if res:
        raise Exception("Error building transformation from string %s" % src_str)

    dest_ref = osr.SpatialReference()
    res = dest_ref.ImportFromProj4(dest_str)

    if res:
        raise Exception("Error building transformation from string %s" % dest_str)

    transform = osr.CoordinateTransformation(src_ref,dest_ref)

    return transform

#Default projection, as per Andrew's R script
_LONGLAT_TO_AEA = None
try:
    _AEA_AUS_PROJ = "+proj=aea +ellps=GRS80, +lat_1=-18.0 +lat_2=-36.0 +units=m +lon_0=134.0 +pm=greenwich"
    _LONGLAT_PROJ = '+proj=longlat +ellps=GRS80 +no_defs'
    _LONGLAT_TO_AEA = build_transform(_LONGLAT_PROJ,_AEA_AUS_PROJ)
except:
    print("WARNING: osr not available, cell area calculations will be approximate")
    pass

class GeospatialReference(_metatypes.ObjectDict):
    def __init__(self, grid_file=None, grid_obj=None,**kw):
        if grid_file is not None:
            self.read_from_grid(grid_file)
        elif grid_obj is not None:
            self.read_from_grid_obj(grid_obj)
        else:
            self['nlats'] = 681
            self['nlons'] = 841
            self['cellsize'] = CELLSIZE
            self['lat_origin'] = -10.
            self['lon_origin'] = 112.
            self['mode'] = 'center'
        self.update(kw)

    def which_style(self):
        '''
        Test to see if gdal or ncdf style GSR
        ie: Corner of cell vs center
        '''
        val = self['lon_origin'] / self['cellsize']
        if val == round(val):
            return 'netcdf'
        else:
            return 'gdal'

    def to_center(self):
        out = GeospatialReference(**self.copy())
        if self.mode == 'corner':
            out['lon_origin'] = _np.around(out['lon_origin'] + self['cellsize'] * 0.5,3)
            out['lat_origin'] = _np.around(out['lat_origin'] - self['cellsize'] * 0.5,3)
            out['mode'] = 'center'
        return out

    def to_corner(self):
        out = GeospatialReference(**self.copy())
        if self.mode == 'center':
            out['lon_origin'] = _np.around(out['lon_origin'] - self['cellsize'] * 0.5,3)
            out['lat_origin'] = _np.around(out['lat_origin'] + self['cellsize'] * 0.5,3)
            out['mode'] = 'corner'
        return out

    def read_from_grid(self, grid_file):
        import osgeo.gdal as gd
        #if grid_file.endswith('\.gz'): # this doesn't work
        if _search('gz$', grid_file):
            grid_file = '/vsigzip/' + grid_file
        source = gd.Open(grid_file)
        self.read_from_grid_obj(source)

    def read_from_grid_obj(self,source):
        gref = source.GetGeoTransform()
        self.update(nlons=source.RasterXSize,
                    nlats=source.RasterYSize,
                    cellsize=gref[1],
                    lon_origin=gref[0], # gdal uses left corner
                    lat_origin=gref[3], # gdal uses upper corner
                    mode='corner')

    def geo_for_cell(self,y,x):
        '''
        Return the lat/lon coords of a (data) cell relative to our local origin
        '''
        return (_helpers.quantize(self.lat_origin - y * self.cellsize,self.cellsize), _helpers.quantize(self.lon_origin + x * self.cellsize,self.cellsize))

    def cell_for_geo(self,lat,lon):
        '''
        Return the data coords of a (geo) cell relative to our local origin
        '''
        return _helpers.iround((self.lat_origin - lat)/self.cellsize),_helpers.iround((lon-self.lon_origin)/self.cellsize)

    def offset(self,y_off,x_off,y_size,x_size):
        '''
        +++
        Note that these offsets are in 'coordinate' terms,
        _not_ data.  Ie the cell numbers used to index the array
        are the opposites of the 'x' and 'y' referred to here
        '''
        self['nlats'] = y_size
        self['nlons'] = x_size
        self['lon_origin'] += x_off * self['cellsize']
        self['lat_origin'] -= y_off * self['cellsize']

    def bounds_args(self):
        '''
        Return a tuple of lat,lon,lat,lon suitable representing the bounding
        corners of this reference
        '''
        min_lat = self.lat_origin - ((self.nlats -1) * self['cellsize'])
        max_lon = self.lon_origin + ((self.nlons-1) * self['cellsize'])
        return (self.lat_origin,self.lon_origin,min_lat,max_lon)

    @property
    def shape(self):
        return (self.nlats,self.nlons)

    def get_geotransform(self):
        '''for gdal method SetGeoTransform - self.mode == "corner"'''
        gr = self.to_corner()
        return (gr.lon_origin,gr.cellsize,0,gr.lat_origin,0,-gr.cellsize)

    @classmethod
    def from_mask(cls,DEFAULT_AWRAL_MASK):
        import h5py
        h = h5py.File(DEFAULT_AWRAL_MASK)
        dims = h['dimensions']
        kw = dict(nlats=len(dims['latitude']),
                  nlons=len(dims['longitude']),
                  lat_origin=dims['latitude'][0],
                  lon_origin=dims['longitude'][0])

        gr = cls(**kw)
        h.close()
        gr.mask = load_mask_grid(DEFAULT_AWRAL_MASK) #h['parameters']['mask'][:]
        return gr

def global_georef():
    import os
    global GLOBAL_GEOREF
    if GLOBAL_GEOREF is None:
        if os.path.splitext(DEFAULT_AWRAL_MASK)[1] == '.flt':
            GLOBAL_GEOREF = GeospatialReference(DEFAULT_AWRAL_MASK).to_center()
        elif os.path.splitext(DEFAULT_AWRAL_MASK)[1] == '.h5':
            GLOBAL_GEOREF = GeospatialReference.from_mask(DEFAULT_AWRAL_MASK)
        else:
            raise Exception("unknown mask grid format: %s" % DEFAULT_AWRAL_MASK)

    return GLOBAL_GEOREF

GLOBAL_GEOREF = None #global_georef()


def cells_in_geodist(dist):
    """
    return the number of cells covered in the (centroid-to-centroid) distance

    :param dist:
    :return:
    """
    return int(abs(round((dist/CELLSIZE))))+1


def bounds_ref(lat0,lon0,lat1,lon1):
    """
    create a geospatial_reference dict from a set of boundary coords

    :param lat0:
    :param lon0:
    :param lat1:
    :param lon1:
    :return:
    """

    lat_origin = max(lat0,lat1)
    lon_origin = min(lon0,lon1)

    lat_min = min(lat0,lat1)
    lon_max = max(lon0,lon1)

    nlats = cells_in_geodist(lat_min - lat_origin)
    nlons = cells_in_geodist(lon_max - lon_origin)

    return GeospatialReference(lat_origin=lat_origin, lon_origin=lon_origin, nlats=nlats, nlons=nlons)

from functools import wraps
def _load_global_georef(fnc):
    '''
    decorator: if parent_ref = None then replace it with global_georef
    '''
    @wraps(fnc)
    def decor(*args, **kwargs):
        if GLOBAL_GEOREF is None:
            global_georef()

        import inspect
        a = inspect.getargspec(fnc)
        pos = a.args.index('parent_ref')

        try: ### explicitly set to None in kwargs
            if kwargs['parent_ref'] is None:
                kwargs['parent_ref'] = GLOBAL_GEOREF
        except KeyError:
            try: ### explicitly set to None in args
                if args[pos] is None:
                    args = list(args)
                    args[pos] = GLOBAL_GEOREF
            except IndexError: ### implicitly set to None in defaults
                kwargs['parent_ref'] = GLOBAL_GEOREF
            pass
        return fnc(*args, **kwargs)
    return decor

@_load_global_georef
def default(parent_ref=GLOBAL_GEOREF):
    """
    standard awral extent

    :param parent_ref:
    :return: Extent
    """
    extent = Extent(parent_ref=parent_ref)
    extent.mask = parent_ref.mask #load_mask_grid()
    extent._set_extents()
    return extent

@_load_global_georef
def from_cell_coords(lat,lon,parent_ref=GLOBAL_GEOREF):
    """
    a single cell extent specified by geographic coordinates

    :param lat:
    :param lon:
    :param parent_ref:
    :return: Extent
    """
    extent = Extent(parent_ref=parent_ref)
    extent.x_size = 1
    extent.y_size = 1
    extent.cell_count = 1
    # Naively assume that if we bothered to generate this cell, it is unmasked...
    extent.mask = _np.zeros((1,1))
    extent.lat_origin = lat
    extent.lon_origin = lon
    extent._set_extents()
    return extent

@_load_global_georef
def from_cell_offset(x,y,parent_ref=GLOBAL_GEOREF):
    """
    a single cell extent specified by data coordinates x,y
    relative to parent_ref

    :param x:
    :param y:
    :param parent_ref:
    :return: Extent
    """
    return from_cell_coords(*parent_ref.geo_for_cell(x,y),parent_ref=parent_ref)

@_load_global_georef
def from_boundary_coords(lat0,lon0,lat1,lon1,parent_ref=GLOBAL_GEOREF,compute_areas=False,mask=None,extent=None):
    """
    multiple cell extent specified by bounding geographic coordinates

    :param lat0:
    :param lon0:
    :param lat1:
    :param lon1:
    :param parent_ref:
    :param compute_areas:
    :param mask:
    :return: Extent
    """
    created_extent = False
    if extent is None:
        created_extent = True
        extent = Extent(parent_ref=parent_ref)

    lat_min = min(lat0,lat1)
    lon_min = min(lon0,lon1)
    lat_max = max(lat0,lat1)
    lon_max = max(lon0,lon1)

    extent.lat_origin = lat_max
    extent.lon_origin = lon_min
    extent.y_size = int(round((lat_max-lat_min)/CELLSIZE + 1))
    extent.x_size = int(round((lon_max-lon_min)/CELLSIZE + 1))

    if mask is None:  ### use default mask
        mask_grid = load_mask_grid()
        lat_off = int(round((parent_ref.lat_origin - extent.lat_origin)/CELLSIZE))
        lon_off = int(round((extent.lon_origin - parent_ref.lon_origin)/CELLSIZE))
        extent.mask = mask_grid[lat_off : lat_off+extent.y_size, lon_off : lon_off+extent.x_size].copy()
    else:
        extent.mask = mask

    if compute_areas:
        extent.compute_areas()

    extent._set_extents()
    if created_extent:
        return extent

@_load_global_georef
def from_boundary_offset(y_min,x_min,y_max,x_max,parent_ref=GLOBAL_GEOREF,compute_areas=False,mask=None):
    """
    bounded (inclusive) region specified by data coordinates,
    relative to parent_ref

    :param y_min:
    :param x_min:
    :param y_max:
    :param x_max:
    :param parent_ref:
    :param compute_areas:
    :param mask:
    :return: Extent
    """
    lat_min = parent_ref.lat_origin - (y_max * CELLSIZE)
    lat_max = parent_ref.lat_origin - (y_min * CELLSIZE)
    lon_min = parent_ref.lon_origin + (x_min * CELLSIZE)
    lon_max = parent_ref.lon_origin + (x_max * CELLSIZE)

    return from_boundary_coords(lat_min,lon_min,lat_max,lon_max,parent_ref=parent_ref,compute_areas=compute_areas,mask=mask)

def from_georef(georef,compute_areas=True):
    """
    extent defined by GeospatialReference
    :param georef:
    :param compute_areas:
    :return: Extent
    """
    return from_boundary_coords(*georef.bounds_args(),compute_areas=compute_areas)

def from_multiple(source_extents,parent_ref=None):
    """
    Treat a list of extents objects as a single extent
    (eg for multi-catchment runs)

    :param source_extents:
    :param parent_ref:
    :return: Extent
    """
    extent = Extent(parent_ref=parent_ref)

    # Convert to dict
    if not hasattr(source_extents,'keys'):
        e_map = {}
        for i,extent in enumerate(source_extents):
            e_map[i] = extent
        source_extents = e_map

    if parent_ref is not None:
        translated = {}
        for k,v in list(source_extents.items()):
            translated[k] = v.translate_to_origin(parent_ref)
        extent.extents = translated
    else:
        extent.extents = source_extents


    extent.x_min = _np.inf
    extent.x_max = 0 - _np.inf
    extent.y_min = _np.inf
    extent.y_max = 0 - _np.inf
    extent.cell_count = 0

    for e in list(extent.extents.values()):
        extent.x_min = min(extent.x_min, e.x_min)
        extent.x_max = max(extent.x_max, e.x_max)
        extent.y_min = min(extent.y_min, e.y_min)
        extent.y_max = max(extent.y_max, e.y_max)

    extent.x_size = (extent.x_max - extent.x_min) + 1
    extent.y_size = (extent.y_max - extent.y_min) + 1

    ref_cell = from_cell_offset(extent.y_min, extent.x_min, extent.parent_ref)

    extent.lat_origin = ref_cell.lat_origin
    extent.lon_origin = ref_cell.lon_origin

    extent.mask = _np.ones(shape=extent.shape,dtype=bool)

    for e in list(extent.extents.values()):
        for cell in e:
            cell = e.localise_cell(cell)
            extent.mask[e.y_min + cell[0] - extent.y_min,e.x_min + cell[1] - extent.x_min] = e.mask[cell[0],cell[1]]

    extent.cell_count = (extent.mask == False).sum()

    return extent

class Extent(object):
    @_load_global_georef
    def __init__(self,parent_ref=GLOBAL_GEOREF):
        """
        Base extent __init__, never called directly

        :param parent_ref:
        :return:
        """

        self.parent_ref = parent_ref
        self.y_size,self.x_size = self.parent_ref.nlats,self.parent_ref.nlons
        self.lat_origin = self.parent_ref['lat_origin']
        self.lon_origin = self.parent_ref['lon_origin']
        self.mask = _np.zeros(self.shape,dtype=bool)
        self.area = None

        self._set_extents()

    def _set_extents(self):
        self.y_min, self.x_min = self.cell_for_geo(self.lat_origin,self.lon_origin)
        self.x_max = (self.x_min + self.x_size) - 1
        self.y_max = (self.y_min + self.y_size) - 1

        self.cell_count = (self.mask == False).sum()

    @property
    def x_index(self):
        return slice(self.x_min,self.x_min+self.x_size)

    @property
    def y_index(self):
        return slice(self.y_min,self.y_min+self.y_size)

    @property
    def shape(self):
        return self.y_size,self.x_size

    def indices(self):
        '''
        Return data indices, latitude first
        '''
        return self.y_index,self.x_index

    def __repr__(self):
        if hasattr(self,'display_name'):
            return self.display_name
        else:
            if self.y_size == 1:
                lat_str = "lat: %s (%s)" % (self.lat_origin,self.y_min)
            else:
                lat_str = "lat_range:%s,%s (%s,%s)" % (self.lat_origin,
                                                       self.lat_origin - (self.y_size - 1)*CELLSIZE,
                                                       self.y_min,
                                                       self.y_min+self.y_size-1)
            if self.x_size == 1:
                lon_str = "lon: %s (%s)" % (self.lon_origin,self.x_min)
            else:
                lon_str = "lon_range:%s,%s (%s,%s)" % (self.lon_origin,
                                                       self.lon_origin + (self.x_size - 1)*CELLSIZE,
                                                       self.x_min,
                                                       self.x_min+self.x_size-1)

            return "%s %s shape:%s" % (lat_str,lon_str,self.shape)


    #+++
    #Duplicated from geospatial_reference
    #Need these calls for translate_to_origin
    def geo_for_cell(self,y,x):
        geo = self.geospatial_reference().geo_for_cell(y,x)
        return (_helpers.quantize(geo[0],0.05),_helpers.quantize(geo[1],0.05))

    def cell_for_geo(self,lat,lon):
        return self.parent_ref.cell_for_geo(lat,lon)

    def contains(self,extent):
        '''
        Test to see if supplied extent is a subset of this one
        '''
        t_ext = extent.translate_to_origin(self.geospatial_reference())
        return self.mask[t_ext.indices()].size == t_ext.shape[0] * t_ext.shape[1]

    def to_dict(self):
        '''
        Dictionary with minimal info required for serialisation
        '''
        raise Exception

    def localise_cell(self,cell):
        '''
        Cells are always specified in terms of continental data
        coordinates.  This function localises the data coordinates
        in terms of the extent
        '''
        return [cell[0]-self.y_min,cell[1]-self.x_min]

    def geospatial_reference(self):
        '''
        Return the geospatial bounds represented by the extent
        '''
        return GeospatialReference(nlats=self.y_size,
                                   nlons=self.x_size,
                                   lon_origin=self.lon_origin,
                                   lat_origin=self.lat_origin,
                                   mode='center')

    def translate(self, new_georef):
        '''
        Translate the dataspace (x,y) coordinates of an Extent
        to those of a parent reference
        ie "What indices will I need to obtain the same data from
        a dataset with a (new) geospatial origin"
        '''
        from copy import deepcopy

        new_extent = deepcopy(self)
        new_extent.parent_ref = new_georef
        new_extent._set_extents()

        return new_extent

    def translate_to_origin(self, parent_ref):
        return self.translate(parent_ref)

    def translate_localise_origin(self):
        '''
        Return a copy of the extent representing the same geo coords,
        but with data coords localised to start at 0,0
        '''
        return self.translate(self.geospatial_reference())

    def get_graph_extent(self):
        '''
        Return a tuple of extents suitable for visualisation with mpl/imshow
        '''
        gsr = self.geospatial_reference()

        halfcell = gsr['cellsize'] * 0.5
        g_ex = []
        g_ex.append(gsr['lon_origin'] - halfcell)
        g_ex.append(gsr['lon_origin']+(gsr['nlons']-1)*gsr['cellsize'] + halfcell)
        g_ex.append(gsr['lat_origin']-(gsr['nlats']-1)*gsr['cellsize'] - halfcell)
        g_ex.append(gsr['lat_origin'] + halfcell)
        return g_ex

    def geo_index(self):
        gsr = self.geospatial_reference()
        g_ex = []
        g_ex.append(gsr['lon_origin'])
        g_ex.append(gsr['lon_origin']+(gsr['nlons']-1)*gsr['cellsize'])
        g_ex.append(gsr['lat_origin']-(gsr['nlats']-1)*gsr['cellsize'])
        g_ex.append(gsr['lat_origin'])

        return (slice(g_ex[2],g_ex[3]),slice(g_ex[0],g_ex[1]))

    def cell_list(self):
        '''
        Return a list of all cells, in order
        '''
        cells = []
        for cell in self:
            cells.append(cell)
        return cells

    def get_locations(self):
        '''
        Return a list of lat/lon pairs of all cells
        '''
        lat,lon = self._flatten_fields()
        return list(zip(lat,lon))

    def iter_points(self):
        '''
        Fast version of iterating cells - only returns tuples (ie no geo information)
        By default just returns cells, can override
        '''
        return self.__iter__()

    def itercells(self):
        def all_points_iter():
            for cell in self:
                yield from_cell_offset(cell[0],cell[1],parent_ref=self.parent_ref)
        return all_points_iter()

    def __iter__(self):
        ### cell list from mask is relative to local origin
        ### need to convert to parent_ref origin
        local_cells = _np.where(_np.logical_not(self.mask))
        cell_list = list(zip(local_cells[0] + self.y_min, local_cells[1] + self.x_min))
        def all_cells_iter():
            for cell in cell_list:
                yield cell
        return all_cells_iter()

    def _flatten_areas(self):
        '''
        Get a flat array of the cell areas
        '''
        return self.areas[self.mask == False].flatten()

    def _flatten_fields(self):
        '''
        Get flat arrays of lats/lons for cells
        '''
        lats = []
        lons = []
        for cell in self:
            gcell = from_cell_offset(cell[0],cell[1])
            lats.append(gcell.lat_origin)
            lons.append(gcell.lon_origin)
        return _helpers.aquantize(_np.array(lats),0.05), _helpers.aquantize(_np.array(lons),0.05)

    def compute_areas(self):
        '''
        Generate array of latitude-correct area values for cells
        +++ Should use optimisation, but AEA results in subtle
            differences accross longitude; calculate cells independently
            for consistency, for now
        '''
        self.areas = _np.ma.zeros(self.shape)
        self.areas.mask = self.mask
        self.weights = _np.ma.zeros(self.shape)
        self.weights.mask = self.mask

        if self.cell_count == 1:
            self.area = self._area()
        else:
            for cell in self:
                gcell = from_cell_offset(cell[0],cell[1])
                lcell = self.localise_cell(cell)
                self.areas[lcell[0],lcell[1]] = gcell._area()
                self.weights[lcell[0],lcell[1]] = gcell._weight

            self.area = self.areas.sum()

    def _area(self):
        if _LONGLAT_TO_AEA is not None:
            poly = self.to_polygon()
            poly.Transform(_LONGLAT_TO_AEA)
            return poly.Area()
        else:
            p = self.get_graph_extent()
            corners = ((p[2],p[0]),(p[3],p[0]),(p[3],p[1]),(p[2],p[1]),(p[2],p[0]))
            ref_lat = (p[2] + p[3])/2.
            area = calc_area(corners,ref_lat)
            return area

    @property
    def _weight(self):
        return _np.cos(self.lat_origin/180. * _np.pi)

    def to_polygon(self):
        '''
        Produce an OGR Polygon object representing the boundaries of the extent
        '''
        from .catchments import polygon_from_cell

        halfcell = CELLSIZE * 0.5
        lat0 = self.lat_origin + halfcell
        lon0 = self.lon_origin - halfcell
        lat1 = self.lat_origin - (self.x_size - 1) * CELLSIZE - halfcell
        lon1 = self.lon_origin + (self.y_size - 1) * CELLSIZE + halfcell
        return polygon_from_cell((lat0,lat1),(lon0,lon1))

def degrees_to_radians(d):
    return d / 180. * _np.pi

def radius(latitude):
    a = 6378137.0         ### equatorial radius GRS80
    b = 6356752.314140347 ### polar radius GRS80
    l = degrees_to_radians(latitude)
    cos = _np.cos(l)
    sin = _np.sin(l)
    return _np.sqrt(((a**2*cos)**2 + (b**2*sin)**2) / ((a*cos)**2 + (b*sin)**2))

def calc_area(corners,ref_lat):
    """
    http://gis.stackexchange.com/questions/711/how-can-i-measure-area-from-geographic-coordinates
    http://trac.osgeo.org/openlayers/browser/trunk/openlayers/lib/OpenLayers/Geometry/LinearRing.js?rev=10116#L233
    http://trs-new.jpl.nasa.gov/dspace/bitstream/2014/40409/3/JPL%20Pub%2007-3%20%20w%20Errata.pdf

    :param corners:
    :param ref_lat:
    :return:
    """
    area = 0.0
    lr = radius(ref_lat)

    for i in range(len(corners)-1):
        p1 = corners[i]
        p2 = corners[i+1]
        area += degrees_to_radians(p2[1] - p1[1]) * (2 + _np.sin(degrees_to_radians(p1[0])) + _np.sin(degrees_to_radians(p2[0])))

    return area * lr**2 / 2.0
