import h5py
import numpy as np
import netCDF4 as nc
from collections import OrderedDict
import types

from awrams.utils.settings import VAR_CHUNK_CACHE_SIZE, VAR_CHUNK_CACHE_NELEMS, VAR_CHUNK_CACHE_PREEMPTION#pylint: disable=no-name-in-module

propfaid = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
settings = list(propfaid.get_cache())
#settings[1]        # size of hash table
settings[2] = 0     #2**17 # =131072 size of chunk cache in bytes
                    # which is big enough for 5x(75, 1, 50 chunks;
                    # default is 2**20 =1048576
settings[3] = 1.    # preemption 1 suited to whole chunk read/write
propfaid.set_cache(*settings)
propfaid.set_fapl_sec2()
propfaid.set_sieve_buf_size(0)
propfaid.set_fclose_degree(h5py.h5f.CLOSE_STRONG)
#propfaid.set_fapl_stdio()

mdc = dict(off  =(False,),
           K1   =(True, 2**10, 2**10, 2**10),
           K32  =(True, 2**15, 2**15, 2**15),
           K262 =(True, 2**18, 2**18, 2**18),
           M1   =(True, 2**20, 2**20, 2**20),
           )

class h5pyOb:
    def __init__(self):
        self.dimensions = OrderedDict()
        for k,v in self.items():
            try:
                v_class = v.attrs['CLASS'].decode()
                if v_class == "DIMENSION_SCALE":
                    self.dimensions[k] = h5pyDimension()
                    if v.maxshape[0] is None:
                        self.dimensions[k]._isunlimited = True
            except:
                pass

    def _init_attrs(self):
        for k in self.attrs.keys():
            try:
                self.__dict__[k] = self.attrs[k].decode()
            except AttributeError:
                self.__dict__[k] = self.attrs[k]

    def _populate(self):
        self.variables = dict()
        self.groups = dict()

        for k in self.keys():
            if isinstance(self[k], h5py.Dataset):
                if 'NAME' in self[k].attrs and self[k].attrs['NAME'].decode().startswith("This is a netCDF dimension but not a netCDF variable"):
                    ### this for dimension 'nv' in monthly files
                    continue
                self.variables[k] = h5pyDataset(self[k])
                try:
                    self.variables[k].units = self[k].attrs['units'].decode()
                except KeyError:
                    self.variables[k].units = None

            elif isinstance(self[k], h5py.Group):
                self.groups[k] = h5pyGroup(self[k])
                #self[k] = self.groups[k]

    def ncattrs(self):
        return list(self.attrs.keys())

    def getncattr(self, key):
        try:
            if self.attrs[key].shape == (1,):
                return self.attrs[key][0] #np.asscalar(self.attrs[key])
        except:
            pass

        return self.__dict__[key]


class h5pyDimension:
    def __init__(self):
        self._isunlimited = False

    def isunlimited(self):
        return self._isunlimited


class h5pyGroup(h5py.Group,h5pyOb):
    def __init__(self, group):
        h5py.Group.__init__(self, group.id)
        h5pyOb.__init__(self)
        self._populate()
        self._init_attrs()

    def createGroup(self, name):
        """
        to mimick netCDF4 createGroup method
        creates group with id=gid (becomes self.id of new group)
        """
        gid = h5py.h5g.create(self.id, name)
        self.groups[name] = h5pyGroup(gid)
        return self.groups[name]

    def setncattr(self, key, value):
        self.attrs[key] = value


class h5pyDataset(h5py.Dataset,h5pyOb):
    def __init__(self, dataset):
        h5py.Dataset.__init__(self, dataset.id)
        self._init_attrs()

        if 'DIMENSION_LIST' in self.attrs:
            self.dimensions = {}
            for d in self.dims:
                for i in d.items():
                    self.dimensions[i[0]] = i[1]

    #    self.name = super(h5pyDataset,self).__getattribute__('name')

    # @property
    # def name(self):
    #     return super(h5pyDataset,self).__getattribute__('name')

    # def __getitem__(self, slice):
    #     """
    #     always return masked array; much better than netCDF4 which will return
    #     numpy.ndarray if no masked_values present = PAINFUL!
    #     """
    #     #logger.info("h5pyDataset.__getitem__")
    #     data = super(h5pyDataset, self).__getitem__(slice)
    #     if data.size == 1: # a scalar #type(slice) == int:
    #         return np.ma.masked_values([data], self.fillvalue)[0]
    #     else:
    #         return np.ma.masked_values(data, self.fillvalue)

    # def __setitem__(self, slice, data):
    #     """
    #     handle resizing here
    #     """
    #     logger.info("h5pyDataset.__setitem__ %s %s %s %s %s %s",self.file,self.name,slice,self.shape,self[slice].shape,len(data))
    #     if self[slice].shape[0] < len(data):
    #         self.resize((len(data),))
    #     logger.info("h5pyDataset.__setitem__")
        #h5py.Dataset.__setitem__(self, slice, data)
    #    h5py.Dataset.write_direct(data, dest_sel=slice)

class _h5py(h5py.File,h5pyOb):
    """
    open database with h5py.File and return object that looks like netCDF4.Dataset
    """
    def __init__(self, file_name, mode='r'):
        self.file_name = file_name
        self._filepath = file_name # mimick netCDF4 attribute

        flags = h5py.h5f.ACC_RDWR
        if mode == 'r':
            flags = h5py.h5f.ACC_RDONLY
        fid = h5py.h5f.open(file_name.encode(), flags=flags, fapl=propfaid)

        self.file_id = fid
        self._id = fid

        h5py.File.__init__(self, fid) #, 'a', driver='sec2')
        h5pyOb.__init__(self)

        self.root = h5pyGroup(self)

        try:
            self.var_name = self.attrs['var_name'].decode()
        except KeyError:
            pass #self.var_name = None

        self._populate()
        self._init_attrs()

    def open(self,fid):
        h5py.File.__init__(self, fid) #, 'a', driver='sec2')
        h5pyOb.__init__(self)

        self.root = h5pyGroup(self)

        self.var_name = self.attrs['var_name'].decode()

        self._populate()
        self._init_attrs()

    def createGroup(self, name):
        self.groups[name] = self.root.createGroup(name)
        return self.groups[name]
        #return self.root.createGroup(name)

    def filepath(self):
        return self._filepath

    def write_direct(self, data, dest_sel):
        self[self.var_name].write_direct(data, dest_sel=dest_sel)

    def flush(self):
        h5py.h5f.flush(self.file_id, h5py.h5f.SCOPE_GLOBAL)

    def close(self):
        self.flush()
        h5py.File.close(self)
        #self.file_id.close()

    def sync(self):
        self.flush()

    def set_mdc(self,mdc):
        h5f = self.file_id
        mdc_cache_config = h5f.get_mdc_config()
        mdc_cache_config.set_initial_size = mdc[0] #True
        mdc_cache_config.initial_size = mdc[1] #1024
        mdc_cache_config.max_size = mdc[2] #1024
        mdc_cache_config.min_size = mdc[3] #1024
        h5f.set_mdc_config(mdc_cache_config)


class _nc(nc.Dataset):
    """
    open database and set chunk cache
    """
    def __init__(self, file_name, mode='r'):
        nc.Dataset.__init__(self, file_name, mode)
        for v in self.variables:
            self.variables[v] = _v(self.variables[v])

    def get_attr(self,key):
        return self.variables[self.var_name].getncattr(key)

    def set_chunk_cache(self, **params):
        p = dict(var_chunk_cache_size=VAR_CHUNK_CACHE_SIZE,
                 var_chunk_cache_nelems=VAR_CHUNK_CACHE_NELEMS,
                 var_chunk_cache_preemption=VAR_CHUNK_CACHE_PREEMPTION)
        p.update(**params)
        self.variables[self.var_name].set_var_chunk_cache(size=p['var_chunk_cache_size'],
                                                          nelems=p['var_chunk_cache_nelems'],
                                                          preemption=p['var_chunk_cache_preemption'])

    def flush(self):
        self.sync()

    def __getitem__(self,idx):
        return self.variables[idx]

class _v:
    '''
    wrap existing Variable to add property attrs
    '''
    def __init__(self,v):
        self.v = v
        self.name = self.v.name
        if hasattr(v,'units'):
            self.units = self.v.units
        if hasattr(v,'bounds'):
            self.bounds = self.v.bounds
        self.dtype = self.v.dtype
        self.shape = self.v.shape
        self.attrs = {}
        for a in self.ncattrs():
            self.attrs[a] = [self.v.getncattr(a)]

    def __getitem__(self, idx):
        return self.v[idx]

    def __setitem__(self, idx, data):
        self.v[idx] = data

    def getncattr(self,k):
        return self.v.getncattr(k)

    def ncattrs(self):
        return self.v.ncattrs()

    def __len__(self):
        return len(self.v)

# def set_mdc(h5_file,mdc):
#     h5f = h5_file.fid
#     mdc_cache_config = h5f.get_mdc_config()
#     mdc_cache_config.set_initial_size = mdc[0] #True
#     mdc_cache_config.initial_size = mdc[1] #1024
#     mdc_cache_config.max_size = mdc[2] #1024
#     mdc_cache_config.min_size = mdc[3] #1024
#     h5f.set_mdc_config(mdc_cache_config)
