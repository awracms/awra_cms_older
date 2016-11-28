import json
import numpy as np
import datetime
import dateutil
from awrams.utils import mapping_types as mt
from awrams.utils.messaging.buffer_group import DataSpec
from awrams.utils.io.data_mapping import FlatFileManager,AnnualSplitFileManager
from awrams.utils.mapping_types import extent_to_spatial_coords,CoordinateSet
from awrams.utils.extents import default

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('nodes')

def callable_to_funcspec(c):
    func_spec = dict(name=c.__name__)
    if hasattr(c,'__module__'):
        func_spec['module'] = c.__module__
    else:
        func_spec['module'] = c.__class__.__module__
    return func_spec

def funcspec_to_callable(func_spec):
    import importlib
    m = importlib.import_module(func_spec['module'])
    c = getattr(m,func_spec['name'])
    return c

class GraphNode:
    def __init__(self,node_type,executor,out_type,inputs=None,properties=None,args=None):
        self.node_type = node_type
        
        if inputs is None:
            inputs = []
        if properties is None:
            properties = {}
        if args is None:
            args = {}
            
        self.inputs = inputs
        self.out_type = out_type
        self.properties = properties
        self.args = args

        if callable(executor):
            executor = callable_to_funcspec(executor)

        self.executor = executor
    
    def get_executor(self):
        return funcspec_to_callable(self.executor)(**self.args)
        
    def __repr__(self):
        return '%s(%s):%s' % (self.node_type,self.inputs,self.args)

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GraphNode):
            return dict(obj_type='GraphNode', data = {
                'node_type': obj.node_type,
                'properties': obj.properties,
                'inputs': obj.inputs,
                'out_type': obj.out_type,
                'args': obj.args,
                'executor': obj.executor
            })
        elif isinstance(obj,np.ndarray):
            return dict(obj_type='ndarray', data = {
                    'data': obj.tolist(),
                    'dtype': obj.dtype.str
            })
        elif isinstance(obj,datetime.datetime):
            return dict(obj_type='datetime', date = {
                    'data': obj.isoformat()
                })
        else:
            return json.JSONEncoder.default(self, obj)
        

class GraphDecoder(json.JSONDecoder):   
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        
    def object_hook(self,obj):
        if 'obj_type' in obj:
            obj_type = obj['obj_type']
            if obj_type=='GraphNode':
                return GraphNode(**obj['data'])
            elif obj_type=='ndarray':
                return np.ndarray(obj['data']['data'],dtype=obj['data']['dtype'])
            elif obj_type=='datetime':
                return dateutil.parser.parse(obj['data'])
            else:
                raise Exception("Unknown type", obj_type)
        else:
            return obj
            
def dump_mapping(mapping,filename):
    with open(filename,'w') as fh:
        json.dump(mapping,fh,cls=GraphEncoder)
        
def load_mapping(filename):
    with open(filename,'r') as fh:
        return json.load(fh,cls=GraphDecoder)
    
'''
Predefined node types and convenience definitions
'''

def get_flattened(data,mask):
    if mask.any():
        return data[...,~mask]
    else:
        return data.reshape(list(data.shape[:data.ndim-2])+[mask.shape[0]*mask.shape[1]])

def get_expanded(data,mask,fill_value=np.nan,as_ma=True):
    outshape = list(data.shape[:data.ndim-1])+[mask.shape[0],mask.shape[1]]

    all_false = False

    if data.shape[-1] == mask.shape[0]*mask.shape[1]:
        out = data.reshape(outshape)
        all_false = True
    else:    
        out = np.empty(outshape,dtype=data.dtype)
        if fill_value is not None:
            out.fill(fill_value)
        out[...,~mask] = data

    if as_ma:
        out = np.ma.MaskedArray(out,fill_value=fill_value)
        if all_false:
            out_mask = False
        else:
            out.mask = mask        
    
    return out

class InputNode:
    def get_data(self,coords):
        raise NotImplementedError

    def get_data_flat(self,coords,mask):
        return get_flattened(self.get_data(coords),mask)

    def get_dataspec(self):
        raise NotImplementedError

    def get_dataspec_flat(self):
        dspec = self.get_dataspec()
        if dspec.valtype == 'array':
            if list(dspec.dims[-2:]) == ['latitude','longitude']:
                dspec.dims = list(dspec.dims[:-2]) + ['cell']
        return dspec

class ForcingNode(InputNode):
    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        return DataSpec('array',dims,self.dtype)

class ProcessNode:
    def process(self,inputs):
        raise NotImplementedError

    def get_dataspec(self,in_specs):
        return promote_dataspecs(in_specs)

class _ExecutionNode:
    def __init__(self):
        pass
    
    def get_data(self,coords,inputs):
        raise NotImplementedError

    def get_data_flat(self,coords,inputs,mask):
        return get_flattened(self.get_data(coords,inputs),mask)

def promote_dataspecs(in_specs):
    curvtype = 'scalar'
    curdtype = None
    dims = []
    for ispec in in_specs:
        if ispec.valtype == 'array':
            curvtype = 'array'
            comp = np.ndarray(shape=(1,),dtype=ispec.dtype)

            #+++ Doesn't actually check if compatible, just assumes
            # we can broadcast to larger arrays....
            if len(ispec.dims) > len(dims):
                dims = ispec.dims
        else:
            comp = np.ndarray(shape=(1),dtype=ispec.dtype)[0]
        if curdtype is None:
            curdtype = comp
        else:
            curdtype = np.result_type(curdtype,comp)

    return DataSpec(curvtype,dims,curdtype)

class TransformNode(ProcessNode):
    def __init__(self,tfunc,func_args):
        self.tfunc = funcspec_to_callable(tfunc)
        self.func_args = func_args
                 
    def process(self,inputs):
        return self.tfunc(*inputs,**self.func_args)


class ConstNode(InputNode):
    def __init__(self,value):
        self.value = value
        
    def get_data(self,coords):
        return self.value

    def get_data_flat(self,coords,mask):
        return self.value

    def get_dataspec(self):
        return DataSpec('scalar',[],type(self.value))

class SpatialConst(InputNode):
    def __init__(self,value,dtype=np.float64):
        self.value = value
        self.dtype = dtype
        
    def get_data(self,coords):
        out = np.empty(shape=coords.shape[1:],dtype=self.dtype)
        out.fill(self.value)
        return out

    def get_data_flat(self,coords,mask):
        out = np.empty(((~mask).sum(),),dtype=self.dtype)
        out.fill(self.value)
        return out

    def get_dataspec(self):
        return DataSpec('array',['latitude','longitude'],self.dtype)    

class AssignNode(ProcessNode):
    def process(self,inputs):
        return inputs[0]

    def get_dataspec(self,in_specs):
        return in_specs[0]

class StaticForcing(InputNode):
    def __init__(self,data_map,nc_var,dims=['time','latitude','longitude']):
        self.data_map = data_map
        self.nc_var = nc_var
        self.dims = dims

    def get_data_prepack(self,cid):
        return self.data_map[cid]

    def get_dataspec(self):
        return DataSpec('array',self.dims,np.float64)

class SplitFileReaderNode(InputNode):
    def __init__(self,path,pattern,nc_var,cache=False):
        from awrams.utils.io.data_mapping import SplitFileManager
        self.sfm = SplitFileManager.open_existing(path,pattern,nc_var)
        self.cache = cache
        self.data = None

    def get_data(self,coords):
        if self.cache:
            if self.data is None:
                d = self.sfm.get_padded_by_coords(coords)
                self.data = np.where(d==self.sfm.fillvalue,np.nan,d)
            return self.data

        d = self.sfm.get_padded_by_coords(coords)
        return np.where(d==self.sfm.fillvalue,np.nan,d)

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.sfm.mapped_var.dtype
        return DataSpec('array',dims,dtype)

class StateFileReaderNode(SplitFileReaderNode):
    def get_data(self,coords):
        from copy import deepcopy
        tc = deepcopy(coords['time'])
        tc.index = [tc.index[0] - 1]
        coords = CoordinateSet([tc,coords['latitude'],coords['longitude']])
        d = self.sfm.get_padded_by_coords(coords)[0,:]
        return np.where(d==self.sfm.fillvalue,np.nan,d)

class InitStateFromArray(InputNode):
    def __init__(self,data,extent=None):
        self.data = data

        if extent is None:
            self.coords = extent_to_spatial_coords(default())
        else:
            self.coords = extent_to_spatial_coords(extent)

        if (len(self.coords[0]),len(self.coords[1])) != data.shape:
            Exception("data array shape does not match extent shape")

    def get_data(self,coords):
        lat_idx = self.coords[0].get_index(coords[1])
        lon_idx = self.coords[1].get_index(coords[2])
        if len(self.data.shape) == 1:
            return self.data
        return self.data[lat_idx,lon_idx]

    def get_dataspec(self):
        return DataSpec('array',['latitude','longitude'],np.float64)

class ForcingGapFiller(InputNode):
    def __init__(self,path,pattern,nc_var,filler_fn):
        self.nc_var = nc_var
        self.force = SplitFileReaderNode(path,pattern,nc_var)
        self.filler = ClimatologyNode(filler_fn,nc_var)

    def get_data_flat(self,coords,mask):
        d = self.force.get_data_flat(coords,mask)
        gaps = np.isnan(d)
        if gaps.any(): ### there are gaps
            fill = self.filler.get_data_flat(coords,mask)
            if gaps.all():
                d = fill
            else:
                # d[gaps] = fill[gaps]
                d = np.where(gaps,fill,d)
        return d

    def get_dataspec(self):
        return self.force.get_dataspec()

class ClimatologyNode(InputNode):
    def __init__(self,filename,variable,preload=False):
        import netCDF4 as nc
        self.fh = nc.Dataset(filename,'r')
        lats,lons = self.fh.variables['latitude'][...], self.fh.variables['longitude'][...]
        self.cs = mt.CoordinateSet(mt.latlon_to_coords(lats,lons))
        if preload:
            self.v = self.fh.variables[variable][...]
        else:
            self.v = self.fh.variables[variable]

    def get_data(self,coords):
        ### convert time to doy (dont forget 29 feb)
        ### accumulate yearwise
        data = []
        for y in np.unique(coords[0].index.year):
            t = [datetime.datetime(2000,ts.month,ts.day).timetuple().tm_yday - 1 for ts in coords[0][coords[0].index.year == y]]
            idx = self.cs.get_index([coords.latitude,coords.longitude])
            idx = (t,idx[0],idx[1])
            data.append(self.v[idx])

        data = np.concatenate(data).reshape(coords.shape)
        return data

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.v.dtype
        return DataSpec('array',dims,dtype)

class SpatialHDF5Node(InputNode):
    def __init__(self,filename,variable,preload=False):
        self.filename = filename
        self.variable = variable
        self.preload = preload

        import h5py
        self.fh = h5py.File(filename,'r')

        lats,lons = self.fh['dimensions/latitude'][...], self.fh['dimensions/longitude'][...]
        self.cs = mt.CoordinateSet(mt.latlon_to_coords(lats,lons))
        if preload:
            self.v = self.fh[variable][...]
        else:
            self.v = self.fh[variable]
        self.dtype = self.v.dtype

    def close(self):
        self.fh.close()

    def get_data(self,coords):
        idx = self.cs.get_index([coords.latitude,coords.longitude])
        return self.v[idx].reshape(coords.shape[1:])

    def get_dataspec(self):
        return DataSpec('array',['latitude','longitude'],self.dtype)

class HypsoHDF5Node(SpatialHDF5Node):
    def __init__(self,filename,variable,preload=False):
        SpatialHDF5Node.__init__(self,filename,variable,preload)

    def get_data(self,coords):
        idx = self.cs.get_index([coords.latitude,coords.longitude])
        data = self.v[:,idx[0],idx[1]]
        return data.reshape((data.shape[0],coords.shape[1],coords.shape[2]))

    def get_dataspec(self):
        return DataSpec('array',['hypsometric_percentile','latitude','longitude'],self.dtype)

class ConstHDF5Node(InputNode):
    def __init__(self,filename,variable,dims):
        import h5py
        fh = h5py.File(filename,'r')
        if len(dims) != len(fh[variable].shape):
            raise Exception("Dimension list does not match shape of data")
        self.value = fh[variable][...]
        self.dims = dims
        fh.close()

    def get_data(self,coords):
        return self.value

    def get_data_flat(self,coords,mask):
        return self.value

    def get_dataspec(self):
        return DataSpec('array',self.dims,self.value.dtype)

class OutputNode(InputNode):
    def __init__(self,var_name,dtype=np.float64):
        self.var_name = var_name
        self.dtype = dtype

    def set_data(self, data):
        self.data = data

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        return DataSpec('array',dims,self.dtype)

class WriterNode:
    def get_dataspec(self):
        raise NotImplementedError

    def get_dataspec_flat(self):
        dspec = self.get_dataspec()
        if dspec.valtype == 'array':
            if list(dspec.dims[-2:]) == ['latitude','longitude']:
                dspec.dims = list(dspec.dims[:-2]) + ['cell']
        return dspec

class FileWriterNode(WriterNode):
    def __init__(self,path,nc_var,mode='r+',extent=None):
        self.file_manager_type = FlatFileManager
        self.fm = None #SplitFileManager.open_existing(path,pattern,nc_var)

        self.path = path
        self.nc_var = nc_var
        self._file_mode = mode
        self.coords = None
        if extent is None:
            self.coords = extent_to_spatial_coords(default())
        else:
            self.coords = extent_to_spatial_coords(extent)
        self._init = True
        self._open = True

    def close(self):
        self.fm.close_all()

    def open_files(self): #period,extent):
        if self._open:
            # self.fm = self.file_manager_type.open_existing(self.path,self.nc_var+'.nc',self.nc_var,mode='r+')
            self.fm = AnnualSplitFileManager.open_existing(self.path,self.nc_var+'.nc',self.nc_var,mode='r+')
            self._open = False

    def init_files(self,time_coords): #period,extent):
        # if self._init:
        v = mt.Variable(self.nc_var,'mm')
        mvar = mt.MappedVariable(v,(time_coords,self.coords[0],self.coords[1]),np.float32) #in_dtype)

        self.fm = self.file_manager_type(self.path,mvar)

        clobber = self._file_mode == 'w'
        self.fm.create_files(False,clobber,chunksize=(1,16,16)) #None) #(64,32,32)) #(1,16,16))

        self._open = True

    def write_data(self,coords,data):
        data = data.filled(-999.)
        self.fm.write_by_coords(coords,data)

    def sync_all(self):
        for f in self.fm.datasetmanager_map.values():
            f.ncd_group.sync()

    def process(self,coords,data):
        self.open_files()
        self.write_data(coords,data)

    def get_dataspec(self):
        dims = ['time','latitude','longitude']
        dtype = self.fm.mapped_var.dtype
        return DataSpec('array',dims,dtype)

class SplitFileWriterNode(FileWriterNode):
    def __init__(self,path,nc_var,mode='w',extent=None):
        super().__init__(path,nc_var,mode,extent)
        self.file_manager_type = AnnualSplitFileManager

    def open_files(self): #period,extent):
        if self._open:
            self.fm = self.file_manager_type.open_existing(self.path,self.nc_var+'_*.nc',self.nc_var,mode='r+')
            self._open = False

def transform(tfunc,inputs,func_args=None):
    if isinstance(inputs,str):
        inputs = [inputs]
    if func_args is None:
        func_args = {}
    if callable(tfunc):
        tfunc = callable_to_funcspec(tfunc)
    elif isinstance(tfunc,str):
        module,name = tfunc.rsplit('.',1)
        tfunc = dict(module=module,name=name)
    if not (isinstance(tfunc,dict) and 'name' in tfunc and 'module' in tfunc):
        raise Exception("Invalid transformation function")
    return GraphNode('transform',callable_to_funcspec(TransformNode),'from_inputs',inputs,args={'tfunc': tfunc, 'func_args': func_args})

def _arithmetic(lhs,rhs,op):
    return transform(np.__getattribute__(op),[lhs,rhs])    

def _mul3(a,b,c):
    return a*b*c

def _mul4(a,b,c,d):
    return a*b*c*d

def _add3(a,b,c):
    return a+b+c

def _add4(a,b,c,d):
    return a+b+c+d

def _mix(lhs,rhs,ratio):
    return (1.0-ratio)*lhs + ratio*rhs

_muls = {2:np.multiply,3:_mul3,4:_mul4}
_adds = {2:np.add,3:_add3,4:_add4}

def add(*args):
    return transform(_adds[len(args)],args)  

def sub(lhs,rhs):
    return transform(np.subtract,[lhs,rhs]) 

def mul(*args):
    return transform(_muls[len(args)],args)  

def div(lhs,rhs):
    return transform(np.divide,[lhs,rhs]) 

def mix(lhs,rhs,ratio):
    return transform(_mix,[lhs,rhs],dict(ratio=ratio))

def isnan(a):
    return transform(np.isnan,[a])

def _where(a,b):
    # return np.where(np.isnan(a),b,a)
    idx = np.isnan(a)
    a[idx] = b[idx]
    return a

def where(a,b):
    return transform(_where,[a,b])

def average(a,b):
    return (a + b) / 2.

def forcing_from_dict(data_map,nc_var,dims):
    return GraphNode('forcing_from_dict',callable_to_funcspec(StaticForcing),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=dict(data_map=data_map,nc_var=nc_var,dims=dims))

def forcing_from_ncfiles(path,pattern,nc_var,cache=False):
    return GraphNode('forcing_from_ncfiles',callable_to_funcspec(SplitFileReaderNode),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=dict(path=path,pattern=pattern,nc_var=nc_var,cache=cache))

def forcing_gap_filler(nc_path,pattern,nc_var,climatology_file):
    return GraphNode('forcing_with_climatology',callable_to_funcspec(ForcingGapFiller),out_type='forcing',inputs=None,properties={'io': 'r'},\
             args=dict(path=nc_path,pattern=pattern,nc_var=nc_var,filler_fn=climatology_file))

def init_state_from_ncfile(path,pattern,nc_var):
    return GraphNode('init_state_from_ncfile',callable_to_funcspec(StateFileReaderNode),out_type='init',inputs=None,properties={'io': 'r'},\
             args=dict(path=path,pattern=pattern,nc_var=nc_var))

def init_state_from_array(data,extent=None):
    return GraphNode('init_state_from_array',callable_to_funcspec(InitStateFromArray),out_type='init',inputs=None,properties={'io': 'r'},\
             args=dict(data=data,extent=extent))

def spatial_from_hdf5(filename,variable,preload=False):
    return GraphNode('spatial_from_hdf5',SpatialHDF5Node,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=dict(filename=filename,variable=variable,preload=preload))

def hypso_from_hdf5(filename,variable,preload=False):
    return GraphNode('hypso_from_hdf5',HypsoHDF5Node,out_type='spatial',inputs=None,properties={'io': 'r'},\
             args=dict(filename=filename,variable=variable,preload=preload))

def climatology(filename,variable,preload=False):
    return GraphNode('climatology',ClimatologyNode,out_type='from_inputs',inputs=None,properties={'io': 'r'},\
             args=dict(filename=filename,variable=variable,preload=preload))

def const_from_hdf5(filename,variable,dims):
    return GraphNode('const_from_hdf5',ConstHDF5Node,out_type='const',inputs=None,properties={},\
         args=dict(filename=filename,variable=variable,dims=dims))

def const(value,**kwargs):
    return GraphNode('constant',ConstNode,out_type='const',inputs=None,properties=kwargs,args=dict(value=value))

def assign(from_node):
    return GraphNode('assign',AssignNode,out_type='from_inputs',inputs=[from_node],properties={})

def write_to_ncfile(path,nc_var,mode='r+',extent=None):
    '''
    save output to a netCDF4 file
    :param path: save path
    :param nc_var: name of variable
    :param mode: 'w' clobber or 'a' or 'r+' append
    :param extent: initialise to this spatial extent or defaults to awrams.utils.extents.default
    :return: GraphNode
    '''
    return GraphNode('write_to_ncfile',callable_to_funcspec(FileWriterNode),out_type='ncfile',inputs=[nc_var],
                     properties={'io': 'w'},args=dict(path=path,nc_var=nc_var,mode=mode,extent=extent))

def write_to_annual_ncfile(path,nc_var,mode='w',extent=None):
    '''
    save output to annually split netCDF4 files
    :param path: save path
    :param nc_var: name of variable
    :param mode: 'w' clobber or 'a' or 'r+' append
    :param extent: initialise to this spatial extent or defaults to awrams.utils.extents.default
    :return: GraphNode
    '''
    return GraphNode('write_to_annual_ncfile',callable_to_funcspec(SplitFileWriterNode),out_type='ncfile',inputs=[nc_var],
                     properties={'io': 'w'},args=dict(path=path,nc_var=nc_var,mode=mode,extent=extent))

def model_output(var_name):
    return GraphNode('output_variable',OutputNode,out_type=None,inputs=None,properties={'io': 'from_model'},
             args=dict(var_name=var_name))

def persist(var_name):
    return GraphNode('output_variable',OutputNode,out_type='save',inputs=[var_name],properties={'io': 'from_model'},
             args=dict(var_name=var_name))

def init_states_from_dict(imap,data_map,extent=None):
    '''
    replace existing initial state mappings with ndarrays in data_map
    :param mapping:
    :param data_map:
    :return:
    '''
    mapping = imap.mapping
    for k in data_map:
        mapping[k] = init_state_from_array(data_map[k],extent=extent)