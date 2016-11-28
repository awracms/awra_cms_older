from awrams.utils.mapping_types import *
import awrams.utils.datetools as dt
import pandas as pd
from awrams.utils.metatypes import ObjectDict as o
import os, shutil
#import netCDF4 as ncd
from collections import OrderedDict
from awrams.utils.io import open_append
from awrams.utils.helpers import quantize

from awrams.utils.messaging.buffers import shm_as_ndarray

# +++ Tidy namespace
from awrams.utils.messaging.general import *


from awrams.utils.io.netcdf_wrapper import NCVariableParameters

from awrams.utils.settings import DB_OPEN_WITH #pylint: disable=no-name-in-module

from awrams.utils.awrams_log import get_module_logger as _get_module_logger
logger = _get_module_logger('data_mapping')

import glob

from awrams.utils.io import db_open_with
db_opener = db_open_with()
import awrams.utils.io.db_helper as dbh

#+++
#Need this because occasionally python 2.7
#mis-initialises open-ended slices

PY_INTMAX = 2**63 - 1


class OutputWriter(object):

    def __init__(self):
        '''
        Coordinates is the complete span of data which this object will be
        responsible for, consisting of an ordered set of Coordinates objects

        '''
        pass

    def write(self,data,indices):
        '''
        write data to the indices specified
        the indices will be within the range of coordinates supplied to the constructor,
        but not specified in coordinate units (ie they will be integers or integer slices,
        not lat/lon/datetime etc)
        '''
        raise Exception("Not defined")



class BufferedRowWriter:
    '''
    Gathers cells into rows then submits for writing.
    +++ Assumes cells are in order (in particular, that all cells for a row arrive before
        any cells for new rows)
    '''
    def __init__(self,ident,extent,buffer_manager,write_queue,control_q,fill_value=-999.):

        self.q = write_queue
        self.control_q = control_q
        self.ident = ident

        self.extent = extent

        self.buffer_manager = buffer_manager

        self.cur_idx = -1
        self.buffer_id = -1

        self.cells_done = 0

        self.fill_value = fill_value

    def finalise(self):
        if self.cur_idx != -1:
            self.flush_to_queue()

    def get_buffer(self):
        self.buffer_id, self.cur_data = self.buffer_manager.get_buffer()
        #self.cur_data[:] = fill_value
        row_mask = self.extent.mask[self.cur_idx] == True
        self.cur_data[:,row_mask] = self.fill_value

    def flush_to_queue(self):
        self.q.put(message('write', id= self.ident, row_idx = index[self.cur_idx], buffer_shape = self.cur_data.shape, buffer_id = self.buffer_id, read_idx = index[0:self.cur_p_len,:]))

    def process_cell(self,data,cell):
        '''
        Process a single cell's timeseries data
        '''
        #+++
        #Correct place to localise cell?
        #Probably yes - our buffers are in 'local' data coords

        lcell = self.extent.localise_cell(cell)
        self[0:self.cur_p_len,lcell[0],lcell[1]] = data

    def set_active_period(self,period):
        self.cur_idx = -1
        self.cur_p_len = len(period)

        #+++ to deal with periods that don't span end of month which causes states period to be empty
        if self.cur_p_len > 0:
            start = period[0]
            end = period[-1]
            if isinstance(start,pd.Period):
                # UGLY UGLY HACK to work around Period objects becoming
                # unpickleable in pandas 0.16
                start = (start.to_timestamp(),start.freq)
                end = (end.to_timestamp(),end.freq)
            self.q.put(message('set_period', id= self.ident, start= start, end= end))

    def __setitem__(self,idx,data):
        if idx[1] != self.cur_idx:
            if self.cur_idx != -1:
                self.flush_to_queue()
            self.cur_idx = idx[1]
            self.get_buffer()
        self.cur_data[idx[0],idx[2]] = data


class SplitIndexOutputWriter(OutputWriter):
    '''
    Base composable class for writing split outputs
    '''
    def __init__(self):
        self.segment_writer_map = OrderedDict()
        self.splitter = None

    def process(self,data,coords):
        data_map = self.splitter.split_data_coords(data,coords)

        for segment in data_map:
            writer = self.segment_writer_map[segment.segment]

            writer[segment.index] = segment.data

    def finalise(self):
            for writer in list(self.segment_writer_map.values()):
                writer.finalise()

class Segment(object):
    '''
    Represents a portion of some global coordinate set
    '''
    def __init__(self,coordinates):
        self.coordinates = coordinates

class Segment1D(Segment):
    '''
    Segment bounded in a single dimension
    '''
    def __init__(self,coordinates,start,end):
        Segment.__init__(self,coordinates)
        self.start = start
        self.end = end

    def __repr__(self):
        return "[%s - %s]:\n%s" % (self.start,self.end,self.coordinates.__repr__())

    def __len__(self):
        return (self.end-self.start)+1

class DimensionalSplitter(object):
    '''
    Base class for splitting data access based
    on boundaries accross a single dimension of the input
    '''
    def __init__(self,coordinates,strategy):
        '''
        coordinates: CoordinateSet to be split
        strategy: string identifying splitting strategy
        '''
        self.segments = []
        self.dim_pos = None
        self.dim_len = 0
        self._build_segments(coordinates)
        self.coordinates = coordinates
        self.strategy = strategy

    def set_coordinates(self,coordinates):
        self.coordinates = coordinates
        self._build_segments(coordinates)

    def _build_segments(self,coordinates):
        '''
        Create the segments for a given CoordinateSet
        Each segment will map to a particular set of indices,
        and be uniquely identifiable.
        '''
        raise Exception("Not implemented")

    def _locate_segment(self,index):
        '''
        index is single integer, return the position in the segment array,
        and the segment
        '''
        for i, segment in enumerate(self.segments):
            if index >= segment.start and index <= segment.end:
                return i, segment

        raise IndexError("Index %i out of range" % index)

    def split_coords_to_indices(self,coordinates):
        return self.split_indices(self.coordinates.get_index(coordinates))

    def split_indices(self,indices):
        '''
        For a given set of (global) indices, return a list of
        (segment,local_index,global_index) dicts, where local index
        maps to the segment, and global_index to a block of data
        the shape of indices, but normalised to start at 0
        '''

        if type(indices) in [slice,int]:
            indices=[indices]


        dim_idx = indices[self.dim_pos]

        if type(dim_idx) == int:
            _,seg = self._locate_segment(dim_idx)
            out_index = self._substitute_index(indices,dim_idx - seg.start)
            return [o(segment=seg,local_index=out_index,global_index=indices)]
        if type(dim_idx) == slice:

            if dim_idx.step != None:
                raise Exception("Stepped slices unsupported")

            out_indices = []

            idx_start = dim_idx.start if dim_idx.start != None else 0
            idx_stop = (dim_idx.stop - 1) if dim_idx.stop not in [PY_INTMAX,None] else self.dim_len-1

            cur_i,cur_seg = self._locate_segment(idx_start)
            cur_start = idx_start
            cur_idx = cur_start + 1
            while cur_idx <= idx_stop:
                if cur_idx > cur_seg.end:
                    local_slice = self._substitute_index(indices,slice(cur_start-cur_seg.start,cur_idx-cur_seg.start))
                    g_i = self._normalize_index(indices)
                    global_slice = self._substitute_index(g_i,slice(cur_start-idx_start,cur_idx-idx_start))
                    out_indices.append(o(segment=cur_seg,local_index=local_slice,global_index=global_slice))
                    cur_i += 1
                    cur_seg = self.segments[cur_i]
                    cur_start = cur_idx
                cur_idx += 1
            local_slice = self._substitute_index(indices,slice(cur_start-cur_seg.start,cur_idx-cur_seg.start)) #-1 ?
            g_i = self._normalize_index(indices)
            global_slice = self._substitute_index(g_i,slice(cur_start-idx_start,cur_idx-idx_start))# -1?
            out_indices.append(o(segment=cur_seg,local_index=local_slice,global_index=global_slice))

            return out_indices


    def _substitute_index(self,indices,i_to_sub):
        return substitute_index(indices,i_to_sub,self.dim_pos)

    def _normalize_index(self,indices):
        '''
        return indices of the same shape, but with slices beginning at 0
        '''
        out_idx = []
        for i, idx in enumerate(indices):
            if type(idx) == slice:
                out_idx.append(normalize_slice(idx,len(self.coordinates[i])))
            else:
                out_idx.append(idx)
        return tuple(out_idx)

    def split_data_coords(self,data,coords):
        '''
        For a given block of data whose shape matches the supplied
        coords, return a list of (segment,index,data) dicts, where index
        is localised to the segment
        '''
        return self.split_data(data, self.coordinates.get_index(coords))


    def split_data(self,data,indices):
        '''
        For a given block of data whose shape matches the supplied
        indices, return a list of (segment,index,data) dicts, where index
        is localised to the segment
        '''
        if isinstance(indices,CoordinateSet):
            indices = self.coordinates.get_index(indices)

        seg_indices = self.split_indices(indices)

        out_data = []

        for seg_i in seg_indices:
            seg_data = data[simplify_indices(seg_i.global_index)]
            out_split = o(segment=seg_i.segment,index=seg_i.local_index,data=seg_data)
            out_data.append(out_split)

        return out_data

class AnnualSplitter(DimensionalSplitter):
    '''
    Splits data access across annual boundaries
    '''

    def __init__(self,coordinates):
        DimensionalSplitter.__init__(self,coordinates,'annual')

    def split_dti(self,dti):
        return dt.split_period(dti,'a')

    def _build_segments(self,coordinates):
        '''
        Builds segments by splitting accross annual boundaries
        '''
        self.segments = []

        dim_i, time = find_time_coord(coordinates)

        self.dim_len = len(time)
        last_idx = self.dim_len - 1

        self.dim_pos = dim_i

        dti_time = time.index

        year_c = (dti_time[-1].year - dti_time[0].year) + 1

        def build_coords(start,end):
            new_time = TimeCoordinates(time.dimension,time.index[start:end+1])
            new_coords = substitute_index(coordinates,new_time,dim_i)
            return CoordinateSet(new_coords)

        if year_c == 1:
            self.segments.append(Segment1D(coordinates,0,last_idx))
        else:
            cur_start_idx = 0
            cur_year = dti_time[0].year
            cur_boundary = dt.end_of_year(cur_year)

            for i, ts in enumerate(dti_time):
                if ts > cur_boundary:
                    start,end = cur_start_idx,i-1
                    self.segments.append(Segment1D(build_coords(start,end),start,end))
                    cur_start_idx = i
                    cur_year = ts.year #+= 1
                    cur_boundary = dt.end_of_year(cur_year)
            start,end = cur_start_idx,last_idx
            self.segments.append(Segment1D(build_coords(start,end),start,end))

class MonthlySplitter(DimensionalSplitter):
    '''
    Splits data access across monthly boundaries
    '''

    def __init__(self,coordinates):
        DimensionalSplitter.__init__(self,coordinates,'month')

    def split_dti(self,dti):
        return dt.split_period(dti,'m')

    def _build_segments(self,coordinates):
        '''
        Builds segments by splitting accross annual boundaries
        '''
        self.segments = []

        dim_i, time = find_time_coord(coordinates)

        self.dim_len = len(time)
        last_idx = self.dim_len - 1

        self.dim_pos = dim_i

        dti_time = time.index

        def build_coords(start,end):
            new_time = TimeCoordinates(time.dimension,time.index[start:end+1])
            new_coords = substitute_index(coordinates,new_time,dim_i)
            return CoordinateSet(new_coords)

        cur_start_idx = 0

        cur_boundary = dt.end_of_month(dti_time[0]) #end_of_year(cur_year)

        for i, ts in enumerate(dti_time):
            if hasattr(ts,'to_timestamp'):
                ts = ts.to_timestamp()
            if ts > cur_boundary:
                start,end = cur_start_idx,i-1
                self.segments.append(Segment1D(build_coords(start,end),start,end))
                cur_start_idx = i
                cur_boundary = dt.end_of_month(ts)
        start,end = cur_start_idx,last_idx
        self.segments.append(Segment1D(build_coords(start,end),start,end))

class FlatFileSplitter(DimensionalSplitter):
    '''
    "Splitter" that really writes to a flat file (transparently)
    '''
    def __init__(self,coordinates):
        DimensionalSplitter.__init__(self,coordinates,'flat')

    def split_dti(self,dti):
        return [dti]

    def _build_segments(self,coordinates):
        dim_i, time = find_time_coord(coordinates)

        last_idx = len(time)- 1

        self.dim_pos = dim_i

        self.segments = [Segment1D(coordinates,0,last_idx)]


def find_time_coord(coordinates):
    '''
    Locate the time dimension in a set of coordinates
    '''
    for i, coord in enumerate(coordinates):
        if isinstance(coord.dimension,TimeDimension):
            return i, coord
    raise Exception("No time dimension found in coordinate set")

def substitute_index(indices,index_to_sub,position):
    '''
    Replace the item in <position> with index_to_sub, return a tuple
    '''
    out_index = []
    for i in range(0,len(indices)):
        if i == position:
            out_index.append(index_to_sub)
        else:
            out_index.append(indices[i])
    return tuple(out_index)

def simplify_indices(indices):
    '''
    Reduce dimensionality if any of the indices are size-1
    '''
    out_index = []
    for i in range(0,len(indices)):
        cur_index = indices[i]
        if type(cur_index) != int:
            out_index.append(cur_index)
    return tuple(out_index)

def normalize_slice(s,dim_len):
    '''
    normalize a slice to start at 0, where dim_len is the max value
    '''
    if s.start == None:
        return s
    else:
        offset = s.start
        stop = dim_len - offset if s.stop == None else s.stop - offset
        return slice(0,stop,s.step)



class SplitArrayOutputWriter(SplitIndexOutputWriter):
    '''
    In this instance, target_arrays should have been generated
    from the same splitter instance, to guarantee 1:1 mapping
    (or the splitter constructed from the source array dimensions)
    '''
    def __init__(self,target_arrays,splitter):
        SplitIndexOutputWriter.__init__(self)
        self.splitter = splitter

        for t, seg in zip(target_arrays,splitter.segments):
            self.segment_writer_map[seg] = t

    def get_by_index(self,indices):
        seg_indices = self.splitter.split_indices(indices)

        out_data = []
        for seg_i in seg_indices:
            seg_data = self.segment_writer_map[seg_i.segment][seg_i.local_index]
            if len(seg_data.shape) == 0:
                seg_data = [seg_data]
            out_data.append(seg_data)
        return np.ma.concatenate(out_data)

    def get_by_coords(self,coords):
        indices = self.splitter.coordinates.get_index(coords)
        return self.get_by_index(indices)

    def set_by_coords(self,coords,data):
        # index = self.splitter.coordinates.get_index(coords)
        return self.process(data,coords)

    def set_by_index(self,index,data):
        return self.process(data,index)

    def finalise(self):
        pass

### DEPRECATED
#class BufferedSplitArrayOutputWriter(SplitIndexOutputWriter):
#    def __init__(self,var_name,target_fns,splitter,write_queue,buffer_manager):
#        SplitIndexOutputWriter.__init__(self)
#        self.splitter = splitter
#
#        for fn, seg in zip(target_fns,splitter.segments):
#            blockshape = (seg.coordinates.shape[0],seg.coordinates.shape[2])
#            self.segment_writer_map[seg] = BufferedRowWriter(fn,blockshape,write_queue,buffer_manager)

class IndexGetter:
    '''
    Helper class for using index creation shorthand
    eg IndexGetter[10:,5] returns [slice(10,None),5]
    '''
    def __getitem__(self,indices):
        return indices

class Indexer:
    '''
    Wrapper class that refers it's get/set item methods to another function
    '''
    def __init__(self,getter_fn,setter_fn = None):
        self.getter_fn = getter_fn
        self.setter_fn = setter_fn

    def __getitem__(self,idx):
        return self.getter_fn(idx)

    def __setitem__(self,idx,value):
        return self.setter_fn(idx,value)

index = IndexGetter()

def filter_years(period):
    from re import match

    years = np.unique(period.year)
    def ff(x):
        m = match('.*([0-9]{4})',x)
        try:
            return int(m.groups()[0]) in years
        except:
            return False
    return ff


def split_padding(period,avail):
    try:
        avail_start = avail[0].to_timestamp()
        avail_end = avail[-1].to_timestamp()
    except AttributeError:
        avail_start = avail[0]
        avail_end = avail[-1]

    if period[0] < avail_start:
        prepad = (period[0], min(period[-1],avail_start-1))
    else:
        prepad = None
    if period[-1] > avail_end:
        postpad = (max(period[0],avail_end+1), period[-1])
    else:
        postpad = None
        
    actual = max(avail_start,period[0]),min(avail_end,period[-1])
        
    if period[-1] < avail_start:
        actual = None
    if period[0] > avail_end:
        actual = None
        
    return prepad,actual,postpad

def index_shape(idx):
    shape = []
    for i in idx:
        if isinstance(i,slice):
            shape.append(i.stop-i.start)
        else:
            shape.append(1)
    return tuple(shape)

def simple_shape(shape):
    return tuple([s for s in shape if s > 1])

def desimplify(target):
    indices = []
    for i,s in enumerate(target):
        if s == 1:
            indices.append(0)
        else:
            indices.append(np.s_[:])
    return indices

class SplitFileManager:
    def __init__(self,path,mapped_var):
        '''
        Manage NetCDF data persistence for a given variable
        '''

        self.mapped_var = mapped_var
        self.path = path
        self.file_map = OrderedDict()
        self.datasetmanager_map = OrderedDict()
        self.array_map = OrderedDict()
        self.get_by_coords = Indexer(self._get_by_coords,self._set_by_coords)

    def locate_day(self,day):
        t_idx = self.splitter.coordinates.time.get_index(day)
        idx, seg = self.splitter._locate_segment(t_idx)
        return self.file_map[seg]

    def get_fn_ds_map(self):
        return dict(var_name=self.mapped_var.variable.name, splitter=self.splitter, file_map=self.file_map)

    def get_frequency(self):
        return self.splitter.coordinates.time.index.freq

    def get_period_map_multi(self,periods):
        '''
        Return the period mapping for a presplit set of periods
        '''
        pmap = {}

        def find_period(p,tfm):
            for k,v in tfm.items():
                try:
                    idx = v.get_index(p)
                    return k,idx
                except:
                    pass
            raise IndexError()

        for i,p in enumerate(periods):
            #+++
            #Don't set when file doesn't exist, also catches
            #0 length periods... still, might hide other exceptions?
            try:
                #s_idx = self.splitter.coordinates.time.get_index(p[0])
                #idx,seg = self.splitter._locate_segment(s_idx)
                #t_idx = seg.coordinates.time.get_index(p)
                #fname = self.file_map[seg]
                fname,t_idx = find_period(p,self.time_file_map)
                pmap[i] = {}
                pmap[i]['filename'] = fname
                pmap[i]['time_index'] = t_idx
            except IndexError:
                pass
        return pmap

    def get_period_map(self,period):
        '''
        Return the period mapping for the given period (splitting if needed)
        '''
        periods = dt.split_discontinuous_dti(period)
        all_periods = []
        for p in periods:
            all_periods += self.splitter.split_dti(p)
        return self.get_period_map_multi(all_periods), all_periods

    def get_chunked_periods(self,chunksize):
        '''
        Return (ordered) DatetimeIndices of all the HDF chunks contained in the dataset
        '''
        chunk_p = []
        for v in self.time_file_map.values():
            chunk_p += dt.split_period_chunks(v,chunksize)
        chunk_p.sort(key = lambda x: x[0])
        return chunk_p

    @classmethod
    def open_existing(self,path,pattern,variable,mode='r',ff=None):
        '''
        classmethod replacement for open_files
        '''
        sfm = SplitFileManager(None,None)
        sfm.open_files(path,pattern,variable,mode=mode,ff=ff)
        return sfm

    def open_files(self,path,pattern,variable,mode='r',ff=None):

        var_name = variable if isinstance(variable,str) else variable.name

        search_pattern = os.path.join(path,pattern)
        files = glob.glob(search_pattern)
        files.sort()

        if ff is None:
            def ff(x):
                return True

        _files = []
        for f in files:
            if ff(f):
                _files.append(f)
        files=_files

        if len(files) == 0:
            raise Exception("No files found in %s matching %s" % (path,pattern))

        #dsm_start = DatasetManager(db_opener(files[0],mode))
        dsm_start = DatasetManager(open_append(db_opener,files[0],mode))

        self.ref_ds = dsm_start

        coords = dsm_start.get_coords()

        time = dsm_start.get_coord('time')
        time_idx = time.index

        self.time_file_map = {}
        self.time_file_map[files[0]] = time

        decide_split_strategy = True
        split_strategy = None
        if len(files) > 1:
            for fn in files[1:]:
                dsm = DatasetManager(db_opener(fn,mode))
                self.time_file_map[fn] = t = dsm.get_coord('time')
                time_idx = time_idx.union(t.index)

                ### guess at file split
                if decide_split_strategy:
                    if len(t) > 31:
                        split_strategy = 'a'
                        decide_split_strategy = False
                    elif len(t) == 1:
                        split_strategy = 'd'
                    else:
                        split_strategy = 'm'

        global_coords = CoordinateSet([TimeCoordinates(time.dimension,time_idx),coords.latitude,coords.longitude])
        ncvar = dsm_start.variables[var_name]
        self.fillvalue = ncvar.attrs['_FillValue'][0]

        v = Variable.from_ncvar(ncvar)
        self.mapped_var = MappedVariable(v,global_coords,ncvar.dtype)

        if len(files) == 1:
            self.splitter = FlatFileSplitter(global_coords)
        else:
            if split_strategy == 'a':
                self.splitter = AnnualSplitter(global_coords)
            elif split_strategy == 'm':
                self.splitter = MonthlySplitter(global_coords)
            else:
               raise Exception("splitting strategy %s not implemented" % split_strategy)

        start_seg = self.splitter.segments[0]
        self.file_map[start_seg] = files[0]
        self.datasetmanager_map[start_seg] = dsm_start
        self.array_map[start_seg] = ncvar

        if len(files) > 1:
            for i, fn in enumerate(files[1:]):
                dsm = DatasetManager(db_opener(fn,mode))

                s_idx = self.splitter.coordinates.time.get_index(self.time_file_map[fn].index[0])
                idx,seg = self.splitter._locate_segment(s_idx)
                
                self.file_map[seg] = fn
                self.datasetmanager_map[seg] = dsm
                self.array_map[seg] = dsm.ncd_group.variables[var_name]

        self.accessor = SplitArrayOutputWriter(list(self.array_map.values()),self.splitter)

    def create_files(self,leave_open=True,clobber=True,chunksize=None,file_creator=None,file_appender=None,create_dirs=True,**kwargs):
        '''
        kwargs are propagated to NCVariableParameters
        '''
        
        if file_creator is None:
            def create_new_nc(fn):
                import netCDF4 as ncd
                try:
                    return ncd.Dataset(fn,'w')
                except RuntimeError:
                    from awrams.utils.io.general import h5py_cleanup_nc_mess
                    h5py_cleanup_nc_mess(fn)
                    return ncd.Dataset(fn,'w')
                #return db_opener(fn,'w')
            file_creator = create_new_nc

        if file_appender is None:
            def append_nc(fn):
                # return ncd.Dataset(fn,'a')
                try:
                    #return db_opener(fn,'a')
                    return open_append(db_opener,fn,'a')
                except:
                    logger.critical("EXCEPTION: %s",fn)
                    raise
            file_appender = append_nc

        if create_dirs:
            os.makedirs(self.path,exist_ok=True)

        # Examine first existing file to see if we need to extend the coordinates
        if clobber == False:
            seg = self.splitter.segments[0]
            fn = os.path.join(self.path,self.gen_file_name(seg))

            if os.path.exists(fn):
                ds = file_appender(fn)
                dsm = DatasetManager(ds)
#                logger.info("filename %s",fn)
                existing_coords = dsm.get_coords()

                if 'time' in existing_coords:
                    #+++
                    # Could definitely generalise this to autoexpand in
                    # other dimensions, hardcoding for time being the 'normal' case...

                    seg_time = seg.coordinates.time

                    existing_time = dsm.get_coord('time').index
                    extension_time = seg_time.index

                    new_seg_tc = period_to_tc(existing_time.union(extension_time))

                    global_extension = self.mapped_var.coordinates.time.index

                    new_global_tc = period_to_tc(existing_time.union(global_extension))

                    dsm.set_time_coords(new_seg_tc)

                    self.mapped_var.coordinates.update_coord(new_global_tc)

                ds.close()
                self.splitter.set_coordinates(self.mapped_var.coordinates)

        for seg in self.splitter.segments:
            fn = os.path.join(self.path,self.gen_file_name(seg))
            self.file_map[seg] = fn

            new_file = True

            if os.path.exists(fn) and not clobber:
                ds = file_appender(fn)
                dsm = DatasetManager(ds)

                if 'time' in seg.coordinates:
                    if len(seg.coordinates.time) > len(ds.variables['time']):
                        dsm.set_time_coords(seg.coordinates.time, resize=True)

                new_file = False
            else:
                # if writing provenance, create / copy template
                ds = file_creator(fn)# ncd.Dataset(fn,'w')

            dsm = DatasetManager(ds)

            '''
            Separate into function ('createCoordinates'?)
            Possibly removing any (direct) reference to netCDF
            '''
            if new_file:
                for coord in seg.coordinates:
                    dsm.create_coordinates(coord)

                from awrams.utils.io.netcdf_wrapper import NCVariableParameters

                if chunksize is None:
                    from awrams.utils.settings import CHUNKSIZE as chunksize #pylint: disable=no-name-in-module

                chunksizes = seg.coordinates.validate_chunksizes(chunksize)
                ncd_params = NCVariableParameters(chunksizes=chunksizes,**kwargs)

                target_var = dsm.create_variable(self.mapped_var,ncd_params)

                dsm.ncd_group.setncattr('var_name',self.mapped_var.variable.name)

                dsm.set_time_coords(seg.coordinates.time, resize=True)

            ds.close()

        if leave_open:
            self.open_all('a')

        self.time_file_map = {}
        for seg in self.splitter.segments:
            fn = self.file_map.get(seg)
            if fn is not None:
                self.time_file_map[fn] = seg.coordinates.time

    def __getitem__(self,idx):
        return self.accessor.get_by_index(idx)

    def _get_by_coords(self,idx):
        return self.accessor.get_by_coords(idx)

    def _set_by_coords(self,idx,value):
        return self.accessor.set_by_coords(idx,value)

    def cell_for_location(self,location):
        lat = quantize(location[0],0.05)
        lon = quantize(location[1],0.05)
        lat_i = self.splitter.coordinates.latitude.get_index(lat)
        lon_i = self.splitter.coordinates.longitude.get_index(lon)
        return (lat_i,lon_i)

    def get_data(self,period,extent):
        '''
        Return a datacube as specified in time and space
        '''
        return self._get_by_coords(gen_coordset(period,extent))

    def retrieve(self,daterange,cell):
        '''
        Compatibility for state initialiser
        '''
        t_idx = self.splitter.coordinates.time.get_index(index[daterange[0]:daterange[1]])
        return self.accessor.get_by_index((t_idx,cell[0],cell[1]))

    def finalise(self):
        '''
        Flush all current writes to disk'
        '''
        self.accessor.finalise()

    def open_all(self,mode='r'):
        for seg in self.splitter.segments:
            ds = db_opener(self.file_map[seg],'a')

            if DB_OPEN_WITH == '_h5py':
                ds.set_mdc(dbh.mdc['K32'])

            dsm = DatasetManager(ds)
            self.datasetmanager_map[seg] = dsm

            target_var = ds.variables[self.mapped_var.variable.name]
            self.array_map[seg] = target_var

        self.accessor = SplitArrayOutputWriter(list(self.array_map.values()),self.splitter)

    def close_all(self):
        '''
        Close all open datasets
        '''
        for ds in list(self.datasetmanager_map.values()):
            ds.ncd_group.close()
        self.datasetmanager_map = OrderedDict()

    def gen_file_name(self,segment):
        '''
        Generate a filename for a given segment
        Subclass depending on splitting strategy to generate appropriate names
        '''
        raise Exception("Not implemented")

    def get_padded_by_coords(self,coords):
        cs = self.splitter.coordinates
        period = coords[0]
        actual = cs.time.index
        split = split_padding(period,actual)

        full_shape = coords.shape
        
        #full_data = np.ma.empty(full_shape,dtype=np.float32) #+++ Get from nc_var dtype
        #full_data.mask = True

        full_data = np.empty(full_shape,dtype=np.float32)
        full_data[...] = np.nan
        
        if split[1] is None:
            return full_data

        actual_idx = cs.get_index([np.s_[split[1][0]:split[1][1]],coords[1],coords[2]])

        actual_data = self.accessor.get_by_index(actual_idx)
        if hasattr(actual_data,'mask'):
            actual_data = actual_data.data

        actual_data[actual_data==self.fillvalue] = np.nan
        
        if split[0] is None and split[2] is None:
            return actual_data.reshape(full_shape)
        
        actual_shape = index_shape(actual_idx)

        #full_shape = (len(period),actual_shape[1],actual_shape[2])

        if split[0] is not None:
            dstart = (split[0][1]-split[0][0]).days + 1
        else:
            dstart = 0

        write_index = desimplify(actual_shape)
        write_index[0] = np.s_[dstart:(dstart+actual_shape[0])]

        full_data[write_index] = actual_data[...]
        
        return full_data

    def write_by_coords(self,coords,data):
        write_index = desimplify(data.shape)
        self.accessor.set_by_coords(coords,data[write_index])


class DatasetManager:
    '''
    Convenience wrapper for a NetCDF group or dataset
    The group must exist before instantiating the DatasetManager
    '''
    def __init__(self,ncd_group):
        self.ncd_group = ncd_group
        self._update_dicts()

    def close(self):
        self.ncd_group.close()

    def _update_dicts(self):
        self.variables = o()
        self.groups = o()
        self.attrs = o()
        for v in self.ncd_group.variables:
            self.variables[v] = self.ncd_group.variables[v]
        if hasattr(self.ncd_group, 'var_name'):
            self.awra_var = self.ncd_group.variables[self.ncd_group.var_name]
        else:
            self.awra_var = self._imply_variable()
        for a in self.ncd_group.ncattrs():
            self.attrs[a] = self.ncd_group.getncattr(a)
        for g in self.ncd_group.groups:
            self.groups[g] = DatasetManager(self.ncd_group.groups[g])

    def _imply_variable(self):
        for v in self.variables:
            if not v.endswith('_bounds') and not v in self.ncd_group.dimensions:
                return self.variables[v]

    def create_coordinates(self, coord):
        '''
        Coord is an AWRAMS Coordinates object;
        This function creates both the dimensions and the coordinate variables
        '''
        coord_len = 0 if coord.unlimited else len(coord)
        self.ncd_group.createDimension(coord.dimension.name,coord_len)
        coord_var = self.ncd_group.createVariable(coord.dimension.name,datatype=coord.dimension.dtype,dimensions=[coord.dimension.name])
        coord_var.setncatts(coord.dimension._dump_attrs())
        # Indices may be native python types, use _persistent_index when writing
        coord_var[:] = coord._persist_index()
        if isinstance(coord,BoundedCoordinates):
            if 'nv' not in self.ncd_group.dimensions:
                self.ncd_group.createDimension('nv',2)
            coord_var.setncattr('bounds',coord.dimension.name+'_bounds')
            bounds_var = self.ncd_group.createVariable(coord.dimension.name+'_bounds',datatype='i',dimensions=[coord.dimension.name,'nv'])
            bounds_var[:] = coord._persist_boundaries()
        self._update_dicts()

    def create_variable(self,mapped_var,ncd_params=None,**kwargs):
        '''
        Takes a MappedVariable object, propogates to NetCDF
        Returns the NetCDF variable
        '''
        if ncd_params is None:
            from awrams.utils.io.netcdf_wrapper import NCVariableParameters
            ncd_params = NCVariableParameters()
        ncd_params.update(**kwargs)
        var_dims = [(dim.name) for dim in mapped_var.coordinates.dimensions]

        for dim in var_dims:
            if dim not in list(self.ncd_group.dimensions.keys()):
                self.create_coordinates(mapped_var.coordinates[dim])

        target_var = self.ncd_group.createVariable(mapped_var.variable.name,datatype=mapped_var.dtype,dimensions=var_dims,**ncd_params)
        target_var.setncatts(mapped_var.variable._dump_attrs())
        self._update_dicts()
        return target_var

    def create_group(self,group_name):
        '''
        Create a new group within the current group; return
        '''
        group = self.ncd_group.createGroup(group_name)
        return group

    def get_coords(self):
        coords = []
        for k in list(self.ncd_group.dimensions.keys()):
            if k in list(self.ncd_group.variables.keys()):
                coords.append(self.get_coord(k))
        #coords = [self.get_coord('time'),self.get_coord('latitude'),self.get_coord('longitude')]
        return CoordinateSet(coords)

    def get_coord(self,coord):
        '''
        Get a Coordinates object whose name matches 'coord'
        '''
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc
        from awrams.utils.helpers import aquantize

        def from_epoch(epoch,ts):
            return epoch + dt.days(int(ts))

        if coord == 'time':
            epoch = epoch_from_nc(self.ncd_group)
            time_var = self.ncd_group.variables['time']
            dti = pd.DatetimeIndex([(epoch + dt.days(int(ts))) for ts in self.ncd_group.variables['time'][:]],freq='d')

            if 'bounds' in time_var.ncattrs():
                bounds_var = self.ncd_group.variables[time_var.bounds]
                boundaries = []
                for b in bounds_var[:]:
                    boundaries.append([from_epoch(epoch,b[0]),from_epoch(epoch,b[1]-1)])

                #Attempt to infer period frequency from boundaries...
                p = infer_period(boundaries)

                return BoundedTimeCoordinates(TimeDimension(epoch),dti.to_period(p),boundaries)
            else:
                return TimeCoordinates(TimeDimension(epoch),dti)
        elif coord == 'latitude':
            lat = np.float64(self.ncd_group.variables['latitude'][:])
            if not hasattr(lat,'__len__'):
                lat = np.array([lat])
            return Coordinates(latitude,aquantize(lat,0.05))
            # return Coordinates(latitude,aquantize(np.float64(self.ncd_group.variables['latitude'][:]),0.05))
        elif coord == 'longitude':
            lon = np.float64(self.ncd_group.variables['longitude'][:])
            if not hasattr(lon,'__len__'):
                lon = np.array([lon])
            return Coordinates(longitude,aquantize(lon,0.05))
            # return Coordinates(longitude,aquantize(np.float64(self.ncd_group.variables['longitude'][:]),0.05))
        else:
            ncvar = self.ncd_group.variables[coord]
            if hasattr(ncvar,'units'):
                units = Units(ncvar.units)
            else:
                units = Units('unknown unit')
            dim = Dimension(ncvar.dimensions[0],units,ncvar.dtype)
            return Coordinates(dim,ncvar[:])

    def get_mapping_var(self,full_map=False):
        '''
        Return a mapping_types.Variable object from the netCDF information
        full_map will include coordinates and datatype
        '''
        
        ncvar = self.awra_var

        attrs = dict([[k,ncvar.getncattr(k)] for k in ncvar.ncattrs()])
        out_var = Variable(attrs['name'],attrs['units'],attrs)
        if full_map:
            cs = self.get_coords()
            return MappedVariable(out_var,cs,ncvar.dtype)
        else:
            return out_var

    def set_time_coords(self,time_coords,resize=False):
#        logger.info("set_time_coords %s %s",self.ncd_group.variables['time'].shape, len(time_coords))
        if resize or self.ncd_group.variables['time'].shape[0] < len(time_coords):
            if isinstance(self.ncd_group.variables['time'], dbh.h5pyDataset):
                self.ncd_group.variables['time'].resize((len(time_coords),))
                for var in self.variables.values():
                    if hasattr(var,'dimensions'):
                        ncdims = var.dimensions
                        if 'time' in ncdims:
                            dims = var.dims
                            tdim = None
                            for i, dim in enumerate(dims):
                                if 'time' in dim.keys():
                                    tdim = i
                            if tdim is not None:
                                new_shape = list(var.shape)
                                new_shape[tdim] = len(time_coords)
                                var.resize(new_shape)

                #self.awra_var.resize((len(time_coords),self.awra_var.shape[1],self.awra_var.shape[2]))
#                logger.info("set_time_coords (is h5pyDataset) %s",self.awra_var.shape)
            else:
                if 'time' in self.awra_var.dimensions:
                    self.awra_var[len(time_coords) - 1,0,0] = -999.
#                logger.info("set_time_coords (is nc.variable) %s",self.awra_var.shape)
        self.ncd_group.variables['time'][:] = time_coords._persist_index()

        if 'bounds' in self.ncd_group.variables['time'].ncattrs():
            bounds_var = self.ncd_group.variables[self.ncd_group.variables['time'].getncattr('bounds')]
            if isinstance(self.ncd_group.variables['time'], dbh.h5pyDataset):
                bounds_var.resize((len(time_coords),bounds_var.shape[1]))
            bounds_var[:] = time_coords._persist_boundaries()

    def get_dates(self):
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc

        epoch = epoch_from_nc(self.ncd_group)
        dti = pd.DatetimeIndex([(epoch + dt.days(int(ts))) for ts in self.ncd_group.variables['time']],freq='d')
        return dti

    def get_daterange(self):
        from awrams.utils.io.netcdf_wrapper import epoch_from_nc

        epoch = epoch_from_nc(self.ncd_group)
        time = self.ncd_group.variables['time']
        return [(epoch + dt.days(int(ts))) for ts in (time[0],time[len(time)-1])]

    def get_extent(self,use_mask=True):
        from awrams.utils.extents import from_boundary_coords
        lat,lon = self.get_coord('latitude'), self.get_coord('longitude')

        if use_mask is True:
            ref_data = self.awra_var[0]
            if not hasattr(ref_data,'mask'):
                try: ### maybe using h5py which doesn't return an ma
                    mask = np.ma.masked_values(ref_data, self.awra_var.attrs['_FillValue'][0])
                    mask = mask.mask
                except AttributeError:
                    mask = False
            else:
                mask = ref_data.mask
        else:
            mask = False

        return from_boundary_coords(lat[0],lon[0],lat[-1],lon[-1],compute_areas = False,mask=mask)

    def extract_period(self,period):
        dti = self.get_dates()
        idx = self.idx_for_period(period) #dti.searchsorted(period)
        return self.awra_var[idx,:]

    def idx_for_period(self,period):
        dti = self.get_dates()
        return dti.searchsorted(period)


class AnnualSplitFileManager(SplitFileManager):
    '''
    Manage files split annually
    '''
    def __init__(self,path,mapped_var,timeframe='daily',filebase=None):
        SplitFileManager.__init__(self,path,mapped_var)

        if filebase is None:
            filebase = mapped_var.variable.name

        self.filebase = filebase

        self.splitter = AnnualSplitter(self.mapped_var.coordinates)
        self.timeframe = timeframe

    def gen_file_name(self,segment):
        '''
        Generate var_year.nc files
        '''
        seg_year = segment.coordinates.time.index[0].year
        tf_str = '' if self.timeframe == 'daily' else self.timeframe + '_'
        #return self.mapped_var.variable.name + '_' + tf_str + str(seg_year) +'.nc'
        return self.filebase + '_' + tf_str + str(seg_year) +'.nc'

class FlatFileManager(SplitFileManager):
    '''
    Manage a single flat file
    '''
    def __init__(self,path,mapped_var,timeframe='daily'):
        SplitFileManager.__init__(self,path,mapped_var)

        self.splitter = FlatFileSplitter(self.mapped_var.coordinates)
        self.timeframe = timeframe

    def gen_file_name(self,segment):
        '''
        Generate var_year.nc files
        '''
        #seg_year = segment.coordinates.time.index[0].year
        #tf_str = '' if self.timeframe == 'daily' else self.timeframe + '_'
        return self.mapped_var.variable.name + '.nc'


def file_manager_from_pattern(pattern,variable):
    import glob
    files = glob.glob(pattern)

def flattened_cell_file(variable,period,extent,dtype=np.float64,filename=None,chunksize=None,leave_open=True):

    import netCDF4 as ncd

    cell_dim = Dimension('cell',Units('cell_idx'),np.int32)
    cell_c = Coordinates(cell_dim,list(range(extent.cell_count)))
    tc = period_to_tc(period)

    cs = CoordinateSet([cell_c,tc])

    m_var = MappedVariable(variable,cs,dtype)

    if filename is None:
        filename = variable.name + '.nc'

    ds = ncd.Dataset(filename,'w')
    dsm = DatasetManager(ds)

    if chunksize is None:
        chunksize = (1,len(period))

    chunksizes = cs.validate_chunksizes(chunksize)

    from awrams.utils.io.netcdf_wrapper import NCVariableParameters
    ncp = NCVariableParameters(chunksizes=chunksizes)

    dsm.create_variable(m_var,ncp)

    lats,lons = extent._flatten_fields()

    lat_v = MappedVariable(Variable('lats',deg_north),cell_c,np.float64)
    lon_v = MappedVariable(Variable('lons',deg_east),cell_c,np.float64)

    dsm.create_variable(lat_v)[:] = lats
    dsm.create_variable(lon_v)[:] = lons

    if not leave_open:
        dsm.close()
    else:
        return dsm

def build_cell_map(ds):
    locs = list(zip(ds.variables['lats'][:],ds.variables['lons'][:]))
    cell_map = {}
    for i, cell in enumerate(locs):
        cell_map[cell] = i
    return cell_map

def managed_dataset(fn,mode='r'):
    '''
    Open a file with a DatasetManager object
    '''
    if mode == 'w':
        import netCDF4 as ncd
        return DatasetManager(ncd.Dataset(fn,mode))
    else:
        return DatasetManager(db_opener(fn,mode))#ncd.Dataset(fn,mode))
