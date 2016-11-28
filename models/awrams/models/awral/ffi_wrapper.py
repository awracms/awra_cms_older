import cffi
import numpy as np
from .template import _SOURCE_FN,_SOURCE_T_FN,_HEADER_FN,_HEADER_T_FN,_LIB_FN
from numbers import Number
import os
import shutil

TYPEMAP = {np.float64: "double *", np.float32: "float *"}

def ccast(ndarr,ffi,to_type=np.float64,promote=True):
    if ndarr.dtype != to_type:
        if promote:
            ndarr = ndarr.astype(to_type)
        else:
            raise Exception("Incorrect dtype",ndarr.dtype,to_type)

    typestr = TYPEMAP[to_type]
    return ffi.cast(typestr,ndarr.ctypes.data)

'''
These are solely used for the sma_awral backwards-compatibility layer
(which is currently broken); can/should be removed when full transition is made
'''

forcing_args = 'pt avpt tat radcskyt rgt u2t'.split(' ')
hrustate_args = 's0 ss sd mleaf'.split(' ')
states_args = 'sg sr'.split(' ')
hruparams_args = 'alb_dry,alb_wet,cgsmax,er_frac_ref,fsoilemax,\
lairef,sla,vc,w0ref_alb,rd,\
us0,ud0,wslimu,wdlimu,w0lime,s_sls,tgrow,tsenc'.split(',')
params_args = 'kr_coeff,slope_coeff,pair'.split(',')
hruspatial_args = 'fhru hveg laimax'.split(' ')
spatial_args = 'k_rout,k_gw,k0sat,kssat,kdsat,kr_0s,kr_sd,s0max,ssmax,sdmax,\
prefr,slope'.split(',')
hypso_args = ['height','hypsperc','ne']


def get_lib_fn(mhash):
    _FILE_PATH = os.path.dirname(__file__)
    return os.path.join(_FILE_PATH,'model','awral_' + mhash + '.so')

def get_header_fn(mhash):
    _FILE_PATH = os.path.dirname(__file__)
    return os.path.join(_FILE_PATH,'model','awral_' + mhash + '.h')

def build_icc(LIB_FN):
    import subprocess

    buildstr = "icc %s -std=c99 -static-intel --shared -fPIC -O3 -o %s" % (_SOURCE_FN,LIB_FN)

    callstr = buildstr.split(' ')
    from subprocess import Popen, PIPE
    pipe = Popen(callstr, stdout=PIPE,stderr=PIPE)
    out,err = pipe.communicate()

    return out,err

def build_gcc(LIB_FN):
    import subprocess

    buildstr = "gcc %s -std=c99 --shared -fPIC -O3 -o %s" % (_SOURCE_FN,LIB_FN)

    callstr = buildstr.split(' ')
    from subprocess import Popen, PIPE
    pipe = Popen(callstr, stdout=PIPE,stderr=PIPE)
    out,err = pipe.communicate()

    return out,err

def filename_for_hash(existing_fn,hashval):
    # sfn = existing_fn.split('.')
    sfn,ext = os.path.splitext(existing_fn)
    # sfn.insert(-1,'_%s.' % hashval)
    # sfn.insert(-1,'_%s' % hashval)
    return ''.join((sfn,'_%s' % hashval,ext))

def model_hash(template,source_fn=_SOURCE_T_FN,header_fn=_HEADER_T_FN):
    from hashlib import md5
    outstr = open(header_fn).read()
    outstr = outstr + open(source_fn).read()
    for k in sorted(template.keys()):
        outstr = outstr + k + str(sorted(template[k]))
    return md5(outstr.encode()).hexdigest()

def validate_or_rebuild(template,source_fn=_SOURCE_FN,header_fn=_HEADER_FN,lib_fn=_LIB_FN,force=False):
    '''
    Checks whether an existing compiled header/library exist for this template
    Will recompile if not
    '''
    mhash = model_hash(template)
    header_fnh = filename_for_hash(header_fn,mhash)
    lib_fnh = filename_for_hash(lib_fn,mhash)

    rebuild=force

    if not os.path.exists(header_fnh):
        rebuild = True

    if not os.path.exists(lib_fnh):
        rebuild = True

    if rebuild:
        print("Rebuilding model")
        process_templates(template)
        out,err = BUILD_MODEL(lib_fnh)
        out,err = out.decode(),err.decode()
        if 'error' in out or 'error' in err:
            print(out)
            print(err)
            raise Exception("Model build failed")
        print(header_fn)
        print(header_fnh)
        shutil.copyfile(header_fn,header_fnh)

    return mhash
        
def process_templates(tdict=None):
    from . import template

    template.transform_awral_files(tdict)

BUILD_MODEL = build_gcc

def template_from_dataspecs(dspec,outputs):
    from .description import get_input_parameters
    from .template import DEFAULT_TEMPLATE

    model_keys = get_input_parameters()

    new_template = dict(zip(DEFAULT_TEMPLATE.keys(),[[] for k in DEFAULT_TEMPLATE]))



    for k in ['OUTPUTS_AVG','OUTPUTS_CELL','OUTPUTS_HRU']:
        new_template[k] = outputs[k]

    hru_keys_base = [k for k in model_keys if '_hru' in k and not k.startswith('init_')]
    hru_keys = np.unique([k[:-6] for k in hru_keys_base])

    for k in hru_keys:
        d0 = dspec[k+'_hrusr']
        d1 = dspec[k+'_hrudr']
        mdims = max(len(d0.dims),len(d1.dims))
        if(mdims):
            ktype = 'INPUTS_SPATIAL_HRU'
        else:
            ktype = 'INPUTS_SCALAR_HRU'
            
        new_template[ktype].append(k)

    for k in model_keys:
        if not k in hru_keys_base \
            and not k.startswith('init_') and not k in ['height','hypsperc']:
            dims = dspec[k].dims
            if dims == ['time','cell']:
                new_template['INPUTS_FORCING'].append(k)
            elif dims == ['cell']:
                new_template['INPUTS_SPATIAL'].append(k)
            elif not len(dims):
                new_template['INPUTS_SCALAR'].append(k)
            else:
                print(k,dims) 
            
    return new_template

class FFIWrapper:
    def __init__(self,force_build=False,template=None,mhash=None):

        if mhash is not None:
            self.template = template
            self._init_ffi(mhash)
        else:
            self.reload(force_build,template)

    def _invalidate(self):
        import gc

        gc.collect()

        self.ffi = None

        self.awralib = None

        self.forcing = None
        self.outputs = None
        self.initial_states = None
        self.final_states = None
        self.parameters = None
        self.spatial = None
        self.hruspatial = None
        self.hruparams = None
        self.hypso = None

        gc.collect()

    def _init_ffi(self,mhash):
        from cffi import FFI
        self.ffi = FFI()

        header_fn = get_header_fn(mhash)
        lib_fn = get_lib_fn(mhash)

        with open(header_fn,'r') as fh:
            self.ffi.cdef(fh.read())

        self.awralib = self.ffi.dlopen(lib_fn)

        self.forcing = self.ffi.new("Forcing*")
        self.outputs = self.ffi.new("Outputs*")
        self.initial_states = self.ffi.new("States *")
        self.final_states = self.ffi.new("States *")
        self.parameters = self.ffi.new("Parameters *")
        self.spatial = self.ffi.new("Spatial *")
        self.hruspatial = self.ffi.new("HRUSpatial[2]")
        self.hruparams = self.ffi.new("HRUParameters[2]")
        self.hypso = self.ffi.new("Hypsometry *")

    def reload(self,force_build=False,template=None):

        self._invalidate()

        if template is None:
            from .template import DEFAULT_TEMPLATE
            template = DEFAULT_TEMPLATE

        mhash = validate_or_rebuild(template,force=force_build)

        self.template = template

        self._init_ffi(mhash)

    def sma_awral(self,**kwargs):
        '''
        This is just broken at the moment.
        '''
        run_settings = dict(kwargs)
        
        cells = 1
        timesteps = run_settings['timesteps']
        forcing_np = {}
        forcealive = []

        for k in forcing_args:
            newforce = np.empty((timesteps,cells))
            #for c in range(cells):
            newforce[:,0] = run_settings[k]
            self.forcing.__setattr__(k,ccast(newforce,self.ffi))
            forcealive.append(newforce)

        
        ALL_OUTPUTS = self.template.OUTPUTS_AVG + self.template.OUTPUTS_CELL

        for k in ALL_OUTPUTS:
            #outputs_np[k] = arr = np.empty((plen,cells))
            full_k = k+'_bal' if k in ['sr','sg','dgw'] else k + '_avg'
            arr = run_settings[full_k]
            self.outputs.__setattr__(k,ccast(arr,self.ffi))

        for hru in range(2):
            for k in self.template.OUTPUTS_HRU:
                full_k = k+'_sr' if hru is 0 else k+'_dr'
                arr = run_settings[full_k]
                self.outputs.hru[hru].__setattr__(k,ccast(arr,self.ffi))   

        for k in states_args:
            rs = run_settings[k]
            nval = np.empty((cells))
            nval[:] = rs
            forcealive.append(nval)
            self.initial_states.__setattr__(k,ccast(nval,self.ffi))
            self.final_states.__setattr__(k,ccast(nval,self.ffi))

        for k in hrustate_args:
            rs = run_settings[k]

            nval0 = np.empty((cells))
            nval0[:] = rs[0]
            self.initial_states.hru[0].__setattr__(k,ccast(nval0,self.ffi))
            self.final_states.hru[0].__setattr__(k,ccast(nval0,self.ffi))
            nval1 = np.empty((cells))
            nval1[:] = rs[1]
            self.initial_states.hru[1].__setattr__(k,ccast(nval1,self.ffi))
            self.final_states.hru[1].__setattr__(k,ccast(nval1,self.ffi))

            forcealive.append(nval0)
            forcealive.append(nval1)

        for k in params_args:
            if k == 'hypsperc':
                rs = run_settings['hypsfsat']
                self.parameters.__setattr__(k,ccast(rs,self.ffi))
                forcealive.append(rs)
            else:
                rs = run_settings[k]
                self.parameters.__setattr__(k,rs)

        for k in hruparams_args:
            rs = run_settings[k]
            self.parameters.hru[0].__setattr__(k,rs[0])
            self.parameters.hru[1].__setattr__(k,rs[1])

        for k in spatial_args:
            rs = run_settings[k]
            if k == 'height':
                nval = np.empty((cells,20))
                nval[0] = rs
                forcealive.append(nval)
            else:
                nval = np.empty((cells))
                forcealive.append(nval)
                if isinstance(rs,np.ndarray):
                    rs = rs[0]
                nval[:] = rs
            self.spatial.__setattr__(k,ccast(nval,self.ffi))

        for k in hruspatial_args:
            rs = run_settings[k]
            nval0 = np.empty((cells))
            nval0[:] = rs[0]
            self.spatial.hru[0].__setattr__(k,ccast(nval0,self.ffi))
            nval1 = np.empty((cells))
            nval1[:] = rs[1]
            self.spatial.hru[1].__setattr__(k,ccast(nval1,self.ffi))
            forcealive.append(nval0)
            forcealive.append(nval1)
            
        self.awralib.awral(self.forcing[0],self.outputs[0],self.initial_states[0],self.final_states[0],self.parameters[0],self.spatial[0],timesteps,cells)

    def _cast(self,ndarr,to_type=np.float64,promote=True):
        '''
        Ensures inputs are in correct datatypes for model.
        '''
        if not ndarr.flags['C_CONTIGUOUS']:
            ndarr = ndarr.flatten()

        if ndarr.dtype != to_type:
            if promote:
                ndarr = ndarr.astype(to_type)
            else:
                raise Exception("Incorrect dtype",ndarr.dtype,to_type)

        self._temp_cast.append(ndarr)

        typestr = TYPEMAP[to_type]

        return self.ffi.cast(typestr,ndarr.ctypes.data)

    def _promote(self,v,shape):
        if isinstance(v,Number):
            out = np.empty(shape,dtype=np.float64)
            out[...] = v
            self._temp_cast.append(out)
            return out
        else:
            return v

    def _promote_except(self,v,shape):
        if isinstance(v,Number):
            raise Exception("Scalar %s supplied for spatial value" %v)
        else:
            return v

    def run_over_dimensions(self,inputs,dims):
        return self.run_from_mapping(inputs,dims['time'],dims['cell'])

    def run_from_mapping(self,mapping,timesteps,cells,scalar_promote=True):
        #forcing_np = {}
        #forcealive = []

        self._temp_cast = []

        if scalar_promote:
            promote = self._promote
        else:
            promote = self._promote_except
        #for k in forcing_args:
        for k in self.template['INPUTS_FORCING']:
            nval = promote(mapping[k],(timesteps,cells,))
            self.forcing.__setattr__(k,self._cast(nval))

        outputs_np = {}
        #outputs_hru_np = []
        
        ALL_OUTPUTS = self.template['OUTPUTS_AVG'] + self.template['OUTPUTS_CELL']

        for k in ALL_OUTPUTS:
            outputs_np[k] = arr = np.empty((timesteps,cells))
            self.outputs.__setattr__(k,self._cast(arr))

        for hru in range(2):
            for k in self.template['OUTPUTS_HRU']:
                full_k = k+'_sr' if hru is 0 else k+'_dr'
                outputs_np[full_k] = arr = np.empty((timesteps,cells))
                self.outputs.hru[hru].__setattr__(k,self._cast(arr))

        outputs_np['final_states'] = {}

        for k in states_args:
            nval = promote(mapping['init_'+k],(cells,))
            self.initial_states.__setattr__(k,self._cast(nval))

            outputs_np['final_states'][k] = sval = np.empty((cells,))
            self.final_states.__setattr__(k,self._cast(sval))

        for k in hrustate_args:
            nval0 = promote(mapping['init_'+k+'_hrusr'],(cells,))
            self.initial_states.hru[0].__setattr__(k,self._cast(nval0))
            nval1 = promote(mapping['init_'+k+'_hrudr'],(cells,))
            self.initial_states.hru[1].__setattr__(k,self._cast(nval1))

            outputs_np['final_states'][k+'_hrusr'] = srval = np.empty((cells,))
            outputs_np['final_states'][k+'_hrudr'] = drval = np.empty((cells,))
            self.final_states.hru[0].__setattr__(k,self._cast(srval))
            self.final_states.hru[1].__setattr__(k,self._cast(drval))


        for k in self.template['INPUTS_SCALAR']:
            rs = mapping[k]
            self.parameters.__setattr__(k,rs)

        for k in self.template['INPUTS_SCALAR_HRU']:
            self.hruparams[0].__setattr__(k,mapping[k+'_hrusr'])
            self.hruparams[1].__setattr__(k,mapping[k+'_hrudr'])

        for k in self.template['INPUTS_SPATIAL']:
            nval = promote(mapping[k],(cells,))
            self.spatial.__setattr__(k,self._cast(nval))

        for k in self.template['INPUTS_SPATIAL_HRU']:
            self.hruspatial[0].__setattr__(k,self._cast(promote(mapping[k+'_hrusr'],(cells,))))
            self.hruspatial[1].__setattr__(k,self._cast(promote(mapping[k+'_hrudr'],(cells,))))
            
        for k in hypso_args:
            nval = mapping[k]
            if k == 'height': #+++ Right now our hypso grids present data in the opposite order
                nval = nval.T.astype(np.float64).flatten()
                self._temp_cast.append(nval)
            self.hypso.__setattr__(k,self._cast(nval))

        self.awralib.awral(self.forcing[0],self.outputs[0],self.initial_states[0],self.final_states[0],\
            self.parameters[0],self.spatial[0],self.hypso[0],self.hruparams,self.hruspatial,timesteps,cells)

        self._temp_cast = []
        
        return outputs_np