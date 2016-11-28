from .settings import DEFAULT_PARAMETER_FILE,SPATIAL_FILE,CLIMATE_DATA

model_name = 'AWRALv5.1'

_forcing_args = 'pt avpt tat radcskyt rgt u2t'.split(' ')
_hrustate_args = 's0 ss sd mleaf'.split(' ')
_states_args = 'sg sr'.split(' ')
_hruparams_args = 'alb_dry,alb_wet,cgsmax,er_frac_ref,fsoilemax,\
lairef,sla,vc,w0ref_alb,\
us0,ud0,wslimu,wdlimu,w0lime,s_sls,tgrow,tsenc,rd'.split(',')
_params_args = 'kr_coeff,slope_coeff,pair'.split(',')
_hruspatial_args = 'fhru hveg laimax'.split(' ')
_spatial_args = 'k_rout,k_gw,k0sat,kssat,kdsat,kr_0s,kr_sd,s0max,ssmax,sdmax,\
prefr,slope'.split(',')
_hypso_args = ['height','hypsperc','ne']

from .template import DEFAULT_TEMPLATE as _DT
OUTPUTS = dict((k,_DT[k]) for k in ['OUTPUTS_HRU','OUTPUTS_AVG','OUTPUTS_CELL'])

_SHARED = None

def set_outputs(outputs):
    global OUTPUTS
    OUTPUTS = outputs

def get_runner(dataspecs,shared=False):
    from . import ffi_wrapper as fw
    if shared:
        if _SHARED is not None:
            return fw.FFIWrapper(mhash=_SHARED['mhash'],template=_SHARED['template'])
        else:
            raise Exception("Call init_shared before using multiprocessing")
    else:
        template = fw.template_from_dataspecs(dataspecs,OUTPUTS)
        return fw.FFIWrapper(False,template)

def init_shared(dataspecs):
    '''
    Call before attempting to use in multiprocessing
    '''
    global _SHARED
    from . import ffi_wrapper as fw
    template = fw.template_from_dataspecs(dataspecs,OUTPUTS)
    mhash = fw.validate_or_rebuild(template)

    _SHARED = dict(mhash=mhash,template=template)

def get_input_parameters():
    '''
    Return the list of keys required as inputs
    '''
    model_keys = []

    model_keys += _forcing_args
    model_keys += ['init_' + k for k in _states_args]
    model_keys += _params_args
    model_keys += _spatial_args
    model_keys += _hypso_args

    for hru in ('_hrusr','_hrudr'):
        model_keys += ['init_' +k+hru for k in _hrustate_args]
        model_keys += [k+hru for k in _hruparams_args]
        model_keys += [k+hru for k in _hruspatial_args]

    return model_keys

def get_state_keys():
    state_keys = _states_args.copy()

    for hru in ('_hrusr','_hrudr'):
        state_keys += [k+hru for k in _hrustate_args]

    return state_keys

def get_default_mapping():
    import json
    from awrams.utils.nodegraph import graph, nodes
    from awrams.utils.metatypes import ObjectDict
    from . import transforms
    import numpy as np

    dparams = json.load(open(DEFAULT_PARAMETER_FILE,'r'))
    #dparams = dict([(k.lower(),v) for k,v in dparams.items()])
    for entry in dparams: 
        entry['MemberName'] = entry['MemberName'].lower()

    mapping = {}

#    for k,v in dparams.items():
#        mapping[k] = nodes.const(v)

    for entry in dparams:
        tmp = entry.copy()
        tmp.pop('MemberName')
        tmp.pop('Value')
        mapping[entry['MemberName']] = nodes.const(entry['Value'],**tmp)
    # Setup a new-style functional input map

    import h5py
    ds = h5py.File(SPATIAL_FILE,mode='r')
    SPATIAL_GRIDS = list(ds['parameters'])
    ds.close()

    # FORCING = {
    #     'tmin': ('tmin*','temp_min_day'),
    #     'tmax': ('tmax*','temp_max_day'),
    #     'precip': ('rr*','rain_day'),
    #     'solar': ('solar*','solar_exposure_day')
    # }

    FORCING = {
        'tmin': ('temp_min*','temp_min_day'),
        'tmax': ('temp_max*','temp_max_day'),
        'precip': ('rain*','rain_day'),
        'solar': ('solar*','solar_exposure_day')
    }
    for k,v in FORCING.items():
        mapping[k+'_f'] = nodes.forcing_from_ncfiles(CLIMATE_DATA,v[0],v[1])
        
    for grid in SPATIAL_GRIDS:
        if grid == 'height':
            mapping['height'] = nodes.hypso_from_hdf5(SPATIAL_FILE,'parameters/height')
        else:
            mapping[grid.lower()+'_grid'] = nodes.spatial_from_hdf5(SPATIAL_FILE,'parameters/%s' % grid)

    mapping.update({
        'tmin': nodes.transform(np.minimum,['tmin_f','tmax_f']),
        'tmax': nodes.transform(np.maximum,['tmin_f','tmax_f']),
        'hypsperc_f': nodes.const_from_hdf5(SPATIAL_FILE,'dimensions/hypsometric_percentile',['hypsometric_percentile']),
        'hypsperc': nodes.mul('hypsperc_f',0.01), # Model needs 0-1.0, file represents as 0-100
        'fday': transforms.fday(),
        'u2t': transforms.u2t('windspeed_grid','fday')
    })

    mapping['er_frac_ref_hrusr'] = nodes.mul('er_frac_ref_hrudr',0.5)

    mapping['k_rout'] = nodes.transform(transforms.k_rout,('k_rout_scale','k_rout_int','meanpet_grid'))
    mapping['k_gw'] = nodes.mul('k_gw_scale','k_gw_grid')

    mapping['s0max'] = nodes.mul('s0max_scale','s0fracawc_grid',100.)
    mapping['ssmax'] = nodes.mul('ssmax_scale','ssfracawc_grid',900.)
    mapping['sdmax'] = nodes.mul('ssmax_scale','sdmax_scale','ssfracawc_grid',5000.)

    mapping['k0sat'] = nodes.mul('k0sat_scale','k0sat_v5_grid')
    mapping['kssat'] = nodes.mul('kssat_scale','kssat_v5_grid')
    mapping['kdsat'] = nodes.mul('kdsat_scale','kdsat_v5_grid')

    mapping['kr_0s'] = nodes.transform(transforms.interlayer_k,('k0sat','kssat'))
    mapping['kr_sd'] = nodes.transform(transforms.interlayer_k,('kssat','kdsat'))

    mapping['prefr'] = nodes.mul('pref_gridscale','pref_grid')
    mapping['fhru_hrusr'] = nodes.sub(1.0,'f_tree_grid')
    mapping['fhru_hrudr'] = nodes.assign('f_tree_grid')
    mapping['ne'] = nodes.mul('ne_scale','ne_grid')
    mapping['slope'] = nodes.assign('slope_grid')
    mapping['hveg_hrudr'] = nodes.assign('hveg_dr_grid')
    mapping['hveg_hrusr'] = nodes.const(0.5)

    mapping['laimax_hrusr'] = nodes.assign('lai_max_grid')
    mapping['laimax_hrudr'] = nodes.assign('lai_max_grid')

    mapping['pair'] = nodes.const(97500.)

    mapping['pt'] = nodes.assign('precip_f')
    mapping['rgt'] = nodes.transform(np.maximum,['solar_f',0.1])
    mapping['tat'] = nodes.mix('tmin','tmax',0.75)
    mapping['avpt'] = nodes.transform(transforms.pe,'tmin')
    mapping['radcskyt'] = transforms.radcskyt()

    mapping['init_sr'] = nodes.const(0.0)
    mapping['init_sg'] = nodes.const(100.0)
    for hru in ('_hrusr','_hrudr'):
        mapping['init_mleaf'+hru] = nodes.div(2.0,'sla'+hru)
        for state in ["s0","ss","sd"]:
            mapping['init_'+state+hru] = nodes.mul(state+'max',0.5)

    # +++dims only required due to having to allocate shared-memory buffer before running...
    dims = ObjectDict(hypsometric_percentile=20,latitude=None,longitude=None,time=None)

    return ObjectDict(mapping=ObjectDict(mapping),dimensions=dims)

def get_default_output_mapping(path='./'):
    from awrams.utils.nodegraph import nodes
    from awrams.utils.metatypes import ObjectDict

    #+++ not dealing with sr and dr versions of HRUS
    outputs = dict((k,_DT[k]) for k in ['OUTPUTS_HRU','OUTPUTS_AVG','OUTPUTS_CELL'])

    mapping = {}
    output_vars = []
    for v in outputs['OUTPUTS_AVG'] + outputs['OUTPUTS_CELL']:
        output_vars.append(v)
    for v in outputs['OUTPUTS_HRU']:
        output_vars.extend([v+'_sr',v+'_dr'])
    for v in output_vars:
            mapping[v] = nodes.write_to_ncfile(path,v)

    return ObjectDict(mapping=ObjectDict(mapping)) #,output_path=output_path)

def get_output_nodes(template):
    from awrams.utils.nodegraph import nodes
    from awrams.utils.metatypes import ObjectDict
    from . import ffi_wrapper as fw

    outputs = dict((k,template[k]) for k in ['OUTPUTS_HRU','OUTPUTS_AVG','OUTPUTS_CELL'])

    mapping = {}
    output_vars = []
    for v in outputs['OUTPUTS_AVG'] + outputs['OUTPUTS_CELL']:
        output_vars.append(v)
    for v in outputs['OUTPUTS_HRU']:
        output_vars.extend([v+'_sr',v+'_dr'])
    for v in output_vars:
        mapping[v] = nodes.model_output(v)

    return ObjectDict(mapping=ObjectDict(mapping)) #,output_path=output_path)

