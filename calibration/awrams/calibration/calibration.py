import numpy as np
import pandas as pd
import pickle
import h5py

from awrams.utils.metatypes import ObjectDict
from awrams.calibration.server import *


def default_node_settings(model):
    import imp
    import awrams.calibration.objectives.multivar_objectives as w
    from awrams.utils.nodegraph import nodes

    ns = {}

    ns['run_period'] = 'UNSPECIFIED'   # the period for which the model is actually run
    ns['eval_period'] = 'UNSPECIFIED'  # the period over which it is evaluated against observations

    from multiprocessing import cpu_count
    ns['num_workers'] = 2 #cpu_count()

    ns['inputs'] = model.get_default_mapping()
    data_path = model.CLIMATE_DATA #'/data/cwd_awra_data/awra_inputs/climate_generated/'
    FORCING = {
        'tmin'  : ('temp_min','temp_min_day'),
        'tmax'  : ('temp_max','temp_max_day'),
        'precip': ('rain','rain_day'),
        'solar' : ('solar','solar_exposure_day')
    }
    for k,v in FORCING.items():
        ns['inputs'].mapping[k+'_f'] = nodes.forcing_from_ncfiles(data_path+'/',v[0]+'*',v[1],cache=False)

    # Example with single catchment...
    ns['catchment_ids'] = []
    ns['catchment_extents'] = {}

    ns['logfile'] = 'calibration.h5'

    # All cal catchments
    #ns['catchment_ids'] = [cid.strip() for cid in open('./Catchment_IDs.csv').readlines()[2:]]

    from .calibrate import get_parameter_df
    ns['default_params'] = get_parameter_df(ns['inputs'].mapping)

    ns['observations'] = ObjectDict(qtot=ObjectDict())

    ns['observations'].qtot.source_type = 'csv'
    ns['observations'].qtot.filename = '/mnt/awramsi_test_data/Calibration/Catchment_Qobs.csv'

    ns['objective'] = ObjectDict({'localf': ObjectDict(), 'globalf':ObjectDict()})

    imp.load_source('lobjf_mod',w.__file__)
    ns['objective']['localf']['filename'] = w.__file__
    ns['objective']['localf']['classname'] = 'LocalEval'
    # Any arguments required by the evaluator are stored in this dict
    ns['objective']['localf']['arguments'] = ObjectDict()
    # e.g
    # ns['objective']['localf']['arguments']['min_valid'] = 15

    ns['objective']['globalf']['filename'] = w.__file__
    ns['objective']['globalf']['classname'] = 'GlobalMultiEval'

    return ns

def default_cal_params(model):
    from awrams.model.awral.settings import DEFAULT_PARAMETER_FILE
    import json
    dparams = json.load(open(DEFAULT_PARAMETER_FILE,'r'))
    return dparams

def default_term_params():
    tp = ObjectDict()

    tp.max_shuffle = 1000 # Max shuffling loops
    tp.max_iter = 20000 # Max model evaluations
    tp.target_score = 1e-8
    tp.max_nsni = 5 # Max shuffle without improvement (as defined below)
    tp.min_imp = 0.01 # Minimum change required for 'improvement' metric

    return tp


class CalibrationInstance(ObjectDict):
    def __init__(self,model):
        ns = default_node_settings(model)
        self.node_settings = ObjectDict(ns)

        # self.hyperp = ObjectDict(complex_sz=43,n_complexes=14,sub_sz=22,n_offspring=1,n_evol=43,min_complexes=1)
        self.hyperp = ObjectDict(complex_sz=5,n_complexes=5,sub_sz=2,n_offspring=1,n_evol=10,min_complexes=2)
        self.num_nodes = 4

        self.termp = default_term_params()
        self.params = ns['default_params'] #default_cal_params(model)
        self.server = None

    def run_local(self,seed=None):
        '''
        Run an optimization locally
        '''
        if self.server is None:
            self.setup_local()

        self.result = None
        try:
            self.result = self.server.run_optimization(seed)
        finally:
            # pass
            self.terminate_server()
        return CalibrationResults(self._LOCALISED_NS.logfile)

    def get_local_results(self):
        return CalibrationResults(self._LOCALISED_NS.logfile)

    def setup_local(self):
        '''
        Initialise a local server without running optimization
        '''
        if self.server is not None:
            self.server.terminate()

        ns = self._localise()
        self.server = CalibrationServerPyMP(ns['catchment_ids'],self.params,ns.objective,self.num_nodes,ns,self.hyperp,self.termp)

    def terminate_server(self,wait_for_results=True):
        if self.server is not None:
            self.server.terminate(wait_for_results)
        self.server = None

    def _dump_to_pickle(self,fn='calibration.pkl'):
        pickle.dump(self,open(fn,'wb'))

    def _localise(self):
        ns = self.node_settings
        self._LOCALISED_NS = ns
        return ns

class CalibrationResults:
    def __init__(self,fn):
        self.filename = fn
        self.fh = h5py.File(fn,'r')
        self.iterations = self.fh.attrs['iterations']
        self.parameters = [k.decode() for k in self.fh['parameter_name'][:]]
        self.local_schema = [k for k in self.fh['local_scores'].keys() if k != 'local_id']
        self.catchment_ids = [k.decode() for k in self.fh['local_scores']['local_id']]
        self.global_score = self.fh['global_score'][...]

    def get_best(self):
        index = np.where(self.fh['global_score'][...]==self.fh['global_score'][...].min())[0][0]
        score = self.fh['global_score'][index]
        parameters = dict(zip(self.parameters,self.fh['parameter_values'][index]))
        return ObjectDict(index=index,score=score,parameters=parameters)

    def get_local_scores(self,key):
        dfd = pd.DataFrame()
        for i,l_id in enumerate(self.fh['local_scores']['local_id']):
            dfd[l_id.decode()] = pd.Series(self.fh['local_scores'][key][:,i])
        return dfd
