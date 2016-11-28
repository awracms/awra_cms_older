import pandas as pd

from awrams.models import awral
from awrams.simulation.ondemand import OnDemandSimulator
from .evaluators import Evaluator
from .objectives import NSE

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('calibrate')

MODEL = None
input_map = None
objective = NSE

def set_model(model=awral):
    global MODEL,input_map
    MODEL = model
    input_map = model.get_default_mapping()
set_model()

def get_parameter_df(mapping):
    params = [(k,v) for (k,v) in mapping.items() if 'Min' in v.properties]
    params = [{'Name':k,'Min':v.properties['Min'],'Max':v.properties['Max'],'Value':v.args['value']} for k,v in params]
    return pd.DataFrame(params).set_index('Name')


class RunoffEvaluator(Evaluator):
    '''
    compare qtot against gauging station data observations
    '''
    def __init__(self,period,extent,observations):
        self.period = period
        self.extent = extent
        self.obs = observations

        self.evaluations = 0

        logger.info("initialising model simulator...")
        self.ods = OnDemandSimulator(MODEL,input_map.mapping)
        self.initial_results,self.inputs = self.ods.run(period,extent,return_inputs=True)
        logger.info("done")
        # print("XXXX",self.inputs)
        self.obj_fn = objective(self.obs[period])

    def new(self):
        return self

    def run_sim(self,parameters):
        for k,v in parameters.iteritems():
            try:
                # self.ods.input_runner.input_graph[k]['exe'].value = v
                self.inputs[k] = v
            except KeyError: ### ignore score column
                pass

        # return self.ods.run(self.period,self.extent)
        return self.ods.run_prepack(self.inputs,self.period,self.extent)

    def evaluate(self,parameters):
        self.evaluations += 1

        results = self.run_sim(parameters)
        qtot_results = results['qtot']

        ### mm_to cumecs is approximate, could be improved by using actual cell areas
        ### this would require osgeo.ogr etc
        catchment_qtot_mod = qtot_results.mean(axis=1) * mm_to_cumecs(qtot_results.shape[1])

        res = 1 - self.obj_fn(catchment_qtot_mod)
        return res

    def plot(self,results,period=None):
        if period is None:
            period = self.period

        cal_catchment_qtot = results['qtot'].mean(axis=1) * mm_to_cumecs(results['qtot'].shape[1])

        ax = pd.DataFrame(cal_catchment_qtot,columns=['modelled'],index=self.period).loc[period].plot(figsize=(18,6),legend=True)
        pd.DataFrame(self.obs[period].values,columns=['observed'],index=period).plot(legend=True,ax=ax)

def mm_to_cumecs(dim):
    ### assumes cell dimensions are 5km X 5km
    return 1e-3 * (dim * 5000 * 5000) / 86400.

