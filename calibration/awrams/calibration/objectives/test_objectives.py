from awrams.calibration.objectives import *
import numpy as np

obs = np.linspace(0.,1.,365)

def test_nse():
	nse = NSE(obs)
	assert( nse(obs) == 1.0 )
	assert( nse(np.repeat(obs.mean(),365)) == 0.0 )

def test_bias():
	bias = Bias(obs)
	assert( bias(obs) == 0.0 )
	assert( bias(obs*2.0) == 1.0 )

class TestLocalSingle:

    schema = ['qtot_nse']

    def __init__(self,obs,eval_period,min_valid=15,flow_variable='qtot_avg'):

        self.valid_idx = {}
        self.nse = {}
        self.flow_variable = flow_variable
        for k in [flow_variable]:

            data = obs[k]

            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
            else:
                self.valid_idx[k] = slice(0,len(eval_period))

            self.nse[k] = NSE(data[self.valid_idx[k]])

    def evaluate(self,modelled):
        qtot_nse = self.nse[self.flow_variable](modelled[self.flow_variable][self.valid_idx[self.flow_variable]])
        return dict(qtot_nse=qtot_nse)

class TestGlobalSingle:
    def evaluate(self,l_results):
        return 1.0 - np.mean(l_results['qtot_nse'])

class TestLocalMulti:

    schema = ['qtot_nse','etot_nse']

    def __init__(self,obs,eval_period,min_valid=15,flow_variable='qtot_avg',et_variable='etot_avg'):

        self.valid_idx = {}
        self.nse = {}

        self.flow_variable = flow_variable
        self.et_variable = et_variable
        
        for k in [flow_variable,et_variable]:

            data = obs[k]

            if np.isnan(data).any():
                nan_mask = np.isnan(data)
                self.valid_idx[k] = np.where(nan_mask == False)
            else:
                self.valid_idx[k] = slice(0,len(eval_period))

            self.nse[k] = NSE(data[self.valid_idx[k]])

    def evaluate(self,modelled):
        qtot_nse = self.nse[self.flow_variable](modelled[self.flow_variable][self.valid_idx[self.flow_variable]])
        etot_nse = self.nse[self.et_variable](modelled[self.et_variable][self.valid_idx[self.et_variable]])
        return dict(qtot_nse=qtot_nse,etot_nse=etot_nse)

class TestGlobalMultiEval:
    def evaluate(self,l_results):
        return 1.0 - np.mean((l_results['qtot_nse'] + l_results['etot_nse']) * 0.5)
