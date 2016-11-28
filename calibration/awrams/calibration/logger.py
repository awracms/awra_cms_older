import h5py
from awrams.utils.messaging.binding import QueueChild, build_queues, bound_proxy
from awrams.utils.messaging.general import message
import multiprocessing as mp
import pandas as pd
import numpy as np

#from Calibration import 
class CalibrationLogger(QueueChild):
    def __init__(self,pipes,parameters,local_ids,schema,filename):
        QueueChild.__init__(self,pipes)
        self.local_ids = local_ids
        self.schema = schema
        self.filename = filename
        self.parameters = parameters
        
        self.plen = len(self.parameters)
        self.llen = len(self.local_ids)

    def run_setup(self):
        self.fh = fh = h5py.File(self.filename,'w')
        
        self.fh.attrs['iterations'] = 0
        
        p_dim = fh.create_dataset('parameter_name',shape=(len(self.parameters),),dtype=h5py.special_dtype(vlen=bytes))
        for i, p in enumerate(self.parameters):
            p_dim[i] = p
            
        p_ds = fh.create_dataset('parameter_values',shape=(0,len(self.parameters)),maxshape=(None,len(self.parameters)),dtype='f8')
        p_ds.dims.create_scale(p_dim,'parameter')
        p_ds.dims[0].label = 'iteration'
        p_ds.dims[1].attach_scale(p_dim)
        
        gs_ds = fh.create_dataset('global_score',shape=(0,),maxshape=(None,),dtype='f8')
        gs_ds.dims[0].label = 'iteration'
    
        l_results = fh.create_group('local_scores')
        
        l_id = l_results.create_dataset('local_id',shape=(len(self.local_ids),),dtype=h5py.special_dtype(vlen=bytes))
        for i, l in enumerate(self.local_ids):
            l_id[i] = l
        
        for name in self.schema:
            ds = l_results.create_dataset(name,shape=(0,len(self.local_ids)),maxshape=(None,len(self.local_ids)),dtype='f8')
            ds.dims.create_scale(l_id,'local_id')
            ds.dims[0].label = 'iteration'
            ds.dims[1].attach_scale(l_id)  
            ds.dims[1].label = 'local_id'

    def log_results(self,parameters,global_score,local_scores):
        #Parameters must support pandas series indexing 
        #ie parameters[param_names] for correct dict sorting
        self.fh.attrs['iterations'] += 1
        i_size = self.fh.attrs['iterations']
        i_idx = i_size-1

        self.fh['parameter_values'].resize((i_size,self.plen))  
        self.fh['parameter_values'][i_idx,:] = parameters[self.parameters]
        
        self.fh['global_score'].resize((i_size,))
        self.fh['global_score'][i_idx] = global_score
        
        for name in self.schema:
            ds = self.fh['local_scores'][name]
            ds.resize((i_size,self.llen))
            ds[i_idx,:] = local_scores[name]

    def cleanup(self):
        self.fh.close()

    def terminate(self,iterations):
        if iterations > 0:
            while not self.fh.attrs['iterations'] == iterations:
                self._handle_message(self._recv_msg())
        self.active = False

