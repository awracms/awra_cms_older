import numpy as np
import threading
import pandas as pd
import imp
from multiprocessing import Process
import multiprocessing as mp

from awrams.utils import extents, catchments
from awrams.utils.mapping_types import period_to_tc,gen_coordset
from awrams.utils.messaging.general import *
from awrams.utils.messaging.buffers import *
from awrams.utils.messaging.binding import *
from awrams.utils.messaging.mp_parent import MultiprocessingParent
from awrams.utils.nodegraph.nodes import ConstNode, forcing_from_dict
import awrams.utils.nodegraph.graph as graph


class AWRALCalNode(MultiprocessingParent):
    def __init__(self,node_id):
        '''
        Initialise a simulation server object with communication ports
        '''
        MultiprocessingParent.__init__(self)

        self.node_id = node_id

    def timed_out(self,failure_source):
        print("Timed out waiting for %s", failure_source)
        self.active = False

    def configure_node(self,node_settings):
        '''
        Create the objects required to run simulations; including buffers for passing data
        '''
        self.run_period = node_settings['run_period']
        self.eval_period = node_settings['eval_period']
        self.timesteps = len(self.run_period)

        rtc = period_to_tc(self.run_period)

        self.eval_idx = rtc.get_index(self.eval_period)

        self.default_params = node_settings['default_params']

        self.catchments = {}
        self.areas = {}

        MAX_WORK_BLOCK = 0

        # Set up observations and local objective functions
        obs_data = {}

        self.active_vars = []

        for obs_name, obs in node_settings['observations'].items():
            obs_type = obs['source_type']
            if obs_type == 'csv':
                obs_df = pd.DataFrame.from_csv(obs['filename'])
                obs_data[obs_name] = obs_df
            else:
                raise Exception("Unknown observation type",obs_type)

            self.active_vars.append(obs_name)

        self.local_objf = {}

        obj_fn = node_settings['objective']['localf']['filename']
        obj_cn = node_settings['objective']['localf']['classname']
        obj_args = node_settings['objective']['localf'].get('arguments')

        if obj_args is None:
            obj_args = {}

        obj_mod = imp.load_source('objf_mod',obj_fn)
        obj_class = getattr(obj_mod,obj_cn)

        cids = node_settings['catchment_ids']
        # sdb = catchments.ShapefileDB()

        for cid in cids:
            # start_idx = self.cell_count
            #+++ Hardcoding against wirada calibration shapefile
            # c = sdb._get_by_field('StationID',cid.zfill(6))
            # print(cid,c.shape,c.cell_count)
            # self.catchments[cid] = c
            c = node_settings['catchment_extents'][cid]
            self.catchments[cid] = c

            self.areas[cid] = c.areas[c.mask==False].flatten()

            if c.cell_count > MAX_WORK_BLOCK:
                MAX_WORK_BLOCK = c.cell_count

            local_obs = {}
            for obs_var,obs_df in obs_data.items():
                # print(cid,obs_var)
                local_obs[obs_var] = np.array(obs_df[cid].loc[self.eval_period])

            self.local_objf[cid] = obj_class(local_obs,self.eval_period,**obj_args)

        '''
        Need to load and split input data
        '''

        ''' Set up runner objects'''

        self.n_workers = node_settings['num_workers']

        self.build_io(node_settings)

        # output_variables = node_settings['output_variables'] #['qtot','w0','etot']
        #
        # self.shm_outputs = create_shm_dict(output_keys,(self.timesteps,MAX_WORK_BLOCK))
        #
        # self.outputs = shm_to_nd_dict(**self.shm_outputs)

        self.feed_q = mp.Queue()

        self.build_workers(node_settings)

    def build_workers(self,node_settings):
        self.workers = []
        self.worker_q = []

        # print("build workers",self.n_workers)
        for i in range(self.n_workers):
            work_q = mp.Queue()
            self.worker_q.append(work_q)
            self.workers.append(CellWorker(self.control_q,work_q,self.feed_q,self.timesteps,self.catchments,self.shm_inputs,self.shm_outputs))
            self.add_child_proc(self.workers[i],work_q)
            work_q.put(message('ack_request',message='worker_ready',id=i))

        self.wait_on_acknowledgements('worker_ready',range(self.n_workers ))
        # print("Setup complete")

    def evaluate_with_params(self,subtask_id,params):
        results = {}
        for cid in self.catchments:
            self.feed_q.put(message('run_cell',params=params,cid=cid))

        for c in range(len(self.catchments)):
            msg = self.control_q.get()
            cid = msg['content']['cid']
            # print(cid,msg)
            # for v in self.outputs[cid]:
            #     print(v,self.outputs[cid][v].shape,self.outputs[cid][v])
            results[cid] = self.local_objf[cid].evaluate(self.aggregate_outputs(cid)) #subtask_id)
        return results

    def aggregate_outputs(self,cid):
        outputs = {}

        for v in self.active_vars:
            areas = self.areas[cid]
            # result = ((self.outputs[v][:,:len(areas)]*areas).sum(1))/self.catchments[subtask_id].area
            result = ((self.outputs[cid][v][:]*areas).sum(1))/self.catchments[cid].area
            outputs[v] = result[self.eval_idx]
        return outputs

    def build_io(self,node_settings): #input_settings,required_inputs):
        '''
        Assume that we have NCD files we can load from - probably there are other sources...
        Build shared memory dictionaries for each of our cell_workers
        '''
        # print("Building inputs...")
        input_settings = node_settings['inputs']

        self.shm_inputs = {}
        self.shm_outputs = {}
        self.outputs = {}
        # inputs = {}

        igraph = graph.ExecutionGraph(input_settings.mapping)
        node_settings['input_dataspecs'] = igraph.get_dataspecs(True)

        for cid in self.catchments:
            ovs = node_settings['output_variables']
            self.shm_outputs[cid] = create_shm_dict(ovs,(self.timesteps,self.catchments[cid].cell_count))
            self.outputs[cid] = shm_to_nd_dict(**self.shm_outputs[cid])

            coords = gen_coordset(self.run_period,self.catchments[cid])
            input_build = igraph.get_data_flat(coords,self.catchments[cid].mask)

            # self.shm_inputs[cid] = {}

            shapes = {}
            for n in igraph.input_graph:
                if not type(igraph.input_graph[n]['exe']) == ConstNode:
                    try:
                        shapes[n] = input_build[n].shape
                    except AttributeError:
                        shapes[n] = None

            self.igraph = igraph

            self.shm_inputs[cid] = create_shm_dict_inputs(shapes)
            _inputs_np = shm_to_nd_dict_inputs(**self.shm_inputs[cid])

            for n in igraph.input_graph:
                if not type(igraph.input_graph[n]['exe']) == ConstNode:

                    # inputs[cid][n] = input_build[n]

                    if shapes[n] is None or len(shapes[n]) == 0:
                        _inputs_np[n][0] = input_build[n]
                    else:
                        _inputs_np[n][...] = input_build[n][...]
        # print("...Done")

    def run(self):
        active = True

        try:
            while active:
                msg = self.recv_msg()
                subject = msg['subject']
                if subject == 'terminate':
                    active = False

                elif subject == 'evaluate':
                    msg = msg['content']
                    out_res = self.evaluate_with_params(None,msg['params'])
                    reply = message('results',results=out_res,node_id=self.node_id,job_meta=msg['job_meta'])
                    self.send_msg(reply)

                elif subject == 'settings':
                    msg = msg['content']
                    node_settings = msg['node_settings']
                    self.configure_node(node_settings)

                else:
                    print("Unknown message : \n%s" % msg)

        except Exception as e:
            self._handle_exception(e)
            raise
        finally:
            self.terminate_children()

    def _handle_exception(self,e,t=None):
        print("Handling exception in Node")
        m = message('exception', node_id=self.node_id, exception=e,traceback=get_traceback())
        self.send_msg(m)

def get_subset_dict(in_dict,idx):
    '''
    Return a dict of subsetted arrays from a dict of arrays
    '''
    out = {}
    for k, v in in_dict.items():
        out[k] = v[idx]
    return out


class CellWorker(Process):
    '''
    Python multiprocessing wrapper around CellArrayProcessor
    Sharedmemory and Queues communication
    '''

    def __init__(self,control_q,input_q,feed_q,len_period,catchments,inputs,outputs):
        Process.__init__(self)

        self.daemon = True
        '''
        control_q : reporting/notifications
        input_q : rpc messages
        feed_q : cell blocks
        '''
        self.control_q = control_q #: reporting/notifications
        self.input_q = input_q # rpc messages
        self.feed_q = feed_q # cell blocks

        self.len_period = len_period
        self.catchments = catchments

        self.inputs = inputs #+++ these are the shared_mem inputs...
        self.outputs = outputs

    def notify_controller(self,subject,**kwargs):
        self.control_q.put(message(subject,**kwargs))

    def get_message(self):
        return self.input_q.get()

    def handle_message(self,msg):
        subject = msg['subject']
        if subject == 'terminate':
            self.feed_q.put(message('terminate'))
            self.active = False
        elif subject == 'ack_request':
            ack = msg['content']
            self.notify_controller('ack',acknowledgement=ack['message'],id=ack['id'])

    def _handle_exception(self,e):
        self.notify_controller('exception', pid=self.pid, exception=e,traceback=get_traceback())

    def _get_cell_messages(self):
        while True:
            msg = self.feed_q.get()
            subject = msg['subject']
            if subject == 'terminate':
                return
            elif subject == 'run_cell':
                content = msg['content']
                self.processor.set_parameters(content['params'])
                self.processor.run_catchment(content['cid']) #+++ cid needs to map to shm_inputs
                self.notify_controller('cell_done',cid=content['cid'])

    def run(self):
        self.active = True

        try:
            # self.outputs_mapped = shm_to_nd_dict(**(self.outputs))

            self.processor = CellProcessor(self.len_period,self.catchments,self.inputs,self.outputs) #_mapped)

            msg_t = threading.Thread(target=self._get_cell_messages)
            msg_t.start()
            while (self.active):
                msg = self.get_message()
                self.handle_message(msg)
        except BaseException as e:
            self._handle_exception(e)
            raise
        finally:
            self.notify_controller('terminated',pid=self.pid)


class CellProcessor:
    '''
    Processes blocks of cells
    '''
    def __init__(self,len_period,catchments,inputs,outputs):

        self.catchments = catchments

        self.inputs = inputs
        self.outputs = outputs
        self.params = None

        self.len_period = len_period

        from awrams.models import awral
        imap = awral.get_default_mapping()

        dspec = {}
        for k in 'tmin_f','tmax_f','precip_f','solar_f','fday','radcskyt':
            dspec[k] = ['time','latitude','longitude']
        sp = [k for k in imap.mapping if k.endswith('_grid')]

        for k in sp:
            dspec[k] = ['latitude','longitude']
        dspec['height'] = ['hypsometric_percentile','latitude','longitude']
        dspec['hypsperc_f'] = ['hypsometric_percentile']

        data_map = {k:{} for k in dspec}

        for cid in self.catchments:
            input = shm_to_nd_dict_inputs(**(self.inputs[cid]))

            for k in dspec:
                data_map[k][cid] = input[k]

        for k in dspec:
            imap.mapping[k] = forcing_from_dict(data_map[k],k,dims=dspec[k])

        self.igraph = graph.ExecutionGraph(imap.mapping)

        self.sim = awral.get_runner(self.igraph.get_dataspecs(True))

    '''
    Core functions (running the model)
    '''

    def run_catchment(self,cid):
        input = self.igraph.get_data_prepack(cid)
        output = self.sim.run_from_mapping(input,self.len_period,self.catchments[cid].cell_count,True)
        output_nd = shm_to_nd_dict(**(self.outputs[cid]))

        for v in output_nd:
        # for v in self.outputs:
            # shape = output[v].shape
            # self.outputs[v][:,:shape[1]] = output[v]
            output_nd[v][...] = output[v]

    '''
    Parameter change functions
    '''
    def set_parameters(self,params):
        self.params = params
        for k,v in params.items():
            try:
                self.igraph.input_graph[k]['exe'].value = v
            except KeyError: ### ignore score column
                pass


class CalNodePyMP(mp.Process):
    def __init__(self,send_q,recv_q):
        mp.Process.__init__(self)
        self.recv_q = recv_q
        self.send_q = send_q

    def send_msg(self,msg):
        self.send_q.put(msg)

    def recv_msg(self):
        return self.recv_q.get()


class AWRALCalNodePyMP(AWRALCalNode,CalNodePyMP):
    def __init__(self,send_q,recv_q,node_id):
        AWRALCalNode.__init__(self,node_id)
        CalNodePyMP.__init__(self,send_q,recv_q)
