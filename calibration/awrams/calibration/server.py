from . import evaluators
from awrams.utils.messaging.general import message
from .communication import *
from awrams.calibration.node import *
from awrams.calibration import sce
from awrams.calibration.logger import CalibrationLogger
from awrams.utils.messaging.binding import build_queues, bound_proxy
import imp

SCE_DEFAULT = dict(complex_sz=43,n_complexes=14,sub_sz=22,n_offspring=1,n_evol=43)

class CalibrationServer:
    def __init__(self,subtask_ids,params,objectives,node_settings,hyperp,termp):
        self.sub_q = MPQueue()
        self.reply_q = {}

        queues = build_queues()
        self.logger_q = queues['input']

        lobj_mod = imp.load_source('lobjf_mod',node_settings.objective.localf.filename)
        local_schema = getattr(lobj_mod,node_settings.objective.localf.classname).schema

        self.logger = CalibrationLogger(queues,list(params.index),subtask_ids,local_schema,node_settings['logfile'])
        self.logger.start()

        self.subtasks = subtask_ids

        self.messaging_factory = evaluators.MEFactory(self.sub_q,len(subtask_ids),subtask_ids,objectives,self.logger_q)

        self.create_comms_manager()
        self.create_nodes()

        self.setup_nodes(node_settings)
        self.comms_manager.start()

        #+++ We should wait for nodes to return OK, or fail if they fail...
        #Currently just gets stuck in optimizer-land without propagating node-fails
        #self.wait_on_nodes()

        self.optimizer = sce.ProxyOptimizer(parameters=params,eval_factory=self.messaging_factory,**hyperp)

        for k,v in termp.items():
            self.optimizer.__setattr__(k,v)

    def create_nodes(self):
        raise NotImplementedError

    def create_comms_manager(self):
        raise NotImplementedError

    def run_optimization(self,seed=None):
        return self.optimizer.run_optimizer(seed)

    def setup_nodes(self,node_settings):
        msg = message('settings',node_settings=node_settings)
        self.comms_manager.broadcast_msg(msg)

    def terminate(self,wait_for_results=True):
        self.comms_manager.terminate()
        iterations = self.optimizer.evaluations if wait_for_results else -1

        self.logger_q.put(message('terminate',iterations=self.optimizer.evaluations))
        self.logger.join()

        self.optimizer.terminate_children()

class CalibrationServerPyMP(CalibrationServer):
    def __init__(self,subtask_ids,params,objectives,node_count,node_settings,hyperp=None,termp=None):

        if termp is None:
            termp = {}

        if hyperp is None:
            hyperp = SCE_DEFAULT

        self.node_count = node_count
        CalibrationServer.__init__(self,subtask_ids,params,objectives,node_settings,hyperp,termp)

    def create_comms_manager(self):
        self.comms_manager = CommunicationManagerPyMP(self.sub_q,self.messaging_factory.reply_queues,self.subtasks,self.node_count)

    def create_nodes(self):
        self.nodes = []

        for i in range(self.node_count):
            node = AWRALCalNodePyMP(self.comms_manager.recv_q,self.comms_manager.out_queues[i],i)
            node.start()
            self.nodes.append(node)

    def terminate(self,wait_for_results=True):
        CalibrationServer.terminate(self,wait_for_results)

        for node in self.nodes:
            node.join()
