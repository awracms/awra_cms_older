from multiprocessing import Process, Queue as MPQueue
from queue import Queue, deque
import threading
from awrams.utils.messaging.general import message
import time
import pickle


class CommunicationManager:
    def __init__(self,submission_q,reply_queues,subtask_ids):
        self.submission_q = submission_q
        self.reply_queues = reply_queues
        self._stopped = threading.Event()
        self.subtask_ids = subtask_ids

        self.nodes = Queue()
        self.node_list = []

        self.jobs = Queue()

        self.children = []

        self.add_child_thread(self._control_loop)
        self.add_child_thread(self._node_work_loop)

    def add_child_thread(self,target,args=None):
        if args is None:
            args = []
        child = threading.Thread(target=target,args=args)
        self.children.append(child)

    def start(self):
        self.active=True
        for child in self.children:
            child.start()

    def _terminate(self):
        #Should shutdown nodes gracefully...
        self._terminate_nodes()
        self.active = False

    def terminate(self):
        self.submission_q.put(message('terminate'))

    def submit_evaluation(self,params,job_meta):
        job_msg = message('evaluate',params=params,job_meta=job_meta)
        self.add_job(job_msg)

    def add_job(self,job_msg):
        self.jobs.put(job_msg)

    def handle_message(self,msg):
        subject = msg['subject']
        if subject == 'evaluate':
            content = msg['content']
            self.submit_evaluation(**content)
        elif subject == 'terminate':
            self._terminate()
        else:
            raise Exception("Unknown message")

    def _control_loop(self):
        while self.active:
            msg = self.submission_q.get()
            self.handle_message(msg)

    def _node_work_loop(self):
        while self.active:
            avail = self._poll_recv()
            if avail:
                result = self._recv_msg()
                subject = result['subject']
                if subject == 'results':
                    content = result['content']
                    self.nodes.put(content['node_id'])
                    self.reply_queues[content['job_meta']['source']].put(result)
                else:
                    raise Exception("Not a result", result)

            if self.jobs.empty() or self.nodes.empty():
                #+++
                time.sleep(0.0001)
            else:
                job = self.jobs.get()
                node = self.nodes.get()
                self._send_msg(job,node)

    def broadcast_msg(self,msg):
        raise NotImplementedError()

    def _send_msg(self,msg,node):
        raise NotImplementedError()

    def _poll_recv(self):
        raise NotImplementedError()

    def _recv_msg(self):
        raise NotImplementedError()

    def _terminate_nodes(self):
        raise NotImplementedError()

class CommunicationManagerPyMP(CommunicationManager):
    def __init__(self,submission_q,reply_q,subtask_ids,node_count):
        CommunicationManager.__init__(self,submission_q,reply_q,subtask_ids)

        self.node_count = node_count
        self.out_queues = {}
        for n in range(self.node_count):
            self.out_queues[n] = MPQueue()
            self.nodes.put(n)

        self.recv_q = MPQueue()

    def broadcast_msg(self,msg):
        for node in range(self.node_count):
            self._send_msg(msg, node)

    def _send_msg(self,msg,node):
        self.out_queues[node].put(msg)

    def _poll_recv(self):
        return not self.recv_q.empty()

    def _recv_msg(self):
        return self.recv_q.get()

    def _terminate_nodes(self):
        for v in list(self.out_queues.values()):
            v.put(message('terminate'))

# class CommunicationManagerMPI(CommunicationManager):
#     def __init__(self,submission_q,reply_q,subtask_ids):
#         CommunicationManager.__init__(self,submission_q,reply_q,subtask_ids)
#
#         #+++
#         #Made this import local so jenkins doesn't see it...
#         from mpi4py import MPI
#
#         self.MPI = MPI
#
#         self.ANY_SOURCE = MPI.ANY_SOURCE
#
#         self.comm = MPI.COMM_WORLD
#         size = self.comm.Get_size()
#
#         self.recv_buf = bytearray(16384)
#
#         self.node_count = size-1
#
#         self.status = None
#
#         for node in range(1,size):
#             self.nodes.put(node)
#             self.node_list.append(node)
#
#     def broadcast_msg(self,msg):
#         for node in self.node_list:
#             self._send_msg(msg, node)
#
#     def _send_msg(self,msg,node):
#         self.comm.send(msg, node)
#
#     def _poll_recv(self):
#         if self.status is None:
#             self.status = self.MPI.Status()
#             self.r = self.comm.Irecv(self.recv_buf,source=self.ANY_SOURCE)
#
#         return self.r.Test(self.status)
#
#     def _recv_msg(self):
#         ret_val = pickle.loads(self.recv_buf)
#         self.status = None
#         return ret_val
#
#     def _terminate_nodes(self):
#         for node in range(1,self.node_count+1):
#             self.comm.send(message('terminate'), node)
#
#
