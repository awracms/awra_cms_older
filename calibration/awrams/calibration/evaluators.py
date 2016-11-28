from awrams.utils.messaging.general import message
from multiprocessing import Queue
import pandas as pd
import imp

class Evaluator:
    '''
    Base evaluator class used by optimizers
    '''
    def __init__(self):
        pass

    def evaluate_population(self,population):
        scores = []
        for k,v in population.iterrows():
            scores.append(self.evaluate(v))
        return scores

    def evaluate(self,parameters):
        raise NotImplementedError()

    def run_setup(self):
        '''
        Any process-safe initialisation should be performed here
        '''
        pass


class MEFactory:
    '''
    Factory class
    '''
    def __init__(self,job_q,n_lres,subtask_ids,objectives,log_q=None):
        self.job_q = job_q
        self.log_q = log_q
        self.n_lres = n_lres
        self.subtask_ids = subtask_ids
        self.objectives = objectives

        lobj_mod = imp.load_source('lobjf_mod',objectives.localf.filename)
        self.local_schema = getattr(lobj_mod,objectives.localf.classname).schema

        self.reply_queues = {}
        self.cur_id = 0

    def new(self,logging=True):
        self.cur_id += 1
        reply_q = Queue()

        self.reply_queues[self.cur_id] = reply_q

        log_q = self.log_q if logging else None

        return MessagingEvaluator(self.job_q,reply_q,self.n_lres,self.subtask_ids,self.objectives['globalf'],self.local_schema,self.cur_id,log_q)

class MessagingEvaluator:
    '''
    Provides a bridge from optimizers who call evaluate(), to message queue evaluators (eg CommunicationManager)
    '''
    def __init__(self,job_q,reply_q,n_lres,subtask_ids,global_objf,local_schema,eval_id,log_q=None):
        self.job_q = job_q
        self.reply_q = reply_q
        self.eval_id = eval_id
        self.subtask_ids = subtask_ids
        self.n_lres = n_lres
        self.results = {}
        self.result_counts = {}
        self.params = {}

        self._gobjf_data = global_objf

        self.local_schema = local_schema

        if log_q is not None:
            self.log_q = log_q
            self.logging = True
        else:
            self.logging = False

    def run_setup(self):
        modname = self._gobjf_data.get('modulename')
        if modname is not None:
            gobj_mod = imp.importlib.import_module(modname)
        else:
            gobj_mod = imp.load_source('gobjf_mod',self._gobjf_data['filename'])

        gobjf_obj = getattr(gobj_mod,self._gobjf_data['classname'])
        self.global_obj = gobjf_obj()

    def log_results(self,parameters,global_score,local_scores):
        self.log_q.put(message('log_results',parameters=parameters,global_score=global_score,local_scores=local_scores))

    def evaluate(self,parameters):
        self._submit_evaluation(parameters)
        lresults = self._retrieve_results()['results']
        gscore = self.global_score(lresults)

        if self.logging:
            self.log_results(parameters,gscore,lresults)

        return gscore

    def _submit_evaluation(self,params,job_tag='default'):
        job_meta = dict(source=self.eval_id,job_tag=job_tag)
        job_msg = message('evaluate',params=dict(params),job_meta=job_meta)
        self.job_q.put(job_msg)
        self.params[job_tag] = dict(params)
        self.results[job_tag] = pd.DataFrame(index=self.subtask_ids,columns=self.local_schema,dtype=float)
        self.result_counts[job_tag] = 0

    def _retrieve_results(self):
        while 1:
            msg = self.reply_q.get()['content']
            tag = msg['job_meta']['job_tag']
            all_res = self._handle_results(msg['results'],tag)
            if all_res is not None:
                return {'tag':tag,'results':all_res}

    def _handle_results(self,in_results,rkey):
        cur_results = self.results[rkey]
        cur_resc = self.result_counts[rkey]
        for k,v in list(in_results.items()):
            cur_results.loc[k] = v
            cur_resc += 1
        if cur_resc == self.n_lres:
            self.results.pop(rkey)
            self.result_counts.pop(rkey)
            return cur_results
        else:
            self.result_counts[rkey] = cur_resc

    def global_score(self,lresults):
        score = self.global_obj.evaluate(lresults)
        # Might be the place to do logging...
        return score

    def evaluate_population(self,population):
        scores = pd.Series(index=population.index)

        for k,v in population.iterrows():
            self._submit_evaluation(v,k)

        for i in range(len(population)):
            lresults = self._retrieve_results()
            gscore = self.global_score(lresults['results'])

            parameters = population.loc[lresults['tag']]

            if self.logging:
                self.log_results(parameters,gscore,lresults['results'])
            # 'tag' -> parameter set...
            scores[lresults['tag']] = gscore
        return scores
