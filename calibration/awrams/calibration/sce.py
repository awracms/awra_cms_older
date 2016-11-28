import numpy as np
import pandas as pd
from awrams.utils.awrams_log import get_module_logger
import multiprocessing as mp
from awrams.utils.messaging.binding import MessageHandler, MultiprocessingParent, QueueChild, bound_proxy
from awrams.utils.messaging.general import message
import time

logger = get_module_logger('SCE')

class ShuffledOptimizer:
    def __init__(self, complex_sz, n_complexes, parameters,eval_fac,min_complexes=1):
        '''
        s : pop_size (initial population)
        m : complex size
        p : number of complexes
        pmin : minimum number of complexes
        '''
        self.complex_sz = complex_sz
        self.n_complexes = n_complexes
        self._n_complexes = n_complexes
        self.pop_size = complex_sz*n_complexes
        self.parameters = parameters
        self.min_complexes = min_complexes

        self.evaluator = eval_fac.new()
        self.evaluator.run_setup()

        self.max_shuffle = np.inf # Max shuffling loops
        self.max_iter = 20000 # Max model evaluations
        self.target_score = 1e-8
        self.max_nsni = 5 # Max shuffle without improvement (as defined below)
        self.min_imp = 0.01 # Minimum proportional change required for 'improvement' metric

        self.evaluations = 0

        self.prev_best = []

    def generate_initial_population(self,seed=None):
        '''
        Should accept distribution and seed values
        For now just generate uniformally from parameters
        '''
        self.population = pd.DataFrame(columns=list(self.parameters.index) + ['Score'])

        self._plist = list(self.parameters.index)

        if seed is not None:
            self.population.loc[0] = seed
            sds = (self.parameters.Max - self.parameters.Min)/6.0
            for i in range(1,self.pop_size):
                self.population.loc[i] = generate_random_point(self.parameters,False,seed,sds)
        else:
            for k,v in self.parameters.iterrows():
                self.population[k] = np.random.uniform(low=v['Min'],high=v['Max'],size=self.pop_size)

        return self.population

    def rank_points(self,ascending=True):
        try:
            self.population.sort_values('Score',ascending=ascending,inplace=True)
        except AttributeError:
            self.population.sort('Score',ascending=ascending,inplace=True)

    def _gen_shuffle_indices(self):
        complex_indices = []
        for c in range(self.n_complexes):
            indices = c + (np.arange(self.complex_sz) * self.n_complexes)
            complex_indices.append(indices)
        return complex_indices

    def partition_into_complexes(self):
        '''
        Generate indices of ranked data
        '''
        shuffle_indices = self._gen_shuffle_indices()
        complexes = []
        for c in shuffle_indices:
            complexes.append(self.population.iloc[c])
        return complexes

    def evaluate_population(self):
        # Iterate through parameters we want evaluated
        # Request evaluation of this parameter set from said node
        # Ideally nodes are divisible by number of simulatenous executed paramsets
        # eg equal to number of complexes
        self.population['Score'] = self.evaluator.evaluate_population(self.population[self._plist])

    def evolve_complexes(self,complex_indices):
        '''
        Call the evolution function of each complex
        '''
        # Logic for each complex could live in evaluation node
        # Or be executed locally (with parameters distributed)
        raise NotImplementedError()

    def reduce_complexes(self):
        if self.n_complexes > self.min_complexes:
            self.population = self.population.iloc[:-self.complex_sz]
            self.n_complexes -= 1
            self.pop_size = self.complex_sz*self.n_complexes

    def check_convergence(self):
        '''
        Evaluate convergence criteria
        '''
        cur_best = self.population['Score'].min()

        if self.evaluations >= self.max_iter:
            print("Max evaluations executed")
            return True
        if cur_best < self.target_score:
            print("Target score achieved")
            return True
        if self.shuffle_loops >= self.max_shuffle:
            print("Maximum shuffle loops executed")
            return True

        if len(self.prev_best) >= self.max_nsni:
            imp_target = self.prev_best[0] - (self.prev_best[0] * self.min_imp)
            if cur_best >= imp_target:
                print("Maximum non-improving shuffle loops executed")
                return True
            else:
                self.prev_best = []

        self.prev_best.append(cur_best)
        if len(self.prev_best) > self.max_nsni:
            self.prev_best.pop(0)
        
        return False

    def run_optimizer(self,seed=None):
        # self.n_complexes = self._n_complexes
        self.generate_initial_population(seed)

        run_start_time = time.time()

        s_start = run_start_time

        self.evaluate_population()
        self.prev_best = [self.population['Score'].min()]

        self.shuffle_loops = 0
        self.evaluations = len(self.population)

        converged = self.check_convergence()

        print("%s (%s): %s, %s, %s, (%.2fs)" % (self.shuffle_loops, self.evaluations, self.population['Score'].min(), self.population['Score'].mean(), self.population['Score'].max(), time.time()-s_start))

        while not converged:
            self.rank_points()
            c_indices = self.partition_into_complexes()
            e_iter = self.evolve_complexes(c_indices,self.population.std())
            self.evaluations += e_iter
            self.shuffle_loops += 1
            self.reduce_complexes()
            s_end = time.time()
            print("%s (%s): %s, %s, %s, (%.2fs)" % (self.shuffle_loops, self.evaluations, self.population['Score'].min(), self.population['Score'].mean(), self.population['Score'].max(), s_end-s_start))
            converged = self.check_convergence()
        return self.population.iloc[0]

class SCEOptimizer(ShuffledOptimizer):
    def __init__(self, complex_sz,n_complexes,sub_sz,n_offspring,n_evol,parameters,evaluator):
        ShuffledOptimizer.__init__(self,complex_sz,n_complexes,parameters,evaluator)

        self.evolvers = []
        for e in range(n_complexes):
            self.evolvers.append(ComplexEvolver(complex_sz,sub_sz,parameters,evaluator,e))
        self.n_offspring = n_offspring
        self.n_evol = n_evol

    def evolve_complexes(self,c_indices,spread=None):
        num_eval = 0
        for i in range(self.n_complexes):
            results = self.evolvers[i].evolve(c_indices[i],self.n_offspring,self.n_evol)
            num_eval += results['evaluations_performed']
            idx = np.arange(i*self.complex_sz,(i+1)*self.complex_sz)
            self.population.iloc[idx] = np.array(results['population'])
        self.rank_points()
        return num_eval

class ProxyOptimizer(ShuffledOptimizer,MultiprocessingParent,MessageHandler):
    def __init__(self, complex_sz,n_complexes,sub_sz,n_offspring,n_evol,parameters,eval_factory,min_complexes=1,
                 random_seed=None):
        MultiprocessingParent.__init__(self)
        ShuffledOptimizer.__init__(self,complex_sz,n_complexes,parameters,eval_factory,min_complexes)

        c = self.control_q

        self.evolvers = []
        #mappings = {'evolve': {'subject': 'handle_results'}}
        for e in range(n_complexes):
            queues = dict(control=c,input=mp.Queue())
            p = PComplexEvolver(queues,complex_sz,sub_sz,parameters,eval_factory,e)
            #proxy,process = instantiate_q_proxy(ComplexEvolver,queues,mappings,complex_sz,sub_sz,parameters,eval_factory,e)
            self.add_child_proc(p,queues['input'])
            proxy = bound_proxy(p)(queues)
            proxy.seed(random_seed)
            self.evolvers.append(proxy)

        self.n_offspring = n_offspring
        self.n_evol = n_evol

    def evolve_complexes(self,c_indices,spread):
        self.num_eval = 0

        self.evolvers_done = set()

        for i in range(self.n_complexes):
            self.evolvers[i].evolve(c_indices[i],self.n_offspring,self.n_evol,spread)

        all_complexes = set(range(self.n_complexes))
        while not (self.evolvers_done == all_complexes):
            self.poll_children()
        
        #+++ May not want to sort here - ie culling worst (pre-sorted) complex vs worst N-points
        self.rank_points()
        return self.num_eval

    def run_optimizer(self,seed=None):
        try:
            return ShuffledOptimizer.run_optimizer(self,seed)
        except:
            self.terminate_children()
            raise

    def handle_results(self,evaluations_performed,population,evolver_id):
        i = evolver_id
        self.num_eval += evaluations_performed
        idx = np.arange(i*self.complex_sz,(i+1)*self.complex_sz)
        self.population.iloc[idx] = np.array(population)
        self.evolvers_done.add(evolver_id)

    def handle_child_message(self,m):
        self._handle_message(m)

    def terminated(self,pid):
        self.child_exited(pid)


class ComplexEvolver:
    def __init__(self, complex_sz,subcomplex_sz,parameters,eval_factory,evolver_id):
        self.size = complex_sz
        self.subcomplex_sz = subcomplex_sz
        self.evaluator = eval_factory.new()
        self.parameters = parameters
        self.evolver_id = evolver_id

    def seed(self,random_seed):
        if not random_seed is None:
            seed = random_seed + self.evolver_id
            np.random.seed(seed)

    def out_of_bounds(self,point):
        return out_of_bounds(point,self.parameters)

    def random_point(self,seed_point=None,spread=None):
        if spread is None:
            spread = self.population.std()
        if seed_point is None:
            return generate_random_point(self.parameters,True)
        else:
            return generate_random_point(self.parameters,False,seed_point,spread)

    def evolve(self,population,num_offspring,num_evolutions,alpha=1.0,beta=0.5,spread=None):

        self.population = population.copy()

        num_i = 0
        for e in range(num_evolutions):
            sub_idx = self._select_simplex()

            #+++
            #iloc vs indexes amibiguous?
            #pandas returning copies vs views?
            #refactor using pure numpy arrays!

            #
            simplex = self.population.iloc[sub_idx].copy()

            try:
                simplex.sort_values('Score',ascending=True,inplace=True)
            except AttributeError: ### pandas pre 0.17.0
                simplex.sort('Score',ascending=True,inplace=True)

            for o in range(num_offspring):
                #Identify worst point, centroid of best point
                centroid = simplex.iloc[:-1].mean()[:-1]
                #+++ Original worst...
                worst_pt = simplex.iloc[-1][:-1]
                best_pt = simplex.iloc[0][:-1]

                #Attempt reflection
                #III

                new_point = centroid + alpha*(centroid-worst_pt)

                if self.out_of_bounds(new_point):
                    new_point = self.random_point(best_pt,spread)

                score = self.evaluator.evaluate(new_point)
                num_i += 1
                # Hardcoding for minimisation +++
                if score < simplex.iloc[-1]['Score']:
                    simplex.iloc[-1] = new_point
                    simplex.iloc[-1]['Score'] = score
                    #print "Reflection : %s" % score
                    #goto VII
                else: #V
                    # Contraction
                    contraction = worst_pt + beta *(centroid-worst_pt)
                    score = self.evaluator.evaluate(contraction)
                    num_i +=1
                    if score < simplex.iloc[-1]['Score']:
                        simplex.iloc[-1] = contraction
                        simplex.iloc[-1]['Score'] = score
                        #print "Contraction : %s" % score
                    else:
                        new_point = self.random_point(best_pt,spread)
                        score = self.evaluator.evaluate(new_point)
                        num_i +=1
                        simplex.iloc[-1] = new_point
                        simplex.iloc[-1]['Score'] = score

                try:
                    simplex.sort_values('Score',ascending=True,inplace=True)
                except AttributeError: ### pandas pre 0.17.0
                    simplex.sort('Score',ascending=True,inplace=True)

            self.population.iloc[sub_idx] = np.array(simplex)
            try:
                self.population.sort_values('Score',ascending=True,inplace=True)
            except AttributeError: ### pandas pre 0.17.0
                self.population.sort('Score',ascending=True,inplace=True)

        self.last_evaluations = num_i
        return dict(evaluations_performed = self.last_evaluations, population=self.population, evolver_id = self.evolver_id)
    
    def _select_simplex(self):
        sub_idx = []
        selected = 0
        while selected != self.subcomplex_sz:
            #+++ Fixed triangular distribution - evaluate alternatives
            idx = int(np.random.triangular(0.,0.,self.size))
            if idx not in sub_idx:
                sub_idx.append(idx)
                selected += 1
        return sub_idx

def generate_random_point(parameters,uniform=True,means=None,sds=None):
    point = {}
    if uniform:
        for k,v in parameters.iterrows():
            point[k] = np.random.uniform(low=v['Min'],high=v['Max'])
    else:
        #Prevent 0 standard deviations
        sds = sds+1.e-99
        for k,v in parameters.iterrows():
            valid = False
            while not valid:
                point[k] = np.random.normal(means[k],sds[k])
                valid = point[k] <= parameters['Max'][k] and point[k] >= parameters['Min'][k]
    return pd.Series(point)

def out_of_bounds(point,parameters):
    return (point[parameters.index] > parameters['Max']).any() or (point[parameters.index] < parameters['Min']).any()

class EvalFacStub:
    @classmethod
    def new(self):
        pass

class PComplexEvolver(ComplexEvolver,QueueChild):
    def __init__(self,pipes,complex_sz,subcomplex_sz,parameters,eval_factory,evolver_id):
        QueueChild.__init__(self,pipes)
        ComplexEvolver.__init__(self,complex_sz,subcomplex_sz,parameters,eval_factory,evolver_id)

    def run_setup(self):
        self.evaluator.run_setup()

    def evolve(self,population,num_offspring,num_evolutions,alpha=1.0,beta=0.5,spread=None):
        results = ComplexEvolver.evolve(self,population,num_offspring,num_evolutions,alpha=1.0,beta=0.5,spread=spread)
        self._send_msg(message('handle_results',**results))