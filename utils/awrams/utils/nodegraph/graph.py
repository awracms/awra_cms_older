from collections import OrderedDict
from numbers import Number
from .nodes import DataSpec, InputNode,ProcessNode,get_expanded,get_flattened

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('graph')

def find_heads(nodes):
    '''
    Separate nodes into no-dependency (heads) and others (tails)
    '''
    #heads = []
    heads = OrderedDict()
    tails = {}
    for k,n in nodes.items():
        if n is None:
            raise Exception("Node value unspecified", k)
        if len(n.inputs) == 0:
            heads[k] = n
            #heads.append(k)
        else:
            tails[k] = n
    return heads, tails
    
def _get_input_tree(nodestr,nodes,tree=None):
    '''
    Return all nodes that are dependencies of the specified output
    UNORDERED
    '''
    if tree is None:
        tree = {}
    node = nodes[nodestr]
    if nodestr not in tree:
        tree[nodestr] = node
        for i in node.inputs:
            if not isinstance(i, Number):
                _get_input_tree(i,nodes,tree)
    return tree
    
def get_input_tree(outputs,nodes):
    '''
    Return all nodes that are dependencies of the specified outputs
    UNORDERED
    '''
    tree = {}
    if isinstance(outputs,str):
        outputs = [outputs]
    for o in outputs:
        tree = _get_input_tree(o,nodes,tree)
    return tree

def build_graph(heads,tails=None):
    '''
    Return a dependency-resolved ordered set of keys
    '''
    if tails is None:
        heads,tails = find_heads(heads)
    
    def all_met(heads,n):
        for i in n.inputs:
            if not isinstance(i,Number):
                if i not in heads:
                    return False
        return True
    
    unmet = {}
    
    found = False
    for k,n in tails.items():
        if all_met(heads,n):
            heads[k] = n
            found = True
        else:
            unmet[k] = n
            
    if len(tails) == 0:
        found = True
            
    if not found:
        print("UNMET:\n",unmet,'\n')
        raise Exception("Unsolvable graph")
    
    if len(unmet) > 0:
        return build_graph(heads,unmet)
    else:
        return heads

class ExecutionGraph:
    def __init__(self,mapping):
        exe_list = build_graph(mapping)

        self.input_graph = OrderedDict()
        self.process_graph = OrderedDict()

        self.mapping = mapping

        self.const_inputs = {}

        for nkey in exe_list:
            inputs = mapping[nkey].inputs
            for i in inputs:
                if isinstance(i,Number):
                    self.const_inputs[i] = i
            if len(inputs): # Processor
                self.process_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
            else: # Input endpoint
                self.input_graph[nkey] = dict(exe=mapping[nkey].get_executor())

    def get_dataspecs(self,flat=False):
        dspecs = {}

        for k,v in self.const_inputs.items():
            dspecs[k] = DataSpec('scalar',[],type(v))

        for k,v in self.input_graph.items():
            if flat:
                dspecs[k] = v['exe'].get_dataspec_flat()
            else:
                dspecs[k] = v['exe'].get_dataspec()

        for k,v in self.process_graph.items():
            dspecs[k] = v['exe'].get_dataspec([dspecs[i] for i in v['inputs']])
        return dspecs

    def get_data(self,coords):

        node_values = self.const_inputs.copy()

        for k,v in self.input_graph.items():
            try:
                node_values[k] = v['exe'].get_data(coords)
            except Exception as e:
                raise Exception("Failed generating %s" % k)

        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                raise Exception("Failed generating %s" % k)
        return node_values

    def get_data_prepack(self,cid):
        node_values = self.const_inputs.copy()

        for k,v in self.input_graph.items():
            try:
                node_values[k] = v['exe'].value
            except AttributeError:
                node_values[k] = v['exe'].get_data_prepack(cid)
            except Exception as e:
                # print(k,coords)
                raise Exception("Failed generating %s" % k)

        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                raise Exception("Failed generating %s" % k)
        return node_values

    def get_data_flat(self,coords,mask=None):
        if mask is None:
            import numpy as np
            #+++ May break in future, but for ease of testing....
            mask_shape = coords.shape[1],coords.shape[2]
            mask = np.empty(mask_shape,dtype=bool) # The 'null mask'
            mask.fill(False)

        node_values = self.const_inputs.copy()

        for k,v in self.input_graph.items():
            try:
                node_values[k] = v['exe'].get_data_flat(coords,mask)
            except Exception as e:
                # print(k,coords)
                # raise
                raise Exception("Failed generating %s" % k)

        for k,v in self.process_graph.items():
            try:
                node_values[k] = v['exe'].process([node_values[i] for i in v['inputs']])
            except Exception as e:
                raise Exception("Failed generating %s" % k)
        return node_values


def build_output_graph(heads,tails=None):
    '''
    Return a dependency-resolved ordered set of keys
    '''
    if tails is None:
        heads,tails = find_heads(heads)

    def all_met(heads,n):
        for i in n.inputs:
            if not isinstance(i,Number):
                if i not in heads:
                    return False
        return True

    unmet = {}

    found = False
    for k,n in tails.items():
        if all_met(heads,n):
            heads[k] = n
            found = True
        else:
            unmet[k] = n

    if len(tails) == 0:
        found = True

    if not found:
        print("UNMET:\n",unmet,'\n')
        raise Exception("Unsolvable graph")

    if len(unmet) > 0:
        return build_graph(heads,unmet)
    else:
        return heads


class OutputGraph:
    def __init__(self,mapping):
        '''

        :param mapping: outputs mapping
        '''
        exe_list = build_output_graph(mapping)

        self.input_graph = OrderedDict()   ### model outputs
        self.save_graph = OrderedDict()    ### persistent outputs
        self.writer_graph = OrderedDict()  ### ncfile writers

        self.mapping = mapping

        # self.const_inputs = {}
        for nkey in exe_list:
            inputs = mapping[nkey].inputs
            if len(inputs):
                if mapping[nkey].out_type == 'ncfile':
                    self.writer_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
                elif mapping[nkey].out_type == 'save':
                    self.save_graph[nkey] = dict(exe=mapping[nkey].get_executor(),inputs=mapping[nkey].inputs)
            else:
                if mapping[nkey].properties['io'] == 'from_model':
                    self.input_graph[nkey] = dict(exe=mapping[nkey].get_executor())

    def get_dataspecs(self,flat=False):
        dspecs = {}

        for k,v in self.input_graph.items():
            if flat:
                dspecs[k] = v['exe'].get_dataspec_flat()
            else:
                dspecs[k] = v['exe'].get_dataspec()

        for k,v in self.writer_graph.items():
            dspecs[k] = v['exe'].get_dataspec()

        return dspecs

    def set_data(self,coords,data_map,mask):
        ### constants for processing
        # node_values = self.const_inputs.copy()
        node_values = {}

        ### outputs from model
        for k,v in self.input_graph.items():
            try:
                v['exe'].set_data(data_map[k])
                node_values[k] = v['exe'].data
            except Exception as e:
                raise Exception("Failed generating %s" % k)

        ### nodes for data persistence (for ondemand not mp server)
        for k,v in self.save_graph.items():
            try:
                node_values[k] = v['exe'].set_data([node_values[i] for i in v['inputs']])
            except Exception as e:
                raise Exception("Failed generating %s" % k)

        ### nodes for writing data to file
        for k,v in self.writer_graph.items():
            try:
                v['exe'].process(coords,get_expanded(node_values[v['inputs'][0]],mask))
            except Exception as e:
                raise #Exception("Failed writing %s" % k)

        return node_values

    def initialise(self,time_coords):
        for k,v in self.writer_graph.items():
            v['exe'].init_files(time_coords)

    def sync_all(self):
        for k,v in self.writer_graph.items():
            v['exe'].sync_all()

    def close_all(self):
        for k,v in self.writer_graph.items():
            v['exe'].close()
