from awrams.utils.nodegraph import graph,nodes
from awrams.utils import mapping_types as mt

class OnDemandSimulator:
    def __init__(self,model,imapping,omapping=None):
        self.input_runner = graph.ExecutionGraph(imapping)
        self.model_runner = model.get_runner(self.input_runner.get_dataspecs(True))

        self.outputs = None
        if omapping is not None:
            output_vars = []
            for v in self.model_runner.template['OUTPUTS_AVG'] + self.model_runner.template['OUTPUTS_CELL']:
                output_vars.append(v)
            for v in self.model_runner.template['OUTPUTS_HRU']:
                output_vars.extend([v+'_sr',v+'_dr'])
            for v in output_vars:
                omapping[v] = nodes.model_output(v)
            self.outputs = graph.OutputGraph(omapping)

    def run(self,period,extent,return_inputs=False):
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(coords[0])

        iresults = self.input_runner.get_data_flat(coords,extent.mask)
        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count,True)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        if return_inputs:
            return mresults,iresults
        else:
            return mresults

    def run_prepack(self,iresults,period,extent):
        '''
        run with pre-packaged inputs for calibration
        :param cid:
        :param period:
        :param extent:
        :return:
        '''
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(coords[0])

        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count,True)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        return mresults
