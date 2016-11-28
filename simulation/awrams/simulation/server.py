from awrams.utils.nodegraph import nodes, graph
from awrams.models.awral import transforms
import numpy as np

from awrams.utils import extents
from awrams.utils import mapping_types as mt
from awrams.utils import datetools as dt
from awrams.utils.catchments import subdivide_extent

#from awrams.utils.messaging.buffers import 
from awrams.utils.messaging import message
from awrams.utils.messaging.buffer_group import BufferGroup, DataSpec, create_managed_buffergroups
from awrams.utils.messaging.robust import SharedMemClient, ControlMaster
from awrams.simulation import input_reader
from awrams.simulation import writer
from awrams.simulation import modelgraph as mg

from awrams.utils.awrams_log import get_module_logger
logger = get_module_logger('server')


import multiprocessing as mp

class Server:
    def __init__(self,model):
        ### defaults
        self.spatial_chunk = 32

        self.num_workers = 4
        self.read_ahead = 1

        self.model = model

        # self.set_max_dims(self.spatial_chunk)

    def _set_max_dims(self,mapping):
        self.max_dims = {'cell': self.spatial_chunk*self.spatial_chunk,'time': 366,'day_of_year': 366}

        for k,v in mapping.dimensions.items():
            if v is not None:
                self.max_dims[k] = v

    def run(self,input_map,output_map,period,extent): #periods,chunks):
        '''
        Should be the basis for new-style sim server
        Currently no file output, but runs inputgraph/model quite happily...
        '''
        import time
        start = time.time()

        self._set_max_dims(input_map)

        chunks = subdivide_extent(extent,self.spatial_chunk)
        periods = dt.split_period(period,'a')

        logger.info("Getting I/O dataspecs...")
        mapping = input_map.mapping
        filtered = graph.get_input_tree(self.model.get_input_parameters(),mapping)

        input_nodes = {}
        worker_nodes = {}
        output_nodes = {}

        for k,v in filtered.items():
            if 'io' in v.properties:
                input_nodes[k] = v
                worker_nodes[k] = nodes.const(None)
            else:
                worker_nodes[k] = v

        for k,v in output_map.mapping.items():
            try:
                if v.properties['io'] == 'from_model':
                    output_nodes[k] = v
            except: # AttributeError:
                pass
                # print("EXCEPTION",k,v)

        input_dspecs = graph.ExecutionGraph(input_nodes).get_dataspecs(True)
        model_dspecs = graph.ExecutionGraph(mapping).get_dataspecs(True)
        output_dspecs = graph.OutputGraph(output_nodes).get_dataspecs(True)

        self.model.init_shared(model_dspecs)

        ### initialise output ncfiles
        logger.info("Initialising output files...")
        outgraph = graph.OutputGraph(output_map.mapping)
        outgraph.initialise(mt.period_to_tc(period))

        #+++ Can we guarantee that statespecs will be 64bit for recycling?

        # NWORKERS = 2
        # READ_AHEAD = 1

        sspec = DataSpec('array',['cell'],np.float64)

        state_specs = {}
        for k in self.model.get_state_keys():
            init_k = 'init_' + k

            input_dspecs[init_k] = sspec
            state_specs[k] = sspec

        logger.info("Building buffers...")
        input_bufs = create_managed_buffergroups(input_dspecs,self.max_dims,self.num_workers+self.read_ahead)
        state_bufs = create_managed_buffergroups(state_specs,self.max_dims,self.num_workers*2)
        output_bufs = create_managed_buffergroups(output_dspecs,self.max_dims,self.num_workers+self.read_ahead)

        all_buffers = dict(inputs=input_bufs,states=state_bufs,outputs=output_bufs)

        smc = SharedMemClient(all_buffers,False)

        control_master = mp.Queue()
        control_status = mp.Queue()

        state_returnq =mp.Queue()

        chunkq = mp.Queue()

        chunkoutq = mp.Queue()

        reader_inq = dict(control=mp.Queue(),state_return=state_returnq)
        reader_outq = dict(control=control_master,chunks=chunkq)

        writer_inq = dict(control=mp.Queue(),chunks=chunkoutq)
        writer_outq = dict(control=control_master,log=mp.Queue()) #,chunks=chunkq)

        child_control_qs = [reader_inq['control'],writer_inq['control'],writer_outq['log']]

        logger.info("Running simulation...")
        workers = []
        for w in range(self.num_workers):
            worker_inq = dict(control=mp.Queue(),chunks=chunkq)
            worker_outq = dict(control=control_master,state_return=state_returnq,chunks=chunkoutq)
            worker_p = mg.ModelGraphRunner(worker_inq,worker_outq,all_buffers,chunks,periods,worker_nodes,self.model)
            worker_p.start()
            workers.append(worker_p)
            child_control_qs.append(worker_inq['control'])

        control = ControlMaster(control_master, control_status, child_control_qs)
        control.start()

        reader_p = input_reader.InputGraphRunner(reader_inq,reader_outq,all_buffers,chunks,periods,input_nodes,self.model.get_state_keys())
        reader_p.start()

        writer_p = writer.OutputGraphRunner(writer_inq,writer_outq,all_buffers,chunks,periods,output_map.mapping)
        writer_p.start()

        log = True
        while log:
            msg = writer_outq['log'].get()
            if msg['subject'] == 'terminate':
                log = False
            else:
                logger.info(msg['subject'])

        writer_p.join()

        for w in workers:
            w.qin['control'].put(message('terminate'))
            # control_master.get_nowait()
            w.join()

        reader_inq['control'].put(message('terminate'))
        control_master.put(message('finished'))

        problem = False
        msg = control_status.get()
        if msg['subject'] == 'exception_raised':
            problem = True
        control.join()

        reader_p.join()

        if problem:
            raise Exception("Problem detected")
        logger.info("elapsed time: %.2f",time.time() - start)

def test_simple_outputs(period,extent,output_mapping=None):
    # from awrams.utils.nodegraph import nodes, graph
    # from awrams.models.awral import transforms
    # import numpy as np

    # from awrams.utils import extents
    from awrams.utils import mapping_types as mt
    from awrams.utils import datetools as dt

    # from awrams.utils.messaging import message

    from awrams.models import awral

    mapping = awral.get_default_mapping()

    if output_mapping is None:
        from awrams.models.awral.template import DEFAULT_TEMPLATE
        output_mapping = awral.get_output_nodes(DEFAULT_TEMPLATE)

    ### initialise output ncfiles
    # coords = [mt.period_to_tc(period)]
    # coords.extend(mt.extent_to_spatial_coords(extent))
    # o = graph.OutputGraph(output_mapping.mapping)
    # o.initialise(coords)

    max_dims = {'cell': 32*32,'time': 366}

    for k,v in mapping.dimensions.items():
        if v is not None:
            max_dims[k] = v

    # periods = dt.split_period(period,'a')

    #chunks = [extents.from_boundary_offset(400,400,431,431),\
    #      extents.from_boundary_offset(200,200,201,201)]

    # from awrams.utils.catchments import subdivide_extent
    # #e = extents.default()
    # e = extent #.from_boundary_offset(200,200,250,250)
    # chunks = subdivide_extent(e,32)

    import time

    start = time.time()

    test_sim(awral,max_dims,mapping,output_mapping,period,extent) #periods,chunks)

    end = time.time()

    print(end-start)

def build_output_graph():
    from awrams.utils.nodegraph import nodes,graph

    from awrams.models import awral
    from awrams.models.awral import ffi_wrapper as fw
    from awrams.models.awral.template import DEFAULT_TEMPLATE

    output_map = awral.get_output_nodes(DEFAULT_TEMPLATE)
    print(output_map)

    outpath = '/data/cwd_awra_data/awra_test_outputs/sbaronha/sim_test_outputs/'
    output_map.mapping.update({
        's0_avg': nodes.transform(nodes.average,['s0_dr','s0_sr']),
        's0_avg_save': nodes.write_to_annual_ncfile(outpath,'s0_avg'),

        'ss_avg': nodes.transform(nodes.average,['ss_dr','ss_sr']),
        'ss_avg_save': nodes.write_to_annual_ncfile(outpath,'ss_avg'),

        # 'sd_avg': nodes.transform(nodes.average,['s0_dr','s0_sr']),
        # 'sd_avg_save': nodes.write_to_annual_ncfile('./','s0_avg'),
        #
        # 'qtot_avg_save': nodes.write_to_annual_ncfile('./','qtot'),
        # 'etot_avg_save': nodes.write_to_annual_ncfile('./','etot')
        })
    outputs = graph.OutputGraph(output_map.mapping)
    # print(outputs.get_dataspecs())
    # print(outputs.get_dataspecs(flat=True))
    return outputs

def test_output_filewrite():
    import awrams.models.awral.settings as settings
    settings.CLIMATE_DATA = '/mnt/awramsi_test_data/AWAP/'
    from awrams.utils import extents
    from awrams.utils import datetools as dt
    outputs = build_output_graph()
    test_simple_outputs(dt.dates('2008-2009'),
                        extents.from_boundary_offset(200,200,250,250),
                        output_mapping=outputs)

def test_big_filewrite():
    import awrams.models.awral.settings as settings
    settings.CLIMATE_DATA = '/mnt/awramsi_test_data/AWAP/'
    from awrams.utils import extents
    from awrams.utils import datetools as dt
    outputs = build_output_graph()
    test_simple_outputs(dt.dates('2000-2015'),
                        extents.default(),
                        output_mapping=outputs)

if __name__ == '__main__':
    # test_simple_outputs()
    # test_output_filewrite()
    test_big_filewrite()
