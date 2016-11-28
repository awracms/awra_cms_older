from nose.tools import nottest,with_setup

def test_imports():
    import awrams.simulation
    # assert False

@nottest
def change_path_to_forcing(imap):
    from os.path import join,dirname
    data_path = join(dirname(__file__),'..','..','test_data','simulation')
    imap.mapping['precip_f'].args['path'] = data_path
    imap.mapping['tmax_f'].args['path'] = data_path
    imap.mapping['tmin_f'].args['path'] = data_path
    imap.mapping['solar_f'].args['path'] = data_path

    # from awrams.utils.nodegraph.nodes import forcing_from_ncfiles
    # data_path = '/data/cwd_awra_data/awra_inputs/climate_generated/' #_sdcvd-awrap01/'
    # imap.mapping['precip_f'] = forcing_from_ncfiles(data_path+'/rr',"rr*",'rain_day') #'bom-rain_day-*'
    # imap.mapping['tmax_f']   = forcing_from_ncfiles(data_path+'/tmax',"tmax*",'temp_max_day') #'bom-tmax_day-*'
    # imap.mapping['tmin_f']   = forcing_from_ncfiles(data_path+'/tmin',"tmin*",'temp_min_day') # 'bom-tmin_day-*'
    # imap.mapping['solar_f']  = forcing_from_ncfiles(data_path+'/solar',"solar*",'solar_exposure_day') # ''bom-rad_day-*'

@nottest
def insert_climatology(imap):
    from os.path import join,dirname
    from awrams.utils.nodegraph import nodes
    data_path = join(dirname(__file__),'..','..','test_data','simulation')
    cpath = join(data_path,'climatology_daily_solar_exposure_day.nc')
    ipath = data_path

    imap.mapping['solar_f'] = nodes.forcing_gap_filler(ipath,'solar*','solar_exposure_day',cpath)

@nottest
def get_initial_states(imap):
    from os.path import join,dirname
    from awrams.utils.nodegraph import nodes

    mapping = imap.mapping
    data_path = join(dirname(__file__),'..','..','test_data','simulation')
    mapping['init_sr'] = nodes.init_state_from_ncfile(data_path,'sr_bal*','sr_bal')
    mapping['init_sg'] = nodes.init_state_from_ncfile(data_path,'sg_bal*','sg_bal')

    HRU = {'_hrusr':'_sr','_hrudr':'_dr'}
    for hru in ('_hrusr','_hrudr'):
        for state in ["s0","ss","sd",'mleaf']:
            mapping['init_'+state+hru] = nodes.init_state_from_ncfile(data_path,state+HRU[hru]+'*',state+HRU[hru])

@nottest
def get_initial_states_dict(imap,period,extent):
    from os.path import join,dirname
    from awrams.utils.io.data_mapping import SplitFileManager
    from awrams.utils.nodegraph import nodes

    data_map = {}
    data_path = join(dirname(__file__),'..','..','test_data','simulation')
    period = [period[0] - 1]
    node_names = {'mleaf_dr': 'init_mleaf_hrudr',
                  'mleaf_sr': 'init_mleaf_hrusr',
                  's0_dr': 'init_s0_hrudr',
                  's0_sr': 'init_s0_hrusr',
                  'ss_dr': 'init_ss_hrudr',
                  'ss_sr': 'init_ss_hrusr',
                  'sd_dr': 'init_sd_hrudr',
                  'sd_sr': 'init_sd_hrusr',
                  'sg_bal': 'init_sg',
                  'sr_bal': 'init_sr'}
    for k in 'mleaf_dr','s0_dr','sd_dr','sg_bal','ss_dr','mleaf_sr','s0_sr','sd_sr','sr_bal','ss_sr':
        sfm = SplitFileManager.open_existing(data_path,k+'*nc',k)
        data_map[node_names[k]] = sfm.get_data(period,extent)
    nodes.init_states_from_dict(imap,data_map,extent)

@nottest
def setup():
    from os.path import join,dirname

    from awrams.utils import datetools as dt

    from awrams.utils.nodegraph import nodes,graph
    from awrams.models import awral
    from awrams.models.awral.template import DEFAULT_TEMPLATE
    from awrams.utils.mapping_types import period_to_tc

    global period
    period = dt.dates('dec 2010 - jan 2011')

    global input_map
    input_map = awral.get_default_mapping()
    change_path_to_forcing(input_map)

    global output_map
    output_map = awral.get_output_nodes(DEFAULT_TEMPLATE)

    global outpath
    outpath = join(dirname(__file__),'..','..','test_data','simulation','outputs')

    output_map.mapping['s0_ncsave'] = nodes.write_to_annual_ncfile(outpath,'s0')
    outgraph = graph.OutputGraph(output_map.mapping)


def tear_down():
    from os import remove
    from os.path import join
    remove(join(outpath,'s0_2010.nc'))
    remove(join(outpath,'s0_2011.nc'))

# @nottest
@with_setup(setup,tear_down)
def test_ondemand_region():
    from awrams.utils import extents
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral
    ### test region
    extent = extents.from_boundary_coords(-32,115,-35,118)
    sim = OnDemandSimulator(awral,input_map.mapping,omapping=output_map.mapping)
    r = sim.run(period,extent)

# @nottest
@with_setup(setup,tear_down)
def test_ondemand_point():
    from awrams.utils import extents
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral
    ### test point
    extent = extents.from_cell_coords(-32.1,115.1)
    sim = OnDemandSimulator(awral,input_map.mapping,omapping=output_map.mapping)
    r = sim.run(period,extent)

# @nottest
@with_setup(setup)
def test_climatology_point():
    import numpy as np
    from awrams.utils import extents
    from awrams.utils import datetools as dt
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    ### test point
    period = dt.dates('dec 2010')
    extent = extents.from_cell_coords(-30,120.5)

    sim = OnDemandSimulator(awral,input_map.mapping) #,omapping=output_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    ### this should be true
    assert np.isnan(i['solar_f']).any()

    insert_climatology(input_map)
    sim = OnDemandSimulator(awral,input_map.mapping) #,omapping=output_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    assert not np.isnan(i['solar_f']).any()

@with_setup(setup)
def test_climatology_region():
    import numpy as np
    from awrams.utils import extents
    from awrams.utils import datetools as dt
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    ### test region
    period = dt.dates('dec 2010')
    extent = extents.from_boundary_coords(-32,115,-35,118)

    sim = OnDemandSimulator(awral,input_map.mapping) #,omapping=output_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    ### this should be true
    assert np.isnan(i['solar_f']).any()

    insert_climatology(input_map)
    sim = OnDemandSimulator(awral,input_map.mapping) #,omapping=output_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    assert not np.isnan(i['solar_f']).any()

# @nottest
@with_setup(setup)
def test_initial_states_point():
    import numpy as np

    from awrams.utils import extents
    from awrams.utils import datetools as dt

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral


    period = dt.dates('dec 2010')

    ### test a single cell
    extent = extents.from_cell_coords(-30,120.5)

    ### simulation with default initial states
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_default = r['final_states']

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with default states simulation
    ### should be different
    for k,o in outputs_init.items():
        assert not o == outputs_default[k]

    ### save initial states to compare
    ini_states = {}
    for k in i:
        try:
            if k.startswith('init'):
                ini_states[k] = i[k]
        except:
            pass

    ### simulation with initial states read from dict
    get_initial_states_dict(input_map,period,extent)
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init_dict = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert o == outputs_init[k]

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass

# @nottest
@with_setup(setup)
def test_initial_states_region():
    import numpy as np

    from awrams.utils import extents
    from awrams.utils import datetools as dt

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral


    period = dt.dates('dec 2010')
    ### test a region
    extent = extents.from_boundary_offset(400,170,407,177)
    print(extent)

    ### simulation with default initial states
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_default = r['final_states']

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with default states simulation
    ### should be different
    for k,o in outputs_init.items():
        assert not (o == outputs_default[k]).any()

    ### save initial states to compare
    ini_states = {}
    for k in i:
        try:
            if k.startswith('init'):
                ini_states[k] = i[k]
        except:
            pass

    ### simulation with initial states read from dict
    get_initial_states_dict(input_map,period,extent)
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init_dict = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert (o == outputs_init[k]).any()

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass


    extent = extents.from_boundary_coords(-30,120.5,-30.35,120.85)
    print(extent)

    ### simulation with initial states read from nc files
    get_initial_states(input_map)
    sim = OnDemandSimulator(awral,input_map.mapping)
    r,i = sim.run(period,extent,return_inputs=True)
    outputs_init = r['final_states']

    ### compare final states with other ini states simulation
    ### should be same
    for k,o in outputs_init_dict.items():
        assert (o == outputs_init[k]).any()

    ### compare initial states from both methods
    ### should be same
    for k in i:
        try:
            if k.startswith('init'):
                assert ini_states[k] == i[k]
        except:
            pass

@with_setup(setup,tear_down)
def test_server():
    from awrams.utils import extents
    from awrams.simulation.server import Server
    from awrams.models import awral

    extent = extents.from_boundary_coords(-32,115,-35,118)

    get_initial_states_dict(input_map,period,extent)
    insert_climatology(input_map)

    sim = Server(awral)
    sim.run(input_map,output_map,period,extent)


if __name__ == '__main__':
    # test_ondemand_point()
    setup()
    # test_climatology()
    # test_initial_states_point()
    # test_initial_states_region()
    test_server()
