from nose.tools import nottest
import os

# @nottest
def test_OutputNode():
    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()
    output_map = awral.get_default_output_mapping()

    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map.mapping)

    print(runner.outputs)

# @nottest
def test_SplitFileWriterNode():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.utils.nodegraph import nodes
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()

    from awrams.utils.nodegraph import nodes
    from awrams.utils.metatypes import ObjectDict

    # output_path = './'
    mapping = {}
    mapping['qtot'] = nodes.write_to_annual_ncfile('./','qtot')

    output_map = ObjectDict(mapping=ObjectDict(mapping)) #,output_path=output_path)

    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map.mapping)

    period = dt.dates('2010-2011')
    extent = extents.from_cell_offset(200,200)
    r = runner.run(period,extent)

# @nottest
def test_output_graph_processing_flatfm():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()
    output_map = {
        's0_save': nodes.write_to_ncfile(os.path.dirname(__file__),'s0',extent=extents.default())
        }
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map)

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2012')
    extent = extents.from_boundary_offset(200,200,201,201)
    r = runner.run(period,extent)

    output_map = {
        's0_save': nodes.write_to_ncfile(os.path.dirname(__file__),'s0',mode='r+')
        }
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2013-2014')
    extent = extents.from_boundary_offset(200,200,201,201)
    r = runner.run(period,extent)

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2015')
    extent = extents.from_cell_offset(202,202)
    r = runner.run(period,extent)

# @nottest
def test_output_graph_processing_splitfm_A():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()
    output_map = {
        's0_save': nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
        }
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map)

    print("RUNNER NEW: multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = extents.from_boundary_offset(200,200,201,201)
    r = runner.run(period,extent)

def test_output_graph_processing_splitfm_B():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()
    output_map = {
        's0_save': nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0',mode='w')
        }
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map)

    print("RUNNER NEW (FILES EXISTING): multiple cells, multiple years")
    period = dt.dates('2010-2011')
    extent = extents.from_boundary_offset(200,200,201,201)
    r = runner.run(period,extent)

    print("RUNNER OLD (FILES EXISTING): single cell, single year")
    period = dt.dates('2015')
    extent = extents.from_cell_offset(202,202)
    r = runner.run(period,extent)


def test_output_graph_processing_splitfm_C():
    from awrams.utils import extents
    from awrams.utils import datetools as dt

    import awrams.models.awral.description
    awrams.models.awral.description.CLIMATE_DATA = os.path.join(os.path.dirname(__file__),'..','..','test_data','simulation')

    from awrams.utils.nodegraph import nodes,graph
    from awrams.simulation.ondemand import OnDemandSimulator
    from awrams.models import awral

    input_map = awral.get_default_mapping()
    output_map = {
        's0_save': nodes.write_to_annual_ncfile(os.path.dirname(__file__),'s0')
        }
    # outputs = graph.OutputGraph(output_map)
    runner = OnDemandSimulator(awral,input_map.mapping,omapping=output_map)

    print("RUNNER NEW: single cell ncf, multiple years")
    period = dt.dates('2010-2011')
    extent = extents.from_cell_offset(202,202)
    r = runner.run(period,extent)


def build_output_graph():
    from awrams.utils.nodegraph import nodes,graph

    from awrams.models import awral
    from awrams.models.awral import ffi_wrapper as fw
    from awrams.models.awral.template import DEFAULT_TEMPLATE

    output_map = awral.get_output_nodes(DEFAULT_TEMPLATE)
    print(output_map)

    output_map.mapping.update({
        's0_avg': nodes.transform(nodes.average,['s0_dr','s0_sr']),
        's0_avg_save': nodes.write_to_annual_ncfile('./','s0_avg')
        })
    outputs = graph.OutputGraph(output_map.mapping)
    print(outputs.get_dataspecs())
    print(outputs.get_dataspecs(flat=True))
    return outputs


if __name__ == '__main__':
    # test_FileWriterNode()
    # test_SplitFileWriterNode()
    # test_output_graph_processing_flatfm()
    # test_output_graph_processing_splitfm()
    # test_output_graph()
    # test_OutputNode()
    pass
